from __future__ import annotations

import os
import tempfile
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest

from querylake_sdk import QueryLakeClient
from querylake_sdk.errors import QueryLakeHTTPStatusError, QueryLakeTransportError


pytestmark = pytest.mark.integration_live
_DEFAULT_CASES_PATH = Path(__file__).with_name("live_query_cases.json")


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def _env_bool(name: str, default: bool = False) -> bool:
    value = _env(name).lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _require_enabled() -> None:
    if _env("QUERYLAKE_LIVE_ENABLE") != "1":
        pytest.skip("Live integration tests are disabled (set QUERYLAKE_LIVE_ENABLE=1).")


def _build_client() -> QueryLakeClient:
    base_url = _env("QUERYLAKE_LIVE_BASE_URL")
    oauth2 = _env("QUERYLAKE_LIVE_OAUTH2")
    api_key = _env("QUERYLAKE_LIVE_API_KEY")
    if not base_url:
        pytest.skip("Missing QUERYLAKE_LIVE_BASE_URL.")
    if not oauth2 and not api_key:
        pytest.skip("Missing QUERYLAKE_LIVE_OAUTH2 / QUERYLAKE_LIVE_API_KEY.")
    timeout_seconds = float(_env("QUERYLAKE_LIVE_TIMEOUT_SECONDS") or "20")
    kwargs = {"base_url": base_url, "timeout_seconds": timeout_seconds}
    if oauth2:
        kwargs["oauth2"] = oauth2
    if api_key:
        kwargs["api_key"] = api_key
    return QueryLakeClient(**kwargs)


def _diagnostics_dir() -> Path:
    value = _env("QUERYLAKE_LIVE_DIAGNOSTICS_DIR") or "docs_tmp/RAG/ci/live_integration"
    path = Path(value).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _extract_request_id_from_headers(headers: Dict[str, str]) -> str:
    for key in ("x-request-id", "x-correlation-id", "request-id"):
        value = (headers.get(key) or "").strip()
        if value:
            return value
    return ""


def _extract_request_id_from_payload(payload: Any) -> str:
    if not isinstance(payload, dict):
        return ""
    # QueryLake routes may include request correlation in metadata envelopes.
    for key in ("request_id", "requestId", "trace_id"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    md = payload.get("md")
    if isinstance(md, dict):
        for key in ("request_id", "trace_id"):
            value = md.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _record_diag(
    test_name: str,
    op: str,
    started_at: float,
    ok: bool,
    note: str = "",
    *,
    request_id: str = "",
    status_code: int | None = None,
) -> None:
    payload = {
        "test": test_name,
        "op": op,
        "ok": bool(ok),
        "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
        "note": note,
        "request_id": request_id,
        "status_code": status_code,
        "run_namespace": _env("QUERYLAKE_LIVE_RUN_NAMESPACE") or "",
    }
    with (_diagnostics_dir() / "live_metrics.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _call_with_retry(test_name: str, op: str, fn):
    attempts = int(_env("QUERYLAKE_LIVE_RETRY_ATTEMPTS") or "3")
    delay_seconds = float(_env("QUERYLAKE_LIVE_RETRY_DELAY_SECONDS") or "1")
    for attempt in range(1, max(1, attempts) + 1):
        started = time.perf_counter()
        try:
            result = fn()
            _record_diag(test_name, f"{op}#{attempt}", started, True)
            return result
        except QueryLakeTransportError as exc:
            _record_diag(test_name, f"{op}#{attempt}", started, False, f"transport:{exc}")
            if attempt >= attempts:
                raise
            time.sleep(delay_seconds)
        except QueryLakeHTTPStatusError as exc:
            _record_diag(test_name, f"{op}#{attempt}", started, False, f"http:{exc.status_code}")
            if attempt >= attempts or exc.status_code < 500:
                raise
            time.sleep(delay_seconds)


def _call_json_route_with_retry(
    test_name: str,
    op: str,
    client: QueryLakeClient,
    method: str,
    path: str,
    *,
    json_payload: Dict[str, Any] | None = None,
) -> Tuple[Any, str]:
    attempts = int(_env("QUERYLAKE_LIVE_RETRY_ATTEMPTS") or "3")
    delay_seconds = float(_env("QUERYLAKE_LIVE_RETRY_DELAY_SECONDS") or "1")
    for attempt in range(1, max(1, attempts) + 1):
        started = time.perf_counter()
        try:
            response = client._request(method, path, json=json_payload)  # noqa: SLF001
            payload = response.json()
            request_id = _extract_request_id_from_headers(dict(response.headers))
            if not request_id:
                request_id = _extract_request_id_from_payload(payload)
            _record_diag(
                test_name,
                f"{op}#{attempt}",
                started,
                True,
                request_id=request_id,
                status_code=response.status_code,
            )
            return payload, request_id
        except QueryLakeTransportError as exc:
            _record_diag(test_name, f"{op}#{attempt}", started, False, f"transport:{exc}")
            if attempt >= attempts:
                raise
            time.sleep(delay_seconds)
        except QueryLakeHTTPStatusError as exc:
            _record_diag(test_name, f"{op}#{attempt}", started, False, f"http:{exc.status_code}")
            if attempt >= attempts or exc.status_code < 500:
                raise
            time.sleep(delay_seconds)
    raise RuntimeError("unreachable retry loop")


def _row_blob_text(row: Dict[str, Any]) -> str:
    pieces: List[str] = []
    for key in (
        "text",
        "content",
        "document_title",
        "document_name",
        "chunk",
        "chunk_text",
        "title",
    ):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            pieces.append(value.strip().lower())
    return "\n".join(pieces)


def _load_query_cases() -> List[Dict[str, Any]]:
    cases_path = _env("QUERYLAKE_LIVE_QUERY_CASES_PATH")
    path = Path(cases_path).expanduser() if cases_path else _DEFAULT_CASES_PATH
    if not path.exists():
        query = _env("QUERYLAKE_LIVE_TEST_QUERY") or "boiler pressure limits"
        return [{"name": "fallback", "query": query, "min_rows": 1, "expected_terms_any": []}]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        pytest.skip(f"Unable to parse query cases file: {path} ({exc})")
    if not isinstance(data, dict) or not isinstance(data.get("cases"), list):
        pytest.skip(f"Query cases file must contain object with 'cases' list: {path}")
    cases: List[Dict[str, Any]] = []
    for idx, raw in enumerate(data.get("cases", []), start=1):
        if not isinstance(raw, dict):
            continue
        query = str(raw.get("query") or "").strip()
        if not query:
            continue
        terms = raw.get("expected_terms_any") or []
        if not isinstance(terms, list):
            terms = []
        cases.append(
            {
                "name": str(raw.get("name") or f"case_{idx}"),
                "query": query,
                "min_rows": int(raw.get("min_rows") or 1),
                "expected_terms_any": [str(term).strip().lower() for term in terms if str(term).strip()],
            }
        )
    return cases


def test_live_health_ready_and_models() -> None:
    test_name = "test_live_health_ready_and_models"
    _require_enabled()
    client = _build_client()
    try:
        health, health_req_id = _call_json_route_with_retry(test_name, "healthz", client, "GET", "/healthz")
        ready, ready_req_id = _call_json_route_with_retry(test_name, "readyz", client, "GET", "/readyz")
        models, models_req_id = _call_json_route_with_retry(test_name, "list_models", client, "GET", "/v1/models")
        assert isinstance(health, dict)
        assert isinstance(ready, dict)
        assert isinstance(models, dict)
        # Keep request-id visibility best-effort; some deployments may not emit one per route.
        assert isinstance(health_req_id, str)
        assert isinstance(ready_req_id, str)
        assert isinstance(models_req_id, str)
    finally:
        client.close()


def test_live_hybrid_search_smoke() -> None:
    test_name = "test_live_hybrid_search_smoke"
    _require_enabled()
    collection_id = _env("QUERYLAKE_LIVE_TEST_COLLECTION_ID")
    if not collection_id:
        pytest.skip("Missing QUERYLAKE_LIVE_TEST_COLLECTION_ID for live search smoke.")
    query_cases = _load_query_cases()
    strict_expectations = _env_bool("QUERYLAKE_LIVE_STRICT_EXPECTATIONS", default=False)

    client = _build_client()
    try:
        for case in query_cases:
            query = str(case.get("query") or "").strip()
            if not query:
                continue
            payload, request_id = _call_json_route_with_retry(
                test_name,
                f"search_hybrid:{case.get('name')}",
                client,
                "POST",
                "/api/search_hybrid",
                json_payload={
                    "query": query,
                    "collection_ids": [collection_id],
                    "limit": 3,
                    "limit_bm25": 6,
                    "limit_similarity": 6,
                    "limit_sparse": 0,
                    "bm25_weight": 0.6,
                    "similarity_weight": 0.4,
                    "sparse_weight": 0.0,
                },
            )
            assert isinstance(payload, dict)
            rows = payload.get("rows")
            assert isinstance(rows, list)
            assert len(rows) >= int(case.get("min_rows") or 1)

            expected_terms = case.get("expected_terms_any") or []
            if strict_expectations and expected_terms:
                blob = "\n".join(_row_blob_text(row) for row in rows if isinstance(row, dict))
                assert any(term in blob for term in expected_terms), (
                    f"Missing expected terms for case '{case.get('name')}': {expected_terms}"
                )
            assert isinstance(request_id, str)
    finally:
        client.close()


def test_live_upload_delete_smoke() -> None:
    test_name = "test_live_upload_delete_smoke"
    _require_enabled()
    if _env("QUERYLAKE_LIVE_ALLOW_WRITE") != "1":
        pytest.skip("Write-path live test disabled (set QUERYLAKE_LIVE_ALLOW_WRITE=1).")
    collection_id = _env("QUERYLAKE_LIVE_TEST_COLLECTION_ID")
    if not collection_id:
        pytest.skip("Missing QUERYLAKE_LIVE_TEST_COLLECTION_ID for write-path smoke.")

    client = _build_client()
    doc_id: str | None = None
    with tempfile.TemporaryDirectory(prefix="qlsdk_live_") as tmp:
        path = Path(tmp) / "live_smoke.txt"
        path.write_text("querylake live integration smoke payload", encoding="utf-8")
        try:
            payload = _call_with_retry(
                test_name,
                "upload_document",
                lambda: client.upload_document(
                    file_path=path,
                    collection_hash_id=collection_id,
                    scan_text=True,
                    create_embeddings=False,
                    create_sparse_embeddings=False,
                    await_embedding=False,
                    document_metadata={
                        "source": "sdk_live_smoke",
                        "run_namespace": _env("QUERYLAKE_LIVE_RUN_NAMESPACE") or "",
                    },
                ),
            )
            assert isinstance(payload, dict)
            doc_id = str(payload.get("hash_id") or "")
            assert doc_id
        finally:
            if doc_id:
                _call_with_retry(test_name, "delete_document", lambda: client.delete_document(document_hash_id=doc_id))
    client.close()
