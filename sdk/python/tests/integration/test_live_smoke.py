from __future__ import annotations

import os
import tempfile
import time
import json
from pathlib import Path

import pytest

from querylake_sdk import QueryLakeClient
from querylake_sdk.errors import QueryLakeHTTPStatusError, QueryLakeTransportError


pytestmark = pytest.mark.integration_live


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


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


def _record_diag(test_name: str, op: str, started_at: float, ok: bool, note: str = "") -> None:
    payload = {
        "test": test_name,
        "op": op,
        "ok": bool(ok),
        "latency_ms": round((time.perf_counter() - started_at) * 1000.0, 2),
        "note": note,
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


def test_live_health_ready_and_models() -> None:
    test_name = "test_live_health_ready_and_models"
    _require_enabled()
    client = _build_client()
    try:
        health = _call_with_retry(test_name, "healthz", lambda: client.healthz())
        ready = _call_with_retry(test_name, "readyz", lambda: client.readyz())
        models = _call_with_retry(test_name, "list_models", lambda: client.list_models())
        assert isinstance(health, dict)
        assert isinstance(ready, dict)
        assert isinstance(models, dict)
    finally:
        client.close()


def test_live_hybrid_search_smoke() -> None:
    test_name = "test_live_hybrid_search_smoke"
    _require_enabled()
    collection_id = _env("QUERYLAKE_LIVE_TEST_COLLECTION_ID")
    if not collection_id:
        pytest.skip("Missing QUERYLAKE_LIVE_TEST_COLLECTION_ID for live search smoke.")
    query = _env("QUERYLAKE_LIVE_TEST_QUERY") or "boiler pressure limits"

    client = _build_client()
    try:
        rows = _call_with_retry(
            test_name,
            "search_hybrid_with_metrics",
            lambda: client.search_hybrid_with_metrics(
                query=query,
                collection_ids=[collection_id],
                limit=3,
                limit_bm25=6,
                limit_similarity=6,
                limit_sparse=0,
                bm25_weight=0.6,
                similarity_weight=0.4,
                sparse_weight=0.0,
            ),
        )
        assert isinstance(rows, dict)
        assert "rows" in rows
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
