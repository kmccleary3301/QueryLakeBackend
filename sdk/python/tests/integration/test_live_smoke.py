from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

from querylake_sdk import QueryLakeClient


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
    kwargs = {"base_url": base_url}
    if oauth2:
        kwargs["oauth2"] = oauth2
    if api_key:
        kwargs["api_key"] = api_key
    return QueryLakeClient(**kwargs)


def test_live_health_ready_and_models() -> None:
    _require_enabled()
    client = _build_client()
    try:
        health = client.healthz()
        ready = client.readyz()
        models = client.list_models()
        assert isinstance(health, dict)
        assert isinstance(ready, dict)
        assert isinstance(models, dict)
    finally:
        client.close()


def test_live_hybrid_search_smoke() -> None:
    _require_enabled()
    collection_id = _env("QUERYLAKE_LIVE_TEST_COLLECTION_ID")
    if not collection_id:
        pytest.skip("Missing QUERYLAKE_LIVE_TEST_COLLECTION_ID for live search smoke.")
    query = _env("QUERYLAKE_LIVE_TEST_QUERY") or "boiler pressure limits"

    client = _build_client()
    try:
        rows = client.search_hybrid_with_metrics(
            query=query,
            collection_ids=[collection_id],
            limit=3,
            limit_bm25=6,
            limit_similarity=6,
            limit_sparse=0,
            bm25_weight=0.6,
            similarity_weight=0.4,
            sparse_weight=0.0,
        )
        assert isinstance(rows, dict)
        assert "rows" in rows
    finally:
        client.close()


def test_live_upload_delete_smoke() -> None:
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
            payload = client.upload_document(
                file_path=path,
                collection_hash_id=collection_id,
                scan_text=True,
                create_embeddings=False,
                create_sparse_embeddings=False,
                await_embedding=False,
                document_metadata={"source": "sdk_live_smoke"},
            )
            assert isinstance(payload, dict)
            doc_id = str(payload.get("hash_id") or "")
            assert doc_id
        finally:
            if doc_id:
                client.delete_document(document_hash_id=doc_id)
    client.close()
