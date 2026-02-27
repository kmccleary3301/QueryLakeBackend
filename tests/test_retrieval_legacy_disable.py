from types import SimpleNamespace
from pathlib import Path
import sys
import asyncio

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.api import search as search_api


class _DummyDB:
    def rollback(self):
        return None


def test_search_bm25_document_chunk_routes_orchestrated_path(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    class _FakeOrchestrator:
        async def run(self, **kwargs):
            return SimpleNamespace(
                candidates=[],
                pipeline_id="orchestrated.search_bm25.document_chunk",
                pipeline_version="v1",
                traces=[],
            )

    monkeypatch.setattr(search_api, "PipelineOrchestrator", lambda: _FakeOrchestrator())

    rows = search_api.search_bm25(
        database=db,
        auth={"username": "tester", "password_prehash": "x"},
        query="boiler pressure",
        collection_ids=["col1"],
        limit=5,
        table="document_chunk",
    )
    assert rows == []


def test_search_hybrid_routes_orchestrated_path(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    class _FakeOrchestrator:
        async def run(self, **kwargs):
            return SimpleNamespace(
                candidates=[],
                pipeline_id="orchestrated.search_hybrid",
                pipeline_version="v1",
                traces=[],
            )

    monkeypatch.setattr(search_api, "PipelineOrchestrator", lambda: _FakeOrchestrator())

    async def _run():
        return await search_api.search_hybrid(
            database=db,
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "q", "embedding": "q"},
            collection_ids=["col1"],
            limit_bm25=4,
            limit_similarity=4,
            bm25_weight=0.7,
            similarity_weight=0.3,
            group_chunks=False,
        )

    result = asyncio.run(_run())
    assert isinstance(result, dict)
    assert result["rows"] == []
