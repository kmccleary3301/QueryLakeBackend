import asyncio
from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.api import search as search_api


class _DummyDB:
    def __init__(self, run):
        self._run = run

    def get(self, model, run_id):
        if run_id == self._run.run_id:
            return self._run
        return None


def test_replay_retrieval_run_hybrid(monkeypatch):
    run = SimpleNamespace(
        run_id="run_h_1",
        route="search_hybrid",
        pipeline_id="orchestrated.search_hybrid",
        pipeline_version="v1",
        query_text='{"bm25":"boiler pressure","embedding":"boiler pressure"}',
        filters={"collection_ids": ["c1"], "web_search": False},
        budgets={"limit_bm25": 6, "limit_similarity": 4, "rerank_enabled": True},
        counters={"bm25_weight": 0.7, "similarity_weight": 0.3, "group_chunks": True},
    )
    db = _DummyDB(run)
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    called = {"kwargs": None}

    async def _fake_search_hybrid(**kwargs):
        called["kwargs"] = kwargs
        return {"rows": []}

    monkeypatch.setattr(search_api, "search_hybrid", _fake_search_hybrid)
    result = asyncio.run(
        search_api.replay_retrieval_run(
            database=db,
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            run_id="run_h_1",
        )
    )
    assert result == {"rows": []}
    assert called["kwargs"]["limit_bm25"] == 6
    assert called["kwargs"]["limit_similarity"] == 4
    assert called["kwargs"]["rerank"] is True
    assert called["kwargs"]["_pipeline_override"]["pipeline_id"] == "orchestrated.search_hybrid"
    assert called["kwargs"]["_pipeline_override"]["pipeline_version"] == "v1"


def test_replay_retrieval_run_bm25(monkeypatch):
    run = SimpleNamespace(
        run_id="run_b_1",
        route="search_bm25.document_chunk",
        pipeline_id="orchestrated.search_bm25.document_chunk",
        pipeline_version="v1",
        query_text="steam limits",
        filters={"collection_ids": ["c2"], "table": "document_chunk", "sort_by": "score", "sort_dir": "DESC"},
        budgets={"limit": 12, "offset": 3},
        counters={"group_chunks": False},
    )
    db = _DummyDB(run)
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    called = {"kwargs": None}

    def _fake_search_bm25(**kwargs):
        called["kwargs"] = kwargs
        return []

    monkeypatch.setattr(search_api, "search_bm25", _fake_search_bm25)
    result = asyncio.run(
        search_api.replay_retrieval_run(
            database=db,
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            run_id="run_b_1",
        )
    )
    assert result == []
    assert called["kwargs"]["limit"] == 12
    assert called["kwargs"]["offset"] == 3
    assert called["kwargs"]["table"] == "document_chunk"
    assert called["kwargs"]["_pipeline_override"]["pipeline_id"] == "orchestrated.search_bm25.document_chunk"
    assert called["kwargs"]["_pipeline_override"]["pipeline_version"] == "v1"


def test_replay_retrieval_run_file_chunks(monkeypatch):
    run = SimpleNamespace(
        run_id="run_f_1",
        route="search_file_chunks",
        pipeline_id="orchestrated.search_file_chunks",
        pipeline_version="v1",
        query_text="compressor alarm",
        filters={"sort_by": "score", "sort_dir": "DESC"},
        budgets={"limit": 9, "offset": 2},
        counters={},
    )
    db = _DummyDB(run)
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    called = {"kwargs": None}

    def _fake_search_file_chunks(**kwargs):
        called["kwargs"] = kwargs
        return {"results": []}

    monkeypatch.setattr(search_api, "search_file_chunks", _fake_search_file_chunks)
    result = asyncio.run(
        search_api.replay_retrieval_run(
            database=db,
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            run_id="run_f_1",
        )
    )
    assert result == {"results": []}
    assert called["kwargs"]["limit"] == 9
    assert called["kwargs"]["offset"] == 2
    assert called["kwargs"]["_pipeline_override"]["pipeline_id"] == "orchestrated.search_file_chunks"
    assert called["kwargs"]["_pipeline_override"]["pipeline_version"] == "v1"
