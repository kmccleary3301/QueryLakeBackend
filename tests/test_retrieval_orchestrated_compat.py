import asyncio
from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.api import search as search_api
from QueryLake.typing.retrieval_primitives import (
    RetrievalCandidate,
    RetrievalExecutionResult,
    RetrievalPipelineSpec,
    RetrievalPipelineStage,
    RetrievalStageTrace,
)


class _DummyDB:
    def rollback(self):
        return None


def test_search_bm25_orchestrated_path_emits_stage_trace(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_ORCHESTRATOR_BM25", "1")

    async def _fake_run(self, **kwargs):
        return RetrievalExecutionResult(
            pipeline_id="orchestrated.search_bm25.document_chunk",
            pipeline_version="v1",
            candidates=[
                RetrievalCandidate(
                    content_id="c1",
                    text="chunk text",
                    metadata={
                        "document_id": "d1",
                        "document_name": "doc",
                        "document_chunk_number": 0,
                        "collection_id": "col1",
                        "creation_timestamp": 1.0,
                        "collection_type": "user",
                        "md": {},
                        "document_md": {},
                    },
                    stage_scores={"bm25_score": 0.8},
                    stage_ranks={"bm25": 1},
                    provenance=["bm25"],
                )
            ],
            traces=[RetrievalStageTrace(stage="retrieve:bm25", duration_ms=2.0)],
            metadata={},
        )

    monkeypatch.setattr(search_api.PipelineOrchestrator, "run", _fake_run)
    monkeypatch.setattr(search_api.metrics, "record_retrieval", lambda **kwargs: None)
    logged = {"kwargs": None}
    monkeypatch.setattr(search_api, "log_retrieval_run", lambda *args, **kwargs: logged.update({"kwargs": kwargs}))

    rows = search_api.search_bm25(
        database=db,
        auth={"username": "tester", "password_prehash": "x"},
        query="boiler pressure",
        collection_ids=["col1"],
        limit=5,
        table="document_chunk",
    )

    assert len(rows) == 1
    assert logged["kwargs"]["pipeline_id"] == "orchestrated.search_bm25.document_chunk"
    assert "stage_trace" in logged["kwargs"]["md"]


def test_search_hybrid_orchestrated_path_emits_stage_trace(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_ORCHESTRATOR_HYBRID", "1")

    async def _fake_run(self, **kwargs):
        return RetrievalExecutionResult(
            pipeline_id="orchestrated.search_hybrid",
            pipeline_version="v1",
            candidates=[
                RetrievalCandidate(
                    content_id="c1",
                    text="hybrid text",
                    metadata={
                        "document_id": "d1",
                        "document_name": "doc",
                        "document_chunk_number": 0,
                        "collection_id": "col1",
                        "creation_timestamp": 1.0,
                        "collection_type": "user",
                        "md": {},
                        "document_md": {},
                    },
                    stage_scores={"bm25_score": 0.6, "similarity_score": 0.7, "weighted_fused": 0.65},
                    stage_ranks={"bm25": 1, "dense": 1, "weighted_fused": 1},
                    provenance=["bm25", "dense"],
                )
            ],
            traces=[RetrievalStageTrace(stage="fusion", duration_ms=3.0)],
            metadata={},
        )

    monkeypatch.setattr(search_api.PipelineOrchestrator, "run", _fake_run)
    monkeypatch.setattr(search_api.metrics, "record_retrieval", lambda **kwargs: None)
    logged = {"kwargs": None}
    monkeypatch.setattr(search_api, "log_retrieval_run", lambda *args, **kwargs: logged.update({"kwargs": kwargs}))

    result = asyncio.run(
        search_api.search_hybrid(
            database=db,
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "q", "embedding": "q"},
            embedding=[0.0] * 1024,
            collection_ids=["col1"],
            limit_bm25=4,
            limit_similarity=4,
            bm25_weight=0.7,
            similarity_weight=0.3,
            group_chunks=False,
        )
    )

    assert "rows" in result and len(result["rows"]) == 1
    assert logged["kwargs"]["pipeline_id"] == "orchestrated.search_hybrid"
    assert "stage_trace" in logged["kwargs"]["md"]


def test_search_hybrid_orchestrated_path_can_return_plan_explain(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_ORCHESTRATOR_HYBRID", "1")

    async def _fake_run(self, **kwargs):
        return RetrievalExecutionResult(
            pipeline_id="orchestrated.search_hybrid",
            pipeline_version="v1",
            candidates=[],
            traces=[RetrievalStageTrace(stage="fusion", duration_ms=1.0)],
            metadata={},
        )

    monkeypatch.setattr(search_api.PipelineOrchestrator, "run", _fake_run)
    monkeypatch.setattr(search_api.metrics, "record_retrieval", lambda **kwargs: None)
    monkeypatch.setattr(search_api, "log_retrieval_run", lambda *args, **kwargs: None)

    result = asyncio.run(
        search_api.search_hybrid(
            database=db,
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "q", "embedding": "q", "explain_plan": True},
            embedding=[0.0] * 1024,
            collection_ids=["col1"],
            limit_bm25=4,
            limit_similarity=4,
            bm25_weight=0.7,
            similarity_weight=0.3,
            group_chunks=False,
        )
    )

    assert "plan_explain" in result
    assert result["plan_explain"]["pipeline"]["pipeline_id"] == "orchestrated.search_hybrid"
    assert result["plan_explain"]["effective"]["fusion"]["primitive"] == "WeightedScoreFusion"


def test_search_hybrid_orchestrated_pre_resolves_dense_embedding(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_ORCHESTRATOR_HYBRID", "1")

    pipeline = RetrievalPipelineSpec(
        pipeline_id="orchestrated.search_hybrid",
        version="v1",
        stages=[RetrievalPipelineStage(stage_id="dense", primitive_id="DenseRetrieverPGVector")],
    )
    monkeypatch.setattr(
        search_api,
        "_resolve_route_pipeline",
        lambda *args, **kwargs: (pipeline, {"source": "test"}),
    )

    captured = {"request": None}

    async def _fake_run(self, **kwargs):
        captured["request"] = kwargs.get("request")
        return RetrievalExecutionResult(
            pipeline_id="orchestrated.search_hybrid",
            pipeline_version="v1",
            candidates=[],
            traces=[RetrievalStageTrace(stage="retrieve:dense", duration_ms=1.0)],
            metadata={},
        )

    embedding_calls = {"count": 0}

    def _toolchain(name):
        async def _fn(auth, payload):
            if name == "embedding":
                embedding_calls["count"] += 1
                return [[float(i) for i in range(1024)]]
            raise AssertionError(f"Unexpected toolchain call: {name}")
        return _fn

    monkeypatch.setattr(search_api.PipelineOrchestrator, "run", _fake_run)
    monkeypatch.setattr(search_api.metrics, "record_retrieval", lambda **kwargs: None)
    monkeypatch.setattr(search_api, "log_retrieval_run", lambda *args, **kwargs: None)

    result = asyncio.run(
        search_api.search_hybrid(
            database=db,
            toolchain_function_caller=_toolchain,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "q", "embedding": "q"},
            embedding=None,
            collection_ids=["col1"],
            limit_bm25=4,
            limit_similarity=4,
            bm25_weight=0.7,
            similarity_weight=0.3,
            group_chunks=False,
        )
    )

    assert result["rows"] == []
    assert embedding_calls["count"] == 1
    assert captured["request"] is not None
    assert isinstance(captured["request"].query_embedding, list)
    assert len(captured["request"].query_embedding) == 1024


def test_search_hybrid_orchestrated_pre_resolves_sparse_value(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_ORCHESTRATOR_HYBRID", "1")

    pipeline = RetrievalPipelineSpec(
        pipeline_id="orchestrated.search_hybrid",
        version="v1",
        stages=[RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25RetrieverParadeDB")],
    )
    monkeypatch.setattr(
        search_api,
        "_resolve_route_pipeline",
        lambda *args, **kwargs: (pipeline, {"source": "test"}),
    )

    captured = {"request": None}

    async def _fake_run(self, **kwargs):
        captured["request"] = kwargs.get("request")
        return RetrievalExecutionResult(
            pipeline_id="orchestrated.search_hybrid",
            pipeline_version="v1",
            candidates=[],
            traces=[RetrievalStageTrace(stage="retrieve:sparse", duration_ms=1.0)],
            metadata={},
        )

    sparse_calls = {"count": 0}

    def _toolchain(name):
        async def _fn(auth, payload):
            if name == "embedding_sparse":
                sparse_calls["count"] += 1
                return [{"indices": [2, 7], "values": [0.9, 0.35], "dimensions": 1024}]
            raise AssertionError(f"Unexpected toolchain call: {name}")
        return _fn

    monkeypatch.setattr(search_api.PipelineOrchestrator, "run", _fake_run)
    monkeypatch.setattr(search_api.metrics, "record_retrieval", lambda **kwargs: None)
    monkeypatch.setattr(search_api, "log_retrieval_run", lambda *args, **kwargs: None)

    result = asyncio.run(
        search_api.search_hybrid(
            database=db,
            toolchain_function_caller=_toolchain,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "q", "sparse": "q"},
            embedding_sparse=None,
            collection_ids=["col1"],
            limit_bm25=4,
            limit_similarity=0,
            limit_sparse=4,
            bm25_weight=0.6,
            similarity_weight=0.0,
            sparse_weight=0.4,
            use_sparse=True,
            use_similarity=False,
            sparse_embedding_function="embedding_sparse",
            group_chunks=False,
        )
    )

    assert result["rows"] == []
    assert sparse_calls["count"] == 1
    assert captured["request"] is not None
    assert captured["request"].options.get("sparse_query_value") is not None


def test_search_file_chunks_orchestrated_path_emits_stage_trace(monkeypatch):
    db = _DummyDB()
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    pipeline = RetrievalPipelineSpec(
        pipeline_id="orchestrated.search_file_chunks",
        version="v1",
        stages=[RetrievalPipelineStage(stage_id="file_bm25", primitive_id="FileChunkBM25RetrieverSQL")],
    )
    monkeypatch.setattr(
        search_api,
        "_resolve_route_pipeline",
        lambda *args, **kwargs: (pipeline, {"source": "test"}),
    )

    async def _fake_run(self, **kwargs):
        return RetrievalExecutionResult(
            pipeline_id="orchestrated.search_file_chunks",
            pipeline_version="v1",
            candidates=[
                RetrievalCandidate(
                    content_id="fc_1",
                    text="file chunk",
                    metadata={
                        "md": {"page": 1},
                        "created_at": 10.0,
                        "file_version_id": "fv_1",
                    },
                    stage_scores={"bm25_score": 0.88},
                    stage_ranks={"bm25": 1},
                    provenance=["bm25"],
                )
            ],
            traces=[RetrievalStageTrace(stage="retrieve:file_bm25", duration_ms=2.0)],
            metadata={},
        )

    monkeypatch.setattr(search_api.PipelineOrchestrator, "run", _fake_run)
    monkeypatch.setattr(search_api.metrics, "record_retrieval", lambda **kwargs: None)
    logged = {"kwargs": None}
    monkeypatch.setattr(search_api, "log_retrieval_run", lambda *args, **kwargs: logged.update({"kwargs": kwargs}))

    result = search_api.search_file_chunks(
        database=db,
        auth={"username": "tester", "password_prehash": "x"},
        query="compressor fault",
        limit=5,
        offset=0,
    )

    assert "results" in result and len(result["results"]) == 1
    assert result["results"][0]["id"] == "fc_1"
    assert logged["kwargs"]["pipeline_id"] == "orchestrated.search_file_chunks"
    assert "stage_trace" in logged["kwargs"]["md"]
