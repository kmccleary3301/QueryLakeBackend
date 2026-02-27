import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_orchestrator import PipelineOrchestrator
from QueryLake.typing.retrieval_primitives import (
    RetrievalCandidate,
    RetrievalPipelineSpec,
    RetrievalPipelineStage,
    RetrievalRequest,
)
from QueryLake.runtime.retrieval_primitives_legacy import RRFusion


class _DummyRetriever:
    def __init__(self, primitive_id: str, rows):
        self.primitive_id = primitive_id
        self.version = "v1"
        self._rows = rows

    async def retrieve(self, request: RetrievalRequest):
        return self._rows


class _DummyReranker:
    primitive_id = "DummyReranker"
    version = "v1"

    async def rerank(self, request: RetrievalRequest, candidates):
        return sorted(candidates, key=lambda c: c.content_id, reverse=True)


class _CaptureReranker:
    primitive_id = "CaptureReranker"
    version = "v1"

    def __init__(self):
        self.seen_ids = []

    async def rerank(self, request: RetrievalRequest, candidates):
        self.seen_ids = [candidate.content_id for candidate in candidates]
        return candidates


def test_pipeline_orchestrator_runs_retrieve_fuse_rerank():
    req = RetrievalRequest(query_text="q", options={"limit": 3})
    pipeline = RetrievalPipelineSpec(
        pipeline_id="p1",
        version="v1",
        stages=[
            RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25"),
            RetrievalPipelineStage(stage_id="dense", primitive_id="Dense"),
        ],
    )
    bm25_rows = [
        RetrievalCandidate(content_id="a", text="A", provenance=["bm25"], stage_ranks={"bm25": 1}),
        RetrievalCandidate(content_id="b", text="B", provenance=["bm25"], stage_ranks={"bm25": 2}),
    ]
    dense_rows = [
        RetrievalCandidate(content_id="b", text="B", provenance=["dense"], stage_ranks={"dense": 1}),
        RetrievalCandidate(content_id="c", text="C", provenance=["dense"], stage_ranks={"dense": 2}),
    ]

    orchestrator = PipelineOrchestrator()
    result = asyncio.run(
        orchestrator.run(
            request=req,
            pipeline=pipeline,
            retrievers={
                "bm25": _DummyRetriever("BM25", bm25_rows),
                "dense": _DummyRetriever("Dense", dense_rows),
            },
            fusion=RRFusion(),
            reranker=_DummyReranker(),
        )
    )

    assert result.pipeline_id == "p1"
    assert len(result.candidates) == 3
    assert len(result.traces) >= 3
    assert any(trace.stage.startswith("retrieve:") for trace in result.traces)


def test_pipeline_orchestrator_applies_acl_filter_before_rerank():
    req = RetrievalRequest(query_text="q", collection_ids=["allowed"], options={"limit": 10})
    pipeline = RetrievalPipelineSpec(
        pipeline_id="p_acl",
        version="v1",
        stages=[
            RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25"),
            RetrievalPipelineStage(stage_id="dense", primitive_id="Dense"),
        ],
    )
    bm25_rows = [
        RetrievalCandidate(content_id="a", text="A", metadata={"collection_id": "allowed"}, provenance=["bm25"]),
        RetrievalCandidate(content_id="b", text="B", metadata={"collection_id": "denied"}, provenance=["bm25"]),
    ]
    dense_rows = [
        RetrievalCandidate(content_id="c", text="C", metadata={"collection_id": "denied"}, provenance=["dense"]),
    ]
    reranker = _CaptureReranker()

    orchestrator = PipelineOrchestrator()
    result = asyncio.run(
        orchestrator.run(
            request=req,
            pipeline=pipeline,
            retrievers={
                "bm25": _DummyRetriever("BM25", bm25_rows),
                "dense": _DummyRetriever("Dense", dense_rows),
            },
            fusion=RRFusion(),
            reranker=reranker,
        )
    )

    assert reranker.seen_ids == ["a"]
    assert [row.content_id for row in result.candidates] == ["a"]
    assert any(trace.stage.startswith("acl:") for trace in result.traces)


def test_pipeline_orchestrator_emits_policy_preflight_trace():
    req = RetrievalRequest(query_text="q", options={"limit": 2})
    pipeline = RetrievalPipelineSpec(
        pipeline_id="p_preflight",
        version="v1",
        stages=[RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25")],
    )
    orchestrator = PipelineOrchestrator()
    result = asyncio.run(
        orchestrator.run(
            request=req,
            pipeline=pipeline,
            retrievers={"bm25": _DummyRetriever("BM25", [])},
            fusion=RRFusion(),
        )
    )
    preflight = [trace for trace in result.traces if trace.stage == "policy_preflight"]
    assert len(preflight) == 1
    assert preflight[0].details["valid"] is True


def test_pipeline_orchestrator_policy_preflight_can_fail_fast():
    req = RetrievalRequest(
        query_text="q",
        options={"limit": -1, "enforce_policy_validation": True},
    )
    pipeline = RetrievalPipelineSpec(
        pipeline_id="p_preflight_fail",
        version="v1",
        stages=[RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25")],
    )
    orchestrator = PipelineOrchestrator()
    try:
        asyncio.run(
            orchestrator.run(
                request=req,
                pipeline=pipeline,
                retrievers={"bm25": _DummyRetriever("BM25", [])},
            )
        )
        assert False, "expected ValueError"
    except ValueError as exc:
        assert "limit_negative" in str(exc)
