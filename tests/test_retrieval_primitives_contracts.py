from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.typing.retrieval_primitives import (
    RetrievalCandidate,
    RetrievalExecutionResult,
    RetrievalPipelineSpec,
    RetrievalPipelineStage,
    RetrievalRequest,
    RetrievalStageTrace,
)


def test_retrieval_request_defaults():
    req = RetrievalRequest(query_text="boiler maintenance")
    assert req.route == "search"
    assert req.collection_ids == []
    assert req.filters == {}
    assert req.budgets == {}


def test_retrieval_pipeline_and_result_serialize():
    pipeline = RetrievalPipelineSpec(
        pipeline_id="orchestrated.search_hybrid",
        version="v1",
        stages=[
            RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25RetrieverParadeDB"),
            RetrievalPipelineStage(stage_id="fuse", primitive_id="RRFusion"),
        ],
        budgets={"limit": 20},
    )
    candidate = RetrievalCandidate(
        content_id="chunk_1",
        text="Sample chunk",
        stage_scores={"bm25": 0.4, "hybrid": 0.7},
        stage_ranks={"bm25": 1, "hybrid": 1},
        provenance=["bm25", "dense"],
    )
    trace = RetrievalStageTrace(stage="bm25", duration_ms=12.4, input_count=0, output_count=20)
    result = RetrievalExecutionResult(
        pipeline_id=pipeline.pipeline_id,
        pipeline_version=pipeline.version,
        candidates=[candidate],
        traces=[trace],
    )
    dump = result.model_dump()
    assert dump["pipeline_id"] == "orchestrated.search_hybrid"
    assert dump["candidates"][0]["content_id"] == "chunk_1"
    assert dump["traces"][0]["stage"] == "bm25"
