from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_explain import build_retrieval_plan_explain
from QueryLake.typing.retrieval_primitives import RetrievalPipelineSpec, RetrievalPipelineStage


def test_build_retrieval_plan_explain_resolves_rrf_effective_fusion():
    pipeline = RetrievalPipelineSpec(
        pipeline_id="orchestrated.search_hybrid",
        version="v2",
        stages=[
            RetrievalPipelineStage(stage_id="bm25", primitive_id="BM25RetrieverParadeDB"),
            RetrievalPipelineStage(stage_id="dense", primitive_id="DenseRetrieverPGVector"),
        ],
        flags={"fusion_primitive": "WeightedScoreFusion", "fusion_normalization": "minmax"},
    )

    explained = build_retrieval_plan_explain(
        route="search_hybrid",
        pipeline=pipeline,
        options={"fusion_primitive": "rrf", "rrf_k": 80, "limit_bm25": 8, "limit_similarity": 12},
        pipeline_resolution={"source": "binding"},
    )

    assert explained["pipeline"]["source"] == "binding"
    assert explained["effective"]["fusion"]["primitive"] == "RRFusion"
    assert explained["effective"]["fusion"]["rrf_k"] == 80
    assert explained["effective"]["limits"]["limit_bm25"] == 8

