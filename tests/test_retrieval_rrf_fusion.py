from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_primitives_legacy import RRFusion, WeightedScoreFusion
from QueryLake.typing.retrieval_primitives import RetrievalCandidate, RetrievalRequest


def test_rrf_fusion_combines_sources_and_ranks():
    fusion = RRFusion(k=60)
    request = RetrievalRequest(query_text="question", options={"fusion_weights": {"bm25": 1.0, "dense": 1.0}})

    bm25 = [
        RetrievalCandidate(content_id="a", text="A", stage_scores={"bm25_score": 0.9}, stage_ranks={"bm25": 1}, provenance=["bm25"]),
        RetrievalCandidate(content_id="b", text="B", stage_scores={"bm25_score": 0.8}, stage_ranks={"bm25": 2}, provenance=["bm25"]),
    ]
    dense = [
        RetrievalCandidate(content_id="b", text="B", stage_scores={"similarity_score": 0.95}, stage_ranks={"dense": 1}, provenance=["dense"]),
        RetrievalCandidate(content_id="c", text="C", stage_scores={"similarity_score": 0.7}, stage_ranks={"dense": 2}, provenance=["dense"]),
    ]

    fused = fusion.fuse(request, {"bm25": bm25, "dense": dense})
    assert len(fused) == 3
    assert fused[0].content_id == "b"
    assert "bm25" in fused[0].provenance and "dense" in fused[0].provenance
    assert "rrf_fused" in fused[0].stage_scores


def test_weighted_score_fusion_minmax_applies_source_weights():
    fusion = WeightedScoreFusion(default_normalization="minmax")
    request = RetrievalRequest(
        query_text="question",
        options={
            "fusion_weights": {"bm25": 0.25, "dense": 0.75},
            "fusion_score_keys": {"bm25": "bm25_score", "dense": "similarity_score"},
        },
    )

    bm25 = [
        RetrievalCandidate(content_id="a", text="A", stage_scores={"bm25_score": 100.0}, provenance=["bm25"]),
        RetrievalCandidate(content_id="b", text="B", stage_scores={"bm25_score": 90.0}, provenance=["bm25"]),
    ]
    dense = [
        RetrievalCandidate(content_id="b", text="B", stage_scores={"similarity_score": 0.90}, provenance=["dense"]),
        RetrievalCandidate(content_id="c", text="C", stage_scores={"similarity_score": 0.50}, provenance=["dense"]),
    ]

    fused = fusion.fuse(request, {"bm25": bm25, "dense": dense})
    assert [row.content_id for row in fused] == ["b", "a", "c"]
    assert "weighted_fused" in fused[0].stage_scores


def test_weighted_score_fusion_rank_fallback_for_missing_scores():
    fusion = WeightedScoreFusion(default_normalization="minmax")
    request = RetrievalRequest(
        query_text="question",
        options={"fusion_weights": {"bm25": 1.0, "dense": 1.0}, "fusion_normalization": "rank"},
    )

    bm25 = [
        RetrievalCandidate(content_id="a", text="A", stage_scores={}, provenance=["bm25"]),
        RetrievalCandidate(content_id="b", text="B", stage_scores={}, provenance=["bm25"]),
    ]
    dense = [
        RetrievalCandidate(content_id="b", text="B", stage_scores={}, provenance=["dense"]),
        RetrievalCandidate(content_id="c", text="C", stage_scores={}, provenance=["dense"]),
    ]

    fused = fusion.fuse(request, {"bm25": bm25, "dense": dense})
    assert len(fused) == 3
    assert fused[0].content_id == "b"
    assert fused[0].stage_ranks["weighted_fused"] == 1
