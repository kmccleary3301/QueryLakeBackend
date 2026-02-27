from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_parity import (
    mean_reciprocal_rank,
    parity_metrics,
    parity_metrics_from_cases,
    reciprocal_rank,
    topk_overlap_ratio,
)


def test_topk_overlap_ratio():
    overlap = topk_overlap_ratio(["a", "b", "c"], ["b", "a", "x"], 2)
    assert overlap == 1.0


def test_parity_metrics_computes_latency_ratio():
    metrics = parity_metrics(
        chunk_results=["a", "b", "c"],
        segment_results=["b", "a", "d"],
        k=2,
        chunk_latency_ms=100.0,
        segment_latency_ms=130.0,
    )
    assert metrics["topk_overlap"] == 1.0
    assert abs(metrics["latency_ratio"] - 1.3) < 1e-9


def test_reciprocal_rank_and_mrr():
    assert reciprocal_rank(["a", "b", "c"], {"b"}) == 0.5
    assert reciprocal_rank(["a", "b", "c"], {"x"}) == 0.0
    mrr = mean_reciprocal_rank(
        cases=[
            {"query": "q1", "expected_ids": ["a"]},
            {"query": "q2", "expected_ids": ["z"]},
        ],
        retrievals_by_query={"q1": ["b", "a"], "q2": ["z", "x"]},
    )
    assert abs(mrr - ((0.5 + 1.0) / 2.0)) < 1e-9


def test_parity_metrics_from_cases_includes_mrr_delta():
    metrics = parity_metrics_from_cases(
        cases=[
            {"query": "q1", "expected_ids": ["a"]},
            {"query": "q2", "expected_ids": ["z"]},
        ],
        chunk_retrievals_by_query={"q1": ["a", "b"], "q2": ["y", "z"]},
        segment_retrievals_by_query={"q1": ["b", "a"], "q2": ["z", "y"]},
        k=2,
        chunk_latency_ms=100.0,
        segment_latency_ms=120.0,
    )
    assert metrics["mode"] == "query_set"
    assert metrics["query_count"] == 2
    assert abs(metrics["topk_overlap_mean"] - 1.0) < 1e-9
    assert abs(metrics["chunk_mrr"] - 0.75) < 1e-9
    assert abs(metrics["segment_mrr"] - 0.75) < 1e-9
    assert abs(metrics["latency_ratio"] - 1.2) < 1e-9
