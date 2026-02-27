from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_promotion import PromotionThresholdPolicy, gate_promotion


def test_gate_promotion_allows_when_thresholds_pass():
    policy = PromotionThresholdPolicy(
        quality_min_delta=0.01,
        p95_latency_max_ratio=1.10,
        cost_max_ratio=1.10,
        min_queries=10,
        quality_metric_key="mrr",
        latency_metric_key="p95_latency_seconds",
        cost_metric_key="avg_cost_usd",
    )
    result = gate_promotion(
        baseline_metrics={"mrr": 0.65, "p95_latency_seconds": 1.0, "avg_cost_usd": 0.010},
        candidate_metrics={"mrr": 0.68, "p95_latency_seconds": 1.05, "avg_cost_usd": 0.0105},
        query_count=24,
        policy=policy,
    )
    assert result["allowed"] is True
    assert result["reasons"] == []


def test_gate_promotion_rejects_with_explicit_reasons():
    policy = PromotionThresholdPolicy(min_queries=20)
    result = gate_promotion(
        baseline_metrics={"mrr": 0.70, "p95_latency_seconds": 1.0, "avg_cost_usd": 0.010},
        candidate_metrics={"mrr": 0.60, "p95_latency_seconds": 1.4, "avg_cost_usd": 0.015},
        query_count=5,
        policy=policy,
    )
    assert result["allowed"] is False
    assert any("query_count" in reason for reason in result["reasons"])
    assert any("quality_delta" in reason for reason in result["reasons"])
    assert any("latency_ratio" in reason for reason in result["reasons"])
    assert any("cost_ratio" in reason for reason in result["reasons"])
