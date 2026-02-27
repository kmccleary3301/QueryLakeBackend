from __future__ import annotations

from typing import Any, Dict, List, Tuple

from pydantic import BaseModel, Field


class PromotionThresholdPolicy(BaseModel):
    quality_min_delta: float = Field(default=0.0, description="Minimum required quality uplift.")
    p95_latency_max_ratio: float = Field(default=1.10, description="Candidate/baseline latency ratio upper bound.")
    cost_max_ratio: float = Field(default=1.10, description="Candidate/baseline cost ratio upper bound.")
    min_queries: int = Field(default=50, ge=1)
    quality_metric_key: str = Field(default="mrr")
    latency_metric_key: str = Field(default="p95_latency_seconds")
    cost_metric_key: str = Field(default="avg_cost_usd")


def evaluate_promotion(
    *,
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    query_count: int,
    policy: PromotionThresholdPolicy,
) -> Tuple[bool, List[str], Dict[str, float]]:
    reasons: List[str] = []
    diagnostics: Dict[str, float] = {}

    if query_count < policy.min_queries:
        reasons.append(f"query_count {query_count} below min_queries {policy.min_queries}")

    base_quality = float(baseline_metrics.get(policy.quality_metric_key, 0.0))
    cand_quality = float(candidate_metrics.get(policy.quality_metric_key, 0.0))
    quality_delta = cand_quality - base_quality
    diagnostics["quality_delta"] = quality_delta
    if quality_delta < policy.quality_min_delta:
        reasons.append(
            f"quality_delta {quality_delta:.6f} below minimum {policy.quality_min_delta:.6f}"
        )

    base_latency = float(baseline_metrics.get(policy.latency_metric_key, 0.0))
    cand_latency = float(candidate_metrics.get(policy.latency_metric_key, 0.0))
    latency_ratio = 1.0 if base_latency <= 0 else (cand_latency / base_latency)
    diagnostics["latency_ratio"] = latency_ratio
    if latency_ratio > policy.p95_latency_max_ratio:
        reasons.append(
            f"latency_ratio {latency_ratio:.6f} exceeds max {policy.p95_latency_max_ratio:.6f}"
        )

    base_cost = float(baseline_metrics.get(policy.cost_metric_key, 0.0))
    cand_cost = float(candidate_metrics.get(policy.cost_metric_key, 0.0))
    cost_ratio = 1.0 if base_cost <= 0 else (cand_cost / base_cost)
    diagnostics["cost_ratio"] = cost_ratio
    if cost_ratio > policy.cost_max_ratio:
        reasons.append(f"cost_ratio {cost_ratio:.6f} exceeds max {policy.cost_max_ratio:.6f}")

    return (len(reasons) == 0), reasons, diagnostics


def gate_promotion(
    *,
    baseline_metrics: Dict[str, Any],
    candidate_metrics: Dict[str, Any],
    query_count: int,
    policy: PromotionThresholdPolicy,
) -> Dict[str, Any]:
    passed, reasons, diagnostics = evaluate_promotion(
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        query_count=query_count,
        policy=policy,
    )
    return {
        "allowed": passed,
        "reasons": reasons,
        "diagnostics": diagnostics,
        "policy": policy.model_dump(),
    }
