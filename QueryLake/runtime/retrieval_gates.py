from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional


def _extract_total_latency_seconds(run_row: Dict[str, Any]) -> Optional[float]:
    timings = run_row.get("timings")
    if not isinstance(timings, dict):
        return None
    value = timings.get("total")
    if isinstance(value, (int, float)) and float(value) >= 0.0:
        return float(value)
    return None


def percentile(values: Iterable[float], q: float) -> float:
    points = sorted(float(v) for v in values if isinstance(v, (int, float)))
    if len(points) == 0:
        return 0.0
    q = max(0.0, min(1.0, float(q)))
    if len(points) == 1:
        return points[0]
    idx = q * (len(points) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return points[lo]
    alpha = idx - lo
    return points[lo] + alpha * (points[hi] - points[lo])


def compute_run_write_coverage(
    run_rows: List[Dict[str, Any]],
    *,
    expected_requests: Optional[int] = None,
) -> Dict[str, Any]:
    total_runs = len(run_rows)
    expected = int(expected_requests) if expected_requests is not None else total_runs
    expected = max(0, expected)
    observed = sum(1 for row in run_rows if isinstance(row.get("run_id"), str) and len(row["run_id"]) > 0)
    coverage = 0.0 if expected == 0 else (float(observed) / float(expected))
    return {
        "expected_requests": expected,
        "observed_run_rows": observed,
        "coverage_ratio": coverage,
        "meets_gate_g0_1": coverage >= 0.95,
    }


def compute_p95_latency_regression(
    *,
    baseline_runs: List[Dict[str, Any]],
    candidate_runs: List[Dict[str, Any]],
) -> Dict[str, Any]:
    baseline_latencies = [
        value for value in (_extract_total_latency_seconds(row) for row in baseline_runs) if value is not None
    ]
    candidate_latencies = [
        value for value in (_extract_total_latency_seconds(row) for row in candidate_runs) if value is not None
    ]
    baseline_p95 = percentile(baseline_latencies, 0.95)
    candidate_p95 = percentile(candidate_latencies, 0.95)
    ratio = 1.0 if baseline_p95 <= 0.0 else (candidate_p95 / baseline_p95)
    return {
        "baseline_count": len(baseline_latencies),
        "candidate_count": len(candidate_latencies),
        "baseline_p95_seconds": baseline_p95,
        "candidate_p95_seconds": candidate_p95,
        "candidate_over_baseline_ratio": ratio,
        "meets_gate_g0_3": ratio < 1.10,
    }

