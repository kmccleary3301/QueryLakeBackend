from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_gates import (
    compute_p95_latency_regression,
    compute_run_write_coverage,
    percentile,
)


def test_percentile_linear_interpolation():
    assert abs(percentile([1, 2, 3, 4], 0.5) - 2.5) < 1e-12
    assert abs(percentile([1, 2, 3, 4], 0.95) - 3.85) < 1e-12


def test_compute_run_write_coverage_uses_expected_requests():
    runs = [{"run_id": "r1"}, {"run_id": "r2"}, {"run_id": ""}]
    report = compute_run_write_coverage(runs, expected_requests=4)
    assert report["expected_requests"] == 4
    assert report["observed_run_rows"] == 2
    assert abs(report["coverage_ratio"] - 0.5) < 1e-12
    assert report["meets_gate_g0_1"] is False


def test_compute_p95_latency_regression():
    baseline = [{"timings": {"total": 1.0}}, {"timings": {"total": 2.0}}, {"timings": {"total": 3.0}}]
    candidate = [{"timings": {"total": 1.1}}, {"timings": {"total": 2.1}}, {"timings": {"total": 3.1}}]
    report = compute_p95_latency_regression(baseline_runs=baseline, candidate_runs=candidate)
    assert report["baseline_count"] == 3
    assert report["candidate_count"] == 3
    assert report["candidate_over_baseline_ratio"] > 1.0
    assert report["meets_gate_g0_3"] is True

