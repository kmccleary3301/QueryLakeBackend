from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_reports import build_experiment_report


def test_build_experiment_report_returns_json_and_markdown():
    report_json, report_md = build_experiment_report(
        experiment_id="exp_1",
        baseline_rows=[{"mrr": 0.6, "p95_latency_seconds": 1.1}],
        candidate_rows=[{"mrr": 0.7, "p95_latency_seconds": 1.2}],
        delta_rows=[{"mrr": 0.1, "p95_latency_seconds": 0.1}],
    )
    assert report_json["experiment_id"] == "exp_1"
    assert "baseline_avg" in report_json and "candidate_avg" in report_json
    assert "Retrieval Experiment Report: exp_1" in report_md
