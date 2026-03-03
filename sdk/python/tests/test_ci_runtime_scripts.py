from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
PROFILE_SCRIPT = REPO_ROOT / "scripts" / "dev" / "ci_runtime_profile.py"
DELTA_SCRIPT = REPO_ROOT / "scripts" / "dev" / "ci_runtime_delta.py"


def _run(script: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(script), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_ci_runtime_profile_from_input_payload(tmp_path: Path) -> None:
    runs_payload = {
        "workflow_runs": [
            {
                "id": 101,
                "name": "SDK Checks",
                "event": "push",
                "status": "completed",
                "conclusion": "success",
                "run_attempt": 1,
                "created_at": "2026-03-02T10:00:00Z",
                "run_started_at": "2026-03-02T10:00:15Z",
                "updated_at": "2026-03-02T10:05:15Z",
                "html_url": "https://example.com/101",
            },
            {
                "id": 102,
                "name": "SDK Checks",
                "event": "pull_request",
                "status": "completed",
                "conclusion": "failure",
                "run_attempt": 2,
                "created_at": "2026-03-02T11:00:00Z",
                "run_started_at": "2026-03-02T11:00:30Z",
                "updated_at": "2026-03-02T11:03:30Z",
                "html_url": "https://example.com/102",
            },
        ]
    }
    input_json = tmp_path / "runs.json"
    out_json = tmp_path / "report.json"
    out_md = tmp_path / "report.md"
    input_json.write_text(json.dumps(runs_payload), encoding="utf-8")

    result = _run(
        PROFILE_SCRIPT,
        "--repo",
        "org/repo",
        "--days",
        "7",
        "--input-runs-json",
        str(input_json),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    )
    assert result.returncode == 0, result.stderr
    report = json.loads(out_json.read_text(encoding="utf-8"))
    assert report["overall"]["run_count"] == 2
    assert "SDK Checks" in report["by_workflow"]
    assert report["by_workflow"]["SDK Checks"]["rerun_rate"] == 0.5
    assert "CI Runtime Profile" in out_md.read_text(encoding="utf-8")


def test_ci_runtime_delta_regression_gate(tmp_path: Path) -> None:
    before = {
        "generated_at_utc": "2026-03-01T00:00:00+00:00",
        "overall": {
            "duration_s": {"p95": 100.0},
            "queue_s": {"p95": 20.0},
            "compute_minutes_total": 40.0,
        },
        "by_workflow": {},
    }
    after = {
        "generated_at_utc": "2026-03-02T00:00:00+00:00",
        "overall": {
            "duration_s": {"p95": 130.0},
            "queue_s": {"p95": 25.0},
            "compute_minutes_total": 55.0,
        },
        "by_workflow": {},
    }
    before_json = tmp_path / "before.json"
    after_json = tmp_path / "after.json"
    out_md = tmp_path / "delta.md"
    before_json.write_text(json.dumps(before), encoding="utf-8")
    after_json.write_text(json.dumps(after), encoding="utf-8")

    result = _run(
        DELTA_SCRIPT,
        "--before-json",
        str(before_json),
        "--after-json",
        str(after_json),
        "--out-md",
        str(out_md),
        "--max-p95-regression-pct",
        "15",
        "--max-compute-regression-pct",
        "20",
        "--fail-on-regression",
    )
    assert result.returncode == 2
    gate = json.loads(result.stdout)
    assert gate["status"] == "fail"
    assert "CI Runtime Delta Report" in out_md.read_text(encoding="utf-8")
