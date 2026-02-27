#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_readiness import evaluate_rollout_readiness


def _run(cmd: list[str]) -> tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _read_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_rows(path: Path) -> list[dict[str, Any]]:
    try:
        payload = _read_json(path)
    except Exception:
        return []
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    return []


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


def _derive_latency_windows_from_all_runs(
    all_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    baseline = [
        row
        for row in all_rows
        if str(row.get("pipeline_id") or "").startswith("legacy.")
    ]
    candidate = [
        row
        for row in all_rows
        if str(row.get("pipeline_id") or "").startswith("orchestrated.")
    ]
    return baseline, candidate


def _slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(value).strip() or "route")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an autonomous retrieval acceptance cycle and emit readiness report.")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs-output", type=Path, default=Path("docs_tmp/RAG/retrieval_runs_cycle_latest.json"))
    parser.add_argument("--gate-output", type=Path, default=Path("docs_tmp/RAG/retrieval_gate_report_cycle_latest.json"))
    parser.add_argument("--smoke-output", type=Path, default=Path("docs_tmp/RAG/retrieval_eval_smoke_cycle_latest.json"))
    parser.add_argument("--nightly-output", type=Path, default=Path("docs_tmp/RAG/retrieval_eval_nightly_cycle_latest.json"))
    parser.add_argument("--baseline-runs", type=Path, default=None)
    parser.add_argument("--candidate-runs", type=Path, default=None)
    parser.add_argument(
        "--baseline-route",
        type=str,
        default="stress_over_baseline",
        help="Fallback route used to export baseline latency runs when --baseline-runs is not provided.",
    )
    parser.add_argument(
        "--candidate-route",
        type=str,
        default="stress_over_candidate",
        help="Fallback route used to export candidate latency runs when --candidate-runs is not provided.",
    )
    parser.add_argument(
        "--baseline-runs-output",
        type=Path,
        default=None,
        help="Auto-export path for baseline latency runs.",
    )
    parser.add_argument(
        "--candidate-runs-output",
        type=Path,
        default=None,
        help="Auto-export path for candidate latency runs.",
    )
    parser.add_argument("--expected-requests", type=int, default=None)
    parser.add_argument("--skip-pytest", action="store_true")
    args = parser.parse_args()

    if args.baseline_runs_output is None:
        args.baseline_runs_output = Path(f"docs_tmp/RAG/retrieval_runs_{_slug(args.baseline_route)}_cycle_latest.json")
    if args.candidate_runs_output is None:
        args.candidate_runs_output = Path(f"docs_tmp/RAG/retrieval_runs_{_slug(args.candidate_route)}_cycle_latest.json")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.runs_output.parent.mkdir(parents=True, exist_ok=True)
    args.gate_output.parent.mkdir(parents=True, exist_ok=True)
    args.smoke_output.parent.mkdir(parents=True, exist_ok=True)
    args.nightly_output.parent.mkdir(parents=True, exist_ok=True)
    args.baseline_runs_output.parent.mkdir(parents=True, exist_ok=True)
    args.candidate_runs_output.parent.mkdir(parents=True, exist_ok=True)

    cycle: Dict[str, Any] = {"commands": {}, "artifacts": {}}

    tests_green = True
    if not args.skip_pytest:
        rc, out, err = _run(["pytest", "-q"])
        cycle["commands"]["pytest_q"] = {"rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
        tests_green = rc == 0

    rc, out, err = _run(
        [
            "python",
            "scripts/retrieval_eval.py",
            "--mode",
            "smoke",
            "--output",
            str(args.smoke_output),
        ]
    )
    cycle["commands"]["retrieval_eval_smoke"] = {"rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
    eval_smoke_pass = rc == 0

    rc, out, err = _run(
        [
            "python",
            "scripts/retrieval_eval.py",
            "--mode",
            "nightly",
            "--output",
            str(args.nightly_output),
        ]
    )
    cycle["commands"]["retrieval_eval_nightly"] = {"rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
    eval_nightly_pass = rc == 0

    rc, out, err = _run(
        [
            "python",
            "scripts/export_retrieval_runs.py",
            "--output",
            str(args.runs_output),
            "--limit",
            "5000",
        ]
    )
    cycle["commands"]["export_retrieval_runs"] = {"rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}

    baseline_runs = args.baseline_runs
    candidate_runs = args.candidate_runs
    baseline_count = 0
    candidate_count = 0

    # If explicit latency windows were not provided, derive them automatically.
    if baseline_runs is None or candidate_runs is None:
        if baseline_runs is None:
            baseline_runs = args.baseline_runs_output
            rc, out, err = _run(
                [
                    "python",
                    "scripts/export_retrieval_runs.py",
                    "--output",
                    str(baseline_runs),
                    "--route",
                    str(args.baseline_route),
                    "--limit",
                    "5000",
                ]
            )
            cycle["commands"]["export_baseline_runs"] = {"rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
        if candidate_runs is None:
            candidate_runs = args.candidate_runs_output
            rc, out, err = _run(
                [
                    "python",
                    "scripts/export_retrieval_runs.py",
                    "--output",
                    str(candidate_runs),
                    "--route",
                    str(args.candidate_route),
                    "--limit",
                    "5000",
                ]
            )
            cycle["commands"]["export_candidate_runs"] = {
                "rc": rc,
                "stdout_tail": out[-2000:],
                "stderr_tail": err[-2000:],
            }

    baseline_rows = _load_rows(baseline_runs) if baseline_runs is not None else []
    candidate_rows = _load_rows(candidate_runs) if candidate_runs is not None else []
    baseline_count = len(baseline_rows)
    candidate_count = len(candidate_rows)

    # Secondary fallback: split the all-runs export into legacy/orchestrated latency windows.
    if baseline_count == 0 or candidate_count == 0:
        all_rows = _load_rows(args.runs_output)
        fallback_baseline, fallback_candidate = _derive_latency_windows_from_all_runs(all_rows)
        if baseline_count == 0 and len(fallback_baseline) > 0:
            baseline_runs = args.baseline_runs_output
            baseline_rows = fallback_baseline
            _write_rows(baseline_runs, baseline_rows)
            baseline_count = len(baseline_rows)
        if candidate_count == 0 and len(fallback_candidate) > 0:
            candidate_runs = args.candidate_runs_output
            candidate_rows = fallback_candidate
            _write_rows(candidate_runs, candidate_rows)
            candidate_count = len(candidate_rows)

    gate_cmd = [
        "python",
        "scripts/retrieval_gate_report.py",
        "--coverage-runs",
        str(args.runs_output),
        "--output",
        str(args.gate_output),
    ]
    if args.expected_requests is not None:
        gate_cmd += ["--expected-requests", str(int(args.expected_requests))]
    if baseline_runs is not None and candidate_runs is not None and baseline_count > 0 and candidate_count > 0:
        gate_cmd += ["--baseline-runs", str(baseline_runs), "--candidate-runs", str(candidate_runs)]

    rc, out, err = _run(gate_cmd)
    cycle["commands"]["retrieval_gate_report"] = {"rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}

    gate_payload = _read_json(args.gate_output) if args.gate_output.exists() else {}
    readiness = evaluate_rollout_readiness(
        tests_green=tests_green,
        eval_smoke_pass=eval_smoke_pass,
        eval_nightly_pass=eval_nightly_pass,
        g0_1=gate_payload.get("g0_1"),
        g0_3=gate_payload.get("g0_3"),
    )

    cycle["artifacts"] = {
        "smoke_eval": str(args.smoke_output),
        "nightly_eval": str(args.nightly_output),
        "runs_export": str(args.runs_output),
        "gate_report": str(args.gate_output),
        "baseline_runs": str(baseline_runs) if baseline_runs is not None else None,
        "candidate_runs": str(candidate_runs) if candidate_runs is not None else None,
        "baseline_run_count": baseline_count,
        "candidate_run_count": candidate_count,
    }
    cycle["readiness"] = readiness
    args.output.write_text(json.dumps(cycle, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "go_for_canary": readiness["go_for_canary"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
