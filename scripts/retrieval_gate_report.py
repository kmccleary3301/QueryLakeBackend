#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_gates import (
    compute_p95_latency_regression,
    compute_run_write_coverage,
)


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        rows = payload.get("rows")
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON shape in {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compute Phase 0 retrieval acceptance gates from run exports.")
    parser.add_argument("--coverage-runs", type=Path, required=True, help="JSON array (or {rows:[...]}) of run rows.")
    parser.add_argument("--expected-requests", type=int, default=None, help="Expected request count for coverage.")
    parser.add_argument("--baseline-runs", type=Path, default=None, help="Baseline run export for latency regression.")
    parser.add_argument("--candidate-runs", type=Path, default=None, help="Candidate run export for latency regression.")
    parser.add_argument("--output", type=Path, default=None, help="Optional output path for JSON report.")
    args = parser.parse_args()

    coverage_rows = _load_rows(args.coverage_runs)
    report: Dict[str, Any] = {
        "g0_1": compute_run_write_coverage(
            coverage_rows,
            expected_requests=args.expected_requests,
        ),
    }

    if args.baseline_runs is not None and args.candidate_runs is not None:
        report["g0_3"] = compute_p95_latency_regression(
            baseline_runs=_load_rows(args.baseline_runs),
            candidate_runs=_load_rows(args.candidate_runs),
        )

    output_text = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(output_text + "\n")
    else:
        print(output_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
