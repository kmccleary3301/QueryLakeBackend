#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_gates import compute_p95_latency_regression


def _window_rows(count: int, mean: float, jitter: float, *, seed: int) -> List[Dict]:
    rng = random.Random(seed)
    rows = []
    for i in range(max(0, int(count))):
        value = max(0.001, rng.gauss(mean, jitter))
        rows.append({"run_id": f"r{i}", "timings": {"total": value}, "status": "ok"})
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Simulate sustained retrieval SLO windows and gate stability.")
    parser.add_argument("--days", type=int, default=7)
    parser.add_argument("--requests-per-day", type=int, default=120)
    parser.add_argument("--baseline-latency", type=float, default=0.30)
    parser.add_argument("--candidate-latency", type=float, default=0.31)
    parser.add_argument("--jitter", type=float, default=0.015)
    parser.add_argument("--seed", type=int, default=20260222)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    daily = []
    all_ok = True
    for day in range(max(1, int(args.days))):
        baseline = _window_rows(
            args.requests_per_day,
            args.baseline_latency,
            args.jitter,
            seed=args.seed + day * 101 + 1,
        )
        candidate = _window_rows(
            args.requests_per_day,
            args.candidate_latency,
            args.jitter,
            seed=args.seed + day * 101 + 2,
        )
        gate = compute_p95_latency_regression(baseline_runs=baseline, candidate_runs=candidate)
        day_report = {
            "day_index": day + 1,
            "baseline_count": len(baseline),
            "candidate_count": len(candidate),
            "p95_ratio": gate["candidate_over_baseline_ratio"],
            "meets_g0_3": gate["meets_gate_g0_3"],
        }
        if not day_report["meets_g0_3"]:
            all_ok = False
        daily.append(day_report)

    payload = {
        "days": len(daily),
        "requests_per_day": int(args.requests_per_day),
        "daily": daily,
        "sustained_window_pass": all_ok,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sustained_window_pass": all_ok}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

