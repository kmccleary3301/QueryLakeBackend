#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List


def _p95(values: List[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int((len(ordered) - 1) * 0.95)
    return ordered[idx]


def _load_json(path: str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate frozen Chandra closure gates from artifacts.")
    parser.add_argument("--benchmark-json", required=True, help="Output JSON from chandra_benchmark_pdf.py.")
    parser.add_argument("--quality-json", required=True, help="Output JSON from chandra_quality_compare.py.")
    parser.add_argument("--warm-max-seconds", type=float, default=30.0)
    parser.add_argument("--warm-p95-max-seconds", type=float, default=33.0)
    parser.add_argument("--require-quality-verdict", default="pass")
    parser.add_argument("--out-json", default=None)
    args = parser.parse_args()

    benchmark = _load_json(args.benchmark_json)
    quality = _load_json(args.quality_json)

    passes = benchmark.get("passes", [])
    warm_runs = [float(item.get("wall_seconds", 0.0)) for item in passes if str(item.get("name", "")).startswith("warm_")]
    if not warm_runs:
        raise RuntimeError("Benchmark artifact does not contain warm_* passes.")

    warm_min = min(warm_runs)
    warm_max = max(warm_runs)
    warm_mean = statistics.mean(warm_runs)
    warm_median = statistics.median(warm_runs)
    warm_p95 = _p95(warm_runs)

    quality_verdict = str((quality.get("recommendation") or {}).get("verdict", "")).strip().lower()
    gate_warm = warm_max <= float(args.warm_max_seconds)
    gate_p95 = warm_p95 <= float(args.warm_p95_max_seconds)
    gate_quality = quality_verdict == str(args.require_quality_verdict).strip().lower()

    result = {
        "benchmark_json": args.benchmark_json,
        "quality_json": args.quality_json,
        "warm_runs_seconds": warm_runs,
        "warm_summary": {
            "count": len(warm_runs),
            "min": round(warm_min, 4),
            "max": round(warm_max, 4),
            "mean": round(warm_mean, 4),
            "median": round(warm_median, 4),
            "p95": round(warm_p95, 4),
        },
        "quality_verdict": quality_verdict,
        "gates": {
            "warm_leq_threshold": {
                "threshold_seconds": float(args.warm_max_seconds),
                "pass": bool(gate_warm),
            },
            "warm_p95_leq_threshold": {
                "threshold_seconds": float(args.warm_p95_max_seconds),
                "pass": bool(gate_p95),
            },
            "quality_verdict_match": {
                "required": str(args.require_quality_verdict).strip().lower(),
                "pass": bool(gate_quality),
            },
        },
    }
    result["overall_pass"] = bool(gate_warm and gate_p95 and gate_quality)

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.out_json:
        out_path = Path(args.out_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
