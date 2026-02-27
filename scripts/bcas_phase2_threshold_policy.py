#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _q(values: List[float], q: float, default: float = 0.0) -> float:
    if not values:
        return default
    if len(values) == 1:
        return float(values[0])
    ordered = sorted(float(v) for v in values)
    idx = (len(ordered) - 1) * q
    lo = int(idx)
    hi = min(lo + 1, len(ordered) - 1)
    frac = idx - lo
    return float((ordered[lo] * (1.0 - frac)) + (ordered[hi] * frac))


def _collect_gate_rows(nightly_root: Path, max_days: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not nightly_root.exists():
        return rows
    day_dirs = sorted([d for d in nightly_root.iterdir() if d.is_dir()])[-max(1, int(max_days)) :]
    for day in day_dirs:
        gate_path = day / f"BCAS_PHASE2_OPERATOR_GATE_{day.name}.json"
        payload = _safe_load(gate_path)
        deltas = payload.get("deltas", {}) if isinstance(payload.get("deltas"), dict) else {}
        rows.append(
            {
                "date": day.name,
                "gate_path": str(gate_path),
                "present": gate_path.exists(),
                "gate_ok": bool(payload.get("gate_ok", False)),
                "overall_pass_delta": float(deltas.get("overall_pass_rate", 0.0) or 0.0),
                "exact_pass_delta": float(deltas.get("exact_phrase_pass_rate", 0.0) or 0.0),
                "overall_latency_delta_ms": float(deltas.get("overall_avg_latency_ms", 0.0) or 0.0),
                "exact_latency_delta_ms": float(deltas.get("exact_phrase_avg_latency_ms", 0.0) or 0.0),
            }
        )
    return rows


def _recommend(rows: List[Dict[str, Any]], min_days: int) -> Dict[str, Any]:
    valid = [r for r in rows if r.get("present")]
    ok_rows = [r for r in valid if r.get("gate_ok")]

    overall_pass = [float(r["overall_pass_delta"]) for r in ok_rows]
    exact_pass = [float(r["exact_pass_delta"]) for r in ok_rows]
    overall_lat = [float(r["overall_latency_delta_ms"]) for r in ok_rows]
    exact_lat = [float(r["exact_latency_delta_ms"]) for r in ok_rows]

    # Tighten conservatively: never require negative pass deltas, and cap latency
    # regression at observed p75 plus a small safety margin.
    min_overall_pass_delta = max(0.0, _q(overall_pass, 0.25, 0.0))
    min_exact_pass_delta = max(0.0, _q(exact_pass, 0.25, 0.0))
    max_overall_latency_regression_ms = max(2.0, _q(overall_lat, 0.75, 5.0) + 2.0)
    max_exact_latency_regression_ms = max(2.0, _q(exact_lat, 0.75, 5.0) + 2.0)

    readiness = len(valid) >= int(min_days) and len(ok_rows) == len(valid)

    return {
        "ready": readiness,
        "window_days": len(rows),
        "valid_days": len(valid),
        "gate_ok_days": len(ok_rows),
        "recommended_thresholds": {
            "min_overall_pass_delta": float(round(min_overall_pass_delta, 6)),
            "min_exact_pass_delta": float(round(min_exact_pass_delta, 6)),
            "max_overall_latency_regression_ms": float(round(max_overall_latency_regression_ms, 3)),
            "max_exact_latency_regression_ms": float(round(max_exact_latency_regression_ms, 3)),
        },
        "observed": {
            "overall_pass_delta": {
                "min": float(min(overall_pass)) if overall_pass else 0.0,
                "median": float(statistics.median(overall_pass)) if overall_pass else 0.0,
                "p75": float(_q(overall_pass, 0.75, 0.0)),
            },
            "exact_pass_delta": {
                "min": float(min(exact_pass)) if exact_pass else 0.0,
                "median": float(statistics.median(exact_pass)) if exact_pass else 0.0,
                "p75": float(_q(exact_pass, 0.75, 0.0)),
            },
            "overall_latency_delta_ms": {
                "min": float(min(overall_lat)) if overall_lat else 0.0,
                "median": float(statistics.median(overall_lat)) if overall_lat else 0.0,
                "p75": float(_q(overall_lat, 0.75, 0.0)),
            },
            "exact_latency_delta_ms": {
                "min": float(min(exact_lat)) if exact_lat else 0.0,
                "median": float(statistics.median(exact_lat)) if exact_lat else 0.0,
                "p75": float(_q(exact_lat, 0.75, 0.0)),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Synthesize recommended operator-gate thresholds from nightly history.")
    parser.add_argument("--nightly-root", type=Path, default=Path("docs_tmp/RAG/nightly"))
    parser.add_argument("--max-days", type=int, default=14)
    parser.add_argument("--min-days", type=int, default=5)
    parser.add_argument("--out", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_GATE_POLICY.json"))
    parser.add_argument(
        "--latest-out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_GATE_POLICY_LATEST.json"),
    )
    args = parser.parse_args()

    rows = _collect_gate_rows(args.nightly_root, args.max_days)
    policy = _recommend(rows, args.min_days)

    payload = {
        "generated_at_unix": time.time(),
        "nightly_root": str(args.nightly_root),
        "max_days": int(args.max_days),
        "min_days": int(args.min_days),
        "rows": rows,
        "policy": policy,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.latest_out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.out, args.latest_out)
    print(
        json.dumps(
            {
                "out": str(args.out),
                "latest_out": str(args.latest_out),
                "ready": policy["ready"],
                "recommended_thresholds": policy["recommended_thresholds"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
