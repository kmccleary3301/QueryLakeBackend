#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _latest_match(day_dir: Path, pattern: str) -> Path | None:
    rows = [path for path in day_dir.glob(pattern) if path.is_file()]
    if not rows:
        return None
    rows.sort(key=lambda p: (p.stat().st_mtime, p.name))
    return rows[-1]


def _extract_eval(payload: Dict[str, Any]) -> Dict[str, float]:
    overall = payload.get("metrics", {}).get("overall") if isinstance(payload.get("metrics"), dict) else {}
    overall = overall if isinstance(overall, dict) else {}
    return {
        "recall_at_k": _safe_float(overall.get("recall_at_k")),
        "mrr": _safe_float(overall.get("mrr")),
        "avg_response_ms": _safe_float(overall.get("avg_response_ms")),
    }


def _extract_stress(payload: Dict[str, Any]) -> Dict[str, float]:
    latency = payload.get("latency_ms", {}) if isinstance(payload.get("latency_ms"), dict) else {}
    throughput = payload.get("throughput", {}) if isinstance(payload.get("throughput"), dict) else {}
    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    return {
        "p95_ms": _safe_float(latency.get("p95")),
        "p99_ms": _safe_float(latency.get("p99")),
        "successful_rps": _safe_float(throughput.get("successful_requests_per_second")),
        "error_rate": _safe_float(throughput.get("error_rate", counts.get("error_rate"))),
    }


def _format_delta(candidate: float, baseline: float, inverse_better: bool = False) -> Dict[str, float]:
    delta = candidate - baseline
    pct = 0.0 if baseline == 0 else (delta / baseline) * 100.0
    improved = delta < 0 if inverse_better else delta > 0
    return {"candidate": candidate, "baseline": baseline, "delta": delta, "delta_pct": pct, "improved": improved}


def _build_markdown(payload: Dict[str, Any]) -> str:
    rows = payload.get("metrics", {})
    lines: List[str] = []
    lines.append("# BCAS Phase2 Nightly Delta")
    lines.append("")
    lines.append(f"- `latest_day`: `{payload.get('latest_day')}`")
    lines.append(f"- `previous_day`: `{payload.get('previous_day')}`")
    lines.append("")
    lines.append("## Eval")
    for key in ["recall_at_k", "mrr", "avg_response_ms"]:
        row = rows.get(key, {})
        direction = "improved" if bool(row.get("improved")) else "regressed_or_flat"
        lines.append(
            f"- `{key}`: latest={row.get('candidate'):.6f}, prev={row.get('baseline'):.6f}, "
            f"delta={row.get('delta'):.6f} ({row.get('delta_pct'):.2f}%), {direction}"
        )
    lines.append("")
    lines.append("## Stress")
    for key in ["p95_ms", "p99_ms", "successful_rps", "error_rate"]:
        row = rows.get(key, {})
        direction = "improved" if bool(row.get("improved")) else "regressed_or_flat"
        lines.append(
            f"- `{key}`: latest={row.get('candidate'):.6f}, prev={row.get('baseline'):.6f}, "
            f"delta={row.get('delta'):.6f} ({row.get('delta_pct'):.2f}%), {direction}"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate latest-vs-previous nightly delta report.")
    parser.add_argument("--nightly-root", type=Path, default=Path("docs_tmp/RAG/nightly"))
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    args = parser.parse_args()

    day_dirs = sorted([path for path in args.nightly_root.iterdir() if path.is_dir()]) if args.nightly_root.exists() else []
    if len(day_dirs) < 2:
        raise SystemExit(f"Need at least 2 nightly day directories under: {args.nightly_root}")

    prev_day = day_dirs[-2]
    latest_day = day_dirs[-1]

    prev_eval_path = _latest_match(prev_day, "BCAS_PHASE2_LIVE_EVAL_METRICS_*.json")
    latest_eval_path = _latest_match(latest_day, "BCAS_PHASE2_LIVE_EVAL_METRICS_*.json")
    prev_stress_path = _latest_match(prev_day, "BCAS_PHASE2_STRESS_*.json")
    latest_stress_path = _latest_match(latest_day, "BCAS_PHASE2_STRESS_*.json")

    if prev_eval_path is None or latest_eval_path is None or prev_stress_path is None or latest_stress_path is None:
        raise SystemExit("Missing eval/stress artifacts for one or both nightly days")

    prev_eval = _extract_eval(_safe_load(prev_eval_path))
    latest_eval = _extract_eval(_safe_load(latest_eval_path))
    prev_stress = _extract_stress(_safe_load(prev_stress_path))
    latest_stress = _extract_stress(_safe_load(latest_stress_path))

    payload = {
        "generated_at_unix": time.time(),
        "nightly_root": str(args.nightly_root),
        "previous_day": prev_day.name,
        "latest_day": latest_day.name,
        "artifacts": {
            "previous_eval": str(prev_eval_path),
            "latest_eval": str(latest_eval_path),
            "previous_stress": str(prev_stress_path),
            "latest_stress": str(latest_stress_path),
        },
        "metrics": {
            "recall_at_k": _format_delta(latest_eval["recall_at_k"], prev_eval["recall_at_k"]),
            "mrr": _format_delta(latest_eval["mrr"], prev_eval["mrr"]),
            "avg_response_ms": _format_delta(latest_eval["avg_response_ms"], prev_eval["avg_response_ms"], inverse_better=True),
            "p95_ms": _format_delta(latest_stress["p95_ms"], prev_stress["p95_ms"], inverse_better=True),
            "p99_ms": _format_delta(latest_stress["p99_ms"], prev_stress["p99_ms"], inverse_better=True),
            "successful_rps": _format_delta(latest_stress["successful_rps"], prev_stress["successful_rps"]),
            "error_rate": _format_delta(latest_stress["error_rate"], prev_stress["error_rate"], inverse_better=True),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(_build_markdown(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
