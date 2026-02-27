#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _collect_day(day_dir: Path) -> Dict[str, Any]:
    date_key = day_dir.name
    retrieval = _safe_load(day_dir / f"BCAS_PHASE2_LIVE_EVAL_RETRIEVAL_EVAL_dense_heavy_augtrivia_{date_key}.json")
    operator_eval = _safe_load(day_dir / f"BCAS_PHASE2_OPERATOR_EVAL_{date_key}.json")
    stress = _safe_load(day_dir / f"BCAS_PHASE2_STRESS_c8_{date_key}.json")
    gate = _safe_load(day_dir / f"BCAS_PHASE2_OPERATOR_GATE_{date_key}.json")

    metrics = retrieval.get("metrics", {}) if isinstance(retrieval.get("metrics"), dict) else {}
    op_overall = operator_eval.get("overall", {}) if isinstance(operator_eval.get("overall"), dict) else {}
    stress_lat = stress.get("latency_ms", {}) if isinstance(stress.get("latency_ms"), dict) else {}
    stress_throughput = stress.get("throughput", {}) if isinstance(stress.get("throughput"), dict) else {}

    return {
        "date": date_key,
        "retrieval_case_count": _float(metrics.get("case_count")),
        "retrieval_recall_at_k": _float(metrics.get("recall_at_k")),
        "retrieval_mrr": _float(metrics.get("mrr")),
        "operator_case_count": _float(op_overall.get("cases")),
        "operator_pass_rate": _float(op_overall.get("pass_rate")),
        "operator_avg_latency_ms": _float(op_overall.get("avg_latency_ms")),
        "stress_rps": _float(stress_throughput.get("successful_requests_per_second")),
        "stress_p50_ms": _float(stress_lat.get("p50")),
        "stress_p95_ms": _float(stress_lat.get("p95")),
        "stress_p99_ms": _float(stress_lat.get("p99")),
        "gate_ok": bool(gate.get("gate_ok", False)),
        "gate_overall_pass_delta": _float((gate.get("deltas") or {}).get("overall_pass_rate")),
        "gate_exact_pass_delta": _float((gate.get("deltas") or {}).get("exact_phrase_pass_rate")),
        "gate_overall_latency_delta_ms": _float((gate.get("deltas") or {}).get("overall_avg_latency_ms")),
        "gate_exact_latency_delta_ms": _float((gate.get("deltas") or {}).get("exact_phrase_avg_latency_ms")),
    }


def _avg(values: List[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize BCAS nightly retrieval trend artifacts.")
    parser.add_argument("--nightly-root", type=Path, default=Path("docs_tmp/RAG/nightly"))
    parser.add_argument("--max-days", type=int, default=7)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_TREND.json"),
    )
    args = parser.parse_args()

    day_dirs = [p for p in args.nightly_root.iterdir()] if args.nightly_root.exists() else []
    day_dirs = sorted([p for p in day_dirs if p.is_dir()])[-max(1, int(args.max_days)) :]

    rows = [_collect_day(day_dir) for day_dir in day_dirs]
    rows = [r for r in rows if r.get("retrieval_case_count", 0.0) > 0.0]

    summary = {
        "days": len(rows),
        "avg_retrieval_recall_at_k": _avg([r["retrieval_recall_at_k"] for r in rows]),
        "avg_retrieval_mrr": _avg([r["retrieval_mrr"] for r in rows]),
        "avg_operator_pass_rate": _avg([r["operator_pass_rate"] for r in rows]),
        "avg_operator_latency_ms": _avg([r["operator_avg_latency_ms"] for r in rows]),
        "avg_stress_rps": _avg([r["stress_rps"] for r in rows]),
        "avg_stress_p95_ms": _avg([r["stress_p95_ms"] for r in rows]),
        "gate_ok_days": int(sum(1 for r in rows if r["gate_ok"])),
    }

    payload = {"nightly_root": str(args.nightly_root), "summary": summary, "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
