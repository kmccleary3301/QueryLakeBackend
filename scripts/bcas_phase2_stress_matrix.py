#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _row(path: Path) -> Dict[str, Any]:
    payload = _load(path)
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    return {
        "path": str(path),
        "label": path.stem,
        "requested_runs": int(summary.get("requested_runs", 0) or 0),
        "successful_runs": int(summary.get("successful_runs", 0) or 0),
        "rps": float(summary.get("median_successful_rps", 0.0) or 0.0),
        "p50_ms": float(summary.get("median_latency_p50_ms", 0.0) or 0.0),
        "p95_ms": float(summary.get("median_latency_p95_ms", 0.0) or 0.0),
        "p99_ms": float(summary.get("median_latency_p99_ms", 0.0) or 0.0),
        "mean_ms": float(summary.get("median_latency_mean_ms", 0.0) or 0.0),
        "max_error_rate": float(summary.get("max_error_rate", 1.0) or 0.0),
    }


def _recommend(rows: List[Dict[str, Any]], p95_budget_ms: float, p99_budget_ms: float) -> Dict[str, Any]:
    eligible: List[Dict[str, Any]] = []
    rejected: List[Dict[str, Any]] = []
    for row in rows:
        ok = (
            row["requested_runs"] > 0
            and row["successful_runs"] == row["requested_runs"]
            and row["max_error_rate"] <= 0.0
            and row["p95_ms"] <= p95_budget_ms
            and row["p99_ms"] <= p99_budget_ms
        )
        if ok:
            eligible.append(row)
        else:
            rejected.append(row)
    if eligible:
        chosen = max(eligible, key=lambda r: (r["rps"], -r["p95_ms"], -r["p99_ms"]))
        reason = "highest_rps_within_budgets"
    elif rows:
        chosen = min(rows, key=lambda r: (r["max_error_rate"], r["p95_ms"], r["p99_ms"], -r["rps"]))
        reason = "fallback_lowest_tail_latency"
    else:
        chosen = {}
        reason = "no_rows"
    return {
        "reason": reason,
        "p95_budget_ms": float(p95_budget_ms),
        "p99_budget_ms": float(p99_budget_ms),
        "eligible_labels": [r["label"] for r in eligible],
        "rejected_labels": [r["label"] for r in rejected],
        "recommended": chosen,
    }


def _render_markdown(rows: List[Dict[str, Any]], recommendation: Dict[str, Any]) -> str:
    lines = []
    lines.append("# BCAS Phase 2 Stress Matrix")
    lines.append("")
    lines.append(f"- Generated at unix: `{int(time.time())}`")
    lines.append(f"- Recommendation reason: `{recommendation.get('reason', 'n/a')}`")
    lines.append(
        f"- Budgets: `p95 <= {recommendation.get('p95_budget_ms')}` ms, "
        f"`p99 <= {recommendation.get('p99_budget_ms')}` ms"
    )
    lines.append("")
    lines.append("| label | runs | rps | p50 ms | p95 ms | p99 ms | error rate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in rows:
        lines.append(
            f"| {row['label']} | {row['successful_runs']}/{row['requested_runs']} | "
            f"{row['rps']:.4f} | {row['p50_ms']:.2f} | {row['p95_ms']:.2f} | "
            f"{row['p99_ms']:.2f} | {row['max_error_rate']:.4f} |"
        )
    rec = recommendation.get("recommended", {}) if isinstance(recommendation.get("recommended"), dict) else {}
    lines.append("")
    if rec:
        lines.append("## Recommendation")
        lines.append("")
        lines.append(f"- Label: `{rec.get('label', 'n/a')}`")
        lines.append(f"- RPS: `{float(rec.get('rps', 0.0)):.4f}`")
        lines.append(f"- p95 ms: `{float(rec.get('p95_ms', 0.0)):.2f}`")
        lines.append(f"- p99 ms: `{float(rec.get('p99_ms', 0.0)):.2f}`")
    else:
        lines.append("## Recommendation")
        lines.append("")
        lines.append("- No valid rows were provided.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize c8/c10/c12 stress suites and recommend an operating point.")
    parser.add_argument("--suite", type=Path, nargs="+", required=True, help="One or more stress suite JSON artifacts.")
    parser.add_argument("--p95-budget-ms", type=float, default=5000.0)
    parser.add_argument("--p99-budget-ms", type=float, default=7000.0)
    parser.add_argument("--out-json", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_STRESS_MATRIX_2026-02-24.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_STRESS_MATRIX_2026-02-24.md"))
    args = parser.parse_args()

    rows = [_row(p) for p in args.suite]
    recommendation = _recommend(rows, p95_budget_ms=float(args.p95_budget_ms), p99_budget_ms=float(args.p99_budget_ms))
    payload = {
        "generated_at_unix": time.time(),
        "rows": rows,
        "recommendation": recommendation,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(_render_markdown(rows, recommendation), encoding="utf-8")
    print(
        json.dumps(
            {
                "out_json": str(args.out_json),
                "out_md": str(args.out_md),
                "recommended": recommendation.get("recommended", {}),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
