#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: Any, digits: int = 4) -> str:
    try:
        f = float(v)
    except Exception:
        return "-"
    return f"{f:.{digits}f}"


def _build_markdown(rows: List[Dict[str, Any]], summary: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# BCAS Phase-2 Daily Status")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Days tracked: {int(summary.get('days', 0) or 0)}")
    lines.append(f"- Avg Recall@k: {_fmt(summary.get('avg_retrieval_recall_at_k'))}")
    lines.append(f"- Avg MRR: {_fmt(summary.get('avg_retrieval_mrr'))}")
    lines.append(f"- Avg Operator Pass: {_fmt(summary.get('avg_operator_pass_rate'))}")
    lines.append(f"- Avg Operator Latency (ms): {_fmt(summary.get('avg_operator_latency_ms'), 2)}")
    lines.append(f"- Avg Stress RPS: {_fmt(summary.get('avg_stress_rps'), 3)}")
    lines.append(f"- Avg Stress p95 (ms): {_fmt(summary.get('avg_stress_p95_ms'), 2)}")
    lines.append(f"- Gate-OK Days: {int(summary.get('gate_ok_days', 0) or 0)}")
    lines.append("")
    lines.append("## Daily Table")
    lines.append("")
    lines.append("| Date | Recall@k | MRR | Operator Pass | Operator Lat ms | Stress RPS | Stress p95 ms | Gate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---|")
    for row in rows:
        lines.append(
            "| {date} | {recall} | {mrr} | {op_pass} | {op_lat} | {rps} | {p95} | {gate} |".format(
                date=row.get("date", "-"),
                recall=_fmt(row.get("retrieval_recall_at_k")),
                mrr=_fmt(row.get("retrieval_mrr")),
                op_pass=_fmt(row.get("operator_pass_rate")),
                op_lat=_fmt(row.get("operator_avg_latency_ms"), 2),
                rps=_fmt(row.get("stress_rps"), 3),
                p95=_fmt(row.get("stress_p95_ms"), 2),
                gate="OK" if bool(row.get("gate_ok", False)) else "FAIL",
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate markdown daily status from BCAS nightly trend JSON.")
    parser.add_argument(
        "--trend-json",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_TREND_2026-02-24.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_DAILY_STATUS_2026-02-24.md"),
    )
    args = parser.parse_args()

    payload = _load(args.trend_json)
    rows = payload.get("rows", [])
    summary = payload.get("summary", {})
    if not isinstance(rows, list):
        rows = []
    if not isinstance(summary, dict):
        summary = {}

    md = _build_markdown(rows, summary)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "rows": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
