#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _fmt_float(v: Any, digits: int = 3) -> str:
    try:
        return f"{float(v):.{digits}f}"
    except Exception:
        return "-"


def _latest_day_dir(nightly_root: Path) -> Path | None:
    if not nightly_root.exists():
        return None
    days = sorted([p for p in nightly_root.iterdir() if p.is_dir()])
    return days[-1] if days else None


def _render(day: str, gate: Dict[str, Any], stress: Dict[str, Any], notify: Dict[str, Any], proposal: Dict[str, Any], policy: Dict[str, Any], trend: Dict[str, Any]) -> str:
    gate_ok = bool(gate.get("gate_ok", False))
    notify_status = str(notify.get("status", "unknown"))
    proposal_ready = bool(proposal.get("approve_ready", False))
    policy_node = policy.get("policy", {}) if isinstance(policy.get("policy"), dict) else {}
    policy_ready = bool(policy_node.get("ready", False))
    thresholds = policy_node.get("recommended_thresholds", {}) if isinstance(policy_node.get("recommended_thresholds"), dict) else {}

    latency = stress.get("latency_ms", {}) if isinstance(stress.get("latency_ms"), dict) else {}
    throughput = stress.get("throughput", {}) if isinstance(stress.get("throughput"), dict) else {}
    counts = stress.get("counts", {}) if isinstance(stress.get("counts"), dict) else {}
    error_rate = throughput.get("error_rate")
    if error_rate is None:
        error_rate = counts.get("error_rate")
    trend_summary = trend.get("summary", {}) if isinstance(trend.get("summary"), dict) else {}

    lines = []
    lines.append("# BCAS Phase-2 Oncall Dashboard")
    lines.append("")
    lines.append(f"- Generated: {datetime.now(timezone.utc).isoformat()}")
    lines.append(f"- Nightly day: {day}")
    lines.append("")
    lines.append("## Health")
    lines.append(f"- Operator gate: {'OK' if gate_ok else 'FAIL'}")
    lines.append(f"- Notification status: {notify_status}")
    lines.append(f"- Baseline proposal ready: {proposal_ready}")
    lines.append(f"- Threshold policy ready: {policy_ready}")
    lines.append("")
    lines.append("## Latest Stress")
    lines.append(f"- RPS: {_fmt_float(throughput.get('successful_requests_per_second'))}")
    lines.append(f"- p50 (ms): {_fmt_float(latency.get('p50'), 2)}")
    lines.append(f"- p95 (ms): {_fmt_float(latency.get('p95'), 2)}")
    lines.append(f"- p99 (ms): {_fmt_float(latency.get('p99'), 2)}")
    lines.append(f"- Error rate: {_fmt_float(error_rate, 4)}")
    lines.append("")
    lines.append("## Gate Deltas")
    deltas = gate.get("deltas", {}) if isinstance(gate.get("deltas"), dict) else {}
    lines.append(f"- Overall pass delta: {_fmt_float(deltas.get('overall_pass_rate'), 6)}")
    lines.append(f"- Exact pass delta: {_fmt_float(deltas.get('exact_phrase_pass_rate'), 6)}")
    lines.append(f"- Overall latency delta (ms): {_fmt_float(deltas.get('overall_avg_latency_ms'), 3)}")
    lines.append(f"- Exact latency delta (ms): {_fmt_float(deltas.get('exact_phrase_avg_latency_ms'), 3)}")
    lines.append("")
    lines.append("## Rolling Trend")
    lines.append(f"- Days tracked: {int(trend_summary.get('days', 0) or 0)}")
    lines.append(f"- Avg recall@k: {_fmt_float(trend_summary.get('avg_retrieval_recall_at_k'), 4)}")
    lines.append(f"- Avg MRR: {_fmt_float(trend_summary.get('avg_retrieval_mrr'), 4)}")
    lines.append(f"- Avg operator pass: {_fmt_float(trend_summary.get('avg_operator_pass_rate'), 4)}")
    lines.append(f"- Avg operator latency (ms): {_fmt_float(trend_summary.get('avg_operator_latency_ms'), 2)}")
    lines.append(f"- Avg stress p95 (ms): {_fmt_float(trend_summary.get('avg_stress_p95_ms'), 2)}")
    lines.append("")
    lines.append("## Suggested Gate Thresholds")
    lines.append(f"- min_overall_pass_delta: {_fmt_float(thresholds.get('min_overall_pass_delta'), 6)}")
    lines.append(f"- min_exact_pass_delta: {_fmt_float(thresholds.get('min_exact_pass_delta'), 6)}")
    lines.append(f"- max_overall_latency_regression_ms: {_fmt_float(thresholds.get('max_overall_latency_regression_ms'), 3)}")
    lines.append(f"- max_exact_latency_regression_ms: {_fmt_float(thresholds.get('max_exact_latency_regression_ms'), 3)}")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate compact BCAS phase-2 oncall dashboard markdown.")
    parser.add_argument("--nightly-root", type=Path, default=Path("docs_tmp/RAG/nightly"))
    parser.add_argument("--trend", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_NIGHTLY_TREND_2026-02-24.json"))
    parser.add_argument("--policy", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_GATE_POLICY_2026-02-24.json"))
    parser.add_argument("--out", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_ONCALL_DASHBOARD_2026-02-24.md"))
    args = parser.parse_args()

    latest = _latest_day_dir(args.nightly_root)
    if latest is None:
        raise SystemExit(f"No nightly day directories found under: {args.nightly_root}")

    day = latest.name
    gate = _safe_load(latest / f"BCAS_PHASE2_OPERATOR_GATE_{day}.json")
    stress = _safe_load(latest / f"BCAS_PHASE2_STRESS_c8_{day}.json")
    notify = _safe_load(latest / f"BCAS_PHASE2_NOTIFICATIONS_{day}.json")
    proposal = _safe_load(latest / f"BCAS_PHASE2_OPERATOR_BASELINE_PROPOSAL_{day}.json")
    trend = _safe_load(args.trend)
    policy = _safe_load(args.policy)

    md = _render(day, gate, stress, notify, proposal, policy, trend)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(md, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "nightly_day": day}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
