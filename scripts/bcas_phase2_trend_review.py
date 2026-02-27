#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
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


def _check(name: str, passed: bool, actual: Any, target: Any) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "actual": actual, "target": target}


def _render_md(payload: Dict[str, Any]) -> str:
    checks: List[Dict[str, Any]] = payload.get("checks", []) if isinstance(payload.get("checks"), list) else []
    summary = payload.get("summary", {}) if isinstance(payload.get("summary"), dict) else {}
    lines = []
    lines.append("# BCAS Phase-2 Trend Review")
    lines.append("")
    lines.append("## Summary")
    lines.append(f"- Review ready: {summary.get('ready')}")
    lines.append(f"- All checks pass: {summary.get('all_checks_pass')}")
    lines.append(f"- Soak ready: {summary.get('soak_ready')}")
    lines.append("")
    lines.append("## Checks")
    lines.append("")
    lines.append("| Check | Passed | Actual | Target |")
    lines.append("|---|---|---|---|")
    for item in checks:
        lines.append(
            f"| {item.get('name')} | {'yes' if item.get('passed') else 'no'} | {item.get('actual')} | {item.get('target')} |"
        )
    lines.append("")
    lines.append("## Manual Signoff")
    lines.append("- Reviewer: (fill)")
    lines.append("- Decision: approve / hold")
    lines.append("- Notes: (fill)")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate BCAS trend stability review and signoff artifact.")
    parser.add_argument("--trend", type=Path, required=True)
    parser.add_argument("--soak", type=Path, required=True)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--min-avg-recall", type=float, default=0.15)
    parser.add_argument("--min-avg-mrr", type=float, default=0.15)
    parser.add_argument("--min-avg-operator-pass", type=float, default=0.90)
    parser.add_argument("--max-avg-stress-p95-ms", type=float, default=4500.0)
    args = parser.parse_args()

    trend = _safe_load(args.trend)
    soak = _safe_load(args.soak)
    tr = trend.get("summary", {}) if isinstance(trend.get("summary"), dict) else {}
    sr = soak.get("summary", {}) if isinstance(soak.get("summary"), dict) else {}

    checks = [
        _check("avg_recall_at_k", _float(tr.get("avg_retrieval_recall_at_k")) >= float(args.min_avg_recall), _float(tr.get("avg_retrieval_recall_at_k")), float(args.min_avg_recall)),
        _check("avg_mrr", _float(tr.get("avg_retrieval_mrr")) >= float(args.min_avg_mrr), _float(tr.get("avg_retrieval_mrr")), float(args.min_avg_mrr)),
        _check("avg_operator_pass_rate", _float(tr.get("avg_operator_pass_rate")) >= float(args.min_avg_operator_pass), _float(tr.get("avg_operator_pass_rate")), float(args.min_avg_operator_pass)),
        _check("avg_stress_p95_ms", _float(tr.get("avg_stress_p95_ms")) <= float(args.max_avg_stress_p95_ms), _float(tr.get("avg_stress_p95_ms")), float(args.max_avg_stress_p95_ms)),
        _check("soak_ready", bool(sr.get("ready", False)), bool(sr.get("ready", False)), True),
    ]

    all_checks_pass = all(bool(c["passed"]) for c in checks)
    payload = {
        "generated_at_unix": time.time(),
        "trend_artifact": str(args.trend),
        "soak_artifact": str(args.soak),
        "thresholds": {
            "min_avg_recall": float(args.min_avg_recall),
            "min_avg_mrr": float(args.min_avg_mrr),
            "min_avg_operator_pass": float(args.min_avg_operator_pass),
            "max_avg_stress_p95_ms": float(args.max_avg_stress_p95_ms),
        },
        "checks": checks,
        "summary": {
            "all_checks_pass": bool(all_checks_pass),
            "soak_ready": bool(sr.get("ready", False)),
            "ready": bool(all_checks_pass),
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    md = _render_md(payload)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.write_text(md, encoding="utf-8")

    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "ready": payload["summary"]["ready"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
