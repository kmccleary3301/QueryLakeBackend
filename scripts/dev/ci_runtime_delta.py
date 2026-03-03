#!/usr/bin/env python3
"""Compare CI runtime profile reports and emit regression deltas."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CI runtime profile delta reporter.")
    parser.add_argument("--before-json", required=True)
    parser.add_argument("--after-json", required=True)
    parser.add_argument("--out-md", required=True)
    parser.add_argument("--max-p95-regression-pct", type=float, default=15.0)
    parser.add_argument("--max-compute-regression-pct", type=float, default=20.0)
    parser.add_argument("--fail-on-regression", action="store_true")
    return parser.parse_args()


def pct_delta(before: float, after: float) -> float:
    if before == 0:
        return 0.0 if after == 0 else 100.0
    return ((after - before) / before) * 100.0


def render(before: dict, after: dict, table_rows: list[str], gate: dict) -> str:
    lines = [
        "# CI Runtime Delta Report",
        "",
        f"- Before: `{before.get('generated_at_utc', 'n/a')}`",
        f"- After: `{after.get('generated_at_utc', 'n/a')}`",
        f"- Gate status: `{gate['status']}`",
        "",
        "## Overall Delta",
        "",
        f"- Duration p95 delta (%): `{gate['overall_duration_p95_delta_pct']}`",
        f"- Compute minutes delta (%): `{gate['overall_compute_delta_pct']}`",
        f"- Queue p95 delta (%): `{gate['overall_queue_p95_delta_pct']}`",
        "",
        "## Workflow Delta",
        "",
        "| Workflow | p95 before | p95 after | p95 delta % | compute before | compute after | compute delta % |",
        "|---|---:|---:|---:|---:|---:|---:|",
        *table_rows,
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    before = json.loads(Path(args.before_json).read_text(encoding="utf-8"))
    after = json.loads(Path(args.after_json).read_text(encoding="utf-8"))

    before_overall = before.get("overall", {})
    after_overall = after.get("overall", {})
    overall_duration_delta = round(
        pct_delta(
            float(before_overall.get("duration_s", {}).get("p95", 0.0)),
            float(after_overall.get("duration_s", {}).get("p95", 0.0)),
        ),
        2,
    )
    overall_compute_delta = round(
        pct_delta(
            float(before_overall.get("compute_minutes_total", 0.0)),
            float(after_overall.get("compute_minutes_total", 0.0)),
        ),
        2,
    )
    overall_queue_delta = round(
        pct_delta(
            float(before_overall.get("queue_s", {}).get("p95", 0.0)),
            float(after_overall.get("queue_s", {}).get("p95", 0.0)),
        ),
        2,
    )

    rows: list[str] = []
    before_wf = before.get("by_workflow", {})
    after_wf = after.get("by_workflow", {})
    all_workflows = sorted(set(before_wf) | set(after_wf))
    for workflow_name in all_workflows:
        b = before_wf.get(workflow_name, {})
        a = after_wf.get(workflow_name, {})
        b_p95 = float(b.get("duration_s", {}).get("p95", 0.0))
        a_p95 = float(a.get("duration_s", {}).get("p95", 0.0))
        b_compute = float(b.get("compute_minutes_total", 0.0))
        a_compute = float(a.get("compute_minutes_total", 0.0))
        rows.append(
            f"| {workflow_name} | {round(b_p95, 2)} | {round(a_p95, 2)} | "
            f"{round(pct_delta(b_p95, a_p95), 2)} | {round(b_compute, 2)} | {round(a_compute, 2)} | "
            f"{round(pct_delta(b_compute, a_compute), 2)} |"
        )

    gate_failed = (
        overall_duration_delta > float(args.max_p95_regression_pct)
        or overall_compute_delta > float(args.max_compute_regression_pct)
    )
    gate = {
        "status": "fail" if gate_failed else "pass",
        "overall_duration_p95_delta_pct": overall_duration_delta,
        "overall_compute_delta_pct": overall_compute_delta,
        "overall_queue_p95_delta_pct": overall_queue_delta,
        "max_p95_regression_pct": float(args.max_p95_regression_pct),
        "max_compute_regression_pct": float(args.max_compute_regression_pct),
    }

    output_path = Path(args.out_md)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render(before=before, after=after, table_rows=rows, gate=gate), encoding="utf-8")

    print(json.dumps(gate, indent=2, sort_keys=True))
    if gate_failed and args.fail_on_regression:
        print("[ci-runtime-delta] regression gate failed", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
