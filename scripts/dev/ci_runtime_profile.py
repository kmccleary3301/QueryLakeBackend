#!/usr/bin/env python3
"""Collect GitHub Actions runtime metrics for profiling and optimization."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import statistics
import sys
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


DEFAULT_WORKFLOWS = [
    "SDK Checks",
    "SDK Publish",
    "SDK Publish Dry-Run (TestPyPI)",
    "SDK Live Integration (Staging)",
    "Docs Checks",
    "Unification Checks",
    "Retrieval Eval",
]


@dataclass(frozen=True)
class RunSample:
    workflow_name: str
    run_id: int
    event: str
    status: str
    conclusion: str | None
    attempt: int
    created_at: dt.datetime
    started_at: dt.datetime
    updated_at: dt.datetime
    duration_s: float
    queue_s: float
    url: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile CI runtime metrics from GitHub Actions.")
    parser.add_argument("--repo", required=True, help="Repository in OWNER/REPO format.")
    parser.add_argument("--days", type=int, default=7, help="Lookback window in days.")
    parser.add_argument(
        "--workflow",
        action="append",
        default=[],
        help="Workflow name to include (repeatable). Defaults to a curated set.",
    )
    parser.add_argument("--out-json", required=True, help="Output JSON report path.")
    parser.add_argument("--out-md", required=True, help="Output markdown summary path.")
    parser.add_argument("--max-runs", type=int, default=1000, help="Safety cap for fetched runs.")
    parser.add_argument(
        "--input-runs-json",
        default="",
        help="Optional local JSON payload with GitHub Actions run objects (workflow_runs list).",
    )
    return parser.parse_args()


def parse_ts(raw: str) -> dt.datetime:
    return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    rank = (len(values) - 1) * pct
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return float(values[int(rank)])
    lower_val = values[lower]
    upper_val = values[upper]
    return float(lower_val + (upper_val - lower_val) * (rank - lower))


def fetch_runs(repo: str, days: int, max_runs: int) -> list[dict]:
    token = os.getenv("GITHUB_TOKEN") or os.getenv("GH_TOKEN")
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "querylake-ci-runtime-profiler/1",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=days)
    page = 1
    rows: list[dict] = []

    while len(rows) < max_runs:
        query = urllib.parse.urlencode({"per_page": 100, "page": page})
        url = f"https://api.github.com/repos/{repo}/actions/runs?{query}"
        request = urllib.request.Request(url=url, headers=headers)
        with urllib.request.urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        batch = payload.get("workflow_runs", [])
        if not batch:
            break

        for row in batch:
            created_at = parse_ts(row["created_at"])
            if created_at < cutoff:
                return rows
            rows.append(row)
            if len(rows) >= max_runs:
                return rows

        page += 1

    return rows


def normalize_samples(raw_runs: Iterable[dict], workflow_allowlist: set[str]) -> list[RunSample]:
    out: list[RunSample] = []
    for row in raw_runs:
        workflow_name = row.get("name") or row.get("workflow_name") or "(unknown)"
        if workflow_allowlist and workflow_name not in workflow_allowlist:
            continue
        created_at = parse_ts(row["created_at"])
        started_at = parse_ts(row["run_started_at"]) if row.get("run_started_at") else created_at
        updated_at = parse_ts(row["updated_at"]) if row.get("updated_at") else started_at
        duration_s = max(0.0, (updated_at - started_at).total_seconds())
        queue_s = max(0.0, (started_at - created_at).total_seconds())

        out.append(
            RunSample(
                workflow_name=workflow_name,
                run_id=int(row.get("id", 0)),
                event=str(row.get("event", "")),
                status=str(row.get("status", "")),
                conclusion=row.get("conclusion"),
                attempt=int(row.get("run_attempt", 1)),
                created_at=created_at,
                started_at=started_at,
                updated_at=updated_at,
                duration_s=duration_s,
                queue_s=queue_s,
                url=str(row.get("html_url", "")),
            )
        )
    return out


def aggregate(samples: list[RunSample]) -> dict:
    if not samples:
        return {
            "run_count": 0,
            "success_rate": 0.0,
            "error_rate": 0.0,
            "rerun_rate": 0.0,
            "duration_s": {"median": 0.0, "p95": 0.0, "mean": 0.0},
            "queue_s": {"median": 0.0, "p95": 0.0, "mean": 0.0},
            "compute_minutes_total": 0.0,
        }

    durations = sorted(sample.duration_s for sample in samples)
    queues = sorted(sample.queue_s for sample in samples)
    success_count = sum(1 for sample in samples if sample.conclusion == "success")
    failure_count = sum(
        1 for sample in samples if sample.conclusion not in (None, "success", "cancelled", "skipped")
    )
    rerun_count = sum(1 for sample in samples if sample.attempt > 1)
    total_compute_minutes = sum(sample.duration_s for sample in samples) / 60.0

    return {
        "run_count": len(samples),
        "success_rate": round(success_count / len(samples), 4),
        "error_rate": round(failure_count / len(samples), 4),
        "rerun_rate": round(rerun_count / len(samples), 4),
        "duration_s": {
            "median": round(statistics.median(durations), 2),
            "p95": round(percentile(durations, 0.95), 2),
            "mean": round(statistics.mean(durations), 2),
        },
        "queue_s": {
            "median": round(statistics.median(queues), 2),
            "p95": round(percentile(queues, 0.95), 2),
            "mean": round(statistics.mean(queues), 2),
        },
        "compute_minutes_total": round(total_compute_minutes, 2),
    }


def render_markdown(report: dict) -> str:
    lines = [
        "# CI Runtime Profile",
        "",
        f"- Generated UTC: `{report['generated_at_utc']}`",
        f"- Repository: `{report['repo']}`",
        f"- Lookback days: `{report['lookback_days']}`",
        f"- Sampled runs: `{report['overall']['run_count']}`",
        "",
        "## Overall",
        "",
        f"- Success rate: `{report['overall']['success_rate']}`",
        f"- Error rate: `{report['overall']['error_rate']}`",
        f"- Rerun rate: `{report['overall']['rerun_rate']}`",
        f"- Duration median/p95/mean (s): `{report['overall']['duration_s']['median']}` / "
        f"`{report['overall']['duration_s']['p95']}` / `{report['overall']['duration_s']['mean']}`",
        f"- Queue median/p95/mean (s): `{report['overall']['queue_s']['median']}` / "
        f"`{report['overall']['queue_s']['p95']}` / `{report['overall']['queue_s']['mean']}`",
        f"- Compute minutes total: `{report['overall']['compute_minutes_total']}`",
        "",
        "## By Workflow",
        "",
        "| Workflow | Runs | Success | Error | Rerun | Dur p50 (s) | Dur p95 (s) | Queue p50 (s) | Queue p95 (s) | Compute min |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for workflow_name, metrics in report["by_workflow"].items():
        lines.append(
            f"| {workflow_name} | {metrics['run_count']} | {metrics['success_rate']} | "
            f"{metrics['error_rate']} | {metrics['rerun_rate']} | "
            f"{metrics['duration_s']['median']} | {metrics['duration_s']['p95']} | "
            f"{metrics['queue_s']['median']} | {metrics['queue_s']['p95']} | "
            f"{metrics['compute_minutes_total']} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    workflow_allowlist = set(args.workflow or DEFAULT_WORKFLOWS)

    try:
        if args.input_runs_json:
            payload = json.loads(Path(args.input_runs_json).read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                raw_runs = payload.get("workflow_runs", payload)
            else:
                raw_runs = payload
            if not isinstance(raw_runs, list):
                raise ValueError("`--input-runs-json` must contain a list or {\"workflow_runs\": [...]} payload.")
        else:
            raw_runs = fetch_runs(repo=args.repo, days=args.days, max_runs=args.max_runs)
        samples = normalize_samples(raw_runs=raw_runs, workflow_allowlist=workflow_allowlist)

        grouped: dict[str, list[RunSample]] = {}
        for sample in samples:
            grouped.setdefault(sample.workflow_name, []).append(sample)

        report = {
            "generated_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "repo": args.repo,
            "lookback_days": int(args.days),
            "workflow_allowlist": sorted(workflow_allowlist),
            "overall": aggregate(samples),
            "by_workflow": {
                workflow_name: aggregate(rows)
                for workflow_name, rows in sorted(grouped.items(), key=lambda pair: pair[0].lower())
            },
            "sample": [
                {
                    "workflow": row.workflow_name,
                    "run_id": row.run_id,
                    "event": row.event,
                    "status": row.status,
                    "conclusion": row.conclusion,
                    "attempt": row.attempt,
                    "duration_s": round(row.duration_s, 2),
                    "queue_s": round(row.queue_s, 2),
                    "url": row.url,
                    "created_at": row.created_at.isoformat(),
                    "started_at": row.started_at.isoformat(),
                    "updated_at": row.updated_at.isoformat(),
                }
                for row in sorted(samples, key=lambda item: item.created_at, reverse=True)[:50]
            ],
        }
    except Exception as exc:  # noqa: BLE001 - CLI boundary.
        print(f"[ci-runtime-profile] FAILED: {exc}", file=sys.stderr)
        return 1

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    out_md.write_text(render_markdown(report), encoding="utf-8")
    print(f"[ci-runtime-profile] wrote {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
