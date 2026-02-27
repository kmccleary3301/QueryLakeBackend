#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_gates import percentile


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return [row for row in payload["rows"] if isinstance(row, dict)]
    raise ValueError(f"Unsupported JSON shape in {path}")


def _latency(row: Dict[str, Any]) -> float:
    timings = row.get("timings")
    if not isinstance(timings, dict):
        return 0.0
    val = timings.get("total")
    if isinstance(val, (int, float)) and float(val) >= 0:
        return float(val)
    return 0.0


def _summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    latencies = [_latency(row) for row in rows]
    errors = sum(1 for row in rows if str(row.get("status", "ok")).lower() != "ok")
    count = len(rows)
    error_rate = 0.0 if count == 0 else float(errors) / float(count)
    return {
        "request_count": count,
        "error_count": errors,
        "error_rate": error_rate,
        "p50_latency_seconds": percentile(latencies, 0.50) if count > 0 else 0.0,
        "p95_latency_seconds": percentile(latencies, 0.95) if count > 0 else 0.0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate retrieval SLO window report from exported retrieval runs.")
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-label", type=str, default="window")
    args = parser.parse_args()

    rows = _load_rows(args.runs)
    routes: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        route = str(row.get("route") or "unknown")
        routes.setdefault(route, []).append(row)

    report = {
        "window_label": args.window_label,
        "overall": _summarize(rows),
        "by_route": {route: _summarize(route_rows) for route, route_rows in sorted(routes.items())},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "routes": len(routes)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

