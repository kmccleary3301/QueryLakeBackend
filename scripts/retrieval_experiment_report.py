#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from QueryLake.runtime.retrieval_reports import build_experiment_report


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path} must contain a JSON list")
    return [row for row in payload if isinstance(row, dict)]


def main() -> int:
    parser = argparse.ArgumentParser(description="Build JSON + Markdown comparison report from experiment metric rows.")
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--baseline-json", required=True)
    parser.add_argument("--candidate-json", required=True)
    parser.add_argument("--delta-json", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-md", required=True)
    args = parser.parse_args()

    baseline_rows = _load_rows(Path(args.baseline_json))
    candidate_rows = _load_rows(Path(args.candidate_json))
    delta_rows = _load_rows(Path(args.delta_json))

    report_json, report_md = build_experiment_report(
        experiment_id=args.experiment_id,
        baseline_rows=baseline_rows,
        candidate_rows=candidate_rows,
        delta_rows=delta_rows,
    )

    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report_json, indent=2), encoding="utf-8")
    out_md.write_text(report_md, encoding="utf-8")
    print(json.dumps(report_json, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
