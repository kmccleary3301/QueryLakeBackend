#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BM25/Dense weight sweep for BCAS live eval.")
    parser.add_argument("--account-config", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json"))
    parser.add_argument("--per-dataset-cases", type=int, default=20)
    parser.add_argument("--max-expected-ids", type=int, default=8)
    parser.add_argument("--limit-bm25", type=int, default=20)
    parser.add_argument("--limit-similarity", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--out", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_WEIGHT_SWEEP_2026-02-24.json"))
    args = parser.parse_args()

    experiments = [
        {"id": "lexical_heavy", "bm25_weight": 0.95, "similarity_weight": 0.05},
        {"id": "balanced", "bm25_weight": 0.70, "similarity_weight": 0.30},
        {"id": "dense_heavy", "bm25_weight": 0.40, "similarity_weight": 0.60},
    ]

    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        metrics_path = Path(f"docs_tmp/RAG/BCAS_PHASE2_LIVE_EVAL_METRICS_{exp['id']}_2026-02-24.json")
        cases_path = Path(f"docs_tmp/RAG/BCAS_PHASE2_LIVE_EVAL_CASES_{exp['id']}_2026-02-24.json")
        cmd = [
            "python",
            "scripts/bcas_phase2_eval.py",
            "--account-config",
            str(args.account_config),
            "--per-dataset-cases",
            str(int(args.per_dataset_cases)),
            "--max-expected-ids",
            str(int(args.max_expected_ids)),
            "--limit-bm25",
            str(int(args.limit_bm25)),
            "--limit-similarity",
            str(int(args.limit_similarity)),
            "--top-k",
            str(int(args.top_k)),
            "--bm25-weight",
            str(float(exp["bm25_weight"])),
            "--similarity-weight",
            str(float(exp["similarity_weight"])),
            "--cases-out",
            str(cases_path),
            "--metrics-out",
            str(metrics_path),
        ]
        rc, out, err = _run(cmd)
        row: Dict[str, Any] = {
            "experiment": exp,
            "rc": rc,
            "stdout_tail": out[-1200:],
            "stderr_tail": err[-1200:],
            "metrics_path": str(metrics_path),
            "cases_path": str(cases_path),
        }
        if rc == 0 and metrics_path.exists():
            payload = _load_json(metrics_path)
            overall = (payload.get("metrics") or {}).get("overall") or {}
            row["overall"] = overall
        rows.append(row)

    successful = [r for r in rows if r.get("rc") == 0 and isinstance(r.get("overall"), dict)]
    best = None
    if len(successful) > 0:
        best = sorted(
            successful,
            key=lambda r: (
                -float(r["overall"].get("recall_at_k", 0.0)),
                -float(r["overall"].get("mrr", 0.0)),
                float(r["overall"].get("avg_response_ms", 1e12)),
            ),
        )[0]

    payload_out = {
        "experiments": rows,
        "best": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload_out, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "successful": len(successful), "best_id": (best or {}).get("experiment", {}).get("id") if isinstance(best, dict) else None}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
