#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List


def _median(values: List[float]) -> float:
    if not values:
        return 0.0
    return float(statistics.median(values))


def _run_one(args: argparse.Namespace, run_idx: int, out_dir: Path) -> Dict[str, Any]:
    out_path = out_dir / f"BCAS_PHASE2_STRESS_{args.profile}_run{run_idx}.json"
    cmd = [
        "python",
        "scripts/bcas_phase2_stress.py",
        "--account-config",
        str(args.account_config),
        "--cases",
        str(args.cases),
        "--duration-s",
        str(args.duration_s),
        "--concurrency",
        str(args.concurrency),
        "--limit-bm25",
        str(args.limit_bm25),
        "--limit-similarity",
        str(args.limit_similarity),
        "--limit-sparse",
        str(args.limit_sparse),
        "--bm25-weight",
        str(args.bm25_weight),
        "--similarity-weight",
        str(args.similarity_weight),
        "--sparse-weight",
        str(args.sparse_weight),
        "--sparse-dimensions",
        str(args.sparse_dimensions),
        "--sparse-embedding-function",
        str(args.sparse_embedding_function),
        "--out",
        str(out_path),
    ]
    if bool(args.use_sparse):
        cmd.append("--use-sparse")
    t0 = time.time()
    proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
    elapsed = time.time() - t0
    payload: Dict[str, Any] = {
        "run_idx": run_idx,
        "command": cmd,
        "rc": int(proc.returncode),
        "elapsed_s": float(elapsed),
        "out": str(out_path),
        "stdout_tail": proc.stdout[-1200:],
        "stderr_tail": proc.stderr[-1200:],
    }
    if proc.returncode == 0 and out_path.exists():
        payload["result"] = json.loads(out_path.read_text(encoding="utf-8"))
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description="Run BCAS stress multiple times and summarize median metrics.")
    parser.add_argument(
        "--account-config",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_LIVE_EVAL_CASES_dense_heavy_augtrivia_2026-02-24.json"),
    )
    parser.add_argument("--duration-s", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit-bm25", type=int, default=12)
    parser.add_argument("--limit-similarity", type=int, default=12)
    parser.add_argument("--limit-sparse", type=int, default=0)
    parser.add_argument("--bm25-weight", type=float, default=0.9)
    parser.add_argument("--similarity-weight", type=float, default=0.1)
    parser.add_argument("--sparse-weight", type=float, default=0.0)
    parser.add_argument("--use-sparse", action="store_true")
    parser.add_argument("--sparse-dimensions", type=int, default=1024)
    parser.add_argument("--sparse-embedding-function", type=str, default="embedding_sparse")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--profile", type=str, default="c8_12x12")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs_tmp/RAG"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_STRESS_SUITE_2026-02-24.json"),
    )
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    run_payloads: List[Dict[str, Any]] = []
    for i in range(1, max(1, int(args.runs)) + 1):
        run_payloads.append(_run_one(args, i, out_dir))

    ok_runs = [r for r in run_payloads if r.get("rc") == 0 and isinstance(r.get("result"), dict)]
    lat_p50 = [float(r["result"]["latency_ms"]["p50"]) for r in ok_runs]
    lat_p95 = [float(r["result"]["latency_ms"]["p95"]) for r in ok_runs]
    lat_p99 = [float(r["result"]["latency_ms"]["p99"]) for r in ok_runs]
    lat_mean = [float(r["result"]["latency_ms"]["mean"]) for r in ok_runs]
    rps = [float(r["result"]["throughput"]["successful_requests_per_second"]) for r in ok_runs]
    error_rates = [float(r["result"]["counts"]["error_rate"]) for r in ok_runs]
    server_total_p50 = [
        float((((r["result"].get("server_timing") or {}).get("server_total_ms") or {}).get("p50", 0.0) or 0.0))
        for r in ok_runs
    ]
    queueing_p50 = [
        float((((r["result"].get("server_timing") or {}).get("queueing_estimate_ms") or {}).get("p50", 0.0) or 0.0))
        for r in ok_runs
    ]
    duration_coverage = [
        float(((r["result"].get("server_timing") or {}).get("duration_coverage_rate", 0.0) or 0.0))
        for r in ok_runs
    ]

    summary = {
        "requested_runs": int(args.runs),
        "successful_runs": len(ok_runs),
        "median_successful_rps": _median(rps),
        "median_latency_p50_ms": _median(lat_p50),
        "median_latency_p95_ms": _median(lat_p95),
        "median_latency_p99_ms": _median(lat_p99),
        "median_latency_mean_ms": _median(lat_mean),
        "median_server_total_p50_ms": _median(server_total_p50),
        "median_queueing_p50_ms": _median(queueing_p50),
        "median_duration_coverage_rate": _median(duration_coverage),
        "max_error_rate": max(error_rates) if error_rates else 1.0,
    }

    payload = {
        "generated_at_unix": time.time(),
        "profile": args.profile,
        "params": {
            "cases": str(args.cases),
            "duration_s": int(args.duration_s),
            "concurrency": int(args.concurrency),
            "limit_bm25": int(args.limit_bm25),
            "limit_similarity": int(args.limit_similarity),
            "limit_sparse": int(args.limit_sparse),
            "bm25_weight": float(args.bm25_weight),
            "similarity_weight": float(args.similarity_weight),
            "sparse_weight": float(args.sparse_weight),
            "use_sparse": bool(args.use_sparse),
            "sparse_dimensions": int(args.sparse_dimensions),
            "sparse_embedding_function": str(args.sparse_embedding_function),
            "runs": int(args.runs),
        },
        "summary": summary,
        "runs": run_payloads,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": summary}, indent=2))
    return 0 if len(ok_runs) > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
