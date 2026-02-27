#!/usr/bin/env python3
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.create_db_session import initialize_database_engine
from QueryLake.runtime.retrieval_runs import log_retrieval_run


def _mk_result_row(i: int) -> dict:
    return {
        "id": f"chunk_{i}",
        "text": f"Synthetic chunk text {i}",
        "md": {"collection_id": "synthetic_col"},
        "bm25_score": 0.2,
        "similarity_score": 0.3,
        "hybrid_score": 0.25,
    }


def _emit_run(db, *, route: str, latency: float, pipeline_id: str, idx: int) -> None:
    log_retrieval_run(
        db,
        route=route,
        actor_user="synthetic_runner",
        query_payload={"bm25": f"synthetic query {idx}", "embedding": f"synthetic query {idx}"},
        collection_ids=["synthetic_col"],
        pipeline_id=pipeline_id,
        pipeline_version="v1",
        filters={"collection_ids": ["synthetic_col"], "synthetic": True},
        budgets={"limit_bm25": 5, "limit_similarity": 5},
        timings={"total": max(0.001, float(latency))},
        counters={"rows_returned": 1, "synthetic": True},
        result_rows=[_mk_result_row(idx)],
        status="ok",
        md={"source": "synthetic_load"},
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic retrieval_run rows for local gate benchmarking.")
    parser.add_argument("--baseline-count", type=int, default=120)
    parser.add_argument("--candidate-count", type=int, default=120)
    parser.add_argument("--baseline-route", type=str, default="search_hybrid_baseline")
    parser.add_argument("--candidate-route", type=str, default="search_hybrid_candidate")
    parser.add_argument("--baseline-latency", type=float, default=0.30)
    parser.add_argument("--candidate-latency", type=float, default=0.31)
    parser.add_argument("--jitter", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=1337)
    args = parser.parse_args()

    random.seed(int(args.seed))
    db, _ = initialize_database_engine()

    for i in range(max(0, int(args.baseline_count))):
        latency = random.gauss(float(args.baseline_latency), float(args.jitter))
        _emit_run(
            db,
            route=args.baseline_route,
            latency=latency,
            pipeline_id="legacy.search_hybrid",
            idx=i,
        )

    for i in range(max(0, int(args.candidate_count))):
        latency = random.gauss(float(args.candidate_latency), float(args.jitter))
        _emit_run(
            db,
            route=args.candidate_route,
            latency=latency,
            pipeline_id="orchestrated.search_hybrid",
            idx=i,
        )

    print(
        {
            "baseline_count": int(args.baseline_count),
            "candidate_count": int(args.candidate_count),
            "baseline_route": args.baseline_route,
            "candidate_route": args.candidate_route,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

