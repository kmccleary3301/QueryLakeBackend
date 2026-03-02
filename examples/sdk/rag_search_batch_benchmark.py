#!/usr/bin/env python3
"""Batch benchmark helper for QueryLake hybrid retrieval profiles.

Usage:
  python examples/sdk/rag_search_batch_benchmark.py \
    --base-url http://127.0.0.1:8000 \
    --username demo --password demo-pass \
    --collection-id <collection_id> \
    --queries-file ./queries.txt \
    --output-file ./artifacts/benchmark.json

Offline tutorial mode:
  python examples/sdk/rag_search_batch_benchmark.py \
    --offline-demo \
    --queries-file ./examples/sdk/fixtures/offline_queries.txt \
    --output-file ./artifacts/benchmark_offline.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Dict, List

from querylake_sdk import QueryLakeClient


def _load_queries(path_value: str) -> List[str]:
    path = Path(path_value).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise SystemExit(f"--queries-file must be an existing file: {path}")
    rows = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        rows.append(line)
    if not rows:
        raise SystemExit(f"--queries-file did not contain any usable queries: {path}")
    return rows


def _offline_rows() -> List[Dict[str, Any]]:
    fixture = Path(__file__).resolve().parent / "fixtures" / "offline_search_rows.json"
    payload = json.loads(fixture.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"Offline fixture must be a JSON list: {fixture}")
    return [row for row in payload if isinstance(row, dict)]


def _duration_ms(entry: Dict[str, Any]) -> float:
    duration = entry.get("duration")
    if isinstance(duration, dict):
        for key in ("total_ms", "total", "ms"):
            value = duration.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return 0.0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password", default=None)
    parser.add_argument("--collection-id", default=None)
    parser.add_argument("--queries-file", required=True)
    parser.add_argument("--output-file", default=None)
    parser.add_argument("--offline-demo", action="store_true")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit-bm25", type=int, default=12)
    parser.add_argument("--limit-similarity", type=int, default=12)
    parser.add_argument("--limit-sparse", type=int, default=0)
    parser.add_argument("--bm25-weight", type=float, default=0.55)
    parser.add_argument("--similarity-weight", type=float, default=0.45)
    parser.add_argument("--sparse-weight", type=float, default=0.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    queries = _load_queries(args.queries_file)

    if args.offline_demo:
        rows = _offline_rows()
        results = []
        for query in queries:
            # Deterministic synthetic latency shape for tutorial mode.
            duration_ms = float(8 + (len(query) % 5))
            results.append(
                {
                    "query": query,
                    "rows": rows[: args.top_k],
                    "total": len(rows),
                    "duration": {"total_ms": duration_ms},
                }
            )
        payload = {
            "mode": "offline-demo",
            "query_count": len(results),
            "summary": {
                "avg_result_count": sum(item["total"] for item in results) / len(results),
                "avg_duration_ms": sum(_duration_ms(item) for item in results) / len(results),
            },
            "results": results,
            "_meta": {
                "generated_at_unix": time.time(),
                "offline_demo": True,
                "fixture": "examples/sdk/fixtures/offline_search_rows.json",
                "queries_file": str(Path(args.queries_file).expanduser().resolve()),
            },
        }
    else:
        if not isinstance(args.collection_id, str) or not args.collection_id.strip():
            raise SystemExit("--collection-id is required unless --offline-demo is set.")
        if not isinstance(args.username, str) or not args.username.strip():
            raise SystemExit("--username is required unless --offline-demo is set.")
        if not isinstance(args.password, str) or not args.password:
            raise SystemExit("--password is required unless --offline-demo is set.")

        with QueryLakeClient(base_url=args.base_url) as client:
            login = client.login(username=args.username, password=args.password)
            if not isinstance(login, dict) or not login.get("auth"):
                raise SystemExit("Login failed (missing auth token).")

            results = []
            for query in queries:
                response = client.search_hybrid_with_metrics(
                    query=query,
                    collection_ids=[args.collection_id],
                    limit_bm25=args.limit_bm25,
                    limit_similarity=args.limit_similarity,
                    limit_sparse=args.limit_sparse,
                    bm25_weight=args.bm25_weight,
                    similarity_weight=args.similarity_weight,
                    sparse_weight=args.sparse_weight,
                    group_chunks=True,
                    rerank=False,
                )
                rows = response.get("rows", []) if isinstance(response, dict) else []
                results.append(
                    {
                        "query": query,
                        "rows": rows[: args.top_k],
                        "total": len(rows),
                        "duration": response.get("duration", {}) if isinstance(response, dict) else {},
                    }
                )

        payload = {
            "mode": "live",
            "base_url": args.base_url,
            "collection_id": args.collection_id,
            "query_count": len(results),
            "summary": {
                "avg_result_count": sum(item["total"] for item in results) / len(results),
                "avg_duration_ms": sum(_duration_ms(item) for item in results) / len(results),
            },
            "results": results,
            "_meta": {
                "generated_at_unix": time.time(),
                "queries_file": str(Path(args.queries_file).expanduser().resolve()),
            },
        }

    if args.output_file:
        destination = Path(args.output_file).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["output_file"] = str(destination)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
