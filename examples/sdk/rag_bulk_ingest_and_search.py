#!/usr/bin/env python3
"""Bulk-ingest a directory and run hybrid retrieval with querylake-sdk.

Usage:
  python examples/sdk/rag_bulk_ingest_and_search.py \
    --base-url http://127.0.0.1:8000 \
    --username demo --password demo-pass \
    --collection "sdk-bulk-demo" \
    --dir ./documents \
    --pattern "*.pdf" \
    --recursive \
    --query "main contribution"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

from querylake_sdk import QueryLakeClient


def resolve_files(directory: str, *, pattern: str, recursive: bool, max_files: int | None) -> List[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--dir must be an existing directory: {root}")
    iterator: Iterable[Path] = root.rglob(pattern) if recursive else root.glob(pattern)
    files = sorted(path for path in iterator if path.is_file())
    if max_files is not None:
        files = files[: max(0, max_files)]
    return files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--collection", required=True, help="Collection name (created if needed).")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--query", required=True)
    parser.add_argument("--await-embedding", action="store_true")
    parser.add_argument("--sparse-embeddings", action="store_true")
    parser.add_argument("--limit-bm25", type=int, default=12)
    parser.add_argument("--limit-similarity", type=int, default=12)
    parser.add_argument("--limit-sparse", type=int, default=0)
    parser.add_argument("--bm25-weight", type=float, default=0.55)
    parser.add_argument("--similarity-weight", type=float, default=0.45)
    parser.add_argument("--sparse-weight", type=float, default=0.0)
    parser.add_argument("--top-k", type=int, default=5)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    files = resolve_files(
        args.dir,
        pattern=args.pattern,
        recursive=args.recursive,
        max_files=args.max_files,
    )
    if not files:
        raise SystemExit(f"No files found in {Path(args.dir).expanduser().resolve()} for pattern {args.pattern!r}.")

    with QueryLakeClient(base_url=args.base_url) as client:
        login_result = client.login(username=args.username, password=args.password)
        if not isinstance(login_result, dict) or not login_result.get("auth"):
            raise SystemExit("Login failed (missing auth token in response).")

        collection = client.create_collection(name=args.collection)
        collection_id = collection["hash_id"]

        uploaded = 0
        for file_path in files:
            client.upload_document(
                file_path=file_path,
                collection_hash_id=collection_id,
                await_embedding=args.await_embedding,
                create_sparse_embeddings=args.sparse_embeddings,
            )
            uploaded += 1

        metrics = client.search_hybrid_with_metrics(
            query=args.query,
            collection_ids=[collection_id],
            limit_bm25=args.limit_bm25,
            limit_similarity=args.limit_similarity,
            limit_sparse=args.limit_sparse,
            bm25_weight=args.bm25_weight,
            similarity_weight=args.similarity_weight,
            sparse_weight=args.sparse_weight,
            group_chunks=True,
            rerank=False,
        )

    rows = metrics.get("rows", []) if isinstance(metrics, dict) else []
    payload = {
        "collection_id": collection_id,
        "uploaded_files": uploaded,
        "query": args.query,
        "results": rows[: args.top_k],
        "duration": metrics.get("duration", {}) if isinstance(metrics, dict) else {},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
