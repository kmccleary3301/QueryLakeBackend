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

from querylake_sdk import QueryLakeClient


def _load_selected_files(selection_path: str) -> list[str]:
    payload = json.loads(Path(selection_path).expanduser().resolve().read_text(encoding="utf-8"))
    selected_files = payload.get("selected_files", payload) if isinstance(payload, dict) else payload
    if not isinstance(selected_files, list):
        raise SystemExit("--from-selection must contain a selected_files list or be a JSON list.")
    return [str(value) for value in selected_files]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--username", required=True)
    parser.add_argument("--password", required=True)
    parser.add_argument("--collection", required=True, help="Collection name (created if needed).")
    parser.add_argument("--dir", default=None)
    parser.add_argument(
        "--from-selection",
        default=None,
        help="Optional selection JSON from CLI upload-dir --selection-output.",
    )
    parser.add_argument("--pattern", default="*")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--extensions",
        default=None,
        help="Optional comma-separated extension filter for directory scan.",
    )
    parser.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Exclude files matching glob relative to --dir (repeatable).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Plan selected files without uploading or searching.")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first upload error.")
    parser.add_argument("--selection-output", default=None, help="Optional path to write selected file report JSON.")
    parser.add_argument("--upload-report-file", default=None, help="Optional path to write upload result JSON.")
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
    if not args.from_selection and not isinstance(args.dir, str):
        raise SystemExit("--dir is required unless --from-selection is provided.")
    include_extensions = None
    if isinstance(args.extensions, str) and args.extensions.strip():
        include_extensions = [part.strip() for part in args.extensions.split(",") if part.strip()]

    with QueryLakeClient(base_url=args.base_url) as client:
        if args.dry_run:
            file_paths = _load_selected_files(args.from_selection) if args.from_selection else None
            upload_result = client.upload_directory(
                collection_hash_id="dry_run",
                directory=args.dir,
                file_paths=file_paths,
                pattern=args.pattern,
                recursive=args.recursive,
                max_files=args.max_files,
                include_extensions=include_extensions,
                exclude_globs=args.exclude_glob,
                dry_run=True,
            )
            if args.selection_output:
                destination = Path(args.selection_output).expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(upload_result, indent=2, sort_keys=True), encoding="utf-8")
                upload_result["selection_output"] = str(destination)
            if args.upload_report_file:
                destination = Path(args.upload_report_file).expanduser().resolve()
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_text(json.dumps(upload_result, indent=2, sort_keys=True), encoding="utf-8")
                upload_result["upload_report_file"] = str(destination)
            print(json.dumps(upload_result, indent=2, sort_keys=True))
            return 0

        login_result = client.login(username=args.username, password=args.password)
        if not isinstance(login_result, dict) or not login_result.get("auth"):
            raise SystemExit("Login failed (missing auth token in response).")

        collection = client.create_collection(name=args.collection)
        collection_id = collection["hash_id"]

        file_paths = _load_selected_files(args.from_selection) if args.from_selection else None

        upload_result = client.upload_directory(
            collection_hash_id=collection_id,
            directory=args.dir,
            file_paths=file_paths,
            pattern=args.pattern,
            recursive=args.recursive,
            max_files=args.max_files,
            include_extensions=include_extensions,
            exclude_globs=args.exclude_glob,
            fail_fast=args.fail_fast,
            await_embedding=args.await_embedding,
            create_sparse_embeddings=args.sparse_embeddings,
        )

        if args.selection_output:
            destination = Path(args.selection_output).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(upload_result, indent=2, sort_keys=True), encoding="utf-8")
            upload_result["selection_output"] = str(destination)
        if args.upload_report_file:
            destination = Path(args.upload_report_file).expanduser().resolve()
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(json.dumps(upload_result, indent=2, sort_keys=True), encoding="utf-8")
            upload_result["upload_report_file"] = str(destination)

        if upload_result.get("uploaded", 0) <= 0:
            print(json.dumps({"collection_id": collection_id, "upload": upload_result, "results": []}, indent=2))
            return 1 if upload_result.get("failed", 0) else 0

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
        "upload": upload_result,
        "query": args.query,
        "results": rows[: args.top_k],
        "duration": metrics.get("duration", {}) if isinstance(metrics, dict) else {},
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
