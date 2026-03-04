#!/usr/bin/env python3
"""Seed deterministic documents for SDK live integration retrieval checks."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List

try:
    from querylake_sdk import QueryLakeClient
except ModuleNotFoundError:  # pragma: no cover - repo-local execution fallback
    import sys

    REPO_ROOT = Path(__file__).resolve().parents[2]
    SDK_SRC = REPO_ROOT / "sdk/python/src"
    if SDK_SRC.exists():
        sys.path.insert(0, str(SDK_SRC))
        from querylake_sdk import QueryLakeClient
    else:
        raise


def _env(name: str) -> str:
    return (os.getenv(name) or "").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seed deterministic live integration fixture documents.")
    parser.add_argument(
        "--fixture-path",
        default="sdk/python/tests/integration/live_query_cases.json",
        help="Path to fixture JSON containing a 'documents' list.",
    )
    parser.add_argument(
        "--collection-id",
        default="",
        help="Collection hash id. Defaults to QUERYLAKE_LIVE_TEST_COLLECTION_ID.",
    )
    parser.add_argument(
        "--create-embeddings",
        action="store_true",
        help="If set, request dense embedding generation during upload.",
    )
    parser.add_argument(
        "--create-sparse-embeddings",
        action="store_true",
        help="If set, request sparse embedding generation during upload.",
    )
    parser.add_argument(
        "--await-embedding",
        action="store_true",
        help="If set, wait for embedding pipeline completion at upload time.",
    )
    parser.add_argument(
        "--output-json",
        default="docs_tmp/RAG/ci/live_integration/seed_fixture_results.json",
        help="Where to write seed result manifest JSON.",
    )
    return parser.parse_args()


def _build_client() -> QueryLakeClient:
    base_url = _env("QUERYLAKE_LIVE_BASE_URL")
    oauth2 = _env("QUERYLAKE_LIVE_OAUTH2")
    api_key = _env("QUERYLAKE_LIVE_API_KEY")
    if not base_url:
        raise SystemExit("Missing QUERYLAKE_LIVE_BASE_URL.")
    if not oauth2 and not api_key:
        raise SystemExit("Missing QUERYLAKE_LIVE_OAUTH2 / QUERYLAKE_LIVE_API_KEY.")

    kwargs: Dict[str, Any] = {"base_url": base_url}
    if oauth2:
        kwargs["oauth2"] = oauth2
    if api_key:
        kwargs["api_key"] = api_key
    return QueryLakeClient(**kwargs)


def _load_documents(path: Path) -> List[Dict[str, str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - utility script
        raise SystemExit(f"Failed to parse fixture JSON: {path} ({exc})")
    docs = payload.get("documents")
    if not isinstance(docs, list) or not docs:
        raise SystemExit(f"Fixture must include non-empty 'documents' list: {path}")
    normalized: List[Dict[str, str]] = []
    for raw in docs:
        if not isinstance(raw, dict):
            continue
        doc_id = str(raw.get("doc_id") or "").strip()
        filename = str(raw.get("filename") or f"{doc_id or 'seed'}.txt").strip()
        content = str(raw.get("content") or "").strip()
        if not content:
            continue
        normalized.append(
            {
                "doc_id": doc_id or filename,
                "filename": filename,
                "content": content,
            }
        )
    if not normalized:
        raise SystemExit("No valid fixture documents found.")
    return normalized


def main() -> int:
    args = parse_args()
    fixture_path = Path(args.fixture_path).expanduser().resolve()
    if not fixture_path.exists():
        raise SystemExit(f"Fixture path does not exist: {fixture_path}")

    collection_id = (args.collection_id or _env("QUERYLAKE_LIVE_TEST_COLLECTION_ID")).strip()
    if not collection_id:
        raise SystemExit("Missing collection id (pass --collection-id or set QUERYLAKE_LIVE_TEST_COLLECTION_ID).")

    docs = _load_documents(fixture_path)
    output_path = Path(args.output_json).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fixture_tag = fixture_path.stem
    results: List[Dict[str, Any]] = []
    client = _build_client()
    try:
        with tempfile.TemporaryDirectory(prefix="qlsdk_live_seed_") as tmp:
            tmp_path = Path(tmp)
            for doc in docs:
                file_path = tmp_path / doc["filename"]
                file_path.write_text(doc["content"], encoding="utf-8")
                idempotency_key = f"qlsdk-live-seed:{fixture_tag}:{doc['doc_id']}"
                response = client.upload_document(
                    file_path=file_path,
                    collection_hash_id=collection_id,
                    scan_text=True,
                    create_embeddings=bool(args.create_embeddings),
                    create_sparse_embeddings=bool(args.create_sparse_embeddings),
                    await_embedding=bool(args.await_embedding),
                    idempotency_key=idempotency_key,
                    document_metadata={
                        "source": "sdk_live_seed_fixture",
                        "fixture": fixture_tag,
                        "seed_doc_id": doc["doc_id"],
                    },
                )
                results.append(
                    {
                        "doc_id": doc["doc_id"],
                        "filename": doc["filename"],
                        "idempotency_key": idempotency_key,
                        "hash_id": response.get("hash_id"),
                        "created": bool(response.get("created", True)),
                    }
                )
    finally:
        client.close()

    summary = {
        "fixture_path": str(fixture_path),
        "collection_id": collection_id,
        "documents_seeded": len(results),
        "results": results,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
