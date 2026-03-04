#!/usr/bin/env python3
"""Bootstrap isolated resources for SDK live integration checks."""

from __future__ import annotations

import argparse
import json
import os
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
    parser = argparse.ArgumentParser(description="Create or validate live integration collection resources.")
    parser.add_argument("--base-url", default="", help="Override QueryLake base URL.")
    parser.add_argument("--oauth2", default="", help="Override OAuth2 token.")
    parser.add_argument("--api-key", default="", help="Override API key.")
    parser.add_argument(
        "--collection-title",
        default="SDK Live Integration Fixture",
        help="Title for the isolated integration collection.",
    )
    parser.add_argument(
        "--output-json",
        default="docs_tmp/RAG/ci/live_integration/bootstrap_contract.json",
        help="Path for bootstrap contract output JSON.",
    )
    return parser.parse_args()


def _build_client(args: argparse.Namespace) -> QueryLakeClient:
    base_url = (args.base_url or _env("QUERYLAKE_LIVE_BASE_URL")).strip()
    oauth2 = (args.oauth2 or _env("QUERYLAKE_LIVE_OAUTH2")).strip()
    api_key = (args.api_key or _env("QUERYLAKE_LIVE_API_KEY")).strip()
    if not base_url:
        raise SystemExit("Missing base URL (use --base-url or QUERYLAKE_LIVE_BASE_URL).")
    if not oauth2 and not api_key:
        raise SystemExit("Missing auth credentials (oauth2/api-key).")
    kwargs: Dict[str, Any] = {"base_url": base_url}
    if oauth2:
        kwargs["oauth2"] = oauth2
    if api_key:
        kwargs["api_key"] = api_key
    return QueryLakeClient(**kwargs)


def _extract_collection_rows(payload: Any) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("collections", "rows", "result", "items"):
            candidate = payload.get(key)
            if isinstance(candidate, list):
                return [row for row in candidate if isinstance(row, dict)]
    return []


def _collection_identity(row: Dict[str, Any]) -> str:
    return str(
        row.get("hash_id")
        or row.get("collection_hash_id")
        or row.get("id")
        or ""
    ).strip()


def main() -> int:
    args = parse_args()
    client = _build_client(args)
    try:
        title = str(args.collection_title or "").strip()
        if not title:
            raise SystemExit("collection title must be non-empty")

        listed = client.list_collections()
        rows = _extract_collection_rows(listed)
        existing = next((row for row in rows if str(row.get("name") or row.get("title") or "").strip() == title), None)

        created = False
        if existing is None:
            created_payload = client.create_collection(
                name=title,
                description="Dedicated deterministic corpus for SDK live integration tests.",
                public=False,
            )
            created = True
            candidate_id = _collection_identity(created_payload)
            if candidate_id:
                collection_id = candidate_id
            else:
                # Fallback to fresh list lookup by title.
                listed = client.list_collections()
                rows = _extract_collection_rows(listed)
                matched = next(
                    (row for row in rows if str(row.get("name") or row.get("title") or "").strip() == title),
                    None,
                )
                if matched is None:
                    raise SystemExit("Collection created but could not be re-discovered by title.")
                existing = matched
                collection_id = _collection_identity(existing)
        else:
            collection_id = _collection_identity(existing)

        if not collection_id:
            raise SystemExit("Unable to resolve collection id for integration collection.")

        contract = {
            "base_url": client.base_url,
            "collection_title": title,
            "collection_id": collection_id,
            "created": created,
            "env_contract": {
                "QUERYLAKE_LIVE_BASE_URL": client.base_url,
                "QUERYLAKE_LIVE_TEST_COLLECTION_ID": collection_id,
                "QUERYLAKE_LIVE_QUERY_CASES_PATH": "sdk/python/tests/integration/live_query_cases.json",
                "QUERYLAKE_LIVE_STRICT_EXPECTATIONS": "1",
            },
        }

        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(contract, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps(contract, indent=2, sort_keys=True))
    finally:
        client.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
