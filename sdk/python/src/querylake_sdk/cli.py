from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from .client import QueryLakeClient
from .errors import QueryLakeError

CONFIG_DIR = Path.home() / ".querylake"
CONFIG_PATH = CONFIG_DIR / "sdk_profiles.json"


def _load_profiles() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"profiles": {}}
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"profiles": {}}
    if not isinstance(payload, dict):
        return {"profiles": {}}
    if "profiles" not in payload or not isinstance(payload["profiles"], dict):
        payload["profiles"] = {}
    return payload


def _save_profiles(payload: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_profile(name: Optional[str]) -> Dict[str, Any]:
    if not name:
        return {}
    profiles = _load_profiles().get("profiles", {})
    value = profiles.get(name, {})
    return value if isinstance(value, dict) else {}


def _resolve_auth(args: argparse.Namespace, profile: Dict[str, Any]) -> Dict[str, str]:
    if getattr(args, "oauth2", None):
        return {"oauth2": args.oauth2}
    if getattr(args, "api_key", None):
        return {"api_key": args.api_key}
    auth = profile.get("auth")
    if isinstance(auth, dict):
        return {str(k): str(v) for k, v in auth.items()}
    return {}


def _resolve_base_url(args: argparse.Namespace, profile: Dict[str, Any]) -> str:
    if getattr(args, "url", None):
        return args.url
    if isinstance(profile.get("base_url"), str) and profile["base_url"].strip():
        return profile["base_url"]
    return os.getenv("QUERYLAKE_BASE_URL", "http://127.0.0.1:8000")


def _build_client(args: argparse.Namespace, *, require_auth: bool = False) -> QueryLakeClient:
    profile = _resolve_profile(getattr(args, "profile", None))
    base_url = _resolve_base_url(args, profile)
    auth = _resolve_auth(args, profile)
    if require_auth and not auth:
        raise SystemExit(
            "No auth found. Use --oauth2/--api-key or run: querylake login --profile <name> ..."
        )
    return QueryLakeClient(base_url=base_url, auth=auth or None)


def _print_output(payload: Any, as_json: bool = True) -> None:
    if as_json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(payload)


def cmd_doctor(args: argparse.Namespace) -> int:
    client = _build_client(args)
    checks: Dict[str, Any] = {"base_url": client.base_url}
    try:
        checks["healthz"] = client.healthz()
        checks["readyz"] = client.readyz()
        checks["ping"] = client.ping()
        checks["models"] = client.list_models()
        checks["ok"] = True
    except QueryLakeError as exc:
        checks["ok"] = False
        checks["error"] = str(exc)
    finally:
        client.close()
    _print_output(checks, as_json=True)
    return 0 if checks.get("ok") else 1


def cmd_login(args: argparse.Namespace) -> int:
    if not args.username or not args.password:
        raise SystemExit("--username and --password are required")
    profile_name = args.profile or "default"
    client = QueryLakeClient(base_url=args.url)
    try:
        result = client.login(username=args.username, password=args.password)
    finally:
        client.close()

    token = result.get("auth") if isinstance(result, dict) else None
    if not isinstance(token, str) or not token:
        raise SystemExit("Login succeeded but no oauth2 token returned.")

    store = _load_profiles()
    profiles = store.setdefault("profiles", {})
    profiles[profile_name] = {
        "base_url": args.url,
        "auth": {"oauth2": token},
        "username": args.username,
    }
    _save_profiles(store)
    _print_output(
        {
            "saved_profile": profile_name,
            "config_path": str(CONFIG_PATH),
            "base_url": args.url,
            "username": args.username,
        },
        as_json=True,
    )
    return 0


def cmd_models(args: argparse.Namespace) -> int:
    client = _build_client(args)
    try:
        data = client.list_models()
    finally:
        client.close()
    _print_output(data, as_json=True)
    return 0


def cmd_profile_list(args: argparse.Namespace) -> int:
    store = _load_profiles()
    _print_output(store, as_json=True)
    return 0


def cmd_rag_create_collection(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    try:
        result = client.create_collection(
            name=args.name,
            description=args.description,
            public=args.public,
            organization_id=args.organization_id,
        )
    finally:
        client.close()
    _print_output(result, as_json=True)
    return 0


def cmd_rag_upload(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    try:
        result = client.upload_document(
            file_path=args.file,
            collection_hash_id=args.collection_id,
            scan_text=not args.no_scan,
            create_embeddings=not args.no_embeddings,
            create_sparse_embeddings=args.sparse_embeddings,
            await_embedding=args.await_embedding,
            sparse_embedding_dimensions=args.sparse_dimensions,
        )
    finally:
        client.close()
    _print_output(result, as_json=True)
    return 0


def cmd_rag_search(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    try:
        if args.mode == "bm25":
            rows = client.api(
                "search_bm25",
                {
                    "query": args.query,
                    "collection_ids": [args.collection_id],
                    "limit": max(args.top_k, args.limit_bm25),
                    "group_chunks": True,
                    "table": "document_chunk",
                },
            )
            if not isinstance(rows, list):
                rows = []
        else:
            rows = client.search_hybrid(
                query=args.query,
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
    finally:
        client.close()
    _print_output({"results": rows[: args.top_k], "total": len(rows)}, as_json=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="querylake",
        description="QueryLake SDK CLI (auth profiles, health checks, and RAG quick workflows).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser.add_argument("--url", default="http://127.0.0.1:8000", help="QueryLake base URL.")
    parser.add_argument("--profile", default=None, help="Saved profile name under ~/.querylake.")
    parser.add_argument("--oauth2", default=None, help="OAuth2 token override.")
    parser.add_argument("--api-key", dest="api_key", default=None, help="API key override.")

    p_doctor = subparsers.add_parser("doctor", help="Run connectivity and readiness checks.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_login = subparsers.add_parser("login", help="Login and save OAuth2 token to a profile.")
    p_login.add_argument("--username", required=True, help="Username.")
    p_login.add_argument("--password", required=True, help="Password.")
    p_login.add_argument(
        "--profile",
        default="default",
        help="Profile name to save. Defaults to 'default'.",
    )
    p_login.add_argument("--url", default="http://127.0.0.1:8000", help="QueryLake base URL.")
    p_login.set_defaults(func=cmd_login)

    p_models = subparsers.add_parser("models", help="List OpenAI-compatible models.")
    p_models.set_defaults(func=cmd_models)

    p_profile = subparsers.add_parser("profile", help="Profile utilities.")
    p_profile_sub = p_profile.add_subparsers(dest="profile_command", required=True)
    p_profile_list = p_profile_sub.add_parser("list", help="List saved profiles.")
    p_profile_list.set_defaults(func=cmd_profile_list)

    p_rag = subparsers.add_parser("rag", help="RAG workflow helpers.")
    rag_sub = p_rag.add_subparsers(dest="rag_command", required=True)

    p_rag_create = rag_sub.add_parser("create-collection", help="Create a document collection.")
    p_rag_create.add_argument("--name", required=True, help="Collection name.")
    p_rag_create.add_argument("--description", default=None, help="Collection description.")
    p_rag_create.add_argument("--public", action="store_true", help="Create a public collection.")
    p_rag_create.add_argument("--organization-id", type=int, default=None)
    p_rag_create.set_defaults(func=cmd_rag_create_collection)

    p_rag_upload = rag_sub.add_parser("upload", help="Upload a document to a collection.")
    p_rag_upload.add_argument("--collection-id", required=True)
    p_rag_upload.add_argument("--file", required=True)
    p_rag_upload.add_argument("--await-embedding", action="store_true")
    p_rag_upload.add_argument("--no-scan", action="store_true")
    p_rag_upload.add_argument("--no-embeddings", action="store_true")
    p_rag_upload.add_argument("--sparse-embeddings", action="store_true")
    p_rag_upload.add_argument("--sparse-dimensions", type=int, default=1024)
    p_rag_upload.set_defaults(func=cmd_rag_upload)

    p_rag_search = rag_sub.add_parser("search", help="Hybrid search over a collection.")
    p_rag_search.add_argument("--collection-id", required=True)
    p_rag_search.add_argument("--query", required=True)
    p_rag_search.add_argument("--mode", choices=["hybrid", "bm25"], default="hybrid")
    p_rag_search.add_argument("--top-k", type=int, default=5)
    p_rag_search.add_argument("--limit-bm25", type=int, default=12)
    p_rag_search.add_argument("--limit-similarity", type=int, default=12)
    p_rag_search.add_argument("--limit-sparse", type=int, default=0)
    p_rag_search.add_argument("--bm25-weight", type=float, default=0.55)
    p_rag_search.add_argument("--similarity-weight", type=float, default=0.45)
    p_rag_search.add_argument("--sparse-weight", type=float, default=0.0)
    p_rag_search.set_defaults(func=cmd_rag_search)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
