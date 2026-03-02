from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import QueryLakeClient
from .errors import QueryLakeError

CONFIG_DIR = Path.home() / ".querylake"
CONFIG_PATH = CONFIG_DIR / "sdk_profiles.json"


def _load_profiles() -> Dict[str, Any]:
    if not CONFIG_PATH.exists():
        return {"profiles": {}, "active_profile": None}
    try:
        payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"profiles": {}, "active_profile": None}
    if not isinstance(payload, dict):
        return {"profiles": {}, "active_profile": None}
    if "profiles" not in payload or not isinstance(payload["profiles"], dict):
        payload["profiles"] = {}
    active = payload.get("active_profile")
    if active is not None and not isinstance(active, str):
        payload["active_profile"] = None
    if "active_profile" not in payload:
        payload["active_profile"] = None
    return payload


def _save_profiles(payload: Dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _resolve_profile(name: Optional[str]) -> Dict[str, Any]:
    if not isinstance(name, str) or not name.strip():
        return {}
    profiles = _load_profiles().get("profiles", {})
    value = profiles.get(name.strip(), {})
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
    explicit = getattr(args, "url", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if isinstance(profile.get("base_url"), str) and profile["base_url"].strip():
        return profile["base_url"]
    return os.getenv("QUERYLAKE_BASE_URL", "http://127.0.0.1:8000")


def _resolve_profile_name(args: argparse.Namespace) -> Optional[str]:
    explicit = getattr(args, "profile", None)
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    store = _load_profiles()
    active = store.get("active_profile")
    profiles = store.get("profiles", {})
    if isinstance(active, str) and active in profiles:
        return active
    return None


def _build_client(args: argparse.Namespace, *, require_auth: bool = False) -> QueryLakeClient:
    profile_name = _resolve_profile_name(args)
    profile = _resolve_profile(profile_name)
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
    if not args.username:
        raise SystemExit("--username is required")
    password = args.password
    if not isinstance(password, str) or not password:
        password = getpass.getpass("QueryLake password: ")
    if not password:
        raise SystemExit("Password is required.")
    profile_name = args.profile or "default"
    base_url = _resolve_base_url(args, {})
    client = QueryLakeClient(base_url=base_url)
    try:
        result = client.login(username=args.username, password=password)
    finally:
        client.close()

    token = result.get("auth") if isinstance(result, dict) else None
    if not isinstance(token, str) or not token:
        raise SystemExit("Login succeeded but no oauth2 token returned.")

    store = _load_profiles()
    profiles = store.setdefault("profiles", {})
    profiles[profile_name] = {
        "base_url": base_url,
        "auth": {"oauth2": token},
        "username": args.username,
    }
    store["active_profile"] = profile_name
    _save_profiles(store)
    _print_output(
        {
            "saved_profile": profile_name,
            "config_path": str(CONFIG_PATH),
            "base_url": base_url,
            "username": args.username,
            "active_profile": profile_name,
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


def cmd_profile_show(args: argparse.Namespace) -> int:
    store = _load_profiles()
    profiles = store.get("profiles", {})
    name = args.name or _resolve_profile_name(args)
    if not name:
        raise SystemExit("No profile selected. Pass --name or set an active profile.")
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        raise SystemExit(f"Profile not found: {name}")
    _print_output(
        {
            "profile": name,
            "active": store.get("active_profile") == name,
            "data": profile,
            "config_path": str(CONFIG_PATH),
        },
        as_json=True,
    )
    return 0


def cmd_profile_set_default(args: argparse.Namespace) -> int:
    store = _load_profiles()
    profiles = store.get("profiles", {})
    name = args.name
    if name not in profiles:
        raise SystemExit(f"Profile not found: {name}")
    store["active_profile"] = name
    _save_profiles(store)
    _print_output({"active_profile": name, "config_path": str(CONFIG_PATH)}, as_json=True)
    return 0


def cmd_profile_set_url(args: argparse.Namespace) -> int:
    store = _load_profiles()
    profiles = store.get("profiles", {})
    name = args.name or _resolve_profile_name(args)
    if not name:
        raise SystemExit("No profile selected. Pass --name or set an active profile.")
    profile = profiles.get(name)
    if not isinstance(profile, dict):
        raise SystemExit(f"Profile not found: {name}")
    profile["base_url"] = args.url.strip()
    profiles[name] = profile
    if store.get("active_profile") is None:
        store["active_profile"] = name
    _save_profiles(store)
    _print_output({"profile": name, "base_url": profile["base_url"]}, as_json=True)
    return 0


def cmd_profile_delete(args: argparse.Namespace) -> int:
    store = _load_profiles()
    profiles = store.get("profiles", {})
    name = args.name
    if name not in profiles:
        raise SystemExit(f"Profile not found: {name}")
    profiles.pop(name, None)
    if store.get("active_profile") == name:
        store["active_profile"] = None
    _save_profiles(store)
    _print_output({"deleted_profile": name, "active_profile": store.get("active_profile")}, as_json=True)
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


def cmd_rag_list_collections(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    try:
        result = client.list_collections(
            organization_id=args.organization_id,
            global_collections=args.global_collections,
        )
    finally:
        client.close()
    _print_output(result, as_json=True)
    return 0


def cmd_rag_list_documents(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    try:
        rows = client.list_collection_documents(
            collection_hash_id=args.collection_id,
            limit=args.limit,
            offset=args.offset,
        )
    finally:
        client.close()
    _print_output({"documents": rows, "count": len(rows)}, as_json=True)
    return 0


def cmd_rag_count_chunks(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    collection_ids = None
    if isinstance(args.collection_ids, str) and args.collection_ids.strip():
        collection_ids = [part.strip() for part in args.collection_ids.split(",") if part.strip()]
    try:
        result = client.count_chunks(collection_ids=collection_ids)
    finally:
        client.close()
    _print_output(result, as_json=True)
    return 0


def cmd_rag_delete_document(args: argparse.Namespace) -> int:
    if not args.yes:
        raise SystemExit("Refusing delete without --yes. This operation is destructive.")
    client = _build_client(args, require_auth=True)
    try:
        result = client.delete_document(document_hash_id=args.document_id)
    finally:
        client.close()
    _print_output({"deleted_document_id": args.document_id, "result": result}, as_json=True)
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


def _resolve_upload_dir_files(
    directory: str,
    *,
    pattern: str,
    recursive: bool,
    max_files: Optional[int],
) -> List[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--dir must be an existing directory: {root}")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    files = [path for path in iterator if path.is_file()]
    files.sort()
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    return files


def cmd_rag_upload_dir(args: argparse.Namespace) -> int:
    files = _resolve_upload_dir_files(
        args.dir,
        pattern=args.pattern,
        recursive=args.recursive,
        max_files=args.max_files,
    )
    if not files:
        raise SystemExit(
            f"No files found under {Path(args.dir).expanduser().resolve()} matching pattern {args.pattern!r}."
        )

    client = _build_client(args, require_auth=True)
    uploaded = 0
    failed = 0
    errors: List[Dict[str, str]] = []
    try:
        for path in files:
            try:
                client.upload_document(
                    file_path=path,
                    collection_hash_id=args.collection_id,
                    scan_text=not args.no_scan,
                    create_embeddings=not args.no_embeddings,
                    create_sparse_embeddings=args.sparse_embeddings,
                    await_embedding=args.await_embedding,
                    sparse_embedding_dimensions=args.sparse_dimensions,
                )
                uploaded += 1
            except Exception as exc:  # noqa: BLE001
                failed += 1
                errors.append({"file": str(path), "error": str(exc)})
                if args.fail_fast:
                    break
    finally:
        client.close()

    payload: Dict[str, Any] = {
        "directory": str(Path(args.dir).expanduser().resolve()),
        "pattern": args.pattern,
        "recursive": bool(args.recursive),
        "requested_files": len(files),
        "uploaded": uploaded,
        "failed": failed,
        "fail_fast": bool(args.fail_fast),
    }
    if errors:
        payload["errors"] = errors
    _print_output(payload, as_json=True)
    return 0 if failed == 0 else 1


def _hybrid_search_defaults_for_preset(preset: str) -> Dict[str, float]:
    profiles: Dict[str, Dict[str, float]] = {
        "balanced": {
            "limit_bm25": 12,
            "limit_similarity": 12,
            "limit_sparse": 0,
            "bm25_weight": 0.55,
            "similarity_weight": 0.45,
            "sparse_weight": 0.0,
        },
        "tri-lane": {
            "limit_bm25": 12,
            "limit_similarity": 12,
            "limit_sparse": 12,
            "bm25_weight": 0.40,
            "similarity_weight": 0.40,
            "sparse_weight": 0.20,
        },
        "lexical-heavy": {
            "limit_bm25": 16,
            "limit_similarity": 8,
            "limit_sparse": 0,
            "bm25_weight": 0.75,
            "similarity_weight": 0.25,
            "sparse_weight": 0.0,
        },
        "semantic-heavy": {
            "limit_bm25": 8,
            "limit_similarity": 16,
            "limit_sparse": 0,
            "bm25_weight": 0.25,
            "similarity_weight": 0.75,
            "sparse_weight": 0.0,
        },
        "sparse-heavy": {
            "limit_bm25": 8,
            "limit_similarity": 8,
            "limit_sparse": 24,
            "bm25_weight": 0.30,
            "similarity_weight": 0.30,
            "sparse_weight": 0.40,
        },
    }
    return dict(profiles[preset])


def _resolve_search_profile(args: argparse.Namespace) -> Dict[str, float]:
    preset = _hybrid_search_defaults_for_preset(args.preset)
    limit_bm25 = int(args.limit_bm25) if args.limit_bm25 is not None else int(preset["limit_bm25"])
    limit_similarity = (
        int(args.limit_similarity)
        if args.limit_similarity is not None
        else int(preset["limit_similarity"])
    )
    limit_sparse = (
        int(args.limit_sparse) if args.limit_sparse is not None else int(preset["limit_sparse"])
    )
    bm25_weight = (
        float(args.bm25_weight) if args.bm25_weight is not None else float(preset["bm25_weight"])
    )
    similarity_weight = (
        float(args.similarity_weight)
        if args.similarity_weight is not None
        else float(preset["similarity_weight"])
    )
    sparse_weight = (
        float(args.sparse_weight) if args.sparse_weight is not None else float(preset["sparse_weight"])
    )
    return {
        "limit_bm25": limit_bm25,
        "limit_similarity": limit_similarity,
        "limit_sparse": limit_sparse,
        "bm25_weight": bm25_weight,
        "similarity_weight": similarity_weight,
        "sparse_weight": sparse_weight,
    }

def _run_rag_search(client: QueryLakeClient, args: argparse.Namespace, query: str) -> Dict[str, Any]:
    profile = _resolve_search_profile(args)
    if args.mode == "bm25":
        rows = client.api(
            "search_bm25",
            {
                "query": query,
                "collection_ids": [args.collection_id],
                "limit": max(args.top_k, int(profile["limit_bm25"])),
                "group_chunks": True,
                "table": "document_chunk",
            },
        )
        if not isinstance(rows, list):
            rows = []
        return {"results": rows[: args.top_k], "total": len(rows)}

    if args.with_metrics:
        metrics_payload = client.search_hybrid_with_metrics(
            query=query,
            collection_ids=[args.collection_id],
            limit_bm25=int(profile["limit_bm25"]),
            limit_similarity=int(profile["limit_similarity"]),
            limit_sparse=int(profile["limit_sparse"]),
            bm25_weight=float(profile["bm25_weight"]),
            similarity_weight=float(profile["similarity_weight"]),
            sparse_weight=float(profile["sparse_weight"]),
            group_chunks=True,
            rerank=False,
        )
        rows = metrics_payload.get("rows", []) if isinstance(metrics_payload, dict) else []
        if not isinstance(rows, list):
            rows = []
        payload: Dict[str, Any] = {"results": rows[: args.top_k], "total": len(rows)}
        if isinstance(metrics_payload, dict):
            if "duration" in metrics_payload:
                payload["duration"] = metrics_payload["duration"]
            if "profile" in metrics_payload:
                payload["profile"] = metrics_payload["profile"]
            if "constraint_hits" in metrics_payload:
                payload["constraint_hits"] = metrics_payload["constraint_hits"]
        return payload

    rows = client.search_hybrid(
        query=query,
        collection_ids=[args.collection_id],
        limit_bm25=int(profile["limit_bm25"]),
        limit_similarity=int(profile["limit_similarity"]),
        limit_sparse=int(profile["limit_sparse"]),
        bm25_weight=float(profile["bm25_weight"]),
        similarity_weight=float(profile["similarity_weight"]),
        sparse_weight=float(profile["sparse_weight"]),
        group_chunks=True,
        rerank=False,
    )
    return {"results": rows[: args.top_k], "total": len(rows)}


def _load_queries(path: str, *, max_queries: Optional[int]) -> List[str]:
    file_path = Path(path).expanduser().resolve()
    if not file_path.exists() or not file_path.is_file():
        raise SystemExit(f"--queries-file must be an existing file: {file_path}")
    queries: List[str] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        queries.append(line)
        if max_queries is not None and len(queries) >= max(0, int(max_queries)):
            break
    if not queries:
        raise SystemExit(f"No queries found in {file_path}")
    return queries


def cmd_rag_search(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    try:
        payload = _run_rag_search(client, args, args.query)
    finally:
        client.close()
    _print_output(payload, as_json=True)
    return 0


def cmd_rag_search_batch(args: argparse.Namespace) -> int:
    queries = _load_queries(args.queries_file, max_queries=args.max_queries)
    client = _build_client(args, require_auth=True)
    outputs: List[Dict[str, Any]] = []
    try:
        for query in queries:
            result = _run_rag_search(client, args, query)
            outputs.append({"query": query, **result})
    finally:
        client.close()

    payload = {
        "query_count": len(outputs),
        "mode": args.mode,
        "preset": args.preset,
        "results": outputs,
    }
    _print_output(payload, as_json=True)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="querylake",
        description="QueryLake SDK CLI (auth profiles, health checks, and RAG quick workflows).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parser.add_argument("--url", default=None, help="QueryLake base URL override.")
    parser.add_argument("--profile", default=None, help="Saved profile name under ~/.querylake.")
    parser.add_argument("--oauth2", default=None, help="OAuth2 token override.")
    parser.add_argument("--api-key", dest="api_key", default=None, help="API key override.")

    p_doctor = subparsers.add_parser("doctor", help="Run connectivity and readiness checks.")
    p_doctor.set_defaults(func=cmd_doctor)

    p_login = subparsers.add_parser("login", help="Login and save OAuth2 token to a profile.")
    p_login.add_argument("--username", required=True, help="Username.")
    p_login.add_argument("--password", default=None, help="Password. If omitted, CLI prompts securely.")
    p_login.add_argument(
        "--profile",
        default="default",
        help="Profile name to save. Defaults to 'default'.",
    )
    p_login.add_argument("--url", default=None, help="QueryLake base URL.")
    p_login.set_defaults(func=cmd_login)

    p_models = subparsers.add_parser("models", help="List OpenAI-compatible models.")
    p_models.set_defaults(func=cmd_models)

    p_profile = subparsers.add_parser("profile", help="Profile utilities.")
    p_profile_sub = p_profile.add_subparsers(dest="profile_command", required=True)
    p_profile_list = p_profile_sub.add_parser("list", help="List saved profiles.")
    p_profile_list.set_defaults(func=cmd_profile_list)
    p_profile_show = p_profile_sub.add_parser("show", help="Show one profile (or active profile).")
    p_profile_show.add_argument("--name", default=None, help="Profile name.")
    p_profile_show.set_defaults(func=cmd_profile_show)
    p_profile_default = p_profile_sub.add_parser("set-default", help="Set active default profile.")
    p_profile_default.add_argument("--name", required=True, help="Profile name.")
    p_profile_default.set_defaults(func=cmd_profile_set_default)
    p_profile_set_url = p_profile_sub.add_parser("set-url", help="Set base URL for a profile.")
    p_profile_set_url.add_argument("--name", default=None, help="Profile name (defaults to active).")
    p_profile_set_url.add_argument("--url", required=True, help="QueryLake base URL.")
    p_profile_set_url.set_defaults(func=cmd_profile_set_url)
    p_profile_delete = p_profile_sub.add_parser("delete", help="Delete a saved profile.")
    p_profile_delete.add_argument("--name", required=True, help="Profile name.")
    p_profile_delete.set_defaults(func=cmd_profile_delete)

    p_rag = subparsers.add_parser("rag", help="RAG workflow helpers.")
    rag_sub = p_rag.add_subparsers(dest="rag_command", required=True)

    p_rag_create = rag_sub.add_parser("create-collection", help="Create a document collection.")
    p_rag_create.add_argument("--name", required=True, help="Collection name.")
    p_rag_create.add_argument("--description", default=None, help="Collection description.")
    p_rag_create.add_argument("--public", action="store_true", help="Create a public collection.")
    p_rag_create.add_argument("--organization-id", type=int, default=None)
    p_rag_create.set_defaults(func=cmd_rag_create_collection)

    p_rag_list_collections = rag_sub.add_parser(
        "list-collections",
        help="List collections visible to the authenticated user.",
    )
    p_rag_list_collections.add_argument("--organization-id", type=int, default=None)
    p_rag_list_collections.add_argument(
        "--global-collections",
        action="store_true",
        help="Request globally visible collections.",
    )
    p_rag_list_collections.set_defaults(func=cmd_rag_list_collections)

    p_rag_list_documents = rag_sub.add_parser(
        "list-documents",
        help="List documents in a collection.",
    )
    p_rag_list_documents.add_argument("--collection-id", required=True)
    p_rag_list_documents.add_argument("--limit", type=int, default=100)
    p_rag_list_documents.add_argument("--offset", type=int, default=0)
    p_rag_list_documents.set_defaults(func=cmd_rag_list_documents)

    p_rag_count_chunks = rag_sub.add_parser(
        "count-chunks",
        help="Count indexed chunks across all or selected collections.",
    )
    p_rag_count_chunks.add_argument(
        "--collection-ids",
        default=None,
        help="Optional comma-separated collection IDs.",
    )
    p_rag_count_chunks.set_defaults(func=cmd_rag_count_chunks)

    p_rag_delete_document = rag_sub.add_parser(
        "delete-document",
        help="Delete one document by hash ID.",
    )
    p_rag_delete_document.add_argument("--document-id", required=True)
    p_rag_delete_document.add_argument(
        "--yes",
        action="store_true",
        help="Required confirmation flag for destructive delete.",
    )
    p_rag_delete_document.set_defaults(func=cmd_rag_delete_document)

    p_rag_upload = rag_sub.add_parser("upload", help="Upload a document to a collection.")
    p_rag_upload.add_argument("--collection-id", required=True)
    p_rag_upload.add_argument("--file", required=True)
    p_rag_upload.add_argument("--await-embedding", action="store_true")
    p_rag_upload.add_argument("--no-scan", action="store_true")
    p_rag_upload.add_argument("--no-embeddings", action="store_true")
    p_rag_upload.add_argument("--sparse-embeddings", action="store_true")
    p_rag_upload.add_argument("--sparse-dimensions", type=int, default=1024)
    p_rag_upload.set_defaults(func=cmd_rag_upload)

    p_rag_upload_dir = rag_sub.add_parser(
        "upload-dir",
        help="Bulk upload documents from a directory to a collection.",
    )
    p_rag_upload_dir.add_argument("--collection-id", required=True)
    p_rag_upload_dir.add_argument("--dir", required=True, help="Directory containing source files.")
    p_rag_upload_dir.add_argument(
        "--pattern",
        default="*",
        help="Glob pattern for file selection (default: '*').",
    )
    p_rag_upload_dir.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively include matching files from subdirectories.",
    )
    p_rag_upload_dir.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Optional cap on number of files to upload after sorting.",
    )
    p_rag_upload_dir.add_argument("--await-embedding", action="store_true")
    p_rag_upload_dir.add_argument("--no-scan", action="store_true")
    p_rag_upload_dir.add_argument("--no-embeddings", action="store_true")
    p_rag_upload_dir.add_argument("--sparse-embeddings", action="store_true")
    p_rag_upload_dir.add_argument("--sparse-dimensions", type=int, default=1024)
    p_rag_upload_dir.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately on first upload failure.",
    )
    p_rag_upload_dir.set_defaults(func=cmd_rag_upload_dir)

    p_rag_search = rag_sub.add_parser("search", help="Hybrid search over a collection.")
    p_rag_search.add_argument("--collection-id", required=True)
    p_rag_search.add_argument("--query", required=True)
    p_rag_search.add_argument("--mode", choices=["hybrid", "bm25"], default="hybrid")
    p_rag_search.add_argument(
        "--preset",
        choices=["balanced", "tri-lane", "lexical-heavy", "semantic-heavy", "sparse-heavy"],
        default="balanced",
        help="Default lane limits/weights profile (can be overridden by explicit --limit/--weight args).",
    )
    p_rag_search.add_argument("--top-k", type=int, default=5)
    p_rag_search.add_argument("--limit-bm25", type=int, default=None)
    p_rag_search.add_argument("--limit-similarity", type=int, default=None)
    p_rag_search.add_argument("--limit-sparse", type=int, default=None)
    p_rag_search.add_argument("--bm25-weight", type=float, default=None)
    p_rag_search.add_argument("--similarity-weight", type=float, default=None)
    p_rag_search.add_argument("--sparse-weight", type=float, default=None)
    p_rag_search.add_argument(
        "--with-metrics",
        action="store_true",
        help="Include duration/profile metadata for hybrid mode.",
    )
    p_rag_search.set_defaults(func=cmd_rag_search)

    p_rag_search_batch = rag_sub.add_parser(
        "search-batch",
        help="Run search over a newline-delimited query file.",
    )
    p_rag_search_batch.add_argument("--collection-id", required=True)
    p_rag_search_batch.add_argument("--queries-file", required=True)
    p_rag_search_batch.add_argument("--mode", choices=["hybrid", "bm25"], default="hybrid")
    p_rag_search_batch.add_argument(
        "--preset",
        choices=["balanced", "tri-lane", "lexical-heavy", "semantic-heavy", "sparse-heavy"],
        default="balanced",
        help="Default lane limits/weights profile (can be overridden by explicit --limit/--weight args).",
    )
    p_rag_search_batch.add_argument("--top-k", type=int, default=5)
    p_rag_search_batch.add_argument("--limit-bm25", type=int, default=None)
    p_rag_search_batch.add_argument("--limit-similarity", type=int, default=None)
    p_rag_search_batch.add_argument("--limit-sparse", type=int, default=None)
    p_rag_search_batch.add_argument("--bm25-weight", type=float, default=None)
    p_rag_search_batch.add_argument("--similarity-weight", type=float, default=None)
    p_rag_search_batch.add_argument("--sparse-weight", type=float, default=None)
    p_rag_search_batch.add_argument(
        "--with-metrics",
        action="store_true",
        help="Include duration/profile metadata for hybrid mode.",
    )
    p_rag_search_batch.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap on number of non-empty queries to run.",
    )
    p_rag_search_batch.set_defaults(func=cmd_rag_search_batch)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
