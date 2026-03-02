from __future__ import annotations

import argparse
import fnmatch
import getpass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from .client import QueryLakeClient
from .errors import QueryLakeError

CONFIG_DIR = Path.home() / ".querylake"
CONFIG_PATH = CONFIG_DIR / "sdk_profiles.json"

INGEST_PROFILE_PRESETS: Dict[str, Dict[str, Any]] = {
    "dense-fast": {
        "scan_text": True,
        "create_embeddings": True,
        "create_sparse_embeddings": False,
        "await_embedding": False,
        "sparse_embedding_dimensions": 1024,
        "dedupe_by_content_hash": False,
        "dedupe_scope": "run-local",
        "idempotency_strategy": "none",
        "idempotency_prefix": "qlsdk",
        "fail_fast": False,
        "checkpoint_save_every": 1,
    },
    "dense-blocking": {
        "scan_text": True,
        "create_embeddings": True,
        "create_sparse_embeddings": False,
        "await_embedding": True,
        "sparse_embedding_dimensions": 1024,
        "dedupe_by_content_hash": True,
        "dedupe_scope": "all",
        "idempotency_strategy": "content-hash",
        "idempotency_prefix": "qlsdk",
        "fail_fast": True,
        "checkpoint_save_every": 1,
    },
    "tri-lane-fast": {
        "scan_text": True,
        "create_embeddings": True,
        "create_sparse_embeddings": True,
        "await_embedding": False,
        "sparse_embedding_dimensions": 1024,
        "dedupe_by_content_hash": True,
        "dedupe_scope": "all",
        "idempotency_strategy": "content-hash",
        "idempotency_prefix": "qlsdk",
        "fail_fast": False,
        "checkpoint_save_every": 10,
    },
    "tri-lane-blocking": {
        "scan_text": True,
        "create_embeddings": True,
        "create_sparse_embeddings": True,
        "await_embedding": True,
        "sparse_embedding_dimensions": 1024,
        "dedupe_by_content_hash": True,
        "dedupe_scope": "all",
        "idempotency_strategy": "content-hash",
        "idempotency_prefix": "qlsdk",
        "fail_fast": True,
        "checkpoint_save_every": 5,
    },
}


def _normalize_ingest_profile(raw: Dict[str, Any], *, source: str) -> Dict[str, Any]:
    profile = dict(raw)
    allowed_keys = {
        "scan_text",
        "create_embeddings",
        "create_sparse_embeddings",
        "await_embedding",
        "sparse_embedding_dimensions",
        "dedupe_by_content_hash",
        "dedupe_scope",
        "idempotency_strategy",
        "idempotency_prefix",
        "fail_fast",
        "checkpoint_save_every",
    }
    extra_keys = sorted(set(profile.keys()) - allowed_keys)
    if extra_keys:
        raise SystemExit(
            f"Ingest profile {source!r}: unsupported key(s): {', '.join(extra_keys)}. "
            f"Allowed keys: {', '.join(sorted(allowed_keys))}"
        )
    bool_keys = {
        "scan_text",
        "create_embeddings",
        "create_sparse_embeddings",
        "await_embedding",
        "dedupe_by_content_hash",
        "fail_fast",
    }
    for key in bool_keys:
        value = profile.get(key)
        if value is None:
            continue
        if not isinstance(value, bool):
            raise SystemExit(f"Ingest profile {source!r}: {key} must be boolean.")

    int_keys = {"sparse_embedding_dimensions", "checkpoint_save_every"}
    for key in int_keys:
        value = profile.get(key)
        if value is None:
            continue
        if not isinstance(value, int) or value < 1:
            raise SystemExit(f"Ingest profile {source!r}: {key} must be an integer >= 1.")

    dedupe_scope = profile.get("dedupe_scope")
    if dedupe_scope is not None:
        dedupe_scope_value = str(dedupe_scope).strip().lower()
        if dedupe_scope_value not in {"run-local", "checkpoint-resume", "all"}:
            raise SystemExit(
                f"Ingest profile {source!r}: dedupe_scope must be one of run-local, checkpoint-resume, all."
            )
        profile["dedupe_scope"] = dedupe_scope_value

    idempotency_strategy = profile.get("idempotency_strategy")
    if idempotency_strategy is not None:
        idempotency_value = str(idempotency_strategy).strip().lower()
        if idempotency_value not in {"none", "content-hash", "path-hash"}:
            raise SystemExit(
                f"Ingest profile {source!r}: idempotency_strategy must be one of none, content-hash, path-hash."
            )
        profile["idempotency_strategy"] = idempotency_value

    if "idempotency_prefix" in profile:
        value = profile.get("idempotency_prefix")
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"Ingest profile {source!r}: idempotency_prefix must be a non-empty string.")
        profile["idempotency_prefix"] = value.strip()
    return profile


def _load_ingest_profile_file(path_value: str) -> Dict[str, Any]:
    path = Path(path_value).expanduser().resolve()
    if not path.exists() or not path.is_file():
        raise SystemExit(f"--ingest-profile-file must be an existing file: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to parse --ingest-profile-file {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"--ingest-profile-file must contain a JSON object: {path}")
    return _normalize_ingest_profile(payload, source=str(path))


def _resolve_upload_ingest_controls(args: argparse.Namespace) -> Dict[str, Any]:
    controls: Dict[str, Any] = dict(INGEST_PROFILE_PRESETS["dense-fast"])
    profile_name = getattr(args, "ingest_profile", None)
    profile_file = getattr(args, "ingest_profile_file", None)

    if isinstance(profile_name, str) and profile_name.strip():
        controls.update(_normalize_ingest_profile(INGEST_PROFILE_PRESETS[profile_name], source=profile_name))
    if isinstance(profile_file, str) and profile_file.strip():
        controls.update(_load_ingest_profile_file(profile_file))

    if args.no_scan:
        controls["scan_text"] = False
    if args.no_embeddings:
        controls["create_embeddings"] = False
    if args.sparse_embeddings:
        controls["create_sparse_embeddings"] = True
    if args.no_sparse_embeddings:
        controls["create_sparse_embeddings"] = False
    if args.await_embedding:
        controls["await_embedding"] = True
    if args.sparse_dimensions is not None:
        controls["sparse_embedding_dimensions"] = max(1, int(args.sparse_dimensions))
    if args.fail_fast:
        controls["fail_fast"] = True
    if args.checkpoint_save_every is not None:
        controls["checkpoint_save_every"] = max(1, int(args.checkpoint_save_every))

    if args.dedupe_content_hash:
        controls["dedupe_by_content_hash"] = True
    if args.no_dedupe_content_hash:
        controls["dedupe_by_content_hash"] = False
    if isinstance(args.dedupe_scope, str) and args.dedupe_scope.strip():
        controls["dedupe_scope"] = args.dedupe_scope.strip().lower()
    if isinstance(args.idempotency_strategy, str) and args.idempotency_strategy.strip():
        controls["idempotency_strategy"] = args.idempotency_strategy.strip().lower()
    if isinstance(args.idempotency_prefix, str) and args.idempotency_prefix.strip():
        controls["idempotency_prefix"] = args.idempotency_prefix.strip()

    return controls


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


def _write_json_file(output_path: str, payload: Any) -> None:
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _selection_sha256(paths: List[str]) -> str:
    normalized = sorted(str(Path(value).expanduser().resolve()) for value in paths)
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


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


def cmd_rag_get_collection(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    try:
        result = client.fetch_collection(collection_hash_id=args.collection_id)
    finally:
        client.close()
    _print_output(result, as_json=True)
    return 0


def cmd_rag_update_collection(args: argparse.Namespace) -> int:
    if args.title is None and args.description is None:
        raise SystemExit("At least one of --title or --description is required.")
    client = _build_client(args, require_auth=True)
    try:
        result = client.modify_collection(
            collection_hash_id=args.collection_id,
            title=args.title,
            description=args.description,
        )
    finally:
        client.close()
    _print_output({"collection_id": args.collection_id, "result": result}, as_json=True)
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


def cmd_rag_random_chunks(args: argparse.Namespace) -> int:
    collection_ids = None
    if isinstance(args.collection_ids, str) and args.collection_ids.strip():
        collection_ids = [part.strip() for part in args.collection_ids.split(",") if part.strip()]
    client = _build_client(args, require_auth=True)
    try:
        rows = client.get_random_chunks(limit=args.limit, collection_ids=collection_ids)
    finally:
        client.close()
    _print_output({"results": rows, "count": len(rows)}, as_json=True)
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
    include_extensions: Optional[List[str]] = None,
    exclude_globs: Optional[List[str]] = None,
) -> List[Path]:
    root = Path(directory).expanduser().resolve()
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"--dir must be an existing directory: {root}")
    iterator = root.rglob(pattern) if recursive else root.glob(pattern)
    files = [path for path in iterator if path.is_file()]
    if include_extensions:
        ext_set = {
            value.lower() if value.startswith(".") else f".{value.lower()}" for value in include_extensions
        }
        files = [path for path in files if path.suffix.lower() in ext_set]
    if exclude_globs:
        patterns = [pattern.strip() for pattern in exclude_globs if isinstance(pattern, str) and pattern.strip()]

        def _is_excluded(path: Path) -> bool:
            rel_posix = path.relative_to(root).as_posix()
            name = path.name
            for pattern in patterns:
                if fnmatch.fnmatch(rel_posix, pattern) or fnmatch.fnmatch(name, pattern):
                    return True
            return False

        files = [path for path in files if not _is_excluded(path)]
    files.sort()
    if max_files is not None:
        files = files[: max(0, int(max_files))]
    return files


def _load_upload_selection_file(
    selection_file: str,
    *,
    base_dir: Optional[str] = None,
) -> tuple[List[Path], Optional[str]]:
    selection_path = Path(selection_file).expanduser().resolve()
    if not selection_path.exists() or not selection_path.is_file():
        raise SystemExit(f"--from-selection must be an existing file: {selection_path}")

    try:
        payload = json.loads(selection_path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        raise SystemExit(f"Failed to parse selection file {selection_path}: {exc}") from exc

    selected_files: Any = payload
    payload_directory: Optional[str] = None
    if isinstance(payload, dict):
        selected_files = payload.get("selected_files")
        directory_value = payload.get("directory")
        if isinstance(directory_value, str) and directory_value.strip():
            payload_directory = directory_value.strip()
    if not isinstance(selected_files, list) or not selected_files:
        raise SystemExit(
            f"Selection file {selection_path} must be a non-empty list of file paths or contain 'selected_files'."
        )

    resolved_base = None
    base_candidate = base_dir or payload_directory
    if isinstance(base_candidate, str) and base_candidate.strip():
        resolved_base = Path(base_candidate).expanduser().resolve()

    resolved_files: List[Path] = []
    for value in selected_files:
        if not isinstance(value, str) or not value.strip():
            continue
        candidate = Path(value).expanduser()
        if not candidate.is_absolute() and resolved_base is not None:
            candidate = resolved_base / candidate
        resolved_files.append(candidate.resolve())

    if not resolved_files:
        raise SystemExit(f"Selection file {selection_path} did not contain any usable file paths.")

    missing = [str(path) for path in resolved_files if not path.exists() or not path.is_file()]
    if missing:
        preview = ", ".join(missing[:5])
        suffix = " ..." if len(missing) > 5 else ""
        raise SystemExit(f"Selection file references missing/non-file paths: {preview}{suffix}")
    return resolved_files, payload_directory


def cmd_rag_upload_dir(args: argparse.Namespace) -> int:
    if args.resume and not args.checkpoint_file:
        raise SystemExit("--resume requires --checkpoint-file.")
    if args.dedupe_content_hash and args.no_dedupe_content_hash:
        raise SystemExit("Choose one of --dedupe-content-hash or --no-dedupe-content-hash, not both.")
    if args.sparse_embeddings and args.no_sparse_embeddings:
        raise SystemExit("Choose one of --sparse-embeddings or --no-sparse-embeddings, not both.")
    ingest_controls = _resolve_upload_ingest_controls(args)
    dedupe_scope = ingest_controls["dedupe_scope"]
    if dedupe_scope not in {"run-local", "checkpoint-resume", "all"}:
        raise SystemExit("--dedupe-scope must be one of: run-local, checkpoint-resume, all")
    idempotency_strategy = ingest_controls["idempotency_strategy"]
    if idempotency_strategy not in {"none", "content-hash", "path-hash"}:
        raise SystemExit("--idempotency-strategy must be one of: none, content-hash, path-hash")
    idempotency_prefix = ingest_controls["idempotency_prefix"]
    checkpoint_cadence = int(max(1, ingest_controls["checkpoint_save_every"]))

    include_extensions: Optional[List[str]] = None
    if isinstance(args.extensions, str) and args.extensions.strip():
        include_extensions = [part.strip() for part in args.extensions.split(",") if part.strip()]
    exclude_globs = list(args.exclude_glob or [])
    selection_source: Optional[str] = None
    selection_mode = "directory-scan"
    payload_directory: str

    if args.from_selection:
        selection_mode = "from-selection"
        files, selected_directory = _load_upload_selection_file(args.from_selection, base_dir=args.dir)
        selection_source = str(Path(args.from_selection).expanduser().resolve())
        include_extensions = None
        exclude_globs = []
        payload_directory = (
            str(Path(selected_directory).expanduser().resolve())
            if isinstance(selected_directory, str) and selected_directory.strip()
            else str(Path(args.dir).expanduser().resolve())
            if isinstance(args.dir, str) and args.dir.strip()
            else "<selection-file>"
        )
    else:
        if not isinstance(args.dir, str) or not args.dir.strip():
            raise SystemExit("--dir is required unless --from-selection is provided.")
        files = _resolve_upload_dir_files(
            args.dir,
            pattern=args.pattern,
            recursive=args.recursive,
            max_files=args.max_files,
            include_extensions=include_extensions,
            exclude_globs=exclude_globs,
        )
        if not files:
            raise SystemExit(
                f"No files found under {Path(args.dir).expanduser().resolve()} matching pattern {args.pattern!r}."
            )
        payload_directory = str(Path(args.dir).expanduser().resolve())

    selected_files = [str(path) for path in files]
    selection_hash = _selection_sha256(selected_files)
    if args.selection_output:
        selection_payload: Dict[str, Any] = {
            "directory": payload_directory,
            "selection_mode": selection_mode,
            "pattern": args.pattern,
            "recursive": bool(args.recursive),
            "requested_files": len(files),
            "selected_files": selected_files,
            "selection_sha256": selection_hash,
        }
        if include_extensions:
            selection_payload["extensions"] = include_extensions
        if exclude_globs:
            selection_payload["exclude_glob"] = exclude_globs
        if selection_source:
            selection_payload["from_selection"] = selection_source
        _write_json_file(args.selection_output, selection_payload)

    if args.dry_run:
        dry_run_payload: Dict[str, Any] = {
            "directory": payload_directory,
            "selection_mode": selection_mode,
            "pattern": args.pattern,
            "recursive": bool(args.recursive),
            "requested_files": len(files),
            "pending_files": len(files),
            "uploaded": 0,
            "failed": 0,
            "dry_run": True,
            "selection_sha256": selection_hash,
            "resumed_from_checkpoint": bool(args.resume),
            "skipped_already_uploaded": 0,
            "dedupe_by_content_hash": bool(args.dedupe_content_hash),
            "dedupe_scope": dedupe_scope if args.dedupe_content_hash else "none",
            "dedupe_skipped": 0,
            "idempotency_strategy": idempotency_strategy,
            "idempotency_prefix": idempotency_prefix,
            "ingest_controls": ingest_controls,
            "ingest_profile": args.ingest_profile,
            "ingest_profile_file": (
                str(Path(args.ingest_profile_file).expanduser().resolve())
                if isinstance(args.ingest_profile_file, str) and args.ingest_profile_file.strip()
                else None
            ),
        }
        if include_extensions:
            dry_run_payload["extensions"] = include_extensions
        if exclude_globs:
            dry_run_payload["exclude_glob"] = exclude_globs
        if selection_source:
            dry_run_payload["from_selection"] = selection_source
        if args.list_files:
            dry_run_payload["selected_files"] = selected_files
        if args.selection_output:
            dry_run_payload["selection_output"] = str(Path(args.selection_output).expanduser().resolve())
        if args.checkpoint_file:
            dry_run_payload["checkpoint_file"] = str(Path(args.checkpoint_file).expanduser().resolve())
            dry_run_payload["checkpoint_save_every"] = checkpoint_cadence
        if args.report_file:
            _write_json_file(args.report_file, dry_run_payload)
            dry_run_payload["report_file"] = str(Path(args.report_file).expanduser().resolve())
        _print_output(dry_run_payload, as_json=True)
        return 0

    client = _build_client(args, require_auth=True)
    try:
        upload_report = client.upload_directory(
            collection_hash_id=args.collection_id,
            directory=payload_directory,
            file_paths=files,
            pattern=args.pattern,
            recursive=args.recursive,
            dry_run=False,
            fail_fast=bool(ingest_controls["fail_fast"]),
            scan_text=bool(ingest_controls["scan_text"]),
            create_embeddings=bool(ingest_controls["create_embeddings"]),
            create_sparse_embeddings=bool(ingest_controls["create_sparse_embeddings"]),
            await_embedding=bool(ingest_controls["await_embedding"]),
            sparse_embedding_dimensions=int(ingest_controls["sparse_embedding_dimensions"]),
            checkpoint_file=args.checkpoint_file,
            resume=args.resume,
            checkpoint_save_every=checkpoint_cadence,
            strict_checkpoint_match=not args.no_checkpoint_strict,
            dedupe_by_content_hash=bool(ingest_controls["dedupe_by_content_hash"]),
            dedupe_scope=dedupe_scope,
            idempotency_strategy=idempotency_strategy,
            idempotency_prefix=idempotency_prefix,
        )
    finally:
        client.close()

    payload: Dict[str, Any] = dict(upload_report)
    payload["directory"] = payload_directory
    payload["selection_mode"] = selection_mode
    payload["pattern"] = args.pattern
    payload["recursive"] = bool(args.recursive)
    payload["requested_files"] = len(files)
    payload["selection_sha256"] = selection_hash
    if include_extensions:
        payload["extensions"] = include_extensions
    if exclude_globs:
        payload["exclude_glob"] = exclude_globs
    if selection_source:
        payload["from_selection"] = selection_source
    payload["ingest_controls"] = ingest_controls
    payload["ingest_profile"] = args.ingest_profile
    if isinstance(args.ingest_profile_file, str) and args.ingest_profile_file.strip():
        payload["ingest_profile_file"] = str(Path(args.ingest_profile_file).expanduser().resolve())
    if args.list_files:
        payload["selected_files"] = selected_files
    if args.selection_output:
        payload["selection_output"] = str(Path(args.selection_output).expanduser().resolve())
    if args.report_file:
        _write_json_file(args.report_file, payload)
        payload["report_file"] = str(Path(args.report_file).expanduser().resolve())
    _print_output(payload, as_json=True)
    return 0 if int(payload.get("failed", 0)) == 0 else 1


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


def _extract_duration_ms(payload: Dict[str, Any]) -> Optional[float]:
    duration = payload.get("duration")
    if not isinstance(duration, dict):
        return None
    candidate = duration.get("total_ms", duration.get("total"))
    if isinstance(candidate, (int, float)):
        return float(candidate)
    return None


def cmd_rag_search(args: argparse.Namespace) -> int:
    client = _build_client(args, require_auth=True)
    try:
        payload = _run_rag_search(client, args, args.query)
    finally:
        client.close()
    failures: List[str] = []
    total = int(payload.get("total", 0)) if isinstance(payload, dict) else 0
    if args.fail_on_empty and total == 0:
        failures.append("total_results == 0")
    if args.min_total_results is not None and total < int(args.min_total_results):
        failures.append(f"total_results < min_total_results ({total} < {int(args.min_total_results)})")
    if failures:
        payload["gate_failed"] = True
        payload["gate_failures"] = failures
    _print_output(payload, as_json=True)
    return 2 if failures else 0


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
        "summary": {
            "avg_result_count": (
                (sum(item.get("total", 0) for item in outputs) / len(outputs)) if outputs else 0.0
            ),
            "avg_duration_ms": (
                (
                    sum(v for v in [_extract_duration_ms(item) for item in outputs] if v is not None)
                    / max(
                        1,
                        sum(1 for v in [_extract_duration_ms(item) for item in outputs] if v is not None),
                    )
                )
                if outputs
                else None
            ),
        },
        "results": outputs,
    }
    failures: List[str] = []
    empty_result_queries = sum(1 for item in outputs if int(item.get("total", 0)) == 0)
    payload["summary"]["empty_result_queries"] = empty_result_queries
    if args.fail_on_empty and empty_result_queries > 0:
        failures.append(f"empty_result_queries > 0 ({empty_result_queries})")
    if args.min_total_results is not None:
        threshold = int(args.min_total_results)
        below = [
            item.get("query", "")
            for item in outputs
            if int(item.get("total", 0)) < threshold
        ]
        if below:
            failures.append(
                f"queries_below_min_total_results > 0 ({len(below)} below {threshold})"
            )
            payload["queries_below_min_total_results"] = below
    if args.output_file:
        output_path = Path(args.output_file).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload["output_file"] = str(output_path)
    if failures:
        payload["gate_failed"] = True
        payload["gate_failures"] = failures
    _print_output(payload, as_json=True)
    return 2 if failures else 0


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

    p_rag_get_collection = rag_sub.add_parser(
        "get-collection",
        help="Fetch collection metadata by collection ID.",
    )
    p_rag_get_collection.add_argument("--collection-id", required=True)
    p_rag_get_collection.set_defaults(func=cmd_rag_get_collection)

    p_rag_update_collection = rag_sub.add_parser(
        "update-collection",
        help="Update collection title/description.",
    )
    p_rag_update_collection.add_argument("--collection-id", required=True)
    p_rag_update_collection.add_argument("--title", default=None)
    p_rag_update_collection.add_argument("--description", default=None)
    p_rag_update_collection.set_defaults(func=cmd_rag_update_collection)

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

    p_rag_random_chunks = rag_sub.add_parser(
        "random-chunks",
        help="Sample random chunks for quick retrieval inspection.",
    )
    p_rag_random_chunks.add_argument("--limit", type=int, default=5)
    p_rag_random_chunks.add_argument(
        "--collection-ids",
        default=None,
        help="Optional comma-separated collection IDs.",
    )
    p_rag_random_chunks.set_defaults(func=cmd_rag_random_chunks)

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
    p_rag_upload_dir.add_argument("--dir", required=False, help="Directory containing source files.")
    p_rag_upload_dir.add_argument(
        "--from-selection",
        default=None,
        help="Optional JSON selection file from a prior upload-dir run (uses selected_files list).",
    )
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
    p_rag_upload_dir.add_argument(
        "--extensions",
        default=None,
        help="Optional comma-separated extension filter (e.g. '.pdf,.md').",
    )
    p_rag_upload_dir.add_argument(
        "--exclude-glob",
        action="append",
        default=[],
        help="Exclude files matching glob relative to --dir (repeatable).",
    )
    p_rag_upload_dir.add_argument(
        "--dry-run",
        action="store_true",
        help="List/validate selected files without uploading.",
    )
    p_rag_upload_dir.add_argument(
        "--list-files",
        action="store_true",
        help="Include selected file paths in output payload.",
    )
    p_rag_upload_dir.add_argument(
        "--selection-output",
        default=None,
        help="Optional path to write selected file list + selection metadata as JSON.",
    )
    p_rag_upload_dir.add_argument(
        "--report-file",
        default=None,
        help="Optional path to write final upload/dry-run JSON payload.",
    )
    p_rag_upload_dir.add_argument(
        "--checkpoint-file",
        default=None,
        help="Optional JSON checkpoint path for resumable runs.",
    )
    p_rag_upload_dir.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint file uploaded set (requires --checkpoint-file).",
    )
    p_rag_upload_dir.add_argument(
        "--checkpoint-save-every",
        type=int,
        default=None,
        help="Persist checkpoint state every N processed files.",
    )
    p_rag_upload_dir.add_argument(
        "--no-checkpoint-strict",
        action="store_true",
        help="Allow resume from checkpoint even when selection_sha256 differs.",
    )
    p_rag_upload_dir.add_argument(
        "--ingest-profile",
        choices=sorted(INGEST_PROFILE_PRESETS.keys()),
        default=None,
        help="Apply a named ingest profile baseline before explicit CLI overrides.",
    )
    p_rag_upload_dir.add_argument(
        "--ingest-profile-file",
        default=None,
        help="Optional JSON file with ingest profile keys (overrides named profile).",
    )
    p_rag_upload_dir.add_argument(
        "--dedupe-content-hash",
        action="store_true",
        help="Skip duplicate files based on SHA-256 content hash.",
    )
    p_rag_upload_dir.add_argument(
        "--no-dedupe-content-hash",
        action="store_true",
        help="Force disable content-hash dedupe (useful with ingest profiles).",
    )
    p_rag_upload_dir.add_argument(
        "--dedupe-scope",
        choices=["run-local", "checkpoint-resume", "all"],
        default=None,
        help="Scope for content-hash dedupe when --dedupe-content-hash is enabled.",
    )
    p_rag_upload_dir.add_argument(
        "--idempotency-strategy",
        choices=["none", "content-hash", "path-hash"],
        default=None,
        help="Client-side idempotency key strategy injected in document metadata.",
    )
    p_rag_upload_dir.add_argument(
        "--idempotency-prefix",
        default=None,
        help="Prefix for generated idempotency keys when strategy is not 'none'.",
    )
    p_rag_upload_dir.add_argument("--await-embedding", action="store_true")
    p_rag_upload_dir.add_argument("--no-scan", action="store_true")
    p_rag_upload_dir.add_argument("--no-embeddings", action="store_true")
    p_rag_upload_dir.add_argument("--sparse-embeddings", action="store_true")
    p_rag_upload_dir.add_argument("--no-sparse-embeddings", action="store_true")
    p_rag_upload_dir.add_argument("--sparse-dimensions", type=int, default=None)
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
    p_rag_search.add_argument(
        "--min-total-results",
        type=int,
        default=None,
        help="Optional gate: fail with exit code 2 if total results are below this threshold.",
    )
    p_rag_search.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Optional gate: fail with exit code 2 when total results are zero.",
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
    p_rag_search_batch.add_argument(
        "--output-file",
        default=None,
        help="Optional path to write full JSON payload.",
    )
    p_rag_search_batch.add_argument(
        "--min-total-results",
        type=int,
        default=None,
        help="Optional gate: fail with exit code 2 if any query has fewer results than this threshold.",
    )
    p_rag_search_batch.add_argument(
        "--fail-on-empty",
        action="store_true",
        help="Optional gate: fail with exit code 2 when any query returns zero results.",
    )
    p_rag_search_batch.set_defaults(func=cmd_rag_search_batch)

    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
