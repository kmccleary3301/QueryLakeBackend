from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Literal, Optional, Union

import httpx

from .errors import QueryLakeAPIError, QueryLakeHTTPStatusError, QueryLakeTransportError
from .models import SearchResultChunk

AuthOverride = Union[Dict[str, str], Literal[False], None]


def _normalize_base_url(base_url: str) -> str:
    value = (base_url or "").strip()
    if not value:
        raise ValueError("base_url must be a non-empty URL")
    return value.rstrip("/")


def _api_function_path(function_name: str) -> str:
    cleaned = (function_name or "").strip().strip("/")
    if not cleaned:
        raise ValueError("function_name must be non-empty")
    if cleaned.startswith("api/"):
        return f"/{cleaned}"
    return f"/api/{cleaned}"


def _ensure_http_success(response: httpx.Response) -> None:
    if 200 <= response.status_code < 300:
        return
    body = response.text
    raise QueryLakeHTTPStatusError(
        status_code=response.status_code,
        url=str(response.request.url),
        body=body[:2000],
    )


def _extract_api_result(function_name: str, payload: Any) -> Any:
    if not isinstance(payload, dict) or "success" not in payload:
        return payload

    if payload.get("success") is False:
        raise QueryLakeAPIError(
            function_name=function_name,
            message=str(payload.get("error") or payload.get("note") or "Unknown API error"),
            trace=(payload.get("trace") if isinstance(payload.get("trace"), str) else None),
            payload=payload,
        )

    if "result" in payload:
        return payload["result"]

    remainder = {k: v for k, v in payload.items() if k != "success"}
    if remainder:
        return remainder
    return None


def _selection_sha256(paths: Iterable[Union[str, Path]]) -> str:
    normalized = [str(Path(value).expanduser().resolve()) for value in paths]
    normalized.sort()
    blob = "\n".join(normalized).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


class QueryLakeClient:
    """Synchronous QueryLake SDK client."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: float = 60.0,
        auth: Optional[Dict[str, str]] = None,
        oauth2: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self._auth: Dict[str, str] = {}
        self.set_auth(auth=auth, oauth2=oauth2, api_key=api_key)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=float(timeout_seconds),
            headers=headers or {},
        )

    @classmethod
    def from_env(cls) -> "QueryLakeClient":
        return cls(
            base_url=os.getenv("QUERYLAKE_BASE_URL", "http://127.0.0.1:8000"),
            oauth2=os.getenv("QUERYLAKE_OAUTH2"),
            api_key=os.getenv("QUERYLAKE_API_KEY"),
            timeout_seconds=float(os.getenv("QUERYLAKE_TIMEOUT_SECONDS", "60")),
        )

    def set_auth(
        self,
        *,
        auth: Optional[Dict[str, str]] = None,
        oauth2: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if auth is not None:
            self._auth = dict(auth)
            return
        if oauth2:
            self._auth = {"oauth2": oauth2}
            return
        if api_key:
            self._auth = {"api_key": api_key}
            return
        self._auth = {}

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "QueryLakeClient":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _resolve_auth(self, auth_override: AuthOverride) -> Optional[Dict[str, str]]:
        if auth_override is False:
            return None
        if isinstance(auth_override, dict):
            return auth_override
        if self._auth:
            return self._auth
        return None

    def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise QueryLakeTransportError(str(exc)) from exc
        _ensure_http_success(response)
        return response

    def api(
        self,
        function_name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        method: Literal["POST", "GET"] = "POST",
        auth: AuthOverride = None,
    ) -> Any:
        body = dict(payload or {})
        resolved_auth = self._resolve_auth(auth)
        if resolved_auth and "auth" not in body:
            body["auth"] = resolved_auth
        response = self._request(method, _api_function_path(function_name), json=body)
        return _extract_api_result(function_name, response.json())

    def healthz(self) -> Dict[str, Any]:
        return self._request("GET", "/healthz").json()

    def readyz(self) -> Dict[str, Any]:
        return self._request("GET", "/readyz").json()

    def ping(self) -> Any:
        return self.api("ping", method="GET", auth=False)

    def list_models(self) -> Dict[str, Any]:
        return self._request("GET", "/v1/models").json()

    def chat_completions(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request("POST", "/v1/chat/completions", json=payload)
        return response.json()

    def embeddings(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self._request("POST", "/v1/embeddings", json=payload)
        return response.json()

    def login(self, *, username: str, password: str) -> Dict[str, Any]:
        result = self.api(
            "login",
            {"auth": {"username": username, "password": password}},
            auth=False,
        )
        if isinstance(result, dict) and isinstance(result.get("auth"), str):
            self.set_auth(oauth2=result["auth"])
        return result

    def add_user(self, *, username: str, password: str) -> Dict[str, Any]:
        result = self.api("add_user", {"username": username, "password": password}, auth=False)
        if isinstance(result, dict) and isinstance(result.get("auth"), str):
            self.set_auth(oauth2=result["auth"])
        return result

    def create_collection(
        self,
        *,
        name: str,
        description: Optional[str] = None,
        public: bool = False,
        organization_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": name,
            "public": bool(public),
        }
        if description is not None:
            payload["description"] = description
        if organization_id is not None:
            payload["organization_id"] = int(organization_id)
        return self.api("create_document_collection", payload)

    def list_collections(
        self,
        *,
        organization_id: Optional[int] = None,
        global_collections: bool = False,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"global_collections": bool(global_collections)}
        if organization_id is not None:
            payload["organization_id"] = int(organization_id)
        return self.api("fetch_document_collections_belonging_to", payload)

    def fetch_collection(self, *, collection_hash_id: Union[str, int]) -> Dict[str, Any]:
        return self.api("fetch_collection", {"collection_hash_id": str(collection_hash_id)})

    def modify_collection(
        self,
        *,
        collection_hash_id: Union[str, int],
        title: Optional[str] = None,
        description: Optional[str] = None,
    ) -> Any:
        payload: Dict[str, Any] = {"collection_hash_id": str(collection_hash_id)}
        if title is not None:
            payload["title"] = title
        if description is not None:
            payload["description"] = description
        return self.api("modify_document_collection", payload)

    def list_collection_documents(
        self,
        *,
        collection_hash_id: Union[str, int],
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return self.api(
            "fetch_collection_documents",
            {
                "collection_hash_id": str(collection_hash_id),
                "limit": int(limit),
                "offset": int(offset),
            },
        )

    def delete_document(self, *, document_hash_id: Union[str, int]) -> Any:
        return self.api("delete_document", {"hash_id": str(document_hash_id)})

    def upload_document(
        self,
        *,
        file_path: Union[str, Path],
        collection_hash_id: Union[str, int],
        scan_text: bool = True,
        create_embeddings: bool = True,
        create_sparse_embeddings: bool = False,
        sparse_embedding_function: str = "embedding_sparse",
        sparse_embedding_dimensions: int = 1024,
        enforce_sparse_dimension_match: bool = True,
        await_embedding: bool = False,
        document_metadata: Optional[Dict[str, Any]] = None,
        auth: AuthOverride = None,
    ) -> Dict[str, Any]:
        resolved_auth = self._resolve_auth(auth)
        if resolved_auth is None:
            raise ValueError("upload_document requires auth. Set oauth2/api_key on client or pass auth=...")
        path = Path(file_path).expanduser().resolve()
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        params_payload: Dict[str, Any] = {
            "auth": resolved_auth,
            "collection_hash_id": str(collection_hash_id),
            "scan_text": bool(scan_text),
            "create_embeddings": bool(create_embeddings),
            "create_sparse_embeddings": bool(create_sparse_embeddings),
            "sparse_embedding_function": sparse_embedding_function,
            "sparse_embedding_dimensions": int(sparse_embedding_dimensions),
            "enforce_sparse_dimension_match": bool(enforce_sparse_dimension_match),
            "await_embedding": bool(await_embedding),
        }
        if document_metadata is not None:
            params_payload["document_metadata"] = document_metadata

        with path.open("rb") as f:
            files = {"file": (path.name, f, "application/octet-stream")}
            response = self._request(
                "POST",
                "/upload_document",
                params={"parameters": json.dumps(params_payload)},
                files=files,
            )
        return _extract_api_result("upload_document", response.json())

    def upload_directory(
        self,
        *,
        collection_hash_id: Union[str, int],
        directory: Optional[Union[str, Path]] = None,
        file_paths: Optional[Iterable[Union[str, Path]]] = None,
        pattern: str = "*",
        recursive: bool = False,
        max_files: Optional[int] = None,
        include_extensions: Optional[Iterable[str]] = None,
        exclude_globs: Optional[Iterable[str]] = None,
        dry_run: bool = False,
        fail_fast: bool = False,
        scan_text: bool = True,
        create_embeddings: bool = True,
        create_sparse_embeddings: bool = False,
        sparse_embedding_function: str = "embedding_sparse",
        sparse_embedding_dimensions: int = 1024,
        enforce_sparse_dimension_match: bool = True,
        await_embedding: bool = False,
        document_metadata: Optional[Dict[str, Any]] = None,
        checkpoint_file: Optional[Union[str, Path]] = None,
        resume: bool = False,
        checkpoint_save_every: int = 1,
        strict_checkpoint_match: bool = True,
        auth: AuthOverride = None,
    ) -> Dict[str, Any]:
        """
        Bulk upload local files for one collection.

        Selection modes:
        - directory scan (`directory`, `pattern`, filters)
        - explicit file list (`file_paths`)
        """
        selection_mode = "directory-scan"
        resolved_files: List[Path] = []
        payload_directory = "<explicit-file-list>"

        if file_paths is not None:
            selection_mode = "explicit-file-list"
            for value in file_paths:
                candidate = Path(value).expanduser().resolve()
                if not candidate.exists() or not candidate.is_file():
                    raise FileNotFoundError(f"File not found: {candidate}")
                resolved_files.append(candidate)
            if isinstance(directory, (str, Path)):
                payload_directory = str(Path(directory).expanduser().resolve())
        else:
            if directory is None:
                raise ValueError("directory is required unless file_paths is provided")
            root = Path(directory).expanduser().resolve()
            if not root.exists() or not root.is_dir():
                raise ValueError(f"directory must be an existing directory: {root}")
            payload_directory = str(root)
            iterator = root.rglob(pattern) if recursive else root.glob(pattern)
            resolved_files = [path for path in iterator if path.is_file()]

            if include_extensions:
                ext_set = {
                    value.lower() if value.startswith(".") else f".{value.lower()}"
                    for value in include_extensions
                    if isinstance(value, str) and value.strip()
                }
                if ext_set:
                    resolved_files = [path for path in resolved_files if path.suffix.lower() in ext_set]

            if exclude_globs:
                patterns = [value.strip() for value in exclude_globs if isinstance(value, str) and value.strip()]

                def _is_excluded(path: Path) -> bool:
                    rel_posix = path.relative_to(root).as_posix()
                    name = path.name
                    for pattern_value in patterns:
                        if fnmatch.fnmatch(rel_posix, pattern_value) or fnmatch.fnmatch(name, pattern_value):
                            return True
                    return False

                if patterns:
                    resolved_files = [path for path in resolved_files if not _is_excluded(path)]

        resolved_files.sort()
        if max_files is not None:
            resolved_files = resolved_files[: max(0, int(max_files))]
        if not resolved_files:
            raise ValueError("No files selected for upload.")

        selected_files = [str(path) for path in resolved_files]
        selection_hash = _selection_sha256(selected_files)

        checkpoint_path: Optional[Path] = None
        resumed_from_checkpoint = False
        skipped_already_uploaded = 0
        uploaded_set: set[str] = set()
        persisted_errors: List[Dict[str, Any]] = []
        checkpoint_started_at_unix: Optional[float] = None
        checkpoint_cadence = int(max(1, checkpoint_save_every))

        if checkpoint_file is not None:
            checkpoint_path = Path(checkpoint_file).expanduser().resolve()
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            if resume and checkpoint_path.exists():
                loaded = json.loads(checkpoint_path.read_text(encoding="utf-8"))
                if not isinstance(loaded, dict):
                    raise ValueError(f"Invalid checkpoint payload at {checkpoint_path}: expected object.")
                checkpoint_hash = loaded.get("selection_sha256")
                if strict_checkpoint_match and checkpoint_hash != selection_hash:
                    raise ValueError(
                        "Checkpoint selection hash mismatch. "
                        f"checkpoint={checkpoint_hash} current={selection_hash}"
                    )
                prior_uploaded = loaded.get("uploaded_files")
                if isinstance(prior_uploaded, list):
                    uploaded_set = {str(value) for value in prior_uploaded if isinstance(value, str)}
                prior_errors = loaded.get("errors")
                if isinstance(prior_errors, list):
                    persisted_errors = [row for row in prior_errors if isinstance(row, dict)]
                started_value = loaded.get("started_at_unix")
                if isinstance(started_value, (int, float)):
                    checkpoint_started_at_unix = float(started_value)
                resumed_from_checkpoint = True
            elif resume and not checkpoint_path.exists():
                raise ValueError(f"Checkpoint file does not exist for resume: {checkpoint_path}")

        if uploaded_set:
            before = len(resolved_files)
            resolved_files = [path for path in resolved_files if str(path) not in uploaded_set]
            skipped_already_uploaded = before - len(resolved_files)

        payload: Dict[str, Any] = {
            "directory": payload_directory,
            "selection_mode": selection_mode,
            "pattern": pattern,
            "recursive": bool(recursive),
            "requested_files": len(selected_files),
            "pending_files": len(resolved_files),
            "uploaded": 0,
            "failed": 0,
            "dry_run": bool(dry_run),
            "selected_files": selected_files,
            "fail_fast": bool(fail_fast),
            "selection_sha256": selection_hash,
            "resumed_from_checkpoint": resumed_from_checkpoint,
            "skipped_already_uploaded": skipped_already_uploaded,
        }
        if checkpoint_path is not None:
            payload["checkpoint_file"] = str(checkpoint_path)
            payload["checkpoint_save_every"] = checkpoint_cadence

        if dry_run:
            return payload

        if not resolved_files:
            payload["status"] = "already_complete"
            return payload

        def _checkpoint_payload(errors_payload: List[Dict[str, Any]], uploaded_payload: set[str], completed: bool) -> Dict[str, Any]:
            now = time.time()
            started = checkpoint_started_at_unix if checkpoint_started_at_unix is not None else now
            return {
                "version": 1,
                "collection_hash_id": str(collection_hash_id),
                "selection_mode": selection_mode,
                "selection_sha256": selection_hash,
                "requested_files": len(selected_files),
                "pending_files": len(resolved_files),
                "uploaded_files_count": len(uploaded_payload),
                "uploaded_files": sorted(uploaded_payload),
                "errors": errors_payload,
                "fail_fast": bool(fail_fast),
                "checkpoint_save_every": checkpoint_cadence,
                "completed": bool(completed),
                "started_at_unix": started,
                "updated_at_unix": now,
            }

        def _flush_checkpoint(errors_payload: List[Dict[str, Any]], uploaded_payload: set[str], completed: bool) -> None:
            if checkpoint_path is None:
                return
            checkpoint_path.write_text(
                json.dumps(_checkpoint_payload(errors_payload, uploaded_payload, completed), indent=2, sort_keys=True),
                encoding="utf-8",
            )

        errors: List[Dict[str, Any]] = list(persisted_errors)
        uploaded = 0
        failed = 0
        since_flush = 0
        for path in resolved_files:
            try:
                self.upload_document(
                    file_path=path,
                    collection_hash_id=collection_hash_id,
                    scan_text=scan_text,
                    create_embeddings=create_embeddings,
                    create_sparse_embeddings=create_sparse_embeddings,
                    sparse_embedding_function=sparse_embedding_function,
                    sparse_embedding_dimensions=sparse_embedding_dimensions,
                    enforce_sparse_dimension_match=enforce_sparse_dimension_match,
                    await_embedding=await_embedding,
                    document_metadata=document_metadata,
                    auth=auth,
                )
                uploaded += 1
                uploaded_set.add(str(path))
            except Exception as exc:  # noqa: BLE001
                failed += 1
                errors.append({"file": str(path), "error": str(exc)})
                if fail_fast:
                    _flush_checkpoint(errors, uploaded_set, completed=False)
                    break
            since_flush += 1
            if since_flush >= checkpoint_cadence:
                _flush_checkpoint(errors, uploaded_set, completed=False)
                since_flush = 0

        completed = failed == 0 and uploaded >= len(resolved_files)
        _flush_checkpoint(errors, uploaded_set, completed=completed)

        payload["uploaded"] = uploaded
        payload["failed"] = failed
        if errors:
            payload["errors"] = errors
        return payload

    def search_hybrid(
        self,
        *,
        query: Union[str, Dict[str, Any]],
        collection_ids: Iterable[Union[str, int]],
        limit_bm25: int = 12,
        limit_similarity: int = 12,
        limit_sparse: int = 0,
        bm25_weight: float = 0.55,
        similarity_weight: float = 0.45,
        sparse_weight: float = 0.0,
        **kwargs: Any,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {
            "query": query,
            "collection_ids": [str(v) for v in collection_ids],
            "limit_bm25": int(limit_bm25),
            "limit_similarity": int(limit_similarity),
            "limit_sparse": int(limit_sparse),
            "bm25_weight": float(bm25_weight),
            "similarity_weight": float(similarity_weight),
            "sparse_weight": float(sparse_weight),
        }
        payload.update(kwargs)
        result = self.api("search_hybrid", payload)
        if isinstance(result, list):
            return result
        if isinstance(result, dict):
            rows = result.get("rows")
            if isinstance(rows, list):
                return rows
        return []

    def search_hybrid_with_metrics(
        self,
        *,
        query: Union[str, Dict[str, Any]],
        collection_ids: Iterable[Union[str, int]],
        limit_bm25: int = 12,
        limit_similarity: int = 12,
        limit_sparse: int = 0,
        bm25_weight: float = 0.55,
        similarity_weight: float = 0.45,
        sparse_weight: float = 0.0,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "query": query,
            "collection_ids": [str(v) for v in collection_ids],
            "limit_bm25": int(limit_bm25),
            "limit_similarity": int(limit_similarity),
            "limit_sparse": int(limit_sparse),
            "bm25_weight": float(bm25_weight),
            "similarity_weight": float(similarity_weight),
            "sparse_weight": float(sparse_weight),
        }
        payload.update(kwargs)
        result = self.api("search_hybrid", payload)
        if isinstance(result, list):
            return {"rows": result}
        if isinstance(result, dict):
            rows = result.get("rows")
            if not isinstance(rows, list):
                result = {"rows": []}
            return result
        return {"rows": []}

    def search_hybrid_chunks(self, **kwargs: Any) -> List[SearchResultChunk]:
        rows = self.search_hybrid(**kwargs)
        parsed: List[SearchResultChunk] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            parsed.append(SearchResultChunk.from_api_dict(row))
        return parsed

    def count_chunks(self, *, collection_ids: Optional[Iterable[Union[str, int]]] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {}
        if collection_ids is not None:
            payload["collection_ids"] = [str(v) for v in collection_ids]
        return self.api("count_chunks", payload)

    def get_random_chunks(
        self,
        *,
        limit: int = 5,
        collection_ids: Optional[Iterable[Union[str, int]]] = None,
    ) -> List[Dict[str, Any]]:
        payload: Dict[str, Any] = {"limit": int(limit)}
        if collection_ids is not None:
            payload["collection_ids"] = [str(v) for v in collection_ids]
        rows = self.api("get_random_chunks", payload)
        return rows if isinstance(rows, list) else []


class AsyncQueryLakeClient:
    """Async QueryLake SDK client."""

    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8000",
        timeout_seconds: float = 60.0,
        auth: Optional[Dict[str, str]] = None,
        oauth2: Optional[str] = None,
        api_key: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self._auth: Dict[str, str] = {}
        self.set_auth(auth=auth, oauth2=oauth2, api_key=api_key)
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=float(timeout_seconds),
            headers=headers or {},
        )

    def set_auth(
        self,
        *,
        auth: Optional[Dict[str, str]] = None,
        oauth2: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if auth is not None:
            self._auth = dict(auth)
            return
        if oauth2:
            self._auth = {"oauth2": oauth2}
            return
        if api_key:
            self._auth = {"api_key": api_key}
            return
        self._auth = {}

    async def aclose(self) -> None:
        await self._client.aclose()

    async def __aenter__(self) -> "AsyncQueryLakeClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.aclose()

    def _resolve_auth(self, auth_override: AuthOverride) -> Optional[Dict[str, str]]:
        if auth_override is False:
            return None
        if isinstance(auth_override, dict):
            return auth_override
        if self._auth:
            return self._auth
        return None

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        try:
            response = await self._client.request(method, path, **kwargs)
        except httpx.HTTPError as exc:
            raise QueryLakeTransportError(str(exc)) from exc
        _ensure_http_success(response)
        return response

    async def api(
        self,
        function_name: str,
        payload: Optional[Dict[str, Any]] = None,
        *,
        method: Literal["POST", "GET"] = "POST",
        auth: AuthOverride = None,
    ) -> Any:
        body = dict(payload or {})
        resolved_auth = self._resolve_auth(auth)
        if resolved_auth and "auth" not in body:
            body["auth"] = resolved_auth
        response = await self._request(method, _api_function_path(function_name), json=body)
        return _extract_api_result(function_name, response.json())

    async def healthz(self) -> Dict[str, Any]:
        return (await self._request("GET", "/healthz")).json()

    async def readyz(self) -> Dict[str, Any]:
        return (await self._request("GET", "/readyz")).json()

    async def ping(self) -> Any:
        return await self.api("ping", method="GET", auth=False)

    async def login(self, *, username: str, password: str) -> Dict[str, Any]:
        result = await self.api(
            "login",
            {"auth": {"username": username, "password": password}},
            auth=False,
        )
        if isinstance(result, dict) and isinstance(result.get("auth"), str):
            self.set_auth(oauth2=result["auth"])
        return result
