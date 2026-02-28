from __future__ import annotations

import json
import os
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
