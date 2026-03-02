from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


AuthDict = Dict[str, str]


@dataclass
class QueryLakeProfile:
    """Serializable profile for local SDK CLI usage."""

    name: str
    base_url: str
    auth: AuthDict = field(default_factory=dict)


@dataclass
class CollectionSummary:
    id: str
    name: str
    document_count: int = 0


@dataclass
class SearchResultChunk:
    id: str
    text: str
    document_name: Optional[str] = None
    collection_id: Optional[str] = None
    hybrid_score: Optional[float] = None
    bm25_score: Optional[float] = None
    similarity_score: Optional[float] = None
    sparse_score: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_api_dict(cls, payload: Dict[str, Any]) -> "SearchResultChunk":
        row_id = payload.get("id")
        if isinstance(row_id, list):
            row_id = ",".join(str(v) for v in row_id)
        return cls(
            id=str(row_id),
            text=str(payload.get("text", "")),
            document_name=payload.get("document_name"),
            collection_id=payload.get("collection_id"),
            hybrid_score=_optional_float(payload.get("hybrid_score")),
            bm25_score=_optional_float(payload.get("bm25_score")),
            similarity_score=_optional_float(payload.get("similarity_score")),
            sparse_score=_optional_float(payload.get("sparse_score")),
            metadata=payload.get("md") or {},
        )


@dataclass
class UploadDirectoryOptions:
    directory: Optional[Union[str, Path]] = None
    file_paths: Optional[List[Union[str, Path]]] = None
    pattern: str = "*"
    recursive: bool = True
    max_files: Optional[int] = None
    include_extensions: Optional[List[str]] = None
    exclude_globs: Optional[List[str]] = None
    dry_run: bool = False
    fail_fast: bool = False
    scan_text: bool = True
    create_embeddings: bool = True
    create_sparse_embeddings: bool = False
    sparse_embedding_function: Optional[str] = None
    sparse_embedding_dimensions: int = 1024
    enforce_sparse_dimension_match: bool = True
    await_embedding: bool = False
    document_metadata: Optional[Dict[str, Any]] = None
    checkpoint_file: Optional[Union[str, Path]] = None
    resume: bool = False
    checkpoint_save_every: int = 1
    strict_checkpoint_match: bool = True
    dedupe_by_content_hash: bool = False
    dedupe_scope: str = "run-local"
    idempotency_strategy: str = "none"
    idempotency_prefix: str = "qlsdk"

    def __post_init__(self) -> None:
        if self.max_files is not None and int(self.max_files) < 0:
            raise ValueError("max_files must be >= 0 when provided.")
        if int(self.sparse_embedding_dimensions) < 1:
            raise ValueError("sparse_embedding_dimensions must be >= 1.")
        if int(self.checkpoint_save_every) < 1:
            raise ValueError("checkpoint_save_every must be >= 1.")
        dedupe_scope = str(self.dedupe_scope).strip().lower()
        if dedupe_scope not in {"run-local", "checkpoint-resume", "all"}:
            raise ValueError("dedupe_scope must be one of: run-local, checkpoint-resume, all.")
        self.dedupe_scope = dedupe_scope
        idempotency_strategy = str(self.idempotency_strategy).strip().lower()
        if idempotency_strategy not in {"none", "content-hash", "path-hash"}:
            raise ValueError("idempotency_strategy must be one of: none, content-hash, path-hash.")
        self.idempotency_strategy = idempotency_strategy
        prefix = str(self.idempotency_prefix).strip()
        if not prefix:
            raise ValueError("idempotency_prefix must be non-empty.")
        self.idempotency_prefix = prefix
        if self.resume and self.checkpoint_file is None:
            raise ValueError("resume=True requires checkpoint_file.")

    def to_kwargs(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "pattern": self.pattern,
            "recursive": bool(self.recursive),
            "dry_run": bool(self.dry_run),
            "fail_fast": bool(self.fail_fast),
            "scan_text": bool(self.scan_text),
            "create_embeddings": bool(self.create_embeddings),
            "create_sparse_embeddings": bool(self.create_sparse_embeddings),
            "sparse_embedding_dimensions": int(self.sparse_embedding_dimensions),
            "enforce_sparse_dimension_match": bool(self.enforce_sparse_dimension_match),
            "await_embedding": bool(self.await_embedding),
            "resume": bool(self.resume),
            "checkpoint_save_every": int(self.checkpoint_save_every),
            "strict_checkpoint_match": bool(self.strict_checkpoint_match),
            "dedupe_by_content_hash": bool(self.dedupe_by_content_hash),
            "dedupe_scope": self.dedupe_scope,
            "idempotency_strategy": self.idempotency_strategy,
            "idempotency_prefix": self.idempotency_prefix,
        }
        if self.directory is not None:
            payload["directory"] = str(Path(self.directory).expanduser().resolve())
        if self.file_paths is not None:
            payload["file_paths"] = [str(Path(v).expanduser().resolve()) for v in self.file_paths]
        if self.max_files is not None:
            payload["max_files"] = int(self.max_files)
        if self.include_extensions is not None:
            payload["include_extensions"] = list(self.include_extensions)
        if self.exclude_globs is not None:
            payload["exclude_globs"] = list(self.exclude_globs)
        if self.sparse_embedding_function is not None:
            payload["sparse_embedding_function"] = str(self.sparse_embedding_function)
        if self.document_metadata is not None:
            payload["document_metadata"] = dict(self.document_metadata)
        if self.checkpoint_file is not None:
            payload["checkpoint_file"] = str(Path(self.checkpoint_file).expanduser().resolve())
        return payload


@dataclass
class HybridSearchOptions:
    limit_bm25: int = 12
    limit_similarity: int = 12
    limit_sparse: int = 0
    bm25_weight: float = 0.55
    similarity_weight: float = 0.45
    sparse_weight: float = 0.0

    def __post_init__(self) -> None:
        for key in ("limit_bm25", "limit_similarity", "limit_sparse"):
            value = int(getattr(self, key))
            if value < 0:
                raise ValueError(f"{key} must be >= 0.")
            setattr(self, key, value)
        for key in ("bm25_weight", "similarity_weight", "sparse_weight"):
            setattr(self, key, float(getattr(self, key)))

    def to_kwargs(self) -> Dict[str, Any]:
        return {
            "limit_bm25": int(self.limit_bm25),
            "limit_similarity": int(self.limit_similarity),
            "limit_sparse": int(self.limit_sparse),
            "bm25_weight": float(self.bm25_weight),
            "similarity_weight": float(self.similarity_weight),
            "sparse_weight": float(self.sparse_weight),
        }


def build_hybrid_search_options(**kwargs: Any) -> HybridSearchOptions:
    return HybridSearchOptions(**kwargs)


def build_upload_directory_options(**kwargs: Any) -> UploadDirectoryOptions:
    return UploadDirectoryOptions(**kwargs)


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_collection_summaries(data: Any) -> List[CollectionSummary]:
    rows = []
    if isinstance(data, dict) and isinstance(data.get("collections"), list):
        rows = data.get("collections", [])
    elif isinstance(data, list):
        rows = data
    parsed: List[CollectionSummary] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed.append(
            CollectionSummary(
                id=str(row.get("id") or row.get("hash_id") or ""),
                name=str(row.get("name") or row.get("title") or ""),
                document_count=int(row.get("document_count") or 0),
            )
        )
    return parsed
