from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


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
