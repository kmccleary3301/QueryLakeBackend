from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List

from QueryLake.runtime.content_fingerprint import content_fingerprint


def dual_write_segments_enabled() -> bool:
    raw = (os.getenv("QUERYLAKE_DUAL_WRITE_SEGMENTS", "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def build_segment_rows_from_chunk_rows(
    *,
    document_version_id: str,
    chunk_rows: Iterable[Dict[str, Any]],
    segment_type: str = "chunk",
) -> List[Dict[str, Any]]:
    rows_out: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunk_rows):
        text = str(chunk.get("text", ""))
        md = chunk.get("md", {}) if isinstance(chunk.get("md", {}), dict) else {}
        row = {
            "document_version_id": document_version_id,
            "artifact_id": None,
            "segment_type": segment_type,
            "segment_index": int(chunk.get("document_chunk_number", idx)),
            "text": text,
            "md": {
                **md,
                "legacy_chunk_id": chunk.get("id"),
                "content_fingerprint": content_fingerprint(text=text, md=md),
            },
            "embedding": chunk.get("embedding"),
        }
        rows_out.append(row)
    return rows_out
