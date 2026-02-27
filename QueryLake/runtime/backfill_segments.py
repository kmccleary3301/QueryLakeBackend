from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Set, Tuple

from QueryLake.runtime.ingestion_dual_write import build_segment_rows_from_chunk_rows


def load_checkpoint(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {"cursor": 0}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"cursor": int(payload.get("cursor", 0))}


def save_checkpoint(path: Path, *, cursor: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"cursor": int(cursor)}, indent=2), encoding="utf-8")


def backfill_chunk_rows_to_segments(
    *,
    chunk_rows: List[Dict],
    document_version_id: str,
    checkpoint_path: Path,
    existing_keys: Set[Tuple[str, str, int]],
    write_segments: Callable[[List[Dict]], None],
    batch_size: int = 100,
) -> Dict[str, int]:
    ckpt = load_checkpoint(checkpoint_path)
    cursor = max(0, int(ckpt["cursor"]))

    total = len(chunk_rows)
    written = 0
    skipped = 0

    while cursor < total:
        next_cursor = min(total, cursor + max(1, int(batch_size)))
        batch_chunks = chunk_rows[cursor:next_cursor]
        segment_rows = build_segment_rows_from_chunk_rows(
            document_version_id=document_version_id,
            chunk_rows=batch_chunks,
            segment_type="chunk",
        )
        to_write = []
        for row in segment_rows:
            key = (row["document_version_id"], row["segment_type"], int(row["segment_index"]))
            if key in existing_keys:
                skipped += 1
                continue
            existing_keys.add(key)
            to_write.append(row)

        if len(to_write) > 0:
            write_segments(to_write)
            written += len(to_write)

        cursor = next_cursor
        save_checkpoint(checkpoint_path, cursor=cursor)

    return {"total": total, "written": written, "skipped": skipped, "cursor": cursor}
