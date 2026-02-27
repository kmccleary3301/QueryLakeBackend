#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from QueryLake.runtime.backfill_segments import backfill_chunk_rows_to_segments


def main() -> int:
    parser = argparse.ArgumentParser(description="Resumable/idempotent chunk->segment backfill scaffold.")
    parser.add_argument("--input-chunks", required=True, help="JSON list of chunk rows")
    parser.add_argument("--document-version-id", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-segments", required=True, help="JSONL output sink for written segment rows")
    parser.add_argument("--batch-size", type=int, default=100)
    args = parser.parse_args()

    chunk_rows = json.loads(Path(args.input_chunks).read_text(encoding="utf-8"))
    if not isinstance(chunk_rows, list):
        raise ValueError("--input-chunks must contain a JSON list")

    out_path = Path(args.output_segments)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    existing_keys = set()

    def _append_segments(rows):
        with out_path.open("a", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    stats = backfill_chunk_rows_to_segments(
        chunk_rows=chunk_rows,
        document_version_id=args.document_version_id,
        checkpoint_path=Path(args.checkpoint),
        existing_keys=existing_keys,
        write_segments=_append_segments,
        batch_size=args.batch_size,
    )
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
