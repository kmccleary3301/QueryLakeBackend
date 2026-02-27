from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.backfill_segments import backfill_chunk_rows_to_segments


def test_backfill_is_resumable_and_idempotent(tmp_path: Path):
    chunk_rows = [
        {"id": "c1", "document_chunk_number": 0, "text": "alpha", "md": {}},
        {"id": "c2", "document_chunk_number": 1, "text": "beta", "md": {}},
        {"id": "c3", "document_chunk_number": 2, "text": "gamma", "md": {}},
    ]
    checkpoint = tmp_path / "ckpt.json"
    writes = []
    existing_keys = set()

    def _write(rows):
        writes.extend(rows)

    first = backfill_chunk_rows_to_segments(
        chunk_rows=chunk_rows,
        document_version_id="dv_1",
        checkpoint_path=checkpoint,
        existing_keys=existing_keys,
        write_segments=_write,
        batch_size=2,
    )
    assert first["written"] == 3
    assert first["cursor"] == 3

    second = backfill_chunk_rows_to_segments(
        chunk_rows=chunk_rows,
        document_version_id="dv_1",
        checkpoint_path=checkpoint,
        existing_keys=existing_keys,
        write_segments=_write,
        batch_size=2,
    )
    assert second["written"] == 0
    assert second["skipped"] == 0
    assert len(writes) == 3
