from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.ingestion_dual_write import (
    build_segment_rows_from_chunk_rows,
    dual_write_segments_enabled,
)


def test_dual_write_segments_enabled_flag(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_DUAL_WRITE_SEGMENTS", "1")
    assert dual_write_segments_enabled() is True
    monkeypatch.setenv("QUERYLAKE_DUAL_WRITE_SEGMENTS", "0")
    assert dual_write_segments_enabled() is False


def test_build_segment_rows_from_chunk_rows():
    rows = build_segment_rows_from_chunk_rows(
        document_version_id="dv_1",
        chunk_rows=[
            {"id": "c1", "document_chunk_number": 0, "text": "hello", "md": {"source": "a"}},
            {"id": "c2", "document_chunk_number": 1, "text": "world", "md": {"source": "a"}},
        ],
    )
    assert len(rows) == 2
    assert rows[0]["document_version_id"] == "dv_1"
    assert rows[0]["segment_index"] == 0
    assert rows[0]["md"]["legacy_chunk_id"] == "c1"
    assert isinstance(rows[0]["md"]["content_fingerprint"], str)
