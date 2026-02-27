from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.sql_db_tables import embedding_record as EmbeddingRecord
from QueryLake.runtime.embedding_records import get_or_create_embedding


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if len(self._rows) > 0 else None


class _DummySession:
    def __init__(self):
        self.rows = []

    def add(self, row):
        if row not in self.rows:
            self.rows.append(row)

    def commit(self):
        return None

    def refresh(self, row):
        return None

    def exec(self, stmt):
        params = stmt.compile().params
        rows = self.rows
        seg = params.get("segment_id_1")
        model = params.get("model_id_1")
        h = params.get("input_hash_1")
        if seg is not None:
            rows = [row for row in rows if row.segment_id == seg]
        if model is not None:
            rows = [row for row in rows if row.model_id == model]
        if h is not None:
            rows = [row for row in rows if row.input_hash == h]
        return _Result(rows)


def test_get_or_create_embedding_uses_record_cache():
    db = _DummySession()
    calls = {"n": 0}

    def _embed(text: str):
        calls["n"] += 1
        return [0.1, 0.2, 0.3]

    first = get_or_create_embedding(
        db,
        segment_id="seg_1",
        text="alpha",
        model_id="m1",
        embedding_fn=_embed,
    )
    assert first["cache_hit"] is False
    assert calls["n"] == 1
    assert isinstance(db.rows[0], EmbeddingRecord)

    second = get_or_create_embedding(
        db,
        segment_id="seg_1",
        text="alpha",
        model_id="m1",
        embedding_fn=_embed,
    )
    assert second["cache_hit"] is True
    assert calls["n"] == 1
