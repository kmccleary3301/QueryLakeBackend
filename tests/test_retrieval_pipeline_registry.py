from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.sql_db_tables import retrieval_pipeline_config as RetrievalPipelineConfig
from QueryLake.runtime.retrieval_pipeline_registry import (
    pipeline_spec_hash,
    register_pipeline_spec,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if len(self._rows) > 0 else None

    def all(self):
        return self._rows


class _DummySession:
    def __init__(self):
        self.rows = []

    def add(self, row):
        self.rows.append(row)

    def commit(self):
        return None

    def refresh(self, row):
        return None

    def exec(self, stmt):
        compiled = stmt.compile()
        params = compiled.params
        rows = self.rows
        pipeline_id = params.get("pipeline_id_1")
        version = params.get("version_1")
        if pipeline_id is not None:
            rows = [row for row in rows if row.pipeline_id == pipeline_id]
        if version is not None:
            rows = [row for row in rows if row.version == version]
        return _Result(rows)


def test_pipeline_spec_hash_is_deterministic():
    spec_a = {"pipeline_id": "p", "version": "v1", "stages": [{"stage_id": "bm25", "enabled": True}]}
    spec_b = {"version": "v1", "stages": [{"enabled": True, "stage_id": "bm25"}], "pipeline_id": "p"}
    assert pipeline_spec_hash(spec_a) == pipeline_spec_hash(spec_b)


def test_register_pipeline_spec_enforces_immutability():
    db = _DummySession()
    spec = {"pipeline_id": "hybrid", "version": "v1", "stages": [{"stage_id": "bm25"}]}

    row_1 = register_pipeline_spec(
        db,
        pipeline_id="hybrid",
        version="v1",
        spec_json=spec,
        created_by="tester",
    )
    assert isinstance(row_1, RetrievalPipelineConfig)
    assert row_1.pipeline_id == "hybrid"

    row_2 = register_pipeline_spec(
        db,
        pipeline_id="hybrid",
        version="v1",
        spec_json=spec,
        created_by="tester",
    )
    assert row_2.id == row_1.id

    try:
        register_pipeline_spec(
            db,
            pipeline_id="hybrid",
            version="v1",
            spec_json={"pipeline_id": "hybrid", "version": "v1", "stages": [{"stage_id": "dense"}]},
        )
        assert False, "Expected immutable registry conflict"
    except ValueError:
        pass
