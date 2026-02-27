from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.sql_db_tables import retrieval_pipeline_binding as RetrievalPipelineBinding
from QueryLake.runtime.retrieval_rollout import rollback_active_pipeline, set_active_pipeline


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
        route = params.get("route_1")
        tenant_scope = params.get("tenant_scope_1")
        rows = self.rows
        if route is not None:
            rows = [row for row in rows if row.route == route]
        if "tenant_scope_1" in params:
            rows = [row for row in rows if row.tenant_scope == tenant_scope]
        return _Result(rows)


def test_set_and_rollback_pipeline_binding():
    db = _DummySession()
    row = set_active_pipeline(
        db,
        route="search_hybrid",
        pipeline_id="p1",
        pipeline_version="v1",
        updated_by="tester",
    )
    assert isinstance(row, RetrievalPipelineBinding)
    assert row.active_pipeline_id == "p1"

    row = set_active_pipeline(
        db,
        route="search_hybrid",
        pipeline_id="p2",
        pipeline_version="v2",
        updated_by="tester",
    )
    assert row.active_pipeline_id == "p2"
    assert row.previous_pipeline_id == "p1"

    row = rollback_active_pipeline(db, route="search_hybrid", updated_by="tester")
    assert row.active_pipeline_id == "p1"
    assert row.previous_pipeline_id == "p2"
