from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.sql_db_tables import document_artifact as DocumentArtifact
from QueryLake.database.sql_db_tables import document_version as DocumentVersion
from QueryLake.runtime.ingestion_lineage import create_upload_lineage_rows


class _Result:
    def __init__(self, row):
        self._row = row

    def first(self):
        return self._row


class _DummySession:
    def __init__(self):
        self.versions = []
        self.artifacts = []

    def exec(self, stmt):
        params = stmt.compile().params
        document_id = params.get("document_id_1")
        versions = [row for row in self.versions if row.document_id == document_id]
        versions = sorted(versions, key=lambda row: row.version_no, reverse=True)
        return _Result(versions[0] if len(versions) > 0 else None)

    def add(self, row):
        if isinstance(row, DocumentVersion):
            self.versions.append(row)
        elif isinstance(row, DocumentArtifact):
            self.artifacts.append(row)

    def commit(self):
        return None

    def refresh(self, row):
        return None

    def rollback(self):
        return None


def test_create_upload_lineage_rows_increments_version(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_INGESTION_LINEAGE_ENABLED", "1")
    db = _DummySession()

    first = create_upload_lineage_rows(
        db,
        document_id="doc_1",
        created_by="tester",
        content_hash="h1",
        storage_ref="blob:dir_a",
    )
    second = create_upload_lineage_rows(
        db,
        document_id="doc_1",
        created_by="tester",
        content_hash="h2",
        storage_ref="blob:dir_b",
    )

    assert first is not None
    assert second is not None
    assert len(db.versions) == 2
    assert db.versions[0].version_no == 1
    assert db.versions[1].version_no == 2
    assert len(db.artifacts) == 2
    assert db.artifacts[0].artifact_type == "source_blob"


def test_create_upload_lineage_rows_respects_flag(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_INGESTION_LINEAGE_ENABLED", "0")
    db = _DummySession()
    out = create_upload_lineage_rows(
        db,
        document_id="doc_1",
        created_by="tester",
        content_hash="h1",
        storage_ref="blob:dir_a",
    )
    assert out is None
    assert len(db.versions) == 0
    assert len(db.artifacts) == 0
