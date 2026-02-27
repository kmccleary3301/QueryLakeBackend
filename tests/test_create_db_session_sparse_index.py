from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database import create_db_session as db_session


def test_create_sparse_vector_index_sql_uses_dimension_cast():
    sql = db_session.create_sparse_vector_index_sql(2048)
    assert "embedding_sparse::sparsevec(2048)" in sql
    assert "sparsevec_cosine_ops" in sql


def test_sparse_index_dimensions_default_guard(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_SPARSE_INDEX_DIMENSIONS", "-1")
    assert db_session._sparse_index_dimensions_default() == 1024
    monkeypatch.setenv("QUERYLAKE_SPARSE_INDEX_DIMENSIONS", "3072")
    assert db_session._sparse_index_dimensions_default() == 3072

