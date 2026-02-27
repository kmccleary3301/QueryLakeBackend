from pathlib import Path
import sys

import pgvector
import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.api.search import _normalize_sparse_query_value
from QueryLake.database.create_db_session import configured_sparse_index_dimensions
from QueryLake.vector_database.embeddings import (
    SparseDimensionMismatchError,
    _extract_sparse_embedding,
)


def test_extract_sparse_embedding_strict_dimension_mismatch_raises():
    value = pgvector.SparseVector({0: 1.0, 7: 0.2}, 16)
    with pytest.raises(SparseDimensionMismatchError):
        _extract_sparse_embedding(
            value,
            dimensions=8,
            strict_dimensions=True,
            source_label="unit_test_sparse",
        )


def test_extract_sparse_embedding_non_strict_coerces_dimension():
    value = pgvector.SparseVector({0: 1.0, 7: 0.2}, 16)
    parsed = _extract_sparse_embedding(
        value,
        dimensions=8,
        strict_dimensions=False,
        source_label="unit_test_sparse",
    )
    assert parsed is not None
    assert int(parsed.dimensions()) == 8


def test_normalize_sparse_query_value_strict_mismatch_raises():
    with pytest.raises(ValueError):
        _normalize_sparse_query_value(
            {"indices": [0, 1], "values": [0.8, 0.2], "dimensions": 16},
            dimensions=8,
            strict_dimensions=True,
            source_label="query.sparse",
        )


def test_configured_sparse_index_dimensions_reads_env(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_SPARSE_INDEX_DIMENSIONS", "2048")
    assert configured_sparse_index_dimensions() == 2048
