import asyncio
from types import SimpleNamespace
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import QueryLake.runtime.retrieval_runs as retrieval_runs
from QueryLake.api import search as search_api


def test_normalize_query_dict_is_deterministic():
    q1 = {"bm25": "alpha", "embedding": "alpha"}
    q2 = {"embedding": "alpha", "bm25": "alpha"}
    s1 = retrieval_runs.normalize_query_text(q1)
    s2 = retrieval_runs.normalize_query_text(q2)
    assert s1 == s2
    assert retrieval_runs.query_fingerprint(s1) == retrieval_runs.query_fingerprint(s2)


def test_build_candidate_rows_handles_list_ids():
    rows = [
        {
            "id": ["chunk_a", "chunk_b"],
            "bm25_score": 0.2,
            "similarity_score": 0.3,
            "hybrid_score": 0.5,
        },
        {"id": "chunk_c", "rerank_score": 0.9},
    ]
    candidates = retrieval_runs.build_candidate_rows("run_1", rows)
    assert len(candidates) == 2
    assert candidates[0].content_id == "chunk_a"
    assert "bm25" in candidates[0].provenance
    assert "dense" in candidates[0].provenance
    assert candidates[1].content_id == "chunk_c"
    assert "rerank" in candidates[1].provenance


def test_build_candidate_rows_from_details_preserves_stage_trace():
    details = [
        {
            "content_id": "chunk_a",
            "stage_scores": {"bm25_score": 0.7, "weighted_fused": 0.9},
            "stage_ranks": {"bm25": 2, "weighted_fused": 1},
            "provenance": ["bm25", "dense"],
            "selected": True,
            "metadata": {"document_id": "doc_1"},
        }
    ]
    rows = retrieval_runs.build_candidate_rows_from_details("run_2", details)
    assert len(rows) == 1
    assert rows[0].content_id == "chunk_a"
    assert rows[0].stage_scores["weighted_fused"] == 0.9
    assert rows[0].stage_ranks["final_rank"] == 1
    assert rows[0].md["document_id"] == "doc_1"


def test_log_retrieval_run_disabled_skips_persist(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_RUN_LOGGING", "0")
    called = {"persist": False}

    def _persist(*args, **kwargs):
        called["persist"] = True

    monkeypatch.setattr(retrieval_runs, "_persist_rows", _persist)
    db = SimpleNamespace(get_bind=lambda: object())
    run_id = retrieval_runs.log_retrieval_run(
        db,
        route="search_hybrid",
        actor_user="tester",
        query_payload="hello world",
        result_rows=[],
    )
    assert run_id is None
    assert called["persist"] is False


def test_log_retrieval_run_pii_safe_mode_redacts_strings(monkeypatch):
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_RUN_LOGGING", "1")
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_CANDIDATE_LOGGING", "1")
    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_PII_SAFE_LOGGING", "1")
    captured = {"run": None, "candidates": None}

    def _persist(bind, run_row, candidate_rows):
        captured["run"] = run_row
        captured["candidates"] = candidate_rows

    monkeypatch.setattr(retrieval_runs, "_persist_rows", _persist)
    db = SimpleNamespace(get_bind=lambda: object())
    raw_query = "contact alice@example.com for boiler pressure note"

    run_id = retrieval_runs.log_retrieval_run(
        db,
        route="search_hybrid",
        actor_user="tester",
        query_payload=raw_query,
        filters={"email": "alice@example.com"},
        md={"notes": "phone 555-0101"},
        candidate_details=[
            {
                "content_id": "chunk_1",
                "metadata": {"user_email": "alice@example.com"},
            }
        ],
        result_rows=[],
        error="oops@private.local",
    )

    assert run_id is not None
    run_row = captured["run"]
    assert run_row is not None
    assert run_row.query_text.startswith("<redacted:")
    assert run_row.query_hash == retrieval_runs.query_fingerprint(raw_query)
    assert run_row.filters["email"].startswith("<redacted:")
    assert run_row.md["notes"].startswith("<redacted:")
    assert run_row.error.startswith("<redacted:")
    assert captured["candidates"][0].md["user_email"].startswith("<redacted:")


class _ListResult:
    def __init__(self, rows):
        self._rows = rows

    def all(self):
        return self._rows


class _CleanupDB:
    def __init__(self, runs, candidates):
        self.runs = runs
        self.candidates = candidates
        self.committed = False

    def exec(self, stmt):
        sql = str(stmt).strip()
        params = stmt.compile().params
        if sql.startswith("SELECT") and "FROM retrieval_run" in sql:
            return _ListResult(self.runs[:])
        if sql.startswith("SELECT") and "FROM retrieval_candidate" in sql:
            run_id = params.get("run_id_1")
            rows = [row for row in self.candidates if row.run_id == run_id]
            return _ListResult(rows)
        if sql.startswith("DELETE FROM retrieval_candidate"):
            run_id = params.get("run_id_1")
            self.candidates = [row for row in self.candidates if row.run_id != run_id]
            return _ListResult([])
        if sql.startswith("DELETE FROM retrieval_run"):
            run_id = params.get("run_id_1")
            self.runs = [row for row in self.runs if row.run_id != run_id]
            return _ListResult([])
        raise AssertionError(f"unexpected statement: {sql}")

    def commit(self):
        self.committed = True


def test_delete_retrieval_logs_dry_run_reports_counts(monkeypatch):
    monkeypatch.setattr(retrieval_runs.time, "time", lambda: 100.0)
    runs = [
        SimpleNamespace(run_id="r1", created_at=10.0, tenant_scope="tenant_a", actor_user="alice"),
        SimpleNamespace(run_id="r2", created_at=90.0, tenant_scope="tenant_a", actor_user="alice"),
        SimpleNamespace(run_id="r3", created_at=5.0, tenant_scope="tenant_b", actor_user="bob"),
    ]
    candidates = [
        SimpleNamespace(run_id="r1"),
        SimpleNamespace(run_id="r1"),
        SimpleNamespace(run_id="r3"),
    ]
    db = _CleanupDB(runs, candidates)
    report = retrieval_runs.delete_retrieval_logs(
        db,
        older_than_seconds=50,
        tenant_scope="tenant_a",
        dry_run=True,
    )
    assert report["run_count"] == 1
    assert report["candidate_count"] == 2
    assert report["deleted"] == 0
    assert db.committed is False


def test_delete_retrieval_logs_deletes_runs_and_candidates(monkeypatch):
    monkeypatch.setattr(retrieval_runs.time, "time", lambda: 100.0)
    runs = [
        SimpleNamespace(run_id="r1", created_at=10.0, tenant_scope="tenant_a", actor_user="alice"),
        SimpleNamespace(run_id="r2", created_at=5.0, tenant_scope="tenant_a", actor_user="bob"),
    ]
    candidates = [
        SimpleNamespace(run_id="r1"),
        SimpleNamespace(run_id="r2"),
    ]
    db = _CleanupDB(runs, candidates)
    report = retrieval_runs.delete_retrieval_logs(
        db,
        older_than_seconds=50,
        tenant_scope="tenant_a",
        actor_user="alice",
        dry_run=False,
    )
    assert report["run_count"] == 1
    assert report["candidate_count"] == 1
    assert report["deleted"] == 1
    assert [row.run_id for row in db.runs] == ["r2"]
    assert [row.run_id for row in db.candidates] == ["r2"]
    assert db.committed is True


def test_search_hybrid_statement_includes_weighted_fusion(monkeypatch):
    class DummyDB:
        pass

    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    statement = asyncio.run(
        search_api.search_hybrid(
            database=DummyDB(),
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query="boiler pressure limits",
            embedding=[0.0] * 1024,
            collection_ids=["abc123"],
            limit_bm25=5,
            limit_similarity=5,
            similarity_weight=0.25,
            bm25_weight=0.75,
            return_statement=True,
            web_search=False,
            rerank=False,
            group_chunks=False,
        )
    )

    assert "semantic_search.rank" in statement
    assert "bm25_ranked.rank" in statement
    assert "* 0.25" in statement
    assert "* 0.75" in statement


def test_search_hybrid_statement_supports_sparse_lane(monkeypatch):
    class DummyDB:
        pass

    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    statement = asyncio.run(
        search_api.search_hybrid(
            database=DummyDB(),
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "boiler", "sparse": "boiler"},
            embedding_sparse={0: 1.0, 4: 0.25},
            collection_ids=["abc123"],
            limit_bm25=0,
            limit_similarity=0,
            limit_sparse=8,
            similarity_weight=0.0,
            bm25_weight=0.0,
            sparse_weight=1.0,
            use_bm25=False,
            use_similarity=False,
            use_sparse=True,
            return_statement=True,
            web_search=False,
            rerank=False,
            group_chunks=False,
        )
    )

    assert "sparse_search" in statement
    assert "embedding_sparse::sparsevec(1024)" in statement
    assert "AS sparsevec(1024)" in statement
    assert "* 1.0" in statement


def test_search_hybrid_adaptive_routing_identifier_profile(monkeypatch):
    class DummyDB:
        pass

    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    statement = asyncio.run(
        search_api.search_hybrid(
            database=DummyDB(),
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query="DOC-12345 pressure threshold",
            embedding=[0.0] * 1024,
            collection_ids=["abc123"],
            adaptive_lane_routing=True,
            use_sparse=False,
            return_statement=True,
            web_search=False,
            rerank=False,
            group_chunks=False,
        )
    )

    assert "* 0.72" in statement
    assert "* 0.18" in statement
    assert "LIMIT 24" in statement
    assert "LIMIT 8" in statement


def test_sparse_pruning_and_calibration_l1():
    sparse_vec = search_api._normalize_sparse_query_value({0: 1.0, 1: 0.5, 2: 0.2, 3: 0.1}, dimensions=8)
    assert sparse_vec is not None

    new_vec, terms_before, terms_after = search_api._apply_sparse_pruning_and_calibration(
        sparse_vec,
        max_terms=2,
        min_abs_weight=0.15,
        calibration="l1",
    )

    assert terms_before == 4
    assert terms_after == 2
    coo = new_vec.to_coo()
    cols = list(coo.col)
    vals = list(coo.data)
    assert set(cols) == {0, 1}
    assert abs(sum(abs(v) for v in vals) - 1.0) < 1e-9


def test_dynamic_lane_limit_schedule_reduces_total_by_weight():
    new_bm25, new_dense, new_sparse, applied = search_api._apply_dynamic_lane_limit_schedule(
        limit_bm25=24,
        limit_similarity=16,
        limit_sparse=12,
        bm25_weight=0.7,
        similarity_weight=0.2,
        sparse_weight=0.1,
        use_bm25=True,
        use_similarity=True,
        use_sparse=True,
        total_cap=20,
        min_per_enabled=3,
    )

    assert applied is True
    assert new_bm25 + new_dense + new_sparse == 20
    assert new_bm25 >= new_dense >= new_sparse
    assert min(new_bm25, new_dense, new_sparse) >= 3


def test_search_hybrid_strict_constraint_prefilter_applies_constraint_query(monkeypatch):
    class DummyDB:
        pass

    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    statement = asyncio.run(
        search_api.search_hybrid(
            database=DummyDB(),
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "boiler pressure", "constraints": "document_name:manual -safety"},
            embedding=[0.0] * 1024,
            collection_ids=["abc123"],
            limit_bm25=8,
            limit_similarity=0,
            limit_sparse=0,
            use_bm25=True,
            use_similarity=False,
            use_sparse=False,
            strict_constraint_prefilter=True,
            return_statement=True,
            web_search=False,
            rerank=False,
            group_chunks=False,
        )
    )
    assert "document_name:manual" in statement
    assert "NOT text:safety" in statement


def test_search_hybrid_can_disable_strict_constraint_prefilter(monkeypatch):
    class DummyDB:
        pass

    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    statement = asyncio.run(
        search_api.search_hybrid(
            database=DummyDB(),
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "boiler pressure", "constraints": "document_name:manual -safety"},
            embedding=[0.0] * 1024,
            collection_ids=["abc123"],
            limit_bm25=8,
            limit_similarity=0,
            limit_sparse=0,
            use_bm25=True,
            use_similarity=False,
            use_sparse=False,
            strict_constraint_prefilter=False,
            return_statement=True,
            web_search=False,
            rerank=False,
            group_chunks=False,
        )
    )
    assert "document_name:manual" not in statement
    assert "NOT text:safety" not in statement


def test_search_hybrid_supports_custom_bm25_catch_all_fields(monkeypatch):
    class DummyDB:
        pass

    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    statement = asyncio.run(
        search_api.search_hybrid(
            database=DummyDB(),
            toolchain_function_caller=lambda name: None,
            auth={"username": "tester", "password_prehash": "x"},
            query={"bm25": "vapor recovery", "bm25_catch_all_fields": ["text", "document_name"]},
            embedding=[0.0] * 1024,
            collection_ids=["abc123"],
            limit_bm25=8,
            limit_similarity=0,
            limit_sparse=0,
            use_bm25=True,
            use_similarity=False,
            use_sparse=False,
            return_statement=True,
            web_search=False,
            rerank=False,
            group_chunks=False,
        )
    )
    assert "text:vapor" in statement
    assert "document_name:vapor" in statement


def test_search_bm25_segment_statement_uses_segment_table(monkeypatch):
    class DummyDB:
        pass

    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_SEGMENT_ENABLED", "1")
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    statement = search_api.search_bm25(
        database=DummyDB(),
        auth={"username": "tester", "password_prehash": "x"},
        query="boiler pressure limits",
        collection_ids=["abc123"],
        table="segment",
        return_statement=True,
        group_chunks=False,
    )
    assert "FROM document_segment" in statement


def test_search_bm25_segment_requires_feature_flag(monkeypatch):
    class DummyDB:
        pass

    monkeypatch.setenv("QUERYLAKE_RETRIEVAL_SEGMENT_ENABLED", "0")
    monkeypatch.setattr(search_api, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))
    monkeypatch.setattr(search_api, "assert_collections_priviledge", lambda database, auth, collection_ids: None)

    try:
        search_api.search_bm25(
            database=DummyDB(),
            auth={"username": "tester", "password_prehash": "x"},
            query="boiler pressure limits",
            collection_ids=["abc123"],
            table="segment",
            return_statement=True,
            group_chunks=False,
        )
        assert False, "expected feature-flag assertion for segment retrieval"
    except AssertionError as exc:
        assert "segment retrieval is disabled" in str(exc)
