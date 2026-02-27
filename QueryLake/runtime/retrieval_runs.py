from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from typing import Any, Dict, Iterable, List, Optional

from sqlmodel import Session, delete, select

from QueryLake.database import sql_db_tables as T

logger = logging.getLogger(__name__)


def _flag_enabled(name: str, default: bool) -> bool:
    raw = (os.getenv(name, "1" if default else "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def normalize_query_text(query_payload: Any) -> str:
    if isinstance(query_payload, str):
        return query_payload.strip()
    if isinstance(query_payload, dict):
        # Deterministic text for hashing/replay.
        return json.dumps(query_payload, sort_keys=True, ensure_ascii=False)
    return str(query_payload)


def query_fingerprint(query_text: str) -> str:
    return hashlib.sha256(query_text.encode("utf-8")).hexdigest()


def _redact_string(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]
    return f"<redacted:{digest}>"


def _sanitize_for_pii(value: Any) -> Any:
    if isinstance(value, str):
        return _redact_string(value)
    if isinstance(value, list):
        return [_sanitize_for_pii(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _sanitize_for_pii(v) for k, v in value.items()}
    return value


def _extract_primary_id(result_id: Any) -> Optional[str]:
    if isinstance(result_id, str):
        return result_id
    if isinstance(result_id, list) and len(result_id) > 0 and isinstance(result_id[0], str):
        return result_id[0]
    return None


def build_candidate_rows(
    run_id: str,
    rows: Iterable[Dict[str, Any]],
) -> List[T.retrieval_candidate]:
    out: List[T.retrieval_candidate] = []
    for idx, row in enumerate(rows):
        raw_id = row.get("id")
        content_id = _extract_primary_id(raw_id)
        if content_id is None:
            continue
        stage_scores = {
            key: row.get(key)
            for key in ["bm25_score", "similarity_score", "hybrid_score", "rerank_score"]
            if row.get(key) is not None
        }
        provenance: List[str] = []
        if row.get("bm25_score") is not None and float(row.get("bm25_score") or 0) > 0:
            provenance.append("bm25")
        if row.get("similarity_score") is not None and float(row.get("similarity_score") or 0) > 0:
            provenance.append("dense")
        if row.get("rerank_score") is not None:
            provenance.append("rerank")
        out.append(
            T.retrieval_candidate(
                run_id=run_id,
                content_id=content_id,
                final_selected=True,
                stage_scores=stage_scores,
                stage_ranks={"final_rank": idx + 1},
                provenance=provenance,
                md={},
            )
        )
    return out


def build_candidate_rows_from_details(
    run_id: str,
    candidates: Iterable[Dict[str, Any]],
    *,
    pii_safe: bool = False,
) -> List[T.retrieval_candidate]:
    out: List[T.retrieval_candidate] = []
    for idx, candidate in enumerate(candidates):
        content_id = candidate.get("content_id")
        if not isinstance(content_id, str):
            continue
        stage_scores = candidate.get("stage_scores")
        if not isinstance(stage_scores, dict):
            stage_scores = {}
        stage_ranks = candidate.get("stage_ranks")
        if not isinstance(stage_ranks, dict):
            stage_ranks = {"final_rank": idx + 1}
        else:
            stage_ranks = {"final_rank": idx + 1, **stage_ranks}
        provenance = candidate.get("provenance")
        if not isinstance(provenance, list):
            provenance = []
        md = candidate.get("metadata")
        if not isinstance(md, dict):
            md = {}
        if pii_safe:
            md = _sanitize_for_pii(md)
        out.append(
            T.retrieval_candidate(
                run_id=run_id,
                content_id=content_id,
                final_selected=bool(candidate.get("selected", True)),
                stage_scores=stage_scores,
                stage_ranks=stage_ranks,
                provenance=provenance,
                md=md,
            )
        )
    return out


def _persist_rows(
    bind: Any,
    run_row: T.retrieval_run,
    candidate_rows: List[T.retrieval_candidate],
) -> None:
    with Session(bind) as run_db:
        run_db.add(run_row)
        if len(candidate_rows) > 0:
            run_db.add_all(candidate_rows)
        run_db.commit()


def log_retrieval_run(
    database: Session,
    *,
    route: str,
    actor_user: Optional[str],
    query_payload: Any,
    collection_ids: Optional[List[str]] = None,
    pipeline_id: Optional[str] = None,
    pipeline_version: Optional[str] = None,
    filters: Optional[Dict[str, Any]] = None,
    budgets: Optional[Dict[str, Any]] = None,
    timings: Optional[Dict[str, Any]] = None,
    counters: Optional[Dict[str, Any]] = None,
    costs: Optional[Dict[str, Any]] = None,
    index_snapshots_used: Optional[Dict[str, Any]] = None,
    result_rows: Optional[List[Dict[str, Any]]] = None,
    candidate_details: Optional[List[Dict[str, Any]]] = None,
    status: str = "ok",
    error: Optional[str] = None,
    md: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if not _flag_enabled("QUERYLAKE_RETRIEVAL_RUN_LOGGING", True):
        return None

    collection_ids = collection_ids or []
    query_text_raw = normalize_query_text(query_payload)
    pii_safe = _flag_enabled("QUERYLAKE_RETRIEVAL_PII_SAFE_LOGGING", False)
    query_text = _redact_string(query_text_raw) if pii_safe else query_text_raw
    run_id: str = T.retrieval_run().run_id  # reserve deterministic id format from table default

    run_row = T.retrieval_run(
        run_id=run_id,
        created_at=time.time(),
        completed_at=time.time(),
        status=status,
        route=route,
        actor_user=actor_user,
        tenant_scope=collection_ids[0] if len(collection_ids) > 0 else None,
        pipeline_id=pipeline_id,
        pipeline_version=pipeline_version,
        query_text=query_text,
        query_hash=query_fingerprint(query_text_raw),
        filters=_sanitize_for_pii(filters or {"collection_ids": collection_ids}) if pii_safe else (filters or {"collection_ids": collection_ids}),
        budgets=_sanitize_for_pii(budgets or {}) if pii_safe else (budgets or {}),
        timings=timings or {},
        counters=counters or {},
        costs=costs or {},
        index_snapshots_used=index_snapshots_used or {},
        result_ids=[rid for rid in [_extract_primary_id((row or {}).get("id")) for row in (result_rows or [])] if rid is not None],
        md=_sanitize_for_pii(md or {}) if pii_safe else (md or {}),
        error=_redact_string(error) if (pii_safe and isinstance(error, str)) else error,
    )

    candidate_rows: List[T.retrieval_candidate] = []
    if _flag_enabled("QUERYLAKE_RETRIEVAL_CANDIDATE_LOGGING", False):
        if candidate_details:
            candidate_rows = build_candidate_rows_from_details(
                run_id,
                candidate_details,
                pii_safe=pii_safe,
            )
        elif result_rows:
            candidate_rows = build_candidate_rows(run_id, result_rows)

    try:
        bind = database.get_bind()
        _persist_rows(bind, run_row, candidate_rows)
        return run_id
    except Exception:
        logger.exception("Failed to persist retrieval run log for route=%s", route)
        return None


def delete_retrieval_logs(
    database: Session,
    *,
    older_than_seconds: Optional[float] = None,
    tenant_scope: Optional[str] = None,
    actor_user: Optional[str] = None,
    dry_run: bool = False,
) -> Dict[str, int]:
    now_ts = time.time()
    cutoff_ts = None
    if older_than_seconds is not None:
        cutoff_ts = now_ts - float(older_than_seconds)

    run_rows = list(database.exec(select(T.retrieval_run)).all())
    run_ids: List[str] = []
    for row in run_rows:
        run_id = getattr(row, "run_id", None)
        if not isinstance(run_id, str):
            continue
        if cutoff_ts is not None and float(getattr(row, "created_at", 0.0)) > float(cutoff_ts):
            continue
        if tenant_scope is not None and getattr(row, "tenant_scope", None) != tenant_scope:
            continue
        if actor_user is not None and getattr(row, "actor_user", None) != actor_user:
            continue
        run_ids.append(run_id)

    candidate_count = 0
    for run_id in run_ids:
        rows = list(
            database.exec(
                select(T.retrieval_candidate).where(T.retrieval_candidate.run_id == run_id)
            ).all()
        )
        candidate_count += len(rows)

    payload = {
        "run_count": len(run_ids),
        "candidate_count": candidate_count,
        "deleted": 0 if dry_run else len(run_ids),
    }
    if dry_run or len(run_ids) == 0:
        return payload

    for run_id in run_ids:
        database.exec(delete(T.retrieval_candidate).where(T.retrieval_candidate.run_id == run_id))
        database.exec(delete(T.retrieval_run).where(T.retrieval_run.run_id == run_id))
    database.commit()
    return payload
