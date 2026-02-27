#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from sqlalchemy import text
from sqlmodel import Session, create_engine, select

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from QueryLake.database.sql_db_tables import DocumentChunk


DEFAULT_DB_URL = os.environ.get(
    "QUERYLAKE_DB_URL",
    "postgresql://querylake_access:querylake_access_password@localhost:5444/querylake_database",
)


PIPELINE_TABLE = "pipeline_work_item"
PIPELINE_KEY_DEFAULT = "documentchunk_embedding_backfill"
PIPELINE_STAGE_DEFAULT = "embed_missing"


@dataclass
class BackfillStats:
    batches_ok: int = 0
    batches_failed: int = 0
    chunks_embedded: int = 0
    docs_promoted_to_finished_4: int = 0
    queue_items_seeded: int = 0
    queue_items_claimed: int = 0
    queue_items_done: int = 0
    queue_items_stale_done: int = 0
    queue_items_retried: int = 0
    started_at_unix: float = 0.0
    ended_at_unix: float = 0.0
    runtime_seconds: float = 0.0
    last_error: Optional[str] = None


def parse_csv(value: str) -> List[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def resolve_default_embedding_model(*, api_base_url: str, api_key: str, timeout_s: int) -> str:
    resp = requests.get(
        f"{api_base_url.rstrip('/')}/api/get_available_models",
        json={"auth": {"api_key": api_key}},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("success") is False:
        raise RuntimeError(body.get("note") or body.get("error") or "get_available_models failed")
    result = body.get("result") or {}
    available_models = result.get("available_models") if isinstance(result, dict) else None
    default_models = available_models.get("default_models") if isinstance(available_models, dict) else None
    embedding_model = default_models.get("embedding") if isinstance(default_models, dict) else None
    if not isinstance(embedding_model, str) or len(embedding_model.strip()) == 0:
        raise RuntimeError("No default embedding model configured in get_available_models.")
    return embedding_model.strip()


def embedding_preflight(
    *,
    api_base_url: str,
    api_key: str,
    timeout_s: int,
    model_override: Optional[str],
) -> Dict[str, Any]:
    model = model_override.strip() if isinstance(model_override, str) and len(model_override.strip()) > 0 else None
    if model is None:
        model = resolve_default_embedding_model(
            api_base_url=api_base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    resp = requests.post(
        f"{api_base_url.rstrip('/')}/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": ["querylake backfill preflight"]},
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"/v1/embeddings returned HTTP {resp.status_code}: {resp.text[:500]}")

    body = resp.json()
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"/v1/embeddings returned no vectors: {body}")
    first = data[0] if isinstance(data[0], dict) else {}
    vector = first.get("embedding") if isinstance(first, dict) else None
    if not isinstance(vector, list) or len(vector) == 0:
        raise RuntimeError(f"/v1/embeddings returned malformed vector payload: {body}")
    return {"ok": True, "model": model, "dimensions": len(vector)}


def embed_batch(
    *,
    api_base_url: str,
    api_key: str,
    model: str,
    texts: Sequence[str],
    timeout_s: int,
) -> List[List[float]]:
    resp = requests.post(
        f"{api_base_url.rstrip('/')}/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": list(texts)},
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"/v1/embeddings returned HTTP {resp.status_code}: {resp.text[:500]}")

    body = resp.json()
    data = body.get("data")
    if not isinstance(data, list):
        raise RuntimeError(f"Malformed embeddings response: {body}")
    vectors: List[List[float]] = []
    for row in data:
        if not isinstance(row, dict) or not isinstance(row.get("embedding"), list):
            raise RuntimeError(f"Malformed embeddings row: {row}")
        vectors.append(row["embedding"])
    if len(vectors) != len(texts):
        raise RuntimeError(f"Embedding cardinality mismatch: requested={len(texts)} got={len(vectors)}")
    return vectors


def resolve_collection_ids(args: argparse.Namespace) -> List[str]:
    ids: List[str] = []
    if args.collection_ids:
        ids.extend(parse_csv(args.collection_ids))
    if args.account_config:
        payload = json.loads(Path(args.account_config).read_text(encoding="utf-8"))
        collections = payload.get("collections")
        if isinstance(collections, dict):
            ids.extend([str(v) for v in collections.values() if isinstance(v, str) and len(v) > 0])
    unique_ids: List[str] = []
    seen = set()
    for value in ids:
        if value not in seen:
            unique_ids.append(value)
            seen.add(value)
    return unique_ids


def pending_counts(database: Session, collection_ids: Sequence[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for cid in collection_ids:
        total = int(
            database.execute(
                text("SELECT COUNT(*) FROM documentchunk WHERE collection_id = :cid"),
                {"cid": cid},
            ).first()[0]
        )
        pending = int(
            database.execute(
                text("SELECT COUNT(*) FROM documentchunk WHERE collection_id = :cid AND embedding IS NULL"),
                {"cid": cid},
            ).first()[0]
        )
        out[cid] = {"chunks_total": total, "chunks_pending_embedding": pending}
    return out


def write_manifest(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_pending_embedding_index(engine) -> None:
    sql = text(
        """
        CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_documentchunk_collection_pending_embedding
        ON documentchunk (collection_id, id)
        WHERE embedding IS NULL
        """
    )
    with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
        conn.execute(sql)


def ensure_pipeline_queue_schema(database: Session) -> None:
    def _ddl(sql: str) -> None:
        try:
            database.execute(text(sql))
        except Exception as exc:
            message = str(exc)
            if "pg_class_relname_nsp_index" in message or "already exists" in message:
                database.rollback()
                return
            raise

    _ddl(
        f"""
        CREATE TABLE IF NOT EXISTS {PIPELINE_TABLE} (
            id TEXT PRIMARY KEY,
            pipeline_key TEXT NOT NULL,
            stage TEXT NOT NULL,
            entity_table TEXT NOT NULL,
            entity_id TEXT NOT NULL,
            collection_id TEXT NULL,
            priority INTEGER NOT NULL DEFAULT 100,
            status TEXT NOT NULL DEFAULT 'pending',
            attempts INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT NULL,
            lease_expires_at DOUBLE PRECISION NULL,
            available_at DOUBLE PRECISION NOT NULL,
            payload JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            last_error TEXT NULL,
            created_at DOUBLE PRECISION NOT NULL,
            updated_at DOUBLE PRECISION NOT NULL
        )
        """
    )
    _ddl(
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS uq_pipeline_work_item_scope_entity
        ON {PIPELINE_TABLE} (pipeline_key, stage, entity_table, entity_id)
        """
    )
    _ddl(
        f"""
        CREATE INDEX IF NOT EXISTS ix_pipeline_work_item_claim
        ON {PIPELINE_TABLE} (pipeline_key, stage, status, available_at, priority, created_at)
        """
    )
    _ddl(
        f"""
        CREATE INDEX IF NOT EXISTS ix_pipeline_work_item_claim_collection
        ON {PIPELINE_TABLE} (pipeline_key, stage, status, collection_id, available_at, priority, created_at)
        """
    )
    _ddl(
        f"""
        CREATE INDEX IF NOT EXISTS ix_pipeline_work_item_lease
        ON {PIPELINE_TABLE} (pipeline_key, stage, lease_expires_at)
        WHERE status = 'leased'
        """
    )
    database.commit()


def queue_reap_expired_leases(database: Session, *, pipeline_key: str, stage: str, now_ts: float) -> int:
    rowcount = (
        database.execute(
            text(
                f"""
                UPDATE {PIPELINE_TABLE}
                SET status = 'pending',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    available_at = :now_ts,
                    updated_at = :now_ts
                WHERE pipeline_key = :pipeline_key
                  AND stage = :stage
                  AND status = 'leased'
                  AND lease_expires_at IS NOT NULL
                  AND lease_expires_at < :now_ts
                """
            ),
            {"pipeline_key": pipeline_key, "stage": stage, "now_ts": now_ts},
        ).rowcount
        or 0
    )
    if rowcount > 0:
        database.commit()
    return int(rowcount)


def queue_seed_missing_work(
    database: Session,
    *,
    collection_ids: Sequence[str],
    pipeline_key: str,
    stage: str,
    max_rows: int,
) -> int:
    now_ts = time.time()
    limit_sql = ""
    params: Dict[str, Any] = {
        "pipeline_key": pipeline_key,
        "stage": stage,
        "collection_ids": list(collection_ids),
        "now_ts": now_ts,
    }
    if max_rows > 0:
        limit_sql = "LIMIT :max_rows"
        params["max_rows"] = int(max_rows)

    result = database.execute(
        text(
            f"""
            INSERT INTO {PIPELINE_TABLE} (
                id, pipeline_key, stage, entity_table, entity_id, collection_id,
                priority, status, attempts, available_at, payload, created_at, updated_at
            )
            SELECT
                md5(random()::text || clock_timestamp()::text || dc.id)::text AS id,
                :pipeline_key,
                :stage,
                'documentchunk',
                dc.id,
                dc.collection_id,
                100,
                'pending',
                0,
                :now_ts,
                '{{}}'::jsonb,
                :now_ts,
                :now_ts
            FROM documentchunk dc
            WHERE dc.collection_id = ANY(:collection_ids)
              AND dc.embedding IS NULL
            ORDER BY dc.id
            {limit_sql}
            ON CONFLICT (pipeline_key, stage, entity_table, entity_id) DO NOTHING
            RETURNING id
            """
        ),
        params,
    )
    inserted_rows = result.fetchall()
    inserted = len(inserted_rows)
    if inserted > 0:
        database.commit()
    return inserted


def queue_claim_batch(
    database: Session,
    *,
    pipeline_key: str,
    stage: str,
    collection_ids: Sequence[str],
    worker_id: str,
    lease_seconds: int,
    batch_size: int,
) -> Tuple[List[DocumentChunk], List[str], int]:
    now_ts = time.time()
    _ = queue_reap_expired_leases(database, pipeline_key=pipeline_key, stage=stage, now_ts=now_ts)
    lease_until = now_ts + max(5, int(lease_seconds))

    claimed_raw = database.execute(
        text(
            f"""
            WITH claimable AS (
                SELECT q.id
                FROM {PIPELINE_TABLE} q
                JOIN documentchunk dc
                  ON dc.id = q.entity_id
                 AND dc.embedding IS NULL
                WHERE q.pipeline_key = :pipeline_key
                  AND q.stage = :stage
                  AND q.status = 'pending'
                  AND q.available_at <= :now_ts
                  AND q.entity_table = 'documentchunk'
                  AND q.collection_id = ANY(:collection_ids)
                ORDER BY q.priority ASC, q.created_at ASC
                LIMIT :batch_size
                FOR UPDATE SKIP LOCKED
            )
            UPDATE {PIPELINE_TABLE} q
            SET status = 'leased',
                lease_owner = :worker_id,
                lease_expires_at = :lease_until,
                attempts = q.attempts + 1,
                updated_at = :now_ts
            FROM claimable c
            WHERE q.id = c.id
            RETURNING q.id, q.entity_id
            """
        ),
        {
            "pipeline_key": pipeline_key,
            "stage": stage,
            "worker_id": worker_id,
            "lease_until": lease_until,
            "now_ts": now_ts,
            "batch_size": int(batch_size),
            "collection_ids": list(collection_ids),
        },
    ).fetchall()
    database.commit()

    claimed_count = len(claimed_raw)
    if claimed_count == 0:
        return [], [], 0

    queue_ids: List[str] = [str(row[0]) for row in claimed_raw]
    entity_ids: List[str] = [str(row[1]) for row in claimed_raw]

    chunk_rows = database.exec(
        select(DocumentChunk)
        .where(DocumentChunk.id.in_(entity_ids))
        .where(DocumentChunk.embedding.is_(None))
    ).all()
    chunk_rows.sort(key=lambda row: entity_ids.index(row.id))
    found_ids = {str(row.id) for row in chunk_rows}
    active_queue_ids = [qid for qid, eid in zip(queue_ids, entity_ids) if eid in found_ids]

    return chunk_rows, active_queue_ids, claimed_count


def queue_cleanup_stale_pending(
    database: Session,
    *,
    pipeline_key: str,
    stage: str,
    collection_ids: Sequence[str],
    cleanup_limit: int,
) -> int:
    if cleanup_limit <= 0:
        return 0
    now_ts = time.time()
    result = database.execute(
        text(
            f"""
            WITH stale AS (
                SELECT q.id
                FROM {PIPELINE_TABLE} q
                WHERE q.pipeline_key = :pipeline_key
                  AND q.stage = :stage
                  AND q.status = 'pending'
                  AND q.entity_table = 'documentchunk'
                  AND q.collection_id = ANY(:collection_ids)
                  AND NOT EXISTS (
                      SELECT 1
                      FROM documentchunk dc
                      WHERE dc.id = q.entity_id
                        AND dc.embedding IS NULL
                  )
                ORDER BY q.created_at ASC
                LIMIT :cleanup_limit
                FOR UPDATE SKIP LOCKED
            )
            UPDATE {PIPELINE_TABLE} q
            SET status = 'done',
                lease_owner = NULL,
                lease_expires_at = NULL,
                updated_at = :now_ts
            FROM stale s
            WHERE q.id = s.id
            RETURNING q.id
            """
        ),
        {
            "pipeline_key": pipeline_key,
            "stage": stage,
            "collection_ids": list(collection_ids),
            "cleanup_limit": int(cleanup_limit),
            "now_ts": now_ts,
        },
    )
    done = len(result.fetchall())
    if done > 0:
        database.commit()
    return done


def queue_mark_done(database: Session, *, queue_ids: Sequence[str]) -> int:
    if len(queue_ids) == 0:
        return 0
    rowcount = (
        database.execute(
            text(
                f"""
                UPDATE {PIPELINE_TABLE}
                SET status = 'done',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    updated_at = :now_ts
                WHERE id = ANY(:queue_ids)
                """
            ),
            {"queue_ids": list(queue_ids), "now_ts": time.time()},
        ).rowcount
        or 0
    )
    database.commit()
    return int(rowcount)


def queue_mark_retry(
    database: Session,
    *,
    queue_ids: Sequence[str],
    retry_seconds: int,
    error_message: str,
) -> int:
    if len(queue_ids) == 0:
        return 0
    now_ts = time.time()
    rowcount = (
        database.execute(
            text(
                f"""
                UPDATE {PIPELINE_TABLE}
                SET status = 'pending',
                    lease_owner = NULL,
                    lease_expires_at = NULL,
                    available_at = :available_at,
                    updated_at = :now_ts,
                    last_error = :error_message
                WHERE id = ANY(:queue_ids)
                """
            ),
            {
                "queue_ids": list(queue_ids),
                "available_at": now_ts + max(1, int(retry_seconds)),
                "now_ts": now_ts,
                "error_message": error_message[:2000],
            },
        ).rowcount
        or 0
    )
    database.commit()
    return int(rowcount)


def claim_scan_batch(
    database: Session,
    *,
    collection_ids: Sequence[str],
    batch_size: int,
) -> List[DocumentChunk]:
    return (
        database.exec(
            select(DocumentChunk)
            .where(DocumentChunk.collection_id.in_(collection_ids))
            .where(DocumentChunk.embedding.is_(None))
            .order_by(DocumentChunk.id)
            .limit(int(batch_size))
        ).all()
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Backfill embeddings for existing documentchunk rows with NULL embedding values, "
            "then promote documents to finished_processing=4 when fully embedded."
        )
    )
    parser.add_argument("--api-base-url", type=str, default="http://localhost:8000")
    parser.add_argument("--api-key", type=str, default=os.getenv("QUERYLAKE_API_KEY", "").strip())
    parser.add_argument("--embedding-model", type=str, default="")
    parser.add_argument("--db-url", type=str, default=DEFAULT_DB_URL)
    parser.add_argument("--collection-ids", type=str, default="")
    parser.add_argument("--account-config", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--http-timeout-s", type=int, default=180)
    parser.add_argument("--max-batches", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--max-chunks", type=int, default=0, help="0 means no cap.")
    parser.add_argument("--strict-preflight", action="store_true", default=True)
    parser.add_argument("--no-strict-preflight", dest="strict_preflight", action="store_false")

    parser.add_argument(
        "--claim-mode",
        choices=["queue", "scan"],
        default="queue",
        help="queue uses SKIP LOCKED leased claims; scan uses legacy sparse scan fallback.",
    )
    parser.add_argument("--pipeline-key", type=str, default=PIPELINE_KEY_DEFAULT)
    parser.add_argument("--pipeline-stage", type=str, default=PIPELINE_STAGE_DEFAULT)
    parser.add_argument(
        "--queue-worker-id",
        type=str,
        default=f"{socket.gethostname()}:{os.getpid()}",
    )
    parser.add_argument("--queue-lease-seconds", type=int, default=300)
    parser.add_argument("--queue-retry-seconds", type=int, default=30)
    parser.add_argument("--queue-seed-limit", type=int, default=250000, help="0 means seed all pending rows.")
    parser.add_argument(
        "--queue-stale-cleanup-limit",
        type=int,
        default=0,
        help="Per-batch stale pending->done cleanup cap. 0 disables stale cleanup loop.",
    )
    parser.add_argument("--ensure-pending-index", action="store_true", default=True)
    parser.add_argument("--no-ensure-pending-index", dest="ensure_pending_index", action="store_false")

    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE1_EMBEDDING_BACKFILL_MANIFEST.json"),
    )
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    if len(args.api_key) == 0:
        raise SystemExit("Missing --api-key (or QUERYLAKE_API_KEY).")

    collection_ids = resolve_collection_ids(args)
    if len(collection_ids) == 0:
        raise SystemExit("No collection ids resolved. Provide --collection-ids and/or --account-config.")

    preflight: Dict[str, Any] = {"enabled": bool(args.strict_preflight)}
    model: Optional[str] = args.embedding_model.strip() or None
    if args.strict_preflight:
        try:
            preflight = {
                "enabled": True,
                **embedding_preflight(
                    api_base_url=args.api_base_url,
                    api_key=args.api_key,
                    timeout_s=max(10, int(args.http_timeout_s)),
                    model_override=model,
                ),
            }
        except Exception as exc:
            raise SystemExit(
                "Embedding preflight failed. Refusing to run backfill.\n"
                f"Reason: {exc}\n"
                "Bring embedding service online, or bypass with --no-strict-preflight."
            )
        model = preflight["model"]
        print(f"[preflight] embedding endpoint OK model={model} dims={preflight['dimensions']}")
    elif model is None:
        model = resolve_default_embedding_model(
            api_base_url=args.api_base_url,
            api_key=args.api_key,
            timeout_s=max(10, int(args.http_timeout_s)),
        )

    stats = BackfillStats(started_at_unix=time.time())
    engine = create_engine(
        args.db_url,
        pool_pre_ping=True,
        connect_args={"connect_timeout": 5},
    )

    if args.ensure_pending_index:
        try:
            ensure_pending_embedding_index(engine)
            print("[db] ensured ix_documentchunk_collection_pending_embedding")
        except Exception as exc:
            print(f"[warn] unable to ensure pending-embedding partial index: {exc}")

    with Session(engine) as database:
        if args.claim_mode == "queue":
            ensure_pipeline_queue_schema(database)
            seeded = queue_seed_missing_work(
                database,
                collection_ids=collection_ids,
                pipeline_key=args.pipeline_key,
                stage=args.pipeline_stage,
                max_rows=int(args.queue_seed_limit),
            )
            stats.queue_items_seeded += int(seeded)
            print(f"[queue] seeded pending work items={seeded}")

        initial_pending = pending_counts(database, collection_ids)
        print("[start] pending chunk counts:", json.dumps(initial_pending))

        while True:
            if args.max_batches and stats.batches_ok >= int(args.max_batches):
                break
            if args.max_chunks and stats.chunks_embedded >= int(args.max_chunks):
                break

            queue_ids: List[str] = []
            claimed_count = 0
            stale_count = 0
            if args.claim_mode == "queue":
                if int(args.queue_stale_cleanup_limit) > 0:
                    stale_count = queue_cleanup_stale_pending(
                        database,
                        pipeline_key=args.pipeline_key,
                        stage=args.pipeline_stage,
                        collection_ids=collection_ids,
                        cleanup_limit=int(args.queue_stale_cleanup_limit),
                    )
                rows, queue_ids, claimed_count = queue_claim_batch(
                    database,
                    pipeline_key=args.pipeline_key,
                    stage=args.pipeline_stage,
                    collection_ids=collection_ids,
                    worker_id=args.queue_worker_id,
                    lease_seconds=int(args.queue_lease_seconds),
                    batch_size=int(args.batch_size),
                )
                stats.queue_items_claimed += int(claimed_count)
                stats.queue_items_stale_done += int(stale_count)

                if len(rows) == 0:
                    if claimed_count > 0:
                        continue
                    seeded = queue_seed_missing_work(
                        database,
                        collection_ids=collection_ids,
                        pipeline_key=args.pipeline_key,
                        stage=args.pipeline_stage,
                        max_rows=int(args.queue_seed_limit),
                    )
                    if seeded > 0:
                        stats.queue_items_seeded += int(seeded)
                        print(f"[queue] reseeded pending work items={seeded}")
                        continue
                    break
            else:
                rows = claim_scan_batch(
                    database,
                    collection_ids=collection_ids,
                    batch_size=int(args.batch_size),
                )
                if len(rows) == 0:
                    break

            texts = [row.text for row in rows]
            touched_doc_ids = sorted({row.document_id for row in rows if isinstance(row.document_id, str)})
            try:
                vectors = embed_batch(
                    api_base_url=args.api_base_url,
                    api_key=args.api_key,
                    model=model,
                    texts=texts,
                    timeout_s=int(args.http_timeout_s),
                )
                for row, vector in zip(rows, vectors):
                    row.embedding = vector
                database.add_all(rows)
                database.commit()

                if len(touched_doc_ids) > 0:
                    promote_stmt = text(
                        """
                        UPDATE document_raw dr
                        SET finished_processing = 4
                        WHERE dr.id = ANY(:doc_ids)
                          AND dr.finished_processing <> 4
                          AND EXISTS (
                              SELECT 1 FROM documentchunk dc2
                              WHERE dc2.document_id = dr.id
                          )
                          AND NOT EXISTS (
                              SELECT 1 FROM documentchunk dc
                              WHERE dc.document_id = dr.id
                                AND dc.embedding IS NULL
                          )
                        """
                    )
                    promoted = database.execute(promote_stmt, {"doc_ids": touched_doc_ids}).rowcount or 0
                    stats.docs_promoted_to_finished_4 += int(promoted)
                    database.commit()

                if args.claim_mode == "queue":
                    done_count = queue_mark_done(database, queue_ids=queue_ids)
                    stats.queue_items_done += int(done_count)

                stats.batches_ok += 1
                stats.chunks_embedded += len(rows)
                if stats.batches_ok % 20 == 0:
                    print(
                        f"[progress] batches={stats.batches_ok} chunks_embedded={stats.chunks_embedded} "
                        f"docs_promoted={stats.docs_promoted_to_finished_4}"
                    )
            except Exception as exc:
                database.rollback()
                stats.batches_failed += 1
                stats.last_error = str(exc)
                if args.claim_mode == "queue" and len(queue_ids) > 0:
                    retried = queue_mark_retry(
                        database,
                        queue_ids=queue_ids,
                        retry_seconds=int(args.queue_retry_seconds),
                        error_message=str(exc),
                    )
                    stats.queue_items_retried += int(retried)
                raise

        final_pending = pending_counts(database, collection_ids)

    stats.ended_at_unix = time.time()
    stats.runtime_seconds = round(stats.ended_at_unix - stats.started_at_unix, 3)

    manifest = {
        "generated_at_unix": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "args": {
            "api_base_url": args.api_base_url,
            "batch_size": int(args.batch_size),
            "max_batches": int(args.max_batches),
            "max_chunks": int(args.max_chunks),
            "strict_preflight": bool(args.strict_preflight),
            "db_url": args.db_url,
            "collection_ids": collection_ids,
            "model": model,
            "claim_mode": args.claim_mode,
            "pipeline_key": args.pipeline_key,
            "pipeline_stage": args.pipeline_stage,
            "queue_worker_id": args.queue_worker_id,
            "queue_lease_seconds": int(args.queue_lease_seconds),
            "queue_retry_seconds": int(args.queue_retry_seconds),
            "queue_seed_limit": int(args.queue_seed_limit),
            "queue_stale_cleanup_limit": int(args.queue_stale_cleanup_limit),
        },
        "preflight": preflight,
        "stats": asdict(stats),
        "initial_pending": initial_pending,
        "final_pending": final_pending,
    }
    write_manifest(args.manifest_out, manifest)
    print(f"[done] manifest -> {args.manifest_out}")
    print(
        "[done] "
        f"batches_ok={stats.batches_ok} chunks_embedded={stats.chunks_embedded} "
        f"docs_promoted={stats.docs_promoted_to_finished_4} failed={stats.batches_failed}"
    )
    if args.claim_mode == "queue":
        print(
            "[done:queue] "
            f"seeded={stats.queue_items_seeded} claimed={stats.queue_items_claimed} "
            f"done={stats.queue_items_done} stale_done={stats.queue_items_stale_done} retried={stats.queue_items_retried}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
