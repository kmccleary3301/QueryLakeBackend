from __future__ import annotations

import os
from typing import Any, Awaitable, Callable, Dict, List, Optional

from sqlalchemy.exc import IntegrityError
from sqlmodel import Session, select

from QueryLake.database.sql_db_tables import embedding_record as EmbeddingRecord
from QueryLake.runtime.content_fingerprint import content_fingerprint


def embedding_input_hash(*, text: str, model_id: str) -> str:
    return content_fingerprint(text=text, md={"model_id": model_id})


def embedding_record_enabled() -> bool:
    raw = (os.getenv("QUERYLAKE_EMBEDDING_RECORD_ENABLED", "0") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def embedding_record_model_id() -> str:
    raw = (os.getenv("QUERYLAKE_EMBEDDING_RECORD_MODEL_ID", "embedding.default") or "").strip()
    return raw if len(raw) > 0 else "embedding.default"


def fetch_embedding_record(
    database: Session,
    *,
    segment_id: str,
    model_id: str,
    input_hash: str,
) -> Optional[EmbeddingRecord]:
    return database.exec(
        select(EmbeddingRecord).where(
            EmbeddingRecord.segment_id == segment_id,
            EmbeddingRecord.model_id == model_id,
            EmbeddingRecord.input_hash == input_hash,
        )
    ).first()


def get_or_create_embedding(
    database: Session,
    *,
    segment_id: str,
    text: str,
    model_id: str,
    embedding_fn: Callable[[str], List[float]],
    md: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_hash = embedding_input_hash(text=text, model_id=model_id)
    existing = fetch_embedding_record(
        database,
        segment_id=segment_id,
        model_id=model_id,
        input_hash=input_hash,
    )
    if existing is not None and existing.embedding is not None:
        return {
            "embedding": list(existing.embedding),
            "cache_hit": True,
            "record_id": existing.id,
            "input_hash": input_hash,
        }

    embedding = embedding_fn(text)
    row = EmbeddingRecord(
        segment_id=segment_id,
        model_id=model_id,
        input_hash=input_hash,
        embedding=embedding,
        md=md or {},
    )
    database.add(row)
    try:
        database.commit()
        database.refresh(row)
    except IntegrityError:
        # Concurrent writer inserted the same key first; reuse existing row.
        database.rollback()
        existing = fetch_embedding_record(
            database,
            segment_id=segment_id,
            model_id=model_id,
            input_hash=input_hash,
        )
        if existing is None or existing.embedding is None:
            raise
        row = existing
    return {
        "embedding": list(row.embedding) if row.embedding is not None else embedding,
        "cache_hit": False,
        "record_id": row.id,
        "input_hash": input_hash,
    }


async def get_or_create_embedding_async(
    database: Session,
    *,
    segment_id: str,
    text: str,
    model_id: str,
    embedding_fn: Callable[[str], Awaitable[List[float]]],
    md: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    input_hash = embedding_input_hash(text=text, model_id=model_id)
    existing = fetch_embedding_record(
        database,
        segment_id=segment_id,
        model_id=model_id,
        input_hash=input_hash,
    )
    if existing is not None and existing.embedding is not None:
        return {
            "embedding": list(existing.embedding),
            "cache_hit": True,
            "record_id": existing.id,
            "input_hash": input_hash,
        }

    embedding = await embedding_fn(text)
    row = EmbeddingRecord(
        segment_id=segment_id,
        model_id=model_id,
        input_hash=input_hash,
        embedding=embedding,
        md=md or {},
    )
    database.add(row)
    try:
        database.commit()
        database.refresh(row)
    except IntegrityError:
        database.rollback()
        existing = fetch_embedding_record(
            database,
            segment_id=segment_id,
            model_id=model_id,
            input_hash=input_hash,
        )
        if existing is None or existing.embedding is None:
            raise
        row = existing

    return {
        "embedding": list(row.embedding) if row.embedding is not None else embedding,
        "cache_hit": False,
        "record_id": row.id,
        "input_hash": input_hash,
    }
