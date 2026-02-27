from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from QueryLake.database.sql_db_tables import retrieval_pipeline_config as RetrievalPipelineConfig


def _canonical_spec(spec_json: Dict[str, Any]) -> str:
    return json.dumps(spec_json, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def pipeline_spec_hash(spec_json: Dict[str, Any]) -> str:
    canonical = _canonical_spec(spec_json)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def register_pipeline_spec(
    database: Session,
    *,
    pipeline_id: str,
    version: str,
    spec_json: Dict[str, Any],
    created_by: Optional[str] = None,
    status: str = "active",
    md: Optional[Dict[str, Any]] = None,
) -> RetrievalPipelineConfig:
    assert isinstance(pipeline_id, str) and len(pipeline_id) > 0, "pipeline_id must be non-empty"
    assert isinstance(version, str) and len(version) > 0, "version must be non-empty"
    assert isinstance(spec_json, dict), "spec_json must be a dict"

    immutable_hash = pipeline_spec_hash(spec_json)
    existing = database.exec(
        select(RetrievalPipelineConfig).where(
            RetrievalPipelineConfig.pipeline_id == pipeline_id,
            RetrievalPipelineConfig.version == version,
        )
    ).first()
    if existing is not None:
        if existing.immutable_hash != immutable_hash:
            raise ValueError(f"Pipeline spec already exists for {pipeline_id}:{version} with different immutable hash")
        return existing

    row = RetrievalPipelineConfig(
        pipeline_id=pipeline_id,
        version=version,
        immutable_hash=immutable_hash,
        spec_json=spec_json,
        created_by=created_by,
        status=status,
        md=md or {},
    )
    database.add(row)
    database.commit()
    database.refresh(row)
    return row


def fetch_pipeline_spec(
    database: Session,
    *,
    pipeline_id: str,
    version: Optional[str] = None,
) -> Optional[RetrievalPipelineConfig]:
    stmt = select(RetrievalPipelineConfig).where(RetrievalPipelineConfig.pipeline_id == pipeline_id)
    if version is not None:
        stmt = stmt.where(RetrievalPipelineConfig.version == version)
    stmt = stmt.order_by(RetrievalPipelineConfig.created_at.desc())
    return database.exec(stmt).first()


def list_pipeline_specs(
    database: Session,
    *,
    pipeline_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 100,
) -> List[RetrievalPipelineConfig]:
    assert isinstance(limit, int) and 1 <= limit <= 1000, "limit must be in [1, 1000]"
    stmt = select(RetrievalPipelineConfig)
    if pipeline_id is not None:
        stmt = stmt.where(RetrievalPipelineConfig.pipeline_id == pipeline_id)
    if status is not None:
        stmt = stmt.where(RetrievalPipelineConfig.status == status)
    stmt = stmt.order_by(RetrievalPipelineConfig.created_at.desc()).limit(limit)
    return list(database.exec(stmt).all())

