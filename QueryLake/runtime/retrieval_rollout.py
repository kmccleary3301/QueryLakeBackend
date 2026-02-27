from __future__ import annotations

from time import time
from typing import Optional

from sqlmodel import Session, select

from QueryLake.database.sql_db_tables import retrieval_pipeline_binding as RetrievalPipelineBinding


def resolve_active_pipeline(
    database: Session,
    *,
    route: str,
    tenant_scope: Optional[str] = None,
) -> Optional[RetrievalPipelineBinding]:
    return database.exec(
        select(RetrievalPipelineBinding).where(
            RetrievalPipelineBinding.route == route,
            RetrievalPipelineBinding.tenant_scope == tenant_scope,
        )
    ).first()


def set_active_pipeline(
    database: Session,
    *,
    route: str,
    pipeline_id: str,
    pipeline_version: str,
    tenant_scope: Optional[str] = None,
    updated_by: Optional[str] = None,
    reason: str = "",
) -> RetrievalPipelineBinding:
    row = resolve_active_pipeline(database, route=route, tenant_scope=tenant_scope)
    if row is None:
        row = RetrievalPipelineBinding(
            route=route,
            tenant_scope=tenant_scope,
            active_pipeline_id=pipeline_id,
            active_pipeline_version=pipeline_version,
            previous_pipeline_id=None,
            previous_pipeline_version=None,
            updated_by=updated_by,
            updated_at=time(),
            md={"reason": reason},
        )
        database.add(row)
    else:
        row.previous_pipeline_id = row.active_pipeline_id
        row.previous_pipeline_version = row.active_pipeline_version
        row.active_pipeline_id = pipeline_id
        row.active_pipeline_version = pipeline_version
        row.updated_by = updated_by
        row.updated_at = time()
        row.md = {**(row.md or {}), "reason": reason}
        database.add(row)
    database.commit()
    database.refresh(row)
    return row


def rollback_active_pipeline(
    database: Session,
    *,
    route: str,
    tenant_scope: Optional[str] = None,
    updated_by: Optional[str] = None,
    reason: str = "rollback",
) -> RetrievalPipelineBinding:
    row = resolve_active_pipeline(database, route=route, tenant_scope=tenant_scope)
    if row is None:
        raise ValueError(f"No active pipeline binding found for route={route} tenant_scope={tenant_scope}")
    if not row.previous_pipeline_id or not row.previous_pipeline_version:
        raise ValueError(f"No previous pipeline is available for rollback on route={route} tenant_scope={tenant_scope}")

    current_id = row.active_pipeline_id
    current_version = row.active_pipeline_version
    row.active_pipeline_id = row.previous_pipeline_id
    row.active_pipeline_version = row.previous_pipeline_version
    row.previous_pipeline_id = current_id
    row.previous_pipeline_version = current_version
    row.updated_by = updated_by
    row.updated_at = time()
    row.md = {**(row.md or {}), "reason": reason}
    database.add(row)
    database.commit()
    database.refresh(row)
    return row
