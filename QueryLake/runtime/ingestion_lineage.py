from __future__ import annotations

import logging
import os
from typing import Dict, Optional

from sqlmodel import Session, select

from QueryLake.database import sql_db_tables as T

logger = logging.getLogger(__name__)


def ingestion_lineage_enabled() -> bool:
    raw = (os.getenv("QUERYLAKE_INGESTION_LINEAGE_ENABLED", "1") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def create_upload_lineage_rows(
    database: Session,
    *,
    document_id: str,
    created_by: Optional[str],
    content_hash: Optional[str],
    storage_ref: Optional[str],
    metadata: Optional[Dict] = None,
) -> Optional[Dict[str, str]]:
    if not ingestion_lineage_enabled():
        return None

    try:
        latest_version = database.exec(
            select(T.document_version)
            .where(T.document_version.document_id == document_id)
            .order_by(T.document_version.version_no.desc())
            .limit(1)
        ).first()
        next_version_no = 1 if latest_version is None else int(latest_version.version_no) + 1

        version_row = T.document_version(
            document_id=document_id,
            version_no=next_version_no,
            content_hash=content_hash,
            status="ready",
            created_by=created_by,
            md={"source": "upload", **(metadata or {})},
        )
        database.add(version_row)
        database.commit()
        database.refresh(version_row)

        artifact_row = T.document_artifact(
            document_version_id=version_row.id,
            artifact_type="source_blob",
            storage_ref=storage_ref,
            md={"source": "upload", **(metadata or {})},
        )
        database.add(artifact_row)
        database.commit()
        database.refresh(artifact_row)

        return {
            "document_version_id": version_row.id,
            "document_artifact_id": artifact_row.id,
        }
    except Exception:
        logger.exception("Failed to create ingestion lineage rows for document_id=%s", document_id)
        try:
            database.rollback()
        except Exception:
            pass
        return None
