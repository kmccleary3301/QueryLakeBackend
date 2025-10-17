from __future__ import annotations

import logging
from time import time
from typing import Any, Dict, Iterable, List, Optional

from pydantic import BaseModel
from sqlalchemy import func, select
from sqlmodel import Session

from QueryLake.database.sql_db_tables import (
    ToolchainDeadLetter,
    ToolchainJob,
    ToolchainSessionEvent,
    ToolchainSessionSnapshot,
)
from QueryLake.observability import metrics

logger = logging.getLogger(__name__)


class EventEnvelope(BaseModel):
    rev: int
    kind: str
    payload: Dict[str, Any]
    actor: Optional[str] = None
    correlation_id: Optional[str] = None
    ts: float


class EventStore:
    def __init__(self, db: Session, snapshot_interval: int = 50):
        self.db = db
        self.snapshot_interval = snapshot_interval

    def _next_rev(self, session_id: str) -> int:
        result = self.db.exec(
            select(func.max(ToolchainSessionEvent.rev)).where(ToolchainSessionEvent.session_id == session_id)
        ).one_or_none()
        max_rev = result[0] if result else None
        return 1 if max_rev is None else max_rev + 1

    def append_event(
        self,
        session_id: str,
        kind: str,
        payload: Dict[str, Any],
        *,
        actor: Optional[str] = None,
        correlation_id: Optional[str] = None,
        snapshot_state: Optional[Dict[str, Any]] = None,
        snapshot_files: Optional[Dict[str, Any]] = None,
    ) -> EventEnvelope:
        rev = self._next_rev(session_id)
        event = ToolchainSessionEvent(
            session_id=session_id,
            rev=rev,
            kind=kind,
            payload=payload,
            actor=actor,
            correlation_id=correlation_id,
        )
        self.db.add(event)
        if snapshot_state is not None and rev % self.snapshot_interval == 0:
            snapshot = ToolchainSessionSnapshot(
                session_id=session_id,
                rev=rev,
                state=snapshot_state,
                files=snapshot_files,
            )
            self.db.add(snapshot)
        self.db.commit()
        self.db.refresh(event)
        metrics.inc_event(kind)
        logger.debug(
            "event.append",
            extra={
                "session_id": session_id,
                "rev": rev,
                "kind": kind,
                "actor": actor,
            },
        )
        return EventEnvelope(
            rev=rev,
            kind=kind,
            payload=payload,
            actor=actor,
            correlation_id=correlation_id,
            ts=event.ts,
        )

    def latest_snapshot(self, session_id: str) -> Optional[ToolchainSessionSnapshot]:
        result = self.db.exec(
            select(ToolchainSessionSnapshot)
            .where(ToolchainSessionSnapshot.session_id == session_id)
            .order_by(ToolchainSessionSnapshot.rev.desc())
            .limit(1)
        ).first()
        return result

    def list_events(self, session_id: str, since_rev: Optional[int] = None) -> List[EventEnvelope]:
        statement = select(ToolchainSessionEvent).where(ToolchainSessionEvent.session_id == session_id)
        if since_rev is not None:
            statement = statement.where(ToolchainSessionEvent.rev > since_rev)
        statement = statement.order_by(ToolchainSessionEvent.rev.asc())
        records = self.db.exec(statement).all()
        envelopes: List[EventEnvelope] = []
        for record in records:
            if hasattr(record, "rev"):
                envelopes.append(
                    EventEnvelope(
                        rev=record.rev,
                        kind=record.kind,
                        payload=record.payload,
                        actor=record.actor,
                        correlation_id=record.correlation_id,
                        ts=record.ts,
                    )
                )
            else:
                mapping = {}
                if hasattr(record, "_mapping"):
                    try:
                        mapping = dict(record._mapping)
                    except Exception:
                        mapping = {}
                envelopes.append(
                    EventEnvelope(
                        rev=mapping.get("rev", 0),
                        kind=mapping.get("kind", ""),
                        payload=mapping.get("payload", {}),
                        actor=mapping.get("actor"),
                        correlation_id=mapping.get("correlation_id"),
                        ts=mapping.get("ts", 0.0),
                    )
                )
        return envelopes

    def add_dead_letter(
        self,
        session_id: str,
        rev: Optional[int],
        event_payload: Dict[str, Any],
        error: str,
    ) -> None:
        self.db.add(
            ToolchainDeadLetter(
                session_id=session_id,
                rev=rev,
                event=event_payload,
                error=error,
            )
        )
        self.db.commit()
        logger.warning(
            "event.dead_letter",
            extra={
                "session_id": session_id,
                "rev": rev,
                "error": error,
            },
        )

    def latest_rev(self, session_id: str) -> int:
        result = self.db.exec(
            select(func.max(ToolchainSessionEvent.rev)).where(ToolchainSessionEvent.session_id == session_id)
        ).one_or_none()
        return result[0] if result and result[0] is not None else 0

    def upsert_job(
        self,
        job_id: str,
        session_id: str,
        node_id: str,
        status: str,
        *,
        request_id: Optional[str] = None,
        progress: Optional[Dict[str, Any]] = None,
        result_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        job = self.db.get(ToolchainJob, job_id)
        if job is None:
            job = ToolchainJob(
                job_id=job_id,
                session_id=session_id,
                node_id=node_id,
                status=status,
                request_id=request_id,
                progress=progress,
                result_meta=result_meta,
            )
            self.db.add(job)
        else:
            job.status = status
            if progress is not None:
                job.progress = progress
            if result_meta is not None:
                job.result_meta = result_meta
            if request_id is not None:
                job.request_id = request_id
            job.updated_at = time()
        self.db.commit()
        logger.debug(
            "job.upsert",
            extra={
                "job_id": job_id,
                "session_id": session_id,
                "node_id": node_id,
                "status": status,
                "request_id": request_id,
            },
        )

    def list_jobs(self, session_id: str) -> List[ToolchainJob]:
        return self.db.exec(
            select(ToolchainJob).where(ToolchainJob.session_id == session_id)
        ).all()

    def get_job(self, job_id: str) -> Optional[ToolchainJob]:
        return self.db.get(ToolchainJob, job_id)
