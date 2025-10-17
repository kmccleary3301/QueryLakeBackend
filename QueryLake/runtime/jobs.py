from __future__ import annotations

import asyncio
import logging
from enum import Enum
from typing import Awaitable, Callable, Dict, Optional

from QueryLake.runtime.events import EventStore
from QueryLake.observability import metrics


class JobStatus(str, Enum):
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    CANCELLING = "CANCELLING"
    CANCELLED = "CANCELLED"


CancelCallback = Callable[[], Awaitable[None]]


class JobRegistry:
    def __init__(self, event_store: EventStore) -> None:
        self.event_store = event_store
        self._callbacks: Dict[str, CancelCallback] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def register(
        self,
        job_id: str,
        session_id: str,
        node_id: str,
        status: JobStatus,
        *,
        request_id: Optional[str] = None,
        cancel_callback: Optional[CancelCallback] = None,
    ) -> None:
        async with self._lock:
            if cancel_callback is not None:
                self._callbacks[job_id] = cancel_callback
        self.event_store.upsert_job(
            job_id,
            session_id,
            node_id,
            status.value,
            request_id=request_id,
        )
        metrics.job_transition(status.value, node_id)
        self._logger.debug(
            "job.register",
            extra={
                "job_id": job_id,
                "session_id": session_id,
                "node_id": node_id,
                "status": status.value,
                "request_id": request_id,
            },
        )

    async def update(
        self,
        job_id: str,
        session_id: str,
        node_id: str,
        status: JobStatus,
        *,
        progress: Optional[dict] = None,
        result_meta: Optional[dict] = None,
    ) -> None:
        self.event_store.upsert_job(
            job_id,
            session_id,
            node_id,
            status.value,
            progress=progress,
            result_meta=result_meta,
        )
        metrics.job_transition(status.value, node_id)
        if status in {JobStatus.COMPLETED, JobStatus.CANCELLED, JobStatus.FAILED}:
            async with self._lock:
                self._callbacks.pop(job_id, None)
        self._logger.debug(
            "job.update",
            extra={
                "job_id": job_id,
                "session_id": session_id,
                "node_id": node_id,
                "status": status.value,
            },
        )

    async def cancel(self, job_id: str) -> bool:
        sentinel = object()
        async with self._lock:
            callback = self._callbacks.get(job_id, sentinel)
        if callback is sentinel:
            return False
        if callback is not None:
            await callback()
        async with self._lock:
            self._callbacks.pop(job_id, None)
        self._logger.info("job.cancel", extra={"job_id": job_id, "result": True})
        return True

    async def attach_metadata(
        self,
        job_id: str,
        session_id: str,
        node_id: str,
        *,
        request_id: Optional[str] = None,
        progress: Optional[dict] = None,
        result_meta: Optional[dict] = None,
    ) -> None:
        job = self.event_store.get_job(job_id)
        status = job.status if job else JobStatus.PENDING.value
        self.event_store.upsert_job(
            job_id,
            session_id,
            node_id,
            status,
            request_id=request_id or (job.request_id if job else None),
            progress=progress if progress is not None else (job.progress if job else None),
            result_meta=result_meta if result_meta is not None else (job.result_meta if job else None),
        )
        self._logger.debug(
            "job.attach_metadata",
            extra={
                "job_id": job_id,
                "session_id": session_id,
                "node_id": node_id,
                "request_id": request_id,
            },
        )
