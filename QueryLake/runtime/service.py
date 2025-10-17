from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from fastapi import HTTPException, status

from QueryLake.api import toolchains as toolchains_api
from QueryLake.database import sql_db_tables
from QueryLake.runtime.events import EventEnvelope, EventStore
from QueryLake.runtime.jobs import JobRegistry, JobStatus
from QueryLake.runtime.session import ToolchainSessionV2
from QueryLake.runtime.sse import SessionStreamHub
from QueryLake.runtime.signals import JobSignalBus
from QueryLake.typing.config import AuthType
from QueryLake.typing.toolchains import ToolChain, ToolChainV2
from QueryLake.observability import metrics


@dataclass
class SessionEntry:
    runtime: ToolchainSessionV2
    lock: asyncio.Lock
    last_rev: int
    author: str
    toolchain_id: str


class ToolchainRuntimeService:
    def __init__(
        self,
        umbrella,
        toolchains_v1: Dict[str, ToolChain],
        toolchains_v2: Dict[str, ToolChainV2],
    ) -> None:
        self.umbrella = umbrella
        self.toolchains_v1 = toolchains_v1
        self.toolchains_v2 = toolchains_v2
        self.database = umbrella.database
        self.event_store = EventStore(self.database)
        self.stream_hub = SessionStreamHub()
        self.job_registry = JobRegistry(self.event_store)
        self.signal_bus = JobSignalBus()
        self._sessions: Dict[str, SessionEntry] = {}
        self._global_lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def create_session(
        self,
        auth: AuthType,
        toolchain_id: str,
        title: Optional[str] = None,
        initial_inputs: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        toolchain_v1 = self.toolchains_v1.get(toolchain_id)
        toolchain_v2 = self.toolchains_v2.get(toolchain_id)
        if toolchain_v1 is None or toolchain_v2 is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Toolchain not found")

        _, auth_info = toolchains_api.get_user(self.database, auth)
        toolchain_function_caller = self.umbrella.api_function_getter
        legacy_session = toolchains_api.create_toolchain_session(
            self.database,
            toolchain_function_caller,
            auth,
            toolchain_id,
            ws=None,
        )
        session_id = legacy_session.session_hash
        author = auth_info.username

        entry = self._build_session_entry(session_id, toolchain_v2, author)
        entry.runtime.state = legacy_session.state
        entry.runtime.files = legacy_session.toolchain_session_files
        entry.runtime.local_cache = legacy_session.local_cache
        if title:
            entry.runtime.state["title"] = title

        payload = {
            "session_id": session_id,
            "toolchain_id": toolchain_id,
            "title": title or entry.runtime.state.get("title", toolchain_v2.name),
            "actor": author,
        }
        envelope = self._emit_event(entry, "SESSION_CREATED", payload, snapshot=True)
        metrics.session_created()
        self._persist_session_state(entry)
        # ensure record stored
        self._sessions[session_id] = entry
        self._logger.info(
            "session.created",
            extra={"session_id": session_id, "toolchain_id": toolchain_id, "actor": author},
        )

        return {
            "session_id": session_id,
            "rev": envelope.rev,
            "state": entry.runtime.state,
            "files": entry.runtime.files,
        }

    async def delete_session(self, session_id: str, auth: AuthType) -> None:
        entry = await self._ensure_session(session_id)
        _, auth_info = toolchains_api.get_user(self.database, auth)
        if auth_info.username != entry.author:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for session")
        payload = {"session_id": session_id}
        self._emit_event(entry, "SESSION_DELETED", payload, snapshot=True)
        metrics.session_deleted()
        async with self._global_lock:
            self._sessions.pop(session_id, None)
        db_session = self.database.get(sql_db_tables.toolchain_session, session_id)
        if db_session:
            self.database.delete(db_session)
            self.database.commit()
        self._logger.info("session.deleted", extra={"session_id": session_id})

    async def get_session_state(self, session_id: str, auth: AuthType) -> Dict[str, Any]:
        entry = await self._ensure_session(session_id)
        _, auth_info = toolchains_api.get_user(self.database, auth)
        if auth_info.username != entry.author:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for session")
        return {
            "session_id": session_id,
            "rev": entry.last_rev,
            "state": entry.runtime.state,
            "files": entry.runtime.files,
            "toolchain_id": entry.toolchain_id,
        }

    async def list_events(self, session_id: str, since_rev: Optional[int], auth: AuthType) -> List[EventEnvelope]:
        entry = await self._ensure_session(session_id)
        _, auth_info = toolchains_api.get_user(self.database, auth)
        if auth_info.username != entry.author:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for session")
        return self.event_store.list_events(session_id, since_rev)

    async def post_event(
        self,
        session_id: str,
        node_id: str,
        inputs: Dict[str, Any],
        expected_rev: Optional[int],
        *,
        auth: AuthType,
        correlation_id: Optional[str],
    ) -> Dict[str, Any]:
        entry = await self._ensure_session(session_id)
        _, auth_info = toolchains_api.get_user(self.database, auth)
        actor = auth_info.username
        if actor != entry.author:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for session")
        async with entry.lock:
            if expected_rev is not None and expected_rev != entry.last_rev:
                raise HTTPException(status_code=409, detail="Revision conflict")
            # Make the current auth available to the runtime for API calls that require it
            # (e.g., llm, embeddings, reranker). This is read by ToolchainSessionV2 when
            # injecting missing parameters based on call signatures.
            entry.runtime.server_context["auth"] = auth
            payload = {
                "session_id": session_id,
                "node_id": node_id,
                "inputs": inputs,
                "actor": actor,
                "correlation_id": correlation_id,
            }
            self._emit_event(entry, "EVENT_RECEIVED", payload, snapshot=False)
            try:
                result = await entry.runtime.process_event(
                    node_id,
                    inputs,
                    actor=actor,
                    correlation_id=correlation_id,
                )
            except asyncio.CancelledError:
                self._emit_event(
                    entry,
                    "EVENT_CANCELLED",
                    {
                        "session_id": session_id,
                        "node_id": node_id,
                        "actor": actor,
                        "correlation_id": correlation_id,
                    },
                    snapshot=True,
                )
                self._persist_session_state(entry)
                raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Event cancelled")

            self._emit_event(
                entry,
                "EVENT_COMPLETED",
                {
                    "session_id": session_id,
                    "node_id": node_id,
                    "actor": actor,
                    "correlation_id": correlation_id,
                },
                snapshot=True,
            )
            self._persist_session_state(entry)
            return {"rev": entry.last_rev, "result": result}

    async def subscribe(self, session_id: str, auth: AuthType):
        entry = await self._ensure_session(session_id)
        _, auth_info = toolchains_api.get_user(self.database, auth)
        if auth_info.username != entry.author:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for session")
        return await self.stream_hub.subscribe(session_id)

    async def unsubscribe(self, session_id: str, subscriber) -> None:
        await self.stream_hub.unsubscribe(session_id, subscriber)

    async def list_jobs(self, session_id: str, auth: AuthType):
        entry = await self._ensure_session(session_id)
        _, auth_info = toolchains_api.get_user(self.database, auth)
        if auth_info.username != entry.author:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for session")
        jobs = self.event_store.list_jobs(session_id)
        result: List[Dict[str, Any]] = []
        for job in jobs:
            if hasattr(job, "model_dump"):
                result.append(job.model_dump())
            else:
                mapping = None
                # SQLAlchemy Row
                if hasattr(job, "_mapping"):
                    try:
                        mapping = dict(job._mapping)
                    except Exception:
                        mapping = None
                if mapping is None:
                    # Fallback: attribute scrape (best-effort)
                    fields = [
                        "job_id",
                        "session_id",
                        "node_id",
                        "status",
                        "request_id",
                        "progress",
                        "result_meta",
                        "created_at",
                        "updated_at",
                    ]
                    mapping = {k: getattr(job, k) for k in fields if hasattr(job, k)}
                result.append(mapping)
        return result

    async def cancel_job(self, session_id: str, job_id: str, auth: AuthType) -> Dict[str, Any]:
        entry = await self._ensure_session(session_id)
        _, auth_info = toolchains_api.get_user(self.database, auth)
        if auth_info.username != entry.author:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized for session")
        import time
        start = time.perf_counter()
        metrics.cancel_requested()
        await self.signal_bus.trigger(job_id)
        success = await self.job_registry.cancel(job_id)
        if not success:
            raise HTTPException(status_code=404, detail="Job not cancellable")
        self.event_store.upsert_job(job_id, session_id, "", JobStatus.CANCELLED.value)
        latency = time.perf_counter() - start
        metrics.cancel_succeeded(latency)
        self._emit_event(
            entry,
            "JOB_CANCELLED",
            {
                "session_id": session_id,
                "job_id": job_id,
                "actor": auth_info.username,
            },
            snapshot=False,
        )
        self._logger.info(
            "job.cancelled",
            extra={"session_id": session_id, "job_id": job_id, "actor": auth_info.username},
        )
        return {"job_id": job_id, "status": JobStatus.CANCELLED.value}

    async def _ensure_session(self, session_id: str) -> SessionEntry:
        async with self._global_lock:
            entry = self._sessions.get(session_id)
            if entry is not None:
                return entry
            db_session = self.database.get(sql_db_tables.toolchain_session, session_id)
            if db_session is None:
                raise HTTPException(status_code=404, detail="Session not found")
            toolchain_v2 = self.toolchains_v2.get(db_session.toolchain_id)
            if toolchain_v2 is None:
                raise HTTPException(status_code=404, detail="Toolchain not found")
            entry = self._build_session_entry(session_id, toolchain_v2, db_session.author)
            if db_session.state:
                entry.runtime.state = json.loads(db_session.state)
            if db_session.file_state:
                entry.runtime.files = json.loads(db_session.file_state)
            entry.last_rev = self.event_store.latest_rev(session_id)
            self._sessions[session_id] = entry
            return entry

    def _build_session_entry(self, session_id: str, toolchain: ToolChainV2, author: str) -> SessionEntry:
        lock = asyncio.Lock()

        def emit(kind: str, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
            snapshot_state = meta.get("state")
            snapshot_files = meta.get("files")
            actor = payload.get("actor")
            correlation_id = payload.get("correlation_id")
            envelope = self.event_store.append_event(
                session_id,
                kind,
                payload,
                actor=actor,
                correlation_id=correlation_id,
                snapshot_state=snapshot_state,
                snapshot_files=snapshot_files,
            )
            entry.last_rev = envelope.rev
            message = {
                "rev": envelope.rev,
                "kind": envelope.kind,
                "payload": payload,
                "actor": envelope.actor,
                "correlation_id": envelope.correlation_id,
                "ts": envelope.ts,
            }
            loop = asyncio.get_running_loop()
            loop.create_task(self.stream_hub.publish(session_id, message))

        runtime = ToolchainSessionV2(
            session_id,
            toolchain,
            author=author,
            server_context={
                **self.umbrella.default_function_arguments,
                "umbrella": self.umbrella,
                "job_signal_bus": self.signal_bus,
            },
            emit_event=emit,
            job_registry=self.job_registry,
        )
        entry = SessionEntry(runtime=runtime, lock=lock, last_rev=0, author=author, toolchain_id=toolchain.id)
        return entry

    def _emit_event(
        self,
        entry: SessionEntry,
        kind: str,
        payload: Dict[str, Any],
        snapshot: bool,
    ) -> EventEnvelope:
        snapshot_state = entry.runtime.state if snapshot else None
        snapshot_files = entry.runtime.files if snapshot else None
        envelope = self.event_store.append_event(
            entry.runtime.session_id,
            kind,
            payload,
            actor=payload.get("actor"),
            correlation_id=payload.get("correlation_id"),
            snapshot_state=snapshot_state,
            snapshot_files=snapshot_files,
        )
        entry.last_rev = envelope.rev
        message = {
            "rev": envelope.rev,
            "kind": envelope.kind,
            "payload": payload,
            "actor": envelope.actor,
            "correlation_id": envelope.correlation_id,
            "ts": envelope.ts,
        }
        loop = asyncio.get_running_loop()
        loop.create_task(self.stream_hub.publish(entry.runtime.session_id, message))
        return envelope

    def _persist_session_state(self, entry: SessionEntry) -> None:
        db_session = self.database.get(sql_db_tables.toolchain_session, entry.runtime.session_id)
        if not db_session:
            return
        db_session.state = json.dumps(entry.runtime.state)
        db_session.file_state = json.dumps(entry.runtime.files)
        db_session.first_event_fired = True
        self.database.add(db_session)
        self.database.commit()
