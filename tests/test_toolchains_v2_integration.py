import asyncio
import json
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from time import time
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
from httpx import ASGITransport

from QueryLake.runtime.service import ToolchainRuntimeService
from QueryLake.runtime.jobs import JobRegistry
from QueryLake.toolchains.legacy_converter import convert_toolchain_dict
from QueryLake.typing.toolchains import ToolChain
from QueryLake.runtime.events import EventEnvelope
from QueryLake.api import toolchains as toolchains_api



class DummyDB:
    def get(self, *args, **kwargs):
        return None

    def add(self, *args, **kwargs):
        return None

    def commit(self):
        return None


class DummyUmbrella:
    def __init__(self, database, toolchains_v1):
        self.database = database
        self.toolchains_v1 = toolchains_v1
        self.default_function_arguments = {
            "database": database,
            "toolchains_available": toolchains_v1,
            "public_key": None,
            "server_private_key": None,
            "toolchain_function_caller": self.api_function_getter,
            "global_config": None,
            "umbrella": self,
        }
        self.special_function_table = {}

    def api_function_getter(self, name: str):
        return getattr(self, name)


@pytest.fixture
def integration_app(monkeypatch):
    database = DummyDB()

    toolchain_json = {
        "name": "transform",
        "id": "transform",
        "category": "demo",
        "initial_state": {"value": 0},
        "nodes": [
            {
                "id": "set_value",
                "type": "transform",
                "inputs": {
                    "new_value": {"ref": {"source": "inputs", "path": "$.new_value"}}
                },
                "mappings": [
                    {
                        "destination": {"kind": "state"},
                        "path": "$.value",
                        "value": {"ref": {"source": "inputs", "path": "$.new_value"}},
                        "mode": "set"
                    }
                ],
            }
        ],
    }

    toolchain_llm_json = {
        "name": "llm_toolchain",
        "id": "llm_toolchain",
        "category": "demo",
        "initial_state": {"messages": []},
        "nodes": [
            {
                "id": "llm",
                "type": "api",
                "api_function": "llm",
                "inputs": {},
                "mappings": []
            }
        ],
    }

    toolchains_v1 = {
        toolchain_json["id"]: ToolChain(**toolchain_json),
        toolchain_llm_json["id"]: ToolChain(**toolchain_llm_json),
    }
    toolchains_v2 = {
        toolchain_json["id"]: convert_toolchain_dict(toolchain_json),
        toolchain_llm_json["id"]: convert_toolchain_dict(toolchain_llm_json),
    }

    umbrella = DummyUmbrella(database, toolchains_v1)
    runtime = ToolchainRuntimeService(umbrella, toolchains_v1, toolchains_v2)

    class FakeJob:
        def __init__(self, **data):
            for key, value in data.items():
                setattr(self, key, value)

        def dict(self):
            return {
                "job_id": self.job_id,
                "session_id": self.session_id,
                "node_id": self.node_id,
                "status": self.status,
                "request_id": self.request_id,
                "progress": self.progress,
                "result_meta": self.result_meta,
                "created_at": self.created_at,
                "updated_at": self.updated_at,
            }

    class FakeEventStore:
        def __init__(self):
            self.events: Dict[str, List[EventEnvelope]] = {}
            self.jobs: Dict[str, Dict[str, FakeJob]] = {}
            self.revisions: Dict[str, int] = {}

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
            rev = self.revisions.get(session_id, 0) + 1
            self.revisions[session_id] = rev
            envelope = EventEnvelope(
                rev=rev,
                kind=kind,
                payload=payload,
                actor=actor,
                correlation_id=correlation_id,
                ts=time(),
            )
            self.events.setdefault(session_id, []).append(envelope)
            return envelope

        def latest_snapshot(self, session_id: str):
            return None

        def list_events(self, session_id: str, since_rev: Optional[int] = None) -> List[EventEnvelope]:
            records = self.events.get(session_id, [])
            if since_rev is None:
                return list(records)
            return [record for record in records if record.rev > since_rev]

        def add_dead_letter(self, *args, **kwargs):
            return None

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
            job = self.jobs.setdefault(session_id, {}).get(job_id)
            if job is None:
                job = FakeJob(
                    job_id=job_id,
                    session_id=session_id,
                    node_id=node_id,
                    status=status,
                    request_id=request_id,
                    progress=progress,
                    result_meta=result_meta,
                    created_at=time(),
                    updated_at=time(),
                )
                self.jobs[session_id][job_id] = job
            else:
                job.status = status
                job.progress = progress
                job.result_meta = result_meta
                job.request_id = request_id or job.request_id
                job.updated_at = time()

        def list_jobs(self, session_id: str) -> List[FakeJob]:
            return list(self.jobs.get(session_id, {}).values())

        def latest_rev(self, session_id: str) -> int:
            return self.revisions.get(session_id, 0)

    runtime.event_store = FakeEventStore()
    runtime.job_registry = JobRegistry(runtime.event_store)

    def fake_get_user(_database, _auth, return_auth_type=False):
        user = SimpleNamespace(
            name="tester",
            password_hash="",
            password_salt="",
            private_key_encryption_salt="",
            private_key_secured="",
        )
        auth_obj = SimpleNamespace(username="tester", password_prehash="stub")
        if return_auth_type:
            return (user, auth_obj, None, 1)
        return (user, auth_obj)

    def fake_create_toolchain_session(_database, _toolchain_function_caller, _auth, toolchain_id, ws=None):
        tc = runtime.toolchains_v2[toolchain_id]
        return SimpleNamespace(
            session_hash=f"sess_{uuid.uuid4().hex}",
            toolchain_id=toolchain_id,
            state=deepcopy(tc.initial_state),
            toolchain_session_files={},
            local_cache={},
            author="tester",
        )

    monkeypatch.setattr(toolchains_api, "get_user", fake_get_user)
    monkeypatch.setattr(toolchains_api, "create_toolchain_session", fake_create_toolchain_session)

    app = FastAPI()

    async def resolve_auth(payload):
        return payload or {"username": "tester", "password_prehash": "stub"}

    @app.post("/sessions")
    async def create_session(body: dict):
        toolchain_id = body.get("toolchain_id")
        if not toolchain_id:
            return JSONResponse(status_code=400, content={"success": False, "error": "toolchain_id required"})
        result = await runtime.create_session(await resolve_auth(body.get("auth")), toolchain_id, body.get("title"), body.get("initial_inputs"))
        return {"success": True, **result}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        state = await runtime.get_session_state(session_id, None)
        return {"success": True, **state}

    @app.post("/sessions/{session_id}/event")
    async def post_event(session_id: str, body: dict):
        node_id = body.get("node_id")
        if not node_id:
            return JSONResponse(status_code=400, content={"success": False, "error": "node_id required"})
        inputs = body.get("inputs", {})
        expected_rev = body.get("rev")
        correlation_id = body.get("correlation_id")
        result = await runtime.post_event(session_id, node_id, inputs, expected_rev, auth=None, correlation_id=correlation_id)
        return {"success": True, **result}

    @app.get("/sessions/{session_id}/events")
    async def list_events(session_id: str, since: int | None = None):
        events = await runtime.list_events(session_id, since, auth=None)
        return {"success": True, "events": [event.model_dump() for event in events]}

    @app.get("/sessions/{session_id}/stream")
    async def stream_session(session_id: str):
        subscriber = await runtime.stream_hub.subscribe(session_id)

        async def event_generator():
            try:
                async for message in subscriber.stream():
                    yield {"event": "message", "data": message}
            finally:
                await runtime.stream_hub.unsubscribe(session_id, subscriber)

        return EventSourceResponse(event_generator())

    @app.get("/sessions/{session_id}/jobs")
    async def list_jobs(session_id: str):
        jobs = await runtime.list_jobs(session_id, auth=None)
        return {"success": True, "jobs": jobs}

    @app.post("/sessions/{session_id}/jobs/{job_id}/cancel")
    async def cancel_job(session_id: str, job_id: str):
        result = await runtime.cancel_job(session_id, job_id, auth=None)
        return {"success": True, **result}

    app.state.runtime = runtime
    app.state.umbrella = umbrella
    return app


@pytest.mark.asyncio
async def test_session_event_flow(integration_app):
    from httpx import AsyncClient

    app = integration_app

    transport = ASGITransport(app=app)
    runtime: ToolchainRuntimeService = app.state.runtime
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/sessions", json={"toolchain_id": "transform"})
        assert resp.status_code == 200
        session_id = resp.json()["session_id"]

        event_resp = await client.post(
            f"/sessions/{session_id}/event",
            json={"node_id": "set_value", "inputs": {"new_value": 42}},
        )
        assert event_resp.status_code == 200
        assert event_resp.json()["rev"] >= 1

        state_resp = await client.get(f"/sessions/{session_id}")
        assert state_resp.status_code == 200
        events_resp = await client.get(f"/sessions/{session_id}/events")
        kinds = [e["kind"] for e in events_resp.json()["events"]]
        assert "EVENT_COMPLETED" in kinds

        events_resp = await client.get(f"/sessions/{session_id}/events")
        kinds = [e["kind"] for e in events_resp.json()["events"]]
        assert "EVENT_RECEIVED" in kinds
        assert "EVENT_COMPLETED" in kinds


@pytest.mark.asyncio
async def test_stream_hub_emits_events(integration_app):
    from httpx import AsyncClient

    app = integration_app
    runtime: ToolchainRuntimeService = app.state.runtime

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post("/sessions", json={"toolchain_id": "transform"})
        session_id = create_resp.json()["session_id"]

        subscriber = await runtime.stream_hub.subscribe(session_id)
        try:
            post_event = asyncio.create_task(
                client.post(
                    f"/sessions/{session_id}/event",
                    json={"node_id": "set_value", "inputs": {"new_value": 7}},
                )
            )
            found = False
            desired = {"EVENT_RECEIVED", "NODE_STARTED", "STATE_PATCH_APPLIED", "NODE_COMPLETED", "EVENT_COMPLETED"}
            for _ in range(6):
                raw_item = await asyncio.wait_for(subscriber.queue.get(), timeout=1.0)
                payload = json.loads(raw_item["data"])
                if payload["kind"] in desired:
                    found = True
                    break
            assert found
            await post_event
        finally:
            await runtime.stream_hub.unsubscribe(session_id, subscriber)


@pytest.mark.asyncio
async def test_job_cancellation_via_rest(integration_app):
    from httpx import AsyncClient

    app = integration_app
    runtime: ToolchainRuntimeService = app.state.runtime
    umbrella = app.state.umbrella

    blocker = asyncio.Event()
    cancellation_notified = asyncio.Event()

    async def blocking_llm(job_signal=None, **kwargs):
        if job_signal is not None:
            await job_signal.on_stop(lambda: cancellation_notified.set())
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            cancellation_notified.set()
            raise
        return {"text": "done"}

    umbrella.llm = blocking_llm

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post("/sessions", json={"toolchain_id": "llm_toolchain"})
        session_id = create_resp.json()["session_id"]

        event_task = asyncio.create_task(
            client.post(
                f"/sessions/{session_id}/event",
                json={"node_id": "llm", "inputs": {}},
            )
        )

        # wait for job registration
        await asyncio.sleep(0.1)
        jobs_resp = await client.get(f"/sessions/{session_id}/jobs")
        job_list = jobs_resp.json()["jobs"]
        assert job_list
        job_id = job_list[0]["job_id"]
        assert job_list[0]["status"] == "RUNNING"

        cancel_resp = await client.post(f"/sessions/{session_id}/jobs/{job_id}/cancel")
        assert cancel_resp.status_code == 200
        cancel_data = cancel_resp.json()
        assert cancel_data["status"] == "CANCELLED"

        event_response = await event_task
        assert event_response.status_code == 409

        jobs_after = await client.get(f"/sessions/{session_id}/jobs")
        assert jobs_after.json()["jobs"][0]["status"] == "CANCELLED"

        events = await client.get(f"/sessions/{session_id}/events")
        kinds = [e["kind"] for e in events.json()["events"]]
        assert "JOB_CANCELLED" in kinds

        await asyncio.wait_for(cancellation_notified.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_subscribe_replays_backlog(integration_app):
    from httpx import AsyncClient

    app = integration_app
    runtime: ToolchainRuntimeService = app.state.runtime

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        create_resp = await client.post("/sessions", json={"toolchain_id": "transform"})
        session_id = create_resp.json()["session_id"]

        await client.post(
            f"/sessions/{session_id}/event",
            json={"node_id": "set_value", "inputs": {"new_value": 21}},
        )

    auth_payload = {"username": "tester", "password_prehash": "stub"}
    events = await runtime.list_events(session_id, None, auth_payload)
    last_rev = events[-1].rev

    subscriber = await runtime.subscribe(session_id, auth_payload)
    try:
        history = await runtime.list_events(session_id, last_rev - 1, auth_payload)
        for event in history:
            await subscriber.push(event.model_dump())

        item = await asyncio.wait_for(subscriber.queue.get(), timeout=1.0)
        payload = json.loads(item["data"])
        assert payload["rev"] == last_rev
    finally:
        await runtime.unsubscribe(session_id, subscriber)
