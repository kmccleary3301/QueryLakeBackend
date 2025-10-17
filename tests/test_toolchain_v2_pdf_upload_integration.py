import asyncio
import json
from io import BytesIO
from pathlib import Path
from typing import Any, Dict

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from httpx import ASGITransport

from QueryLake.typing.toolchains import ToolChainV2
from QueryLake.runtime.service import ToolchainRuntimeService
from QueryLake.api import toolchains as toolchains_api
from QueryLake.runtime.jobs import JobRegistry


class FakeEventStore:
    def __init__(self):
        self.events: Dict[str, list] = {}
        self.jobs: Dict[str, Dict[str, Any]] = {}
        self.revisions: Dict[str, int] = {}

    def append_event(self, session_id: str, kind: str, payload: Dict[str, Any], *, actor=None, correlation_id=None, snapshot_state=None, snapshot_files=None):
        rev = self.revisions.get(session_id, 0) + 1
        self.revisions[session_id] = rev
        envelope = type("Envelope", (), {"rev": rev, "kind": kind, "payload": payload, "actor": actor, "correlation_id": correlation_id, "ts": 0})
        self.events.setdefault(session_id, []).append(envelope)
        return envelope

    def list_events(self, session_id: str, since_rev: int | None = None):
        return [e for e in self.events.get(session_id, []) if since_rev is None or e.rev > since_rev]

    def latest_snapshot(self, session_id: str):
        return None

    def upsert_job(self, job_id: str, session_id: str, node_id: str, status: str, *, request_id=None, progress=None, result_meta=None):
        job = self.jobs.setdefault(session_id, {}).get(job_id)
        if job is None:
            job = type("Job", (), {"job_id": job_id, "session_id": session_id, "node_id": node_id, "status": status, "progress": progress, "result_meta": result_meta, "request_id": request_id, "created_at": 0, "updated_at": 0})
            self.jobs[session_id][job_id] = job
        else:
            job.status = status
            job.progress = progress
            job.result_meta = result_meta
            job.request_id = request_id or job.request_id

    def list_jobs(self, session_id: str):
        return list(self.jobs.get(session_id, {}).values())

    def get_job(self, job_id: str):
        for jobs in self.jobs.values():
            if job_id in jobs:
                return jobs[job_id]
        return None

    def dead_letter(self, *args, **kwargs):
        return None


class DummyDB:
    def get(self, *args, **kwargs):
        return None

    def add(self, *args, **kwargs):
        return None

    def commit(self):
        return None


class DummyFilesRuntime:
    def __init__(self) -> None:
        self.calls: list[Dict[str, Any]] = []

    async def process_version(self, file_id: str, version_id: str, auth: Any = None):
        self.calls.append({"file_id": file_id, "version_id": version_id, "auth": auth})
        return {"job_id": f"jb_{version_id}", "status": "COMPLETED"}


class DummyUmbrella:
    def __init__(self, database, toolchains_v1, files_runtime):
        self.database = database
        self.toolchains_v1 = toolchains_v1
        self.files_runtime = files_runtime
        self.default_function_arguments = {
            "database": database,
            "toolchains_available": toolchains_v1,
            "public_key": None,
            "server_private_key": None,
            "toolchain_function_caller": self.api_function_getter,
            "global_config": None,
            "umbrella": self,
            "job_signal_bus": None,
        }
        self.special_function_table = {}

    def api_function_getter(self, name: str):
        if name == "files_process_version":
            async def call(database=None, auth=None, umbrella=None, file_id: str | None = None, version_id: str | None = None):
                return await self.files_runtime.process_version(file_id, version_id, auth=auth)

            return call
        raise ValueError(f"Unknown function {name}")


@pytest.fixture
def pdf_upload_app(monkeypatch):
    database = DummyDB()
    files_runtime = DummyFilesRuntime()

    toolchain_path = ROOT / "toolchains" / "demo_pdf_upload_v2.json"
    toolchain_data = json.loads(toolchain_path.read_text())
    tc_v2 = ToolChainV2.model_validate(toolchain_data)

    toolchains_v1 = {tc_v2.id: object()}
    toolchains_v2 = {tc_v2.id: tc_v2}

    umbrella = DummyUmbrella(database, toolchains_v1, files_runtime)
    runtime = ToolchainRuntimeService(umbrella, toolchains_v1, toolchains_v2)
    runtime.event_store = FakeEventStore()
    runtime.job_registry = JobRegistry(runtime.event_store)

    def fake_get_user(_database, _auth, return_auth_type=False):
        user = type("User", (), {"name": "tester"})()
        auth_obj = type("Auth", (), {"username": "tester", "password_prehash": "stub"})()
        if return_auth_type:
            return (user, auth_obj, None, 1)
        return (user, auth_obj)

    def fake_create_toolchain_session(_database, _toolchain_function_caller, _auth, toolchain_id, ws=None):
        return type(
            "LegacySession",
            (),
            {
                "session_hash": f"sess_{toolchain_id}",
                "toolchain_id": toolchain_id,
                "state": json.loads(json.dumps(toolchains_v2[toolchain_id].initial_state)),
                "toolchain_session_files": {},
                "local_cache": {},
                "author": "tester",
            },
        )()

    monkeypatch.setattr(toolchains_api, "get_user", fake_get_user)
    monkeypatch.setattr(toolchains_api, "create_toolchain_session", fake_create_toolchain_session)

    app = FastAPI()

    async def resolve_auth(payload):
        return payload or {"username": "tester", "password_prehash": "stub"}

    @app.post("/files")
    async def upload_file(file: UploadFile = File(...)):
        data = await file.read()
        if not data:
            return JSONResponse(status_code=400, content={"success": False, "error": "empty"})
        file_id = "fl_demo"
        version_id = "fv_demo"
        meta = {
            "file_id": file_id,
            "version_id": version_id,
            "logical_name": file.filename,
            "bytes_cas": "cas_demo",
        }
        app.state.upload_meta = meta
        return {"success": True, **meta}

    @app.post("/sessions")
    async def create_session(body: dict):
        toolchain_id = body.get("toolchain_id")
        result = await runtime.create_session(await resolve_auth(body.get("auth")), toolchain_id, body.get("title"), body.get("initial_inputs"))
        return {"success": True, **result}

    @app.post("/sessions/{session_id}/event")
    async def post_event(session_id: str, body: dict):
        node_id = body.get("node_id")
        inputs = body.get("inputs", {})
        result = await runtime.post_event(session_id, node_id, inputs, expected_rev=body.get("rev"), auth=await resolve_auth(body.get("auth")), correlation_id=None)
        return {"success": True, **result}

    @app.get("/sessions/{session_id}")
    async def get_session(session_id: str):
        result = await runtime.get_session_state(session_id, auth=await resolve_auth(None))
        return {"success": True, **result}

    app.state.runtime = runtime
    app.state.umbrella = umbrella
    app.state.files_runtime = files_runtime
    return app


@pytest.mark.asyncio
async def test_pdf_toolchain_end_to_end(pdf_upload_app):
    from httpx import AsyncClient

    app = pdf_upload_app
    transport = ASGITransport(app=app)

    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Upload a PDF
        resp = await client.post(
            "/files",
            files={"file": ("sample.pdf", b"%PDF-1.4\n%EOF", "application/pdf")},
        )
        assert resp.status_code == 200
        upload_meta = resp.json()
        assert upload_meta["success"] is True

        # Create session
        session_resp = await client.post("/sessions", json={"toolchain_id": "demo_pdf_upload_v2"})
        assert session_resp.status_code == 200
        session_id = session_resp.json()["session_id"]

        # Trigger toolchain event with file metadata
        event_payload = {
            "node_id": "attach_pdf",
            "inputs": {"file_meta": {k: upload_meta[k] for k in ["file_id", "version_id", "logical_name", "bytes_cas"]}},
        }
        event_resp = await client.post(f"/sessions/{session_id}/event", json=event_payload)
        assert event_resp.status_code == 200

        # Verify files runtime was invoked
        calls = app.state.files_runtime.calls
        assert calls and calls[-1]["file_id"] == upload_meta["file_id"]
        assert calls[-1]["version_id"] == upload_meta["version_id"]

        # Verify session state recorded job result
        state_resp = await client.get(f"/sessions/{session_id}")
        assert state_resp.status_code == 200
        last_job = state_resp.json()["state"]["last_job"]
        assert last_job["status"] == "COMPLETED"
