import asyncio
import json
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

import pytest

import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.session import ToolchainSessionV2
from QueryLake.toolchains.legacy_converter import convert_toolchain_dict
from QueryLake.typing.toolchains import MappingDestination, ValueExpression, ValueRef, NodeV2, Mapping, ToolChainV2
from QueryLake.runtime.jobs import JobRegistry, JobStatus
from QueryLake.runtime.signals import JobSignalBus
from QueryLake.runtime.sse import SSESubscriber


def test_converter_produces_v2_nodes():
    toolchain_path = ROOT / "toolchains" / "chat_session_normal_new_scheme.json"
    data = json.loads(toolchain_path.read_text())
    converted = convert_toolchain_dict(data)
    assert isinstance(converted, ToolChainV2)
    assert converted.id == data["id"]
    assert all(isinstance(node, NodeV2) for node in converted.nodes)


@pytest.mark.asyncio
async def test_session_applies_state_patch_events():
    toolchain = ToolChainV2(
        name="test",
        id="test",
        category="demo",
        initial_state={"counter": 0},
        nodes=[
            NodeV2(
                id="increment",
                type="transform",
                mappings=[
                    Mapping(
                        destination=MappingDestination(kind="state"),
                        path="$.counter",
                        value=ValueExpression(literal=2),
                        mode="set",
                    ),
                ],
            )
        ],
    )

    events = []

    def emit(kind: str, payload: dict, meta: dict) -> None:
        events.append((kind, payload, meta))

    session = ToolchainSessionV2(
        session_id="sess",
        toolchain=toolchain,
        author="tester",
        server_context={},
        emit_event=emit,
    )
    session.state["counter"] = 1

    result = await session.process_event("increment", {}, actor="tester", correlation_id=None)
    assert result == {}
    kinds = [event[0] for event in events]
    assert "NODE_STARTED" in kinds
    assert "NODE_COMPLETED" in kinds
    assert "STATE_PATCH_APPLIED" in kinds
    assert session.state["counter"] == 2


class StubJobRegistry:
    def __init__(self) -> None:
        self.calls: List[Any] = []
        self.callbacks: Dict[str, Any] = {}

    async def register(self, job_id, session_id, node_id, status, cancel_callback=None):
        self.calls.append(("register", job_id, status))
        if cancel_callback is not None:
            self.callbacks[job_id] = cancel_callback

    async def update(self, job_id, session_id, node_id, status, progress=None, result_meta=None):
        self.calls.append(("update", job_id, status, result_meta))

    async def cancel(self, job_id: str) -> bool:
        callback = self.callbacks.get(job_id)
        if callback is None:
            return False
        await callback()
        self.calls.append(("cancel", job_id))
        return True


class DummyEventStore:
    def __init__(self) -> None:
        self.jobs: Dict[str, Any] = {}

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
        existing = self.jobs.get(job_id)
        self.jobs[job_id] = SimpleNamespace(
            job_id=job_id,
            session_id=session_id,
            node_id=node_id,
            status=status,
            request_id=request_id if request_id is not None else (existing.request_id if existing else None),
            progress=progress if progress is not None else (existing.progress if existing else None),
            result_meta=result_meta if result_meta is not None else (existing.result_meta if existing else None),
        )

    def list_jobs(self, session_id: str) -> List[Any]:
        return [job for job in self.jobs.values() if job.session_id == session_id]

    def get_job(self, job_id: str) -> Optional[Any]:
        return self.jobs.get(job_id)


class DummyUmbrella:
    def __init__(self, result: Dict[str, Any]):
        self._result = result

    def api_function_getter(self, name: str):
        return getattr(self, name)

    async def llm(self, **kwargs):
        return self._result


@pytest.mark.asyncio
async def test_job_registry_tracks_llm_calls():
    toolchain = ToolChainV2(
        name="job-test",
        id="job-test",
        category="demo",
        initial_state={"messages": []},
        nodes=[
            NodeV2(
                id="llm",
                type="api",
                api_function="llm",
                mappings=[
                    Mapping(
                        destination=MappingDestination(kind="state"),
                        path="$.messages",
                        value=ValueExpression(literal=["done"]),
                        mode="set",
                    )
                ],
            )
        ],
    )

    events: List[Any] = []
    job_registry = StubJobRegistry()
    umbrella = DummyUmbrella({"text": "hi"})

    def emit(kind: str, payload: dict, meta: dict) -> None:
        events.append((kind, payload, meta))

    session = ToolchainSessionV2(
        session_id="sess",
        toolchain=toolchain,
        author="tester",
        server_context={"umbrella": umbrella},
        emit_event=emit,
        job_registry=job_registry,
    )

    await session.process_event("llm", {}, actor="tester", correlation_id=None)

    assert any(call[0] == "register" for call in job_registry.calls)
    assert any(call[0] == "update" and call[2] == JobStatus.COMPLETED for call in job_registry.calls)
    kinds = [event[0] for event in events]
    assert "JOB_ENQUEUED" in kinds
    assert "JOB_COMPLETED" in kinds


@pytest.mark.asyncio
async def test_job_registry_records_failure():
    toolchain = ToolChainV2(
        name="job-fail",
        id="job-fail",
        category="demo",
        initial_state={},
        nodes=[
            NodeV2(
                id="llm",
                type="api",
                api_function="llm",
                mappings=[],
            )
        ],
    )

    events: List[Any] = []
    job_registry = StubJobRegistry()

    class FailingUmbrella(DummyUmbrella):
        async def llm(self, **kwargs):
            raise RuntimeError("boom")

    umbrella = FailingUmbrella({})

    def emit(kind: str, payload: dict, meta: dict) -> None:
        events.append((kind, payload, meta))

    session = ToolchainSessionV2(
        session_id="sess",
        toolchain=toolchain,
        author="tester",
        server_context={"umbrella": umbrella},
        emit_event=emit,
        job_registry=job_registry,
    )

    with pytest.raises(RuntimeError):
        await session.process_event("llm", {}, actor="tester", correlation_id=None)

    assert any(call[0] == "register" for call in job_registry.calls)
    statuses = [call[2] for call in job_registry.calls if call[0] == "update"]
    assert JobStatus.FAILED in statuses
    kinds = [event[0] for event in events]
    assert "JOB_FAILED" in kinds


@pytest.mark.asyncio
async def test_job_registry_handles_cancellation():
    toolchain = ToolChainV2(
        name="job-cancel",
        id="job-cancel",
        category="demo",
        initial_state={},
        nodes=[
            NodeV2(
                id="llm",
                type="api",
                api_function="llm",
                mappings=[],
            )
        ],
    )

    events: List[Any] = []
    job_registry = StubJobRegistry()
    blocker = asyncio.Event()

    class BlockingUmbrella(DummyUmbrella):
        async def llm(self, **kwargs):
            try:
                await blocker.wait()
            except asyncio.CancelledError:
                raise
            return {"text": "done"}

    umbrella = BlockingUmbrella({})

    def emit(kind: str, payload: dict, meta: dict) -> None:
        events.append((kind, payload, meta))

    session = ToolchainSessionV2(
        session_id="sess",
        toolchain=toolchain,
        author="tester",
        server_context={"umbrella": umbrella},
        emit_event=emit,
        job_registry=job_registry,
    )

    task = asyncio.create_task(session.process_event("llm", {}, actor="tester", correlation_id=None))

    while not job_registry.callbacks:
        await asyncio.sleep(0)

    job_id = next(iter(job_registry.callbacks.keys()))
    await job_registry.cancel(job_id)

    with pytest.raises(asyncio.CancelledError):
        await task

    updates = [call for call in job_registry.calls if call[0] == "update"]
    assert any(call[2] == JobStatus.CANCELLED for call in updates)
    kinds = [event[0] for event in events]
    assert "JOB_CANCELLED" in kinds


@pytest.mark.asyncio
async def test_jsonlogic_condition_and_merge_mapping():
    toolchain = ToolChainV2(
        name="logic",
        id="logic",
        category="demo",
        initial_state={"data": {"count": 0}, "items": []},
        nodes=[
            NodeV2(
                id="logic",
                type="transform",
                inputs={"value": ValueExpression(ref=ValueRef(source="inputs", path="$.value"))},
                mappings=[
                    Mapping(
                        destination=MappingDestination(kind="state"),
                        path="$.data",
                        value=ValueExpression(literal={"count": {"ref": {"source": "inputs", "path": "$.value"}}}),
                        mode="merge",
                        condition={"==": [{"var": "inputs.value"}, 5]},
                    ),
                    Mapping(
                        destination=MappingDestination(kind="state"),
                        path="$.items",
                        value=ValueExpression(ref=ValueRef(source="inputs", path="$.value")),
                        mode="append",
                    ),
                ],
            )
        ],
    )

    events: List[Any] = []

    def emit(kind: str, payload: dict, meta: dict) -> None:
        events.append(kind)

    session = ToolchainSessionV2(
        session_id="logic",
        toolchain=toolchain,
        author="tester",
        server_context={},
        emit_event=emit,
    )

    await session.process_event("logic", {"value": 5}, actor="tester", correlation_id=None)
    assert session.state["data"]["count"] == 5
    assert session.state["items"] == [5]

    await session.process_event("logic", {"value": 2}, actor="tester", correlation_id=None)
    # merge should not run when condition fails; append still runs
    assert session.state["data"]["count"] == 5
    assert session.state["items"] == [5, 2]


class _FixedUUID:
    def __init__(self, value: str) -> None:
        self.hex = value


@pytest.mark.asyncio
async def test_job_signal_injected_and_cleanup(monkeypatch):
    job_id_value = "job-signal-test"
    monkeypatch.setattr(uuid, "uuid4", lambda: _FixedUUID(job_id_value))

    signal_bus = JobSignalBus()
    event_store = DummyEventStore()
    job_registry = JobRegistry(event_store)

    class DummyUmbrella:
        def __init__(self) -> None:
            self.job_signal: Optional[Any] = None

        def api_function_getter(self, name: str):
            return getattr(self, name)

        async def llm(self, job_signal=None, **_: Any):
            self.job_signal = job_signal
            if job_signal is not None:
                await job_signal.set_request_id("req-test")
            return {"text": "ok"}

    umbrella = DummyUmbrella()

    events: List[str] = []

    def emit(kind: str, payload: dict, meta: dict) -> None:
        events.append(kind)

    toolchain = ToolChainV2(
        name="job",
        id="job",
        category="demo",
        initial_state={},
        nodes=[
            NodeV2(
                id="llm",
                type="api",
                api_function="llm",
                mappings=[],
            )
        ],
    )

    session = ToolchainSessionV2(
        session_id="sess",
        toolchain=toolchain,
        author="tester",
        server_context={"umbrella": umbrella, "job_signal_bus": signal_bus},
        emit_event=emit,
        job_registry=job_registry,
    )

    await session.process_event("llm", {}, actor="tester", correlation_id=None)

    assert umbrella.job_signal is not None
    assert umbrella.job_signal.job_id == job_id_value
    assert not umbrella.job_signal.triggered()
    assert event_store.jobs[job_id_value].request_id == "req-test"

    # Signal bus should have been cleaned up once the job completed.
    triggered = await signal_bus.trigger(job_id_value)
    assert triggered is False


@pytest.mark.asyncio
async def test_job_signal_cancellation_triggers_callbacks(monkeypatch):
    job_id_value = "job-cancel-test"
    monkeypatch.setattr(uuid, "uuid4", lambda: _FixedUUID(job_id_value))

    signal_bus = JobSignalBus()
    event_store = DummyEventStore()
    job_registry = JobRegistry(event_store)

    class BlockingUmbrella:
        def __init__(self) -> None:
            self.started = asyncio.Event()
            self.cancelled = asyncio.Event()
            self.job_signal: Optional[Any] = None

        def api_function_getter(self, name: str):
            return getattr(self, name)

        async def llm(self, job_signal=None, **_: Any):
            self.job_signal = job_signal
            self.started.set()
            if job_signal is not None:
                await job_signal.set_request_id("req-cancel")
                await job_signal.on_stop(lambda: self.cancelled.set())
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                self.cancelled.set()
                raise

    umbrella = BlockingUmbrella()

    events: List[str] = []

    def emit(kind: str, payload: dict, meta: dict) -> None:
        events.append(kind)

    toolchain = ToolChainV2(
        name="cancel",
        id="cancel",
        category="demo",
        initial_state={},
        nodes=[
            NodeV2(
                id="llm",
                type="api",
                api_function="llm",
                mappings=[],
            )
        ],
    )

    session = ToolchainSessionV2(
        session_id="sess",
        toolchain=toolchain,
        author="tester",
        server_context={"umbrella": umbrella, "job_signal_bus": signal_bus},
        emit_event=emit,
        job_registry=job_registry,
    )

    task = asyncio.create_task(session.process_event("llm", {}, actor="tester", correlation_id=None))

    await asyncio.wait_for(umbrella.started.wait(), timeout=1.0)

    triggered = await signal_bus.trigger(job_id_value)
    assert triggered is True
    cancel_success = await job_registry.cancel(job_id_value)
    assert cancel_success is True

    with pytest.raises(asyncio.CancelledError):
        await task

    await asyncio.wait_for(umbrella.cancelled.wait(), timeout=1.0)

    # Job registry should have marked the job as cancelled.
    assert event_store.jobs[job_id_value].status == JobStatus.CANCELLED.value
    assert event_store.jobs[job_id_value].request_id == "req-cancel"

    # Signal bus should discard the job once cleanup runs.
    triggered_again = await signal_bus.trigger(job_id_value)
    assert triggered_again is False


@pytest.mark.asyncio
async def test_sse_backpressure_injects_meta():
    subscriber = SSESubscriber("sess", queue_size=1)
    await subscriber.push({"rev": 1, "kind": "A", "payload": {}})
    await subscriber.push({"rev": 2, "kind": "B", "payload": {}})
    item = await subscriber.queue.get()
    data = json.loads(item["data"])
    assert data["rev"] == 2
    assert data.get("meta", {}).get("backpressure_drop", {}).get("dropped") == 1
