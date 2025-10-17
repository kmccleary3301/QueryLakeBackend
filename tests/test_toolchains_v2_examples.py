import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.typing.toolchains import ToolChainV2
from QueryLake.runtime.session import ToolchainSessionV2


class DemoUmbrella:
    """Stub umbrella that exposes a deterministic LLM function."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def api_function_getter(self, name: str):
        assert name == "llm", f"demo umbrella only supports 'llm', got {name}"
        return self.llm

    async def llm(
        self,
        chat_history: List[Dict[str, Any]] | None = None,
        model_parameters: Dict[str, Any] | None = None,
        stream_callables: Dict[str, Any] | None = None,
        **_: Any,
    ) -> Dict[str, Any]:
        chat_history = chat_history or []
        last_message = chat_history[-1]["content"] if chat_history else ""
        text = f"LLM::{last_message}"
        self.calls.append({"chat": chat_history, "text": text})

        if stream_callables and "output" in stream_callables:
            # Emit streaming chunks based on whitespace splits
            chunks = text.split(" ")
            for idx, chunk in enumerate(chunks):
                suffix = " " if idx < len(chunks) - 1 else ""
                await stream_callables["output"](chunk + suffix)
        return {"output": text}


def _load_toolchain(toolchain_id: str) -> ToolChainV2:
    path = ROOT / "toolchains_v2_examples" / f"{toolchain_id}.json"
    data = json.loads(path.read_text())
    return ToolChainV2.model_validate(data)


def test_demo_cancelable_research():
    tool = _load_toolchain("demo_cancelable_research")
    umbrella = DemoUmbrella()
    events: List[str] = []

    def emit(kind: str, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
        events.append(kind)

    sess = ToolchainSessionV2(
        session_id="demo-cancel",
        toolchain=tool,
        author="tester",
        server_context={"umbrella": umbrella},
        emit_event=emit,
        job_registry=None,
    )

    async def run():
        return await sess.process_event("prepare", {"prompt": "AI for logistics"}, actor="tester")

    user_payload = asyncio.run(run())

    assert sess.state["status"] == "complete"
    assert sess.state["report"]["draft"].startswith("LLM::")
    assert len(sess.state["report"]["chunks"]) >= 1
    assert user_payload["report"]["status"] == "complete"
    assert "STREAM_OPEN" in events
    assert "STREAM_CLOSE" in events


def test_demo_parallel_brainstorm():
    tool = _load_toolchain("demo_parallel_brainstorm")
    umbrella = DemoUmbrella()
    events: List[str] = []

    def emit(kind: str, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
        events.append(kind)

    sess = ToolchainSessionV2(
        session_id="demo-brainstorm",
        toolchain=tool,
        author="tester",
        server_context={"umbrella": umbrella},
        emit_event=emit,
        job_registry=None,
    )

    async def run():
        return await sess.process_event("ingest", {"topic": "Product launch"}, actor="tester")

    user_payload = asyncio.run(run())

    assert "brainstorm" in sess.state["ideas"]
    assert "critic" in sess.state["ideas"]
    assert sess.state["ideas"]["brainstorm"].startswith("LLM::")
    assert sess.state["ideas"]["critic"].startswith("LLM::")
    if user_payload["summary"]["brainstorm"] is not None:
        assert user_payload["summary"]["brainstorm"].startswith("LLM::")
    assert user_payload["summary"]["critic"].startswith("LLM::")


def test_demo_streaming_coauthor():
    tool = _load_toolchain("demo_streaming_coauthor")
    umbrella = DemoUmbrella()
    events: List[str] = []

    def emit(kind: str, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
        events.append(kind)

    sess = ToolchainSessionV2(
        session_id="demo-coauthor",
        toolchain=tool,
        author="tester",
        server_context={"umbrella": umbrella},
        emit_event=emit,
        job_registry=None,
    )

    async def run():
        return await sess.process_event("compose", {"topic": "Outline bullets"}, actor="tester")

    user_payload = asyncio.run(run())

    assert sess.state["draft"]["text"].startswith("LLM::")
    assert len(sess.state["draft"]["chunks"]) >= 1
    assert user_payload["draft"]["text"].startswith("LLM::")
    assert "STREAM_OPEN" in events
    assert "STREAM_CHUNK" in events
    assert "STREAM_CLOSE" in events
