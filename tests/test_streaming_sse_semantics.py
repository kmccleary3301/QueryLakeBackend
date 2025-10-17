import asyncio
from pathlib import Path
import sys
from typing import Any, Dict

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.typing.toolchains import (
    ToolChainV2, NodeV2, Mapping, MappingDestination, ValueExpression, ValueRef
)
from QueryLake.runtime.session import ToolchainSessionV2


async def _fake_llm_api(**kwargs) -> Dict[str, Any]:
    # Simulate a streaming LLM that yields chunks then returns the final text
    stream = kwargs.get("stream_callables") or {}
    on_token = stream.get("output")
    text = "IoT in healthcare."
    if on_token is not None:
        for ch in ["IoT ", "in ", "healthcare."]:
            await on_token(ch)
    return {"output": text}


def test_streaming_events_emitted_and_state_updates():
    tool = ToolChainV2(
        name="stream-demo",
        id="stream-demo",
        category="demo",
        initial_state={"acc": {"text": "", "chunks": []}},
        nodes=[
            NodeV2(
                id="compose",
                type="api",
                api_function="llm",
                inputs={
                    "chat_history": ValueExpression(literal=[{"role": "user", "content": "topic"}]),
                    "model_parameters": ValueExpression(literal={"stream": True}),
                },
                mappings=[
                    Mapping(
                        destination=MappingDestination(kind="state"),
                        path="$.acc.text",
                        value=ValueExpression(ref=ValueRef(source="outputs", path="$.output")),
                        mode="append",
                        stream={"enabled": True, "mode": "append", "initial": ""},
                    ),
                    Mapping(
                        destination=MappingDestination(kind="state"),
                        path="$.acc.chunks",
                        value=ValueExpression(ref=ValueRef(source="outputs", path="$.output")),
                        mode="append",
                        stream={"enabled": True, "mode": "append", "initial": []},
                    ),
                ],
            )
        ],
    )

    events = []

    def emit(kind: str, payload: Dict[str, Any], meta: Dict[str, Any]) -> None:
        events.append(kind)

    class DummyUmbrella:
        def api_function_getter(self, name: str):
            return _fake_llm_api

    sess = ToolchainSessionV2(
        session_id="sess1",
        toolchain=tool,
        author="tester",
        server_context={"umbrella": DummyUmbrella()},
        emit_event=emit,
        job_registry=None,
    )

    asyncio.run(sess.process_event("compose", {}, actor="tester"))

    # Check streaming events exist
    assert "STREAM_OPEN" in events
    assert "STREAM_CHUNK" in events
    assert "STREAM_CLOSE" in events

    # And state reflects the appended text and chunks
    assert sess.state["acc"]["text"].endswith("healthcare.")
    assert len(sess.state["acc"]["chunks"]) >= 1


def test_iterate_mapping_appends_loop_items_to_state():
    tool = ToolChainV2(
        name="iterate-demo",
        id="iterate-demo",
        category="demo",
        initial_state={"words": [], "out": []},
        nodes=[
            NodeV2(
                id="copy",
                type="transform",
                mappings=[
                    Mapping(
                        destination=MappingDestination(kind="state"),
                        path="$.out",
                        value=ValueExpression(ref=ValueRef(source="server", path="$.loop")),
                        mode="append",
                        iterate=ValueRef(source="inputs", path="$.words"),
                    )
                ],
            )
        ],
    )

    events = []
    def emit(kind, payload, meta):
        events.append(kind)

    sess = ToolchainSessionV2(
        session_id="sess2",
        toolchain=tool,
        author="tester",
        server_context={"umbrella": object()},
        emit_event=emit,
        job_registry=None,
    )

    asyncio.run(sess.process_event("copy", {"words": ["a", "b", "c"]}, actor="tester"))

    assert sess.state["out"] == ["a", "b", "c"]

