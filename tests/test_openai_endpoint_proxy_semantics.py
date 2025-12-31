import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi.responses import JSONResponse, StreamingResponse

from QueryLake.routing import openai_completions as oc


class _DummyRawRequest:
    def __init__(self, body: Dict[str, Any], headers: Optional[Dict[str, str]] = None):
        self._body = body
        self.headers = headers or {}
        self.state = object()

    async def json(self) -> Dict[str, Any]:
        return self._body


class _DummyAsyncGen:
    def __init__(self, items: Iterable[str]):
        self._items = list(items)
        self._idx = 0

    def __aiter__(self):
        return self

    async def __anext__(self) -> str:
        if self._idx >= len(self._items):
            raise StopAsyncIteration
        item = self._items[self._idx]
        self._idx += 1
        return item


class _DummyChatCompletionMethod:
    def __init__(self, items: List[str]):
        self._items = items

    def remote(self, *_args, **_kwargs) -> _DummyAsyncGen:
        return _DummyAsyncGen(self._items)


class _DummyLLMHandle:
    def __init__(self, items: List[str]):
        self.create_chat_completion_original = _DummyChatCompletionMethod(items)


class _DummyUmbrella:
    def __init__(self, llm_handle: _DummyLLMHandle):
        self.database = object()
        self.config = type("Cfg", (), {"vllm_upstream_base_url": None, "vllm_upstream_model_map": None})()
        self.llm_handles = {"demo-model": llm_handle}


def test_openai_chat_completion_streaming_response(monkeypatch):
    monkeypatch.setattr(oc, "get_user", lambda *_args, **_kwargs: (None, None))

    umbrella = _DummyUmbrella(
        _DummyLLMHandle(
            [
                ">>>>>>>>>>>STREAM",
                "data: {\"id\":\"x\",\"choices\":[]}\n\n",
                "data: [DONE]\n\n",
            ]
        )
    )
    raw_request = _DummyRawRequest(
        {"model": "demo-model", "messages": [{"role": "user", "content": "hi"}], "stream": True},
        headers={"authorization": "Bearer test"},
    )

    resp = asyncio.run(oc.openai_chat_completion(umbrella, raw_request))
    assert isinstance(resp, StreamingResponse)
    assert resp.media_type == "text/event-stream"


def test_openai_chat_completion_standard_json_response(monkeypatch):
    monkeypatch.setattr(oc, "get_user", lambda *_args, **_kwargs: (None, None))

    payload = {"id": "cmpl_x", "object": "chat.completion", "choices": [], "model": "demo-model"}
    umbrella = _DummyUmbrella(
        _DummyLLMHandle(
            [
                ">>>>>>>>>>>STANDARD | " + json.dumps(payload),
            ]
        )
    )
    raw_request = _DummyRawRequest(
        {"model": "demo-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"authorization": "Bearer test"},
    )

    resp = asyncio.run(oc.openai_chat_completion(umbrella, raw_request))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 200
    assert json.loads(resp.body) == payload


def test_openai_chat_completion_error_json_response(monkeypatch):
    monkeypatch.setattr(oc, "get_user", lambda *_args, **_kwargs: (None, None))

    payload = {"object": "error", "type": "Upstream", "message": "boom", "code": 500}
    umbrella = _DummyUmbrella(
        _DummyLLMHandle(
            [
                ">>>>>>>>>>>ERROR | " + json.dumps(payload),
            ]
        )
    )
    raw_request = _DummyRawRequest(
        {"model": "demo-model", "messages": [{"role": "user", "content": "hi"}]},
        headers={"authorization": "Bearer test"},
    )

    resp = asyncio.run(oc.openai_chat_completion(umbrella, raw_request))
    assert isinstance(resp, JSONResponse)
    assert resp.status_code == 500
    assert json.loads(resp.body) == payload


def test_openai_embedding_endpoint_shapes_response(monkeypatch):
    monkeypatch.setattr(oc, "get_user", lambda *_args, **_kwargs: (None, None))

    async def _fake_embedding_call(*_args, **kwargs):
        assert kwargs.get("return_tokens_usage") is True
        return ([[0.1, 0.2, 0.3]], 7)

    monkeypatch.setattr(oc, "embedding_call", _fake_embedding_call)

    class _DummyEmbUmbrella:
        def __init__(self):
            self.database = object()
            self.config = type("Cfg", (), {"vllm_upstream_base_url": None, "vllm_upstream_model_map": None})()

    raw_request = _DummyRawRequest(
        {"model": "demo-emb", "input": "hello"},
        headers={"authorization": "Bearer test"},
    )

    resp = asyncio.run(oc.openai_create_embedding(_DummyEmbUmbrella(), raw_request))
    assert isinstance(resp, JSONResponse)
    payload = json.loads(resp.body)
    assert payload["model"] == "demo-emb"
    assert payload["object"] == "list"
    assert payload["usage"]["total_tokens"] == 7
    assert payload["data"][0]["index"] == 0
