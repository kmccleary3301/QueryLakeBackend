import asyncio
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.operation_classes.ray_chandra_class import ChandraDeployment, ChandraRequest, _MicroBatcher

_ChandraImpl = ChandraDeployment.func_or_class


@pytest.mark.asyncio
async def test_microbatcher_batches_requests():
    batches = []

    async def process(requests):
        batches.append(len(requests))
        return [req.prompt for req in requests]

    batcher = _MicroBatcher(process, max_batch_size=4, max_batch_wait_ms=10)
    batcher.start()

    async def submit(prompt):
        return await batcher.submit(ChandraRequest(image="img", prompt=prompt, max_new_tokens=1))

    results = await asyncio.gather(
        submit("a"),
        submit("b"),
        submit("c"),
        submit("d"),
    )
    assert results == ["a", "b", "c", "d"]
    assert batches[0] == 4
    await batcher.shutdown()


def test_filter_vllm_constructor_kwargs_filters_unknown_and_none():
    candidate = {
        "model": "models/chandra",
        "dtype": "auto",
        "max_model_len": 131072,
        "enable_chunked_prefill": None,
        "unknown_key": "drop-me",
    }
    accepted = {"model", "dtype", "max_model_len", "enable_chunked_prefill"}
    filtered = _ChandraImpl._filter_vllm_constructor_kwargs(candidate, accepted)
    assert filtered == {
        "model": "models/chandra",
        "dtype": "auto",
        "max_model_len": 131072,
    }


def test_extract_vllm_output_text_handles_common_shapes():
    class _Output:
        def __init__(self, text):
            self.text = text

    class _WithOutputs:
        def __init__(self, text):
            self.outputs = [_Output(text)]

    assert _ChandraImpl._extract_vllm_output_text(None) == ""
    assert _ChandraImpl._extract_vllm_output_text(_WithOutputs("hello")) == "hello"
    assert _ChandraImpl._extract_vllm_output_text("raw") == "raw"


def test_extract_openai_message_content_handles_string_and_parts():
    assert _ChandraImpl._extract_openai_message_content("hello") == "hello"
    assert _ChandraImpl._extract_openai_message_content(None) == ""
    structured = [
        {"type": "text", "text": "line one"},
        {"type": "output_text", "text": "line two"},
        {"type": "ignored", "content": "line three"},
    ]
    assert _ChandraImpl._extract_openai_message_content(structured) == "line one\nline two\nline three"


def test_runtime_backend_vllm_server_can_initialize_without_probe():
    deployment = _ChandraImpl(
        model_path="models/chandra",
        runtime_backend="vllm_server",
        vllm_server_probe_on_init=False,
        vllm_server_fallback_to_hf_on_error=False,
    )
    assert deployment._runtime_backend == "vllm_server"
    assert deployment._vllm_server_base_url.endswith("/v1")


def test_normalize_vllm_server_base_urls_from_csv_and_deduplicate():
    normalized = _ChandraImpl._normalize_vllm_server_base_urls(
        "http://127.0.0.1:8022,http://127.0.0.1:8022/v1,http://127.0.0.1:8023/",
        default_url="http://127.0.0.1:9000/v1",
    )
    assert normalized == [
        "http://127.0.0.1:8022/v1",
        "http://127.0.0.1:8023/v1",
    ]


def test_vllm_server_round_robin_endpoint_selection():
    deployment = _ChandraImpl(
        model_path="models/chandra",
        runtime_backend="vllm_server",
        vllm_server_probe_on_init=False,
        vllm_server_fallback_to_hf_on_error=False,
        vllm_server_base_urls=["http://127.0.0.1:8022", "http://127.0.0.1:8023/v1"],
    )
    sequence = [deployment._next_vllm_server_base_url() for _ in range(5)]
    assert sequence == [
        "http://127.0.0.1:8022/v1",
        "http://127.0.0.1:8023/v1",
        "http://127.0.0.1:8022/v1",
        "http://127.0.0.1:8023/v1",
        "http://127.0.0.1:8022/v1",
    ]


def test_fixed_output_token_budget_ignores_profile_token_knobs():
    deployment = _ChandraImpl(
        model_path="models/chandra",
        runtime_backend="vllm_server",
        vllm_server_probe_on_init=False,
        vllm_server_fallback_to_hf_on_error=False,
        max_new_tokens=64,
        profile_speed_max_new_tokens=128,
        profile_balanced_max_new_tokens=256,
        profile_quality_max_new_tokens=512,
        adaptive_high_max_new_tokens=2048,
    )
    assert deployment._fixed_max_new_tokens == 768
