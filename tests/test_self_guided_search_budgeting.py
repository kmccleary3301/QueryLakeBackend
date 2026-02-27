import asyncio
from pathlib import Path
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.api.custom_model_functions import self_guided_search as sgs


def test_self_guided_search_returns_budget_and_cost_fields(monkeypatch):
    monkeypatch.setattr(sgs, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))

    calls = {"n": 0}

    async def _llm_call(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return {
                "output": "ready",
                "function_calls": [{"function": "ready_to_answer", "arguments": {}}],
                "input_token_count": 11,
                "output_token_count": 7,
            }
        return {
            "output": "final answer",
            "function_calls": [],
            "input_token_count": 13,
            "output_token_count": 9,
        }

    def _toolchain(name):
        if name == "llm":
            return _llm_call
        if name == "search_bm25":
            return lambda **kwargs: []
        if name == "search_hybrid":
            async def _hybrid(**kwargs):
                return {"rows": []}
            return _hybrid
        raise KeyError(name)

    result = asyncio.run(
        sgs.self_guided_search(
            database=SimpleNamespace(),
            auth={"username": "tester", "password_prehash": "x"},
            toolchain_function_caller=_toolchain,
            question="What is BASF?",
            collection_ids=["abc123"],
            max_searches=2,
            budget_policy={"timeout_seconds": 120},
        )
    )

    assert result["output"] == "final answer"
    assert "budget_policy" in result
    assert "budget_counters" in result
    assert "cost_accounting" in result
    assert result["cost_accounting"]["totals"]["model_calls"] == 2


def test_self_guided_search_guardrails_malformed_function_calls(monkeypatch):
    monkeypatch.setattr(sgs, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))

    calls = {"n": 0}

    async def _llm_call(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] <= 2:
            return {
                "output": "thinking...",
                "function_calls": [],
                "input_token_count": 5,
                "output_token_count": 3,
            }
        return {
            "output": "best-effort final answer",
            "function_calls": [],
            "input_token_count": 7,
            "output_token_count": 4,
        }

    def _toolchain(name):
        if name == "llm":
            return _llm_call
        if name == "search_bm25":
            return lambda **kwargs: []
        if name == "search_hybrid":
            async def _hybrid(**kwargs):
                return {"rows": []}
            return _hybrid
        raise KeyError(name)

    result = asyncio.run(
        sgs.self_guided_search(
            database=SimpleNamespace(),
            auth={"username": "tester", "password_prehash": "x"},
            toolchain_function_caller=_toolchain,
            question="What is BASF?",
            collection_ids=["abc123"],
            max_searches=2,
            budget_policy={"max_invalid_function_calls": 2, "timeout_seconds": 120},
        )
    )

    assert result["output"] == "best-effort final answer"
    assert result["stop_reason"] == "malformed_function_calls"


def test_self_guided_search_strict_mode_sets_seed(monkeypatch):
    monkeypatch.setattr(sgs, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))

    observed = {"seed": None}
    calls = {"n": 0}

    async def _llm_call(*args, **kwargs):
        calls["n"] += 1
        model_parameters = kwargs.get("model_parameters", {})
        if observed["seed"] is None:
            observed["seed"] = model_parameters.get("seed")
        if calls["n"] == 1:
            return {
                "output": "ready",
                "function_calls": [{"function": "ready_to_answer", "arguments": {}}],
                "input_token_count": 3,
                "output_token_count": 2,
            }
        return {
            "output": "final",
            "function_calls": [],
            "input_token_count": 3,
            "output_token_count": 2,
        }

    def _toolchain(name):
        if name == "llm":
            return _llm_call
        if name == "search_bm25":
            return lambda **kwargs: []
        if name == "search_hybrid":
            async def _hybrid(**kwargs):
                return {"rows": []}
            return _hybrid
        raise KeyError(name)

    result = asyncio.run(
        sgs.self_guided_search(
            database=SimpleNamespace(),
            auth={"username": "tester", "password_prehash": "x"},
            toolchain_function_caller=_toolchain,
            question="What is BASF?",
            collection_ids=["abc123"],
            budget_policy={"strict_deterministic_mode": True, "timeout_seconds": 120},
        )
    )

    assert result["output"] == "final"
    assert observed["seed"] == 0


def test_self_guided_search_hybrid_uses_default_orchestrated_path(monkeypatch):
    monkeypatch.setattr(sgs, "get_user", lambda database, auth: (SimpleNamespace(), SimpleNamespace(username="tester")))

    llm_calls = {"n": 0}
    hybrid_calls = {"kwargs": []}

    async def _llm_call(*args, **kwargs):
        llm_calls["n"] += 1
        if llm_calls["n"] == 1:
            return {
                "output": "search now",
                "function_calls": [{"function": "search_database", "arguments": {"question": "vapor recovery"}}],
                "input_token_count": 4,
                "output_token_count": 3,
            }
        if llm_calls["n"] == 2:
            return {
                "output": "ready",
                "function_calls": [{"function": "ready_to_answer", "arguments": {}}],
                "input_token_count": 4,
                "output_token_count": 3,
            }
        return {
            "output": "final",
            "function_calls": [],
            "input_token_count": 4,
            "output_token_count": 3,
        }

    async def _hybrid(**kwargs):
        hybrid_calls["kwargs"].append(kwargs)
        return {
            "rows": [
                {
                    "id": "chunk_1",
                    "text": "Recovered text",
                    "document_name": "doc",
                    "collection_id": "abc123",
                }
            ]
        }

    def _toolchain(name):
        if name == "llm":
            return _llm_call
        if name == "search_bm25":
            return lambda **kwargs: []
        if name == "search_hybrid":
            return _hybrid
        raise KeyError(name)

    result = asyncio.run(
        sgs.self_guided_search(
            database=SimpleNamespace(),
            auth={"username": "tester", "password_prehash": "x"},
            toolchain_function_caller=_toolchain,
            question="What is BASF?",
            collection_ids=["abc123"],
            use_hybrid=True,
            budget_policy={"timeout_seconds": 120},
        )
    )

    assert result["output"] == "final"
    assert len(hybrid_calls["kwargs"]) == 1
    assert "_orchestrator_bypass" not in hybrid_calls["kwargs"][0]
