from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.search_budgeting import (
    SearchBudgetCounters,
    effective_max_searches,
    evaluate_stop_conditions,
    normalize_budget_policy,
    record_step_cost,
    resolve_budget_policy,
)


def test_normalize_budget_policy_with_defaults():
    policy = normalize_budget_policy(None, defaults={"max_searches": 3, "max_reranks": 2})
    assert policy.max_searches == 3
    assert policy.max_reranks == 2


def test_evaluate_stop_conditions_hits_search_limit():
    policy = normalize_budget_policy({"max_searches": 2, "timeout_seconds": 999})
    counters = SearchBudgetCounters(searches=2)
    stop = evaluate_stop_conditions(
        policy=policy,
        counters=counters,
        start_time=10**12,
        confidence_history=[],
    )
    assert stop.should_stop is True
    assert stop.reason == "max_searches"


def test_evaluate_stop_conditions_hits_diminishing_return():
    policy = normalize_budget_policy(
        {
            "max_searches": 10,
            "timeout_seconds": 999,
            "min_confidence_gain": 0.05,
            "plateau_window": 2,
        }
    )
    counters = SearchBudgetCounters(searches=1)
    stop = evaluate_stop_conditions(
        policy=policy,
        counters=counters,
        start_time=10**12,
        confidence_history=[0.1, 0.13, 0.14],
    )
    assert stop.should_stop is True
    assert stop.reason == "diminishing_return"


def test_record_step_cost_accumulates_totals():
    ledger = {}
    ledger = record_step_cost(
        ledger,
        step="llm_round",
        prompt_tokens=100,
        completion_tokens=25,
        model_calls=1,
        estimated_usd=0.001,
    )
    ledger = record_step_cost(
        ledger,
        step="search_call",
        web_calls=1,
    )
    assert ledger["totals"]["prompt_tokens"] == 100
    assert ledger["totals"]["completion_tokens"] == 25
    assert ledger["totals"]["model_calls"] == 1
    assert ledger["totals"]["web_calls"] == 1
    assert abs(ledger["totals"]["estimated_usd"] - 0.001) < 1e-12
    assert len(ledger["steps"]) == 2


def test_effective_max_searches_adaptive_escalates_on_low_confidence():
    policy = normalize_budget_policy(
        {
            "max_searches": 5,
            "adaptive_depth_enabled": True,
            "adaptive_depth_tiers": [1, 2, 3],
            "adaptive_depth_min_confidence": 0.9,
            "adaptive_depth_min_gain": 0.2,
            "plateau_window": 1,
        }
    )
    counters = SearchBudgetCounters(searches=1)
    cap = effective_max_searches(
        policy=policy,
        counters=counters,
        confidence_history=[0.1],
    )
    assert cap == 2


def test_effective_max_searches_adaptive_holds_when_confident():
    policy = normalize_budget_policy(
        {
            "max_searches": 5,
            "adaptive_depth_enabled": True,
            "adaptive_depth_tiers": [1, 2, 3],
            "adaptive_depth_min_confidence": 0.6,
            "adaptive_depth_min_gain": 0.05,
            "plateau_window": 1,
        }
    )
    counters = SearchBudgetCounters(searches=1)
    cap = effective_max_searches(
        policy=policy,
        counters=counters,
        confidence_history=[0.8],
    )
    assert cap == 1


def test_resolve_budget_policy_scoped_overrides_apply_in_order():
    policy = resolve_budget_policy(
        {
            "defaults": {"max_searches": 4, "timeout_seconds": 120},
            "tenant_overrides": {"tenant_a": {"max_searches": 6}},
            "toolchain_overrides": {"tc_rag": {"timeout_seconds": 60}},
            "tenant_toolchain_overrides": {
                "tenant_a": {"tc_rag": {"max_searches": 8}}
            },
        },
        defaults={"max_searches": 3, "timeout_seconds": 180},
        tenant_scope="tenant_a",
        toolchain_id="tc_rag",
    )
    assert policy.max_searches == 8
    assert int(policy.timeout_seconds) == 60


def test_resolve_budget_policy_without_scoped_keys_preserves_legacy_shape():
    policy = resolve_budget_policy(
        {"max_searches": 7, "max_reranks": 3},
        defaults={"max_searches": 5, "max_reranks": 1},
        tenant_scope="tenant_a",
        toolchain_id="tc_any",
    )
    assert policy.max_searches == 7
    assert policy.max_reranks == 3
