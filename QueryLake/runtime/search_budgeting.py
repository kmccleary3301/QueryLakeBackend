from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union
import time

from pydantic import BaseModel, Field


class SearchBudgetPolicy(BaseModel):
    max_searches: int = Field(default=5, ge=0)
    max_reranks: int = Field(default=5, ge=0)
    max_prompt_tokens: int = Field(default=100_000, ge=0)
    max_completion_tokens: int = Field(default=40_000, ge=0)
    max_web_calls: int = Field(default=0, ge=0)
    max_depth: int = Field(default=30, ge=1)
    timeout_seconds: float = Field(default=180.0, ge=1.0)

    min_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    min_confidence_gain: Optional[float] = Field(default=None, ge=0.0)
    plateau_window: int = Field(default=2, ge=1)
    adaptive_depth_enabled: bool = Field(default=False)
    adaptive_depth_tiers: List[int] = Field(default_factory=lambda: [1, 2, 3])
    adaptive_depth_min_confidence: float = Field(default=0.75, ge=0.0, le=1.0)
    adaptive_depth_min_gain: float = Field(default=0.05, ge=0.0)
    strict_deterministic_mode: bool = Field(default=False)
    max_invalid_function_calls: int = Field(default=3, ge=1)
    max_duplicate_search_retries: int = Field(default=2, ge=0)


class SearchBudgetCounters(BaseModel):
    searches: int = 0
    reranks: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    web_calls: int = 0
    depth: int = 0


class StopEvaluation(BaseModel):
    should_stop: bool = False
    reason: Optional[str] = None


def confidence_snapshot(confidence_history: Optional[list], *, window: int) -> Tuple[float, float]:
    values = [float(v) for v in (confidence_history or [])]
    if len(values) == 0:
        return 0.0, 0.0
    latest = float(values[-1])
    if len(values) <= int(window):
        baseline = float(values[0])
    else:
        baseline = float(values[-1 - int(window)])
    gain = max(0.0, latest - baseline)
    return latest, gain


def _normalized_adaptive_tiers(policy: SearchBudgetPolicy) -> List[int]:
    tiers = []
    for raw in policy.adaptive_depth_tiers:
        try:
            value = int(raw)
        except Exception:
            continue
        if value <= 0:
            continue
        tiers.append(value)
    if len(tiers) == 0:
        tiers = [1, 2, 3]
    tiers = sorted(set(tiers))
    cap = max(1, int(policy.max_searches))
    tiers = [min(v, cap) for v in tiers]
    if tiers[-1] != cap:
        tiers.append(cap)
    return sorted(set(tiers))


def effective_max_searches(
    *,
    policy: SearchBudgetPolicy,
    counters: SearchBudgetCounters,
    confidence_history: Optional[list] = None,
) -> int:
    hard_cap = max(1, int(policy.max_searches))
    if not bool(policy.adaptive_depth_enabled):
        return hard_cap

    tiers = _normalized_adaptive_tiers(policy)
    latest, gain = confidence_snapshot(confidence_history, window=max(1, int(policy.plateau_window)))
    cap = tiers[0]
    idx = 0
    searches_done = int(counters.searches)
    while idx < len(tiers) - 1 and searches_done >= cap:
        should_escalate = (
            latest < float(policy.adaptive_depth_min_confidence)
            and gain < float(policy.adaptive_depth_min_gain)
        )
        if not should_escalate:
            break
        idx += 1
        cap = tiers[idx]
    return min(hard_cap, int(cap))


def normalize_budget_policy(
    policy: Optional[Union[SearchBudgetPolicy, Dict]],
    *,
    defaults: Optional[Dict[str, Union[int, float]]] = None,
) -> SearchBudgetPolicy:
    if isinstance(policy, SearchBudgetPolicy):
        return policy
    merged = dict(defaults or {})
    if isinstance(policy, dict):
        merged.update(policy)
    return SearchBudgetPolicy(**merged)


def resolve_budget_policy(
    policy: Optional[Union[SearchBudgetPolicy, Dict]],
    *,
    defaults: Optional[Dict[str, Union[int, float]]] = None,
    tenant_scope: Optional[str] = None,
    toolchain_id: Optional[str] = None,
) -> SearchBudgetPolicy:
    if isinstance(policy, SearchBudgetPolicy):
        return policy
    if not isinstance(policy, dict):
        return normalize_budget_policy(policy, defaults=defaults)

    # If no scoped profiles are provided, preserve simple legacy behavior.
    scoped_keys = {
        "defaults",
        "tenant_overrides",
        "toolchain_overrides",
        "tenant_toolchain_overrides",
    }
    if len(scoped_keys.intersection(set(policy.keys()))) == 0:
        return normalize_budget_policy(policy, defaults=defaults)

    merged: Dict[str, Union[int, float, bool, list, None]] = dict(defaults or {})
    model_fields = set(SearchBudgetPolicy.model_fields.keys())

    scoped_defaults = policy.get("defaults")
    if isinstance(scoped_defaults, dict):
        merged.update({k: v for k, v in scoped_defaults.items() if k in model_fields})

    # Allow top-level policy fields to act as baseline too.
    merged.update({k: v for k, v in policy.items() if k in model_fields})

    tenant_overrides = policy.get("tenant_overrides")
    if isinstance(tenant_overrides, dict) and isinstance(tenant_scope, str):
        scoped = tenant_overrides.get(tenant_scope)
        if isinstance(scoped, dict):
            merged.update({k: v for k, v in scoped.items() if k in model_fields})

    toolchain_overrides = policy.get("toolchain_overrides")
    if isinstance(toolchain_overrides, dict) and isinstance(toolchain_id, str):
        scoped = toolchain_overrides.get(toolchain_id)
        if isinstance(scoped, dict):
            merged.update({k: v for k, v in scoped.items() if k in model_fields})

    tenant_toolchain_overrides = policy.get("tenant_toolchain_overrides")
    if (
        isinstance(tenant_toolchain_overrides, dict)
        and isinstance(tenant_scope, str)
        and isinstance(toolchain_id, str)
    ):
        tenant_map = tenant_toolchain_overrides.get(tenant_scope)
        if isinstance(tenant_map, dict):
            scoped = tenant_map.get(toolchain_id)
            if isinstance(scoped, dict):
                merged.update({k: v for k, v in scoped.items() if k in model_fields})

    return SearchBudgetPolicy(**merged)


def record_step_cost(
    ledger: Optional[Dict],
    *,
    step: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    model_calls: int = 0,
    web_calls: int = 0,
    estimated_usd: float = 0.0,
) -> Dict:
    payload = dict(ledger or {})
    payload.setdefault(
        "totals",
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "model_calls": 0,
            "web_calls": 0,
            "estimated_usd": 0.0,
        },
    )
    payload.setdefault("steps", [])
    entry = {
        "step": str(step),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "model_calls": int(model_calls),
        "web_calls": int(web_calls),
        "estimated_usd": float(estimated_usd),
    }
    payload["steps"].append(entry)
    payload["totals"]["prompt_tokens"] += int(prompt_tokens)
    payload["totals"]["completion_tokens"] += int(completion_tokens)
    payload["totals"]["model_calls"] += int(model_calls)
    payload["totals"]["web_calls"] += int(web_calls)
    payload["totals"]["estimated_usd"] += float(estimated_usd)
    return payload


def evaluate_stop_conditions(
    *,
    policy: SearchBudgetPolicy,
    counters: SearchBudgetCounters,
    start_time: float,
    confidence_history: Optional[list] = None,
) -> StopEvaluation:
    elapsed = max(0.0, time.time() - float(start_time))
    if elapsed >= float(policy.timeout_seconds):
        return StopEvaluation(should_stop=True, reason="timeout")
    if int(counters.depth) >= int(policy.max_depth):
        return StopEvaluation(should_stop=True, reason="max_depth")
    search_limit = effective_max_searches(
        policy=policy,
        counters=counters,
        confidence_history=confidence_history,
    )
    if int(counters.searches) >= int(search_limit):
        return StopEvaluation(should_stop=True, reason="max_searches")
    if int(counters.reranks) >= int(policy.max_reranks):
        return StopEvaluation(should_stop=True, reason="max_reranks")
    if int(counters.prompt_tokens) >= int(policy.max_prompt_tokens):
        return StopEvaluation(should_stop=True, reason="max_prompt_tokens")
    if int(counters.completion_tokens) >= int(policy.max_completion_tokens):
        return StopEvaluation(should_stop=True, reason="max_completion_tokens")
    if int(counters.web_calls) >= int(policy.max_web_calls) and int(policy.max_web_calls) > 0:
        return StopEvaluation(should_stop=True, reason="max_web_calls")

    confidence_history = confidence_history or []
    if policy.min_confidence is not None and len(confidence_history) > 0:
        if float(confidence_history[-1]) >= float(policy.min_confidence):
            return StopEvaluation(should_stop=True, reason="target_confidence_reached")
    if (
        policy.min_confidence_gain is not None
        and len(confidence_history) > int(policy.plateau_window)
    ):
        window = int(policy.plateau_window)
        gain = float(confidence_history[-1]) - float(confidence_history[-1 - window])
        if gain < float(policy.min_confidence_gain):
            return StopEvaluation(should_stop=True, reason="diminishing_return")
    return StopEvaluation(should_stop=False, reason=None)
