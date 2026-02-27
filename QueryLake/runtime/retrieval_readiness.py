from __future__ import annotations

from typing import Any, Dict, Optional


def evaluate_rollout_readiness(
    *,
    tests_green: bool,
    eval_smoke_pass: bool,
    eval_nightly_pass: bool,
    g0_1: Optional[Dict[str, Any]] = None,
    g0_3: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    coverage_ok = bool((g0_1 or {}).get("meets_gate_g0_1", False))
    latency_known = g0_3 is not None
    latency_ok = bool((g0_3 or {}).get("meets_gate_g0_3", False)) if latency_known else False

    checks = {
        "tests_green": bool(tests_green),
        "eval_smoke_pass": bool(eval_smoke_pass),
        "eval_nightly_pass": bool(eval_nightly_pass),
        "g0_1_coverage_ok": coverage_ok,
        "g0_3_latency_ok": latency_ok,
        "g0_3_latency_available": latency_known,
    }
    missing = []
    if not coverage_ok:
        missing.append("G0.1 coverage threshold not met or not measured")
    if not latency_known:
        missing.append("G0.3 latency baseline/candidate comparison not provided")
    elif not latency_ok:
        missing.append("G0.3 latency threshold not met")
    if not checks["tests_green"]:
        missing.append("test suite failing")
    if not checks["eval_smoke_pass"]:
        missing.append("smoke eval failing")
    if not checks["eval_nightly_pass"]:
        missing.append("nightly eval failing")

    go = len(missing) == 0
    return {
        "go_for_canary": go,
        "checks": checks,
        "blocking_reasons": missing,
    }

