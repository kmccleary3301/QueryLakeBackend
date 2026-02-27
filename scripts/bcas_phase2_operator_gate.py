#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

DEFAULT_BASELINE = Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_EVAL_2026-02-24.json")
DEFAULT_POINTER = Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_BASELINE_POINTER.json")
PROFILE_KEYS = ("per_dataset", "limit", "http_timeout_s")


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(payload: Dict[str, Any], group: str, key: str) -> float:
    section = payload.get(group, {})
    if not isinstance(section, dict):
        return 0.0
    value = section.get(key, 0.0)
    try:
        return float(value)
    except Exception:
        return 0.0


def _resolve_baseline(
    *,
    explicit_baseline: Path | None,
    baseline_pointer: Path,
    baseline_fallback: Path,
) -> tuple[Path, str]:
    if explicit_baseline is not None:
        if not explicit_baseline.exists():
            raise SystemExit(f"Baseline file missing: {explicit_baseline}")
        return explicit_baseline, "explicit"

    if baseline_pointer.exists():
        payload = _load(baseline_pointer)
        pointer_path = payload.get("baseline_path")
        if isinstance(pointer_path, str) and len(pointer_path.strip()) > 0:
            resolved = Path(pointer_path)
            if resolved.exists():
                return resolved, "pointer"

    # Bootstrap pointer with fallback for subsequent runs.
    baseline_pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer_payload = {
        "baseline_path": str(baseline_fallback),
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
        "source": "operator_gate_fallback_bootstrap",
    }
    baseline_pointer.write_text(json.dumps(pointer_payload, indent=2), encoding="utf-8")
    if not baseline_fallback.exists():
        raise SystemExit(f"Baseline file missing: {baseline_fallback}")
    return baseline_fallback, "fallback"


def _resolve_thresholds(
    *,
    min_overall_pass_delta: float,
    min_exact_pass_delta: float,
    max_overall_latency_regression_ms: float,
    max_exact_latency_regression_ms: float,
    policy_path: Path,
    use_policy_if_ready: bool,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    thresholds = {
        "min_overall_pass_delta": float(min_overall_pass_delta),
        "min_exact_pass_delta": float(min_exact_pass_delta),
        "max_overall_latency_regression_ms": float(max_overall_latency_regression_ms),
        "max_exact_latency_regression_ms": float(max_exact_latency_regression_ms),
    }
    policy_info: Dict[str, Any] = {
        "path": str(policy_path),
        "present": False,
        "used": False,
        "ready": False,
    }
    if not use_policy_if_ready or (not policy_path.exists()):
        return thresholds, policy_info

    policy_info["present"] = True
    payload = _load(policy_path)
    policy_node = payload.get("policy", {}) if isinstance(payload.get("policy"), dict) else {}
    recommended = policy_node.get("recommended_thresholds", {}) if isinstance(policy_node.get("recommended_thresholds"), dict) else {}
    ready = bool(policy_node.get("ready", False))
    policy_info["ready"] = ready
    if not ready:
        return thresholds, policy_info

    for key in list(thresholds.keys()):
        if key in recommended:
            try:
                thresholds[key] = float(recommended[key])
            except Exception:
                continue
    policy_info["used"] = True
    policy_info["recommended_thresholds"] = {k: recommended.get(k) for k in thresholds.keys()}
    return thresholds, policy_info


def _profiles_equal(left: Any, right: Any) -> bool:
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return float(left) == float(right)
    return left == right


def _extract_eval_profile(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = payload.get("params")
    if not isinstance(params, dict):
        return {}
    profile: Dict[str, Any] = {}
    for key in PROFILE_KEYS:
        if key in params:
            profile[key] = params[key]
    return profile


def _profile_mismatches(
    baseline_profile: Dict[str, Any],
    candidate_profile: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    mismatches: Dict[str, Dict[str, Any]] = {}
    for key in PROFILE_KEYS:
        if key not in baseline_profile or key not in candidate_profile:
            continue
        baseline_value = baseline_profile.get(key)
        candidate_value = candidate_profile.get(key)
        if not _profiles_equal(baseline_value, candidate_value):
            mismatches[key] = {
                "baseline": baseline_value,
                "candidate": candidate_value,
            }
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Gate operator regression against a baseline eval artifact.")
    parser.add_argument(
        "--baseline",
        type=Path,
        default=None,
    )
    parser.add_argument("--baseline-pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--baseline-fallback", type=Path, default=DEFAULT_BASELINE)
    parser.add_argument(
        "--candidate",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_EVAL_phrase_candidate_rerank_2026-02-24.json"),
    )
    parser.add_argument("--min-overall-pass-delta", type=float, default=0.0)
    parser.add_argument("--min-exact-pass-delta", type=float, default=0.0)
    parser.add_argument("--max-overall-latency-regression-ms", type=float, default=20.0)
    parser.add_argument("--max-exact-latency-regression-ms", type=float, default=20.0)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_GATE_POLICY_LATEST.json"),
    )
    parser.add_argument("--use-policy-if-ready", action="store_true")
    parser.add_argument(
        "--enforce-profile-match",
        action="store_true",
        help="Fail gate when baseline/candidate eval profiles mismatch on key params.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_GATE_2026-02-24.json"),
    )
    args = parser.parse_args()

    baseline_path, baseline_source = _resolve_baseline(
        explicit_baseline=args.baseline,
        baseline_pointer=args.baseline_pointer,
        baseline_fallback=args.baseline_fallback,
    )
    thresholds, policy_info = _resolve_thresholds(
        min_overall_pass_delta=float(args.min_overall_pass_delta),
        min_exact_pass_delta=float(args.min_exact_pass_delta),
        max_overall_latency_regression_ms=float(args.max_overall_latency_regression_ms),
        max_exact_latency_regression_ms=float(args.max_exact_latency_regression_ms),
        policy_path=args.policy,
        use_policy_if_ready=bool(args.use_policy_if_ready),
    )
    baseline = _load(baseline_path)
    candidate = _load(args.candidate)
    baseline_profile = _extract_eval_profile(baseline)
    candidate_profile = _extract_eval_profile(candidate)
    profile_mismatches = _profile_mismatches(baseline_profile, candidate_profile)
    profile_comparable = len(baseline_profile) > 0 and len(candidate_profile) > 0
    profile_enforced = bool(args.enforce_profile_match)
    profile_match_ok = (not profile_enforced) or (not profile_comparable) or (len(profile_mismatches) == 0)

    base_overall_pass = _metric(baseline, "overall", "pass_rate")
    cand_overall_pass = _metric(candidate, "overall", "pass_rate")
    base_overall_lat = _metric(baseline, "overall", "avg_latency_ms")
    cand_overall_lat = _metric(candidate, "overall", "avg_latency_ms")

    base_exact = baseline.get("by_operator_type", {}).get("exact_phrase", {})
    cand_exact = candidate.get("by_operator_type", {}).get("exact_phrase", {})
    base_exact_pass = float(base_exact.get("pass_rate", 0.0) or 0.0)
    cand_exact_pass = float(cand_exact.get("pass_rate", 0.0) or 0.0)
    base_exact_lat = float(base_exact.get("avg_latency_ms", 0.0) or 0.0)
    cand_exact_lat = float(cand_exact.get("avg_latency_ms", 0.0) or 0.0)

    overall_pass_delta = cand_overall_pass - base_overall_pass
    exact_pass_delta = cand_exact_pass - base_exact_pass
    overall_latency_delta = cand_overall_lat - base_overall_lat
    exact_latency_delta = cand_exact_lat - base_exact_lat

    checks = {
        "overall_pass_delta_ok": overall_pass_delta >= float(thresholds["min_overall_pass_delta"]),
        "exact_pass_delta_ok": exact_pass_delta >= float(thresholds["min_exact_pass_delta"]),
        "overall_latency_ok": overall_latency_delta <= float(thresholds["max_overall_latency_regression_ms"]),
        "exact_latency_ok": exact_latency_delta <= float(thresholds["max_exact_latency_regression_ms"]),
        "eval_profile_match_ok": profile_match_ok,
    }
    gate_ok = all(checks.values())

    payload = {
        "generated_at_unix": time.time(),
        "baseline": str(baseline_path),
        "baseline_source": baseline_source,
        "baseline_pointer": str(args.baseline_pointer),
        "candidate": str(args.candidate),
        "policy": policy_info,
        "eval_profile": {
            "keys": list(PROFILE_KEYS),
            "baseline": baseline_profile,
            "candidate": candidate_profile,
            "comparable": profile_comparable,
            "enforced": profile_enforced,
            "mismatches": profile_mismatches,
        },
        "thresholds": thresholds,
        "deltas": {
            "overall_pass_rate": overall_pass_delta,
            "exact_phrase_pass_rate": exact_pass_delta,
            "overall_avg_latency_ms": overall_latency_delta,
            "exact_phrase_avg_latency_ms": exact_latency_delta,
        },
        "baseline_metrics": {
            "overall_pass_rate": base_overall_pass,
            "exact_phrase_pass_rate": base_exact_pass,
            "overall_avg_latency_ms": base_overall_lat,
            "exact_phrase_avg_latency_ms": base_exact_lat,
        },
        "candidate_metrics": {
            "overall_pass_rate": cand_overall_pass,
            "exact_phrase_pass_rate": cand_exact_pass,
            "overall_avg_latency_ms": cand_overall_lat,
            "exact_phrase_avg_latency_ms": cand_exact_lat,
        },
        "checks": checks,
        "gate_ok": gate_ok,
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "gate_ok": gate_ok, "deltas": payload["deltas"]}, indent=2))
    return 0 if gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
