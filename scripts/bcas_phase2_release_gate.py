#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_POLICY = {
    "version": "v1",
    "name": "strict_queue_confirm",
    "thresholds": {
        "min_recall_delta": -0.01,
        "min_mrr_delta": -0.01,
        "max_eval_latency_ratio": 1.10,
        "max_stress_p95_ratio": 1.10,
        "max_stress_p99_ratio": 1.15,
        "min_success_rps_ratio": 0.90,
        "max_error_rate": 0.0,
    },
}


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        return payload
    return {}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _extract_eval_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    overall = payload.get("metrics", {}).get("overall") if isinstance(payload.get("metrics"), dict) else None
    if not isinstance(overall, dict):
        overall = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    overall = overall if isinstance(overall, dict) else {}
    return {
        "recall_at_k": _safe_float(overall.get("recall_at_k")),
        "mrr": _safe_float(overall.get("mrr")),
        "avg_response_ms": _safe_float(overall.get("avg_response_ms")),
    }


def _extract_stress_metrics(payload: Dict[str, Any]) -> Dict[str, float]:
    latency = payload.get("latency_ms", {}) if isinstance(payload.get("latency_ms"), dict) else {}
    throughput = payload.get("throughput", {}) if isinstance(payload.get("throughput"), dict) else {}
    counts = payload.get("counts", {}) if isinstance(payload.get("counts"), dict) else {}
    return {
        "p95_ms": _safe_float(latency.get("p95")),
        "p99_ms": _safe_float(latency.get("p99")),
        "mean_ms": _safe_float(latency.get("mean")),
        "successful_requests_per_second": _safe_float(throughput.get("successful_requests_per_second")),
        "error_rate": _safe_float(throughput.get("error_rate", counts.get("error_rate"))),
    }


def _ratio(numerator: float, denominator: float, default: float = 1.0) -> float:
    if denominator <= 0:
        return float(default)
    return float(numerator) / float(denominator)


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# BCAS Phase2 Unified Release Gate")
    lines.append("")
    lines.append(f"- `gate_ok`: `{report.get('gate_ok')}`")
    lines.append(f"- `policy`: `{report.get('policy_name')}`")
    lines.append("")
    lines.append("## Checks")
    for check in report.get("checks", []):
        if not isinstance(check, dict):
            continue
        status = "PASS" if bool(check.get("ok")) else "FAIL"
        lines.append(f"- **{status}** `{check.get('id')}`: {check.get('message')}")
    lines.append("")
    lines.append("## Diagnostic Summary")
    for key, value in (report.get("diagnostics") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Unified strict release gate across eval + stress artifacts.")
    parser.add_argument("--baseline-eval", type=Path, required=True)
    parser.add_argument("--candidate-eval", type=Path, required=True)
    parser.add_argument("--baseline-stress", type=Path, required=True)
    parser.add_argument("--candidate-stress", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=None)
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--out-md", type=Path, required=True)
    parser.add_argument("--fail-on-gate", action="store_true")
    args = parser.parse_args()

    policy = dict(DEFAULT_POLICY)
    if args.policy is not None and args.policy.exists():
        loaded = _load_json(args.policy)
        if isinstance(loaded, dict):
            policy.update({k: v for k, v in loaded.items() if k != "thresholds"})
            thresholds = dict(policy.get("thresholds", {}))
            thresholds.update((loaded.get("thresholds") or {}))
            policy["thresholds"] = thresholds
    thresholds = policy.get("thresholds", {})

    baseline_eval = _extract_eval_metrics(_load_json(args.baseline_eval))
    candidate_eval = _extract_eval_metrics(_load_json(args.candidate_eval))
    baseline_stress = _extract_stress_metrics(_load_json(args.baseline_stress))
    candidate_stress = _extract_stress_metrics(_load_json(args.candidate_stress))

    recall_delta = candidate_eval["recall_at_k"] - baseline_eval["recall_at_k"]
    mrr_delta = candidate_eval["mrr"] - baseline_eval["mrr"]
    eval_latency_ratio = _ratio(candidate_eval["avg_response_ms"], baseline_eval["avg_response_ms"])
    stress_p95_ratio = _ratio(candidate_stress["p95_ms"], baseline_stress["p95_ms"])
    stress_p99_ratio = _ratio(candidate_stress["p99_ms"], baseline_stress["p99_ms"])
    success_rps_ratio = _ratio(candidate_stress["successful_requests_per_second"], baseline_stress["successful_requests_per_second"])
    error_rate = candidate_stress["error_rate"]

    checks: List[Dict[str, Any]] = []
    checks.append(
        {
            "id": "eval_recall_delta",
            "ok": recall_delta >= _safe_float(thresholds.get("min_recall_delta"), -0.01),
            "message": f"recall_delta={recall_delta:.6f} threshold={_safe_float(thresholds.get('min_recall_delta')):.6f}",
        }
    )
    checks.append(
        {
            "id": "eval_mrr_delta",
            "ok": mrr_delta >= _safe_float(thresholds.get("min_mrr_delta"), -0.01),
            "message": f"mrr_delta={mrr_delta:.6f} threshold={_safe_float(thresholds.get('min_mrr_delta')):.6f}",
        }
    )
    checks.append(
        {
            "id": "eval_latency_ratio",
            "ok": eval_latency_ratio <= _safe_float(thresholds.get("max_eval_latency_ratio"), 1.10),
            "message": f"eval_latency_ratio={eval_latency_ratio:.6f} threshold={_safe_float(thresholds.get('max_eval_latency_ratio')):.6f}",
        }
    )
    checks.append(
        {
            "id": "stress_p95_ratio",
            "ok": stress_p95_ratio <= _safe_float(thresholds.get("max_stress_p95_ratio"), 1.10),
            "message": f"stress_p95_ratio={stress_p95_ratio:.6f} threshold={_safe_float(thresholds.get('max_stress_p95_ratio')):.6f}",
        }
    )
    checks.append(
        {
            "id": "stress_p99_ratio",
            "ok": stress_p99_ratio <= _safe_float(thresholds.get("max_stress_p99_ratio"), 1.15),
            "message": f"stress_p99_ratio={stress_p99_ratio:.6f} threshold={_safe_float(thresholds.get('max_stress_p99_ratio')):.6f}",
        }
    )
    checks.append(
        {
            "id": "stress_success_rps_ratio",
            "ok": success_rps_ratio >= _safe_float(thresholds.get("min_success_rps_ratio"), 0.90),
            "message": f"success_rps_ratio={success_rps_ratio:.6f} threshold={_safe_float(thresholds.get('min_success_rps_ratio')):.6f}",
        }
    )
    checks.append(
        {
            "id": "stress_error_rate",
            "ok": error_rate <= _safe_float(thresholds.get("max_error_rate"), 0.0),
            "message": f"error_rate={error_rate:.6f} threshold={_safe_float(thresholds.get('max_error_rate')):.6f}",
        }
    )

    gate_ok = all(bool(check.get("ok")) for check in checks)
    failed = [check for check in checks if not bool(check.get("ok"))]

    report: Dict[str, Any] = {
        "generated_at_unix": time.time(),
        "policy_name": policy.get("name", "strict_queue_confirm"),
        "policy_version": policy.get("version", "v1"),
        "thresholds": thresholds,
        "baseline": {"eval": baseline_eval, "stress": baseline_stress},
        "candidate": {"eval": candidate_eval, "stress": candidate_stress},
        "diagnostics": {
            "recall_delta": recall_delta,
            "mrr_delta": mrr_delta,
            "eval_latency_ratio": eval_latency_ratio,
            "stress_p95_ratio": stress_p95_ratio,
            "stress_p99_ratio": stress_p99_ratio,
            "success_rps_ratio": success_rps_ratio,
            "error_rate": error_rate,
            "failed_checks": [check.get("id") for check in failed],
        },
        "checks": checks,
        "gate_ok": gate_ok,
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.out_md.write_text(_build_markdown(report), encoding="utf-8")
    print(json.dumps({"gate_ok": gate_ok, "out_json": str(args.out_json), "out_md": str(args.out_md)}, indent=2))

    if args.fail_on_gate and not gate_ok:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
