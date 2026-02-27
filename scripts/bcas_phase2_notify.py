#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import urllib.error
import urllib.request
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _post_webhook(url: str, payload: Dict[str, Any], timeout_s: float) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            return {
                "sent": True,
                "status_code": int(getattr(response, "status", 200) or 200),
                "reason": "ok",
            }
    except urllib.error.HTTPError as exc:
        return {
            "sent": False,
            "status_code": int(exc.code),
            "reason": f"http_error:{exc.reason}",
        }
    except Exception as exc:
        return {"sent": False, "status_code": 0, "reason": f"error:{exc}"}


def _load_secret_from_file(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


def _resolve_webhook_url(cli_webhook_url: str, cli_secret_file: Path | None) -> tuple[str, Dict[str, Any]]:
    # Highest precedence: explicit CLI URL.
    if cli_webhook_url.strip():
        return cli_webhook_url.strip(), {"source": "cli", "secret_file_used": None}

    # Next: direct environment URL.
    env_url = os.getenv("BCAS_PHASE2_NOTIFY_WEBHOOK", "").strip()
    if env_url:
        return env_url, {"source": "env", "secret_file_used": None}

    # Next: secret file from CLI or env.
    secret_file = cli_secret_file
    if secret_file is None:
        env_secret_file = os.getenv("BCAS_PHASE2_NOTIFY_WEBHOOK_SECRET_FILE", "").strip()
        if env_secret_file:
            secret_file = Path(env_secret_file)
    if secret_file is not None and secret_file.exists():
        secret_url = _load_secret_from_file(secret_file)
        if secret_url:
            return secret_url, {"source": "secret_file", "secret_file_used": str(secret_file)}

    return "", {"source": "none", "secret_file_used": str(secret_file) if secret_file else None}


def _signature_for_events(status: str, events: List[Dict[str, Any]]) -> str:
    reduced = [{"severity": e.get("severity"), "type": e.get("type"), "message": e.get("message")} for e in events]
    raw = json.dumps({"status": status, "events": reduced}, sort_keys=True)
    return sha256(raw.encode("utf-8")).hexdigest()


def _apply_dedupe_throttle(
    *,
    dedupe_key: str,
    status: str,
    events: List[Dict[str, Any]],
    state_file: Path,
    cooldown_seconds: int,
) -> tuple[bool, Dict[str, Any]]:
    now = time.time()
    signature = _signature_for_events(status, events)
    payload: Dict[str, Any] = {}
    if state_file.exists():
        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    dedupe = payload.get("dedupe", {}) if isinstance(payload.get("dedupe"), dict) else {}
    prior = dedupe.get(dedupe_key, {}) if isinstance(dedupe.get(dedupe_key), dict) else {}
    prior_signature = str(prior.get("signature", ""))
    prior_sent_at = float(prior.get("sent_at_unix", 0.0) or 0.0)
    elapsed = now - prior_sent_at
    throttle_active = (prior_signature == signature) and (elapsed < float(cooldown_seconds))
    info = {
        "dedupe_key": dedupe_key,
        "signature": signature,
        "prior_signature": prior_signature if prior_signature else None,
        "prior_sent_at_unix": prior_sent_at if prior_sent_at > 0 else None,
        "elapsed_seconds_since_prior": elapsed if prior_sent_at > 0 else None,
        "cooldown_seconds": int(cooldown_seconds),
        "throttled": throttle_active,
    }
    return (not throttle_active), info


def _update_dedupe_state(*, state_file: Path, dedupe_key: str, signature: str) -> None:
    payload: Dict[str, Any] = {}
    if state_file.exists():
        try:
            payload = json.loads(state_file.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    dedupe = payload.get("dedupe", {}) if isinstance(payload.get("dedupe"), dict) else {}
    dedupe[dedupe_key] = {"signature": signature, "sent_at_unix": time.time()}
    payload["dedupe"] = dedupe
    state_file.parent.mkdir(parents=True, exist_ok=True)
    state_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _severity_rank(severity: str) -> int:
    if severity == "critical":
        return 3
    if severity == "warn":
        return 2
    return 1


def _build_events(gate: Dict[str, Any], stress: Dict[str, Any], p95_threshold_ms: float, p99_threshold_ms: float) -> List[Dict[str, Any]]:
    events: List[Dict[str, Any]] = []

    gate_ok = bool(gate.get("gate_ok", False))
    checks_payload = gate.get("checks")
    failed_checks: List[str] = []
    if isinstance(checks_payload, dict):
        failed_checks = [str(key) for key, ok in checks_payload.items() if not bool(ok)]
    elif isinstance(checks_payload, list):
        for row in checks_payload:
            if not isinstance(row, dict):
                continue
            if not bool(row.get("ok", False)):
                failed_checks.append(str(row.get("id", "unknown")))
    if not gate_ok:
        events.append(
            {
                "severity": "critical",
                "type": "gate",
                "message": "Operator/release gate failed",
                "details": {
                    "failed_checks": failed_checks,
                    "raw_checks": checks_payload,
                },
            }
        )

    deltas = gate.get("deltas", {}) if isinstance(gate.get("deltas"), dict) else {}
    if len(deltas) == 0 and isinstance(gate.get("diagnostics"), dict):
        deltas = gate.get("diagnostics", {})
    exact_pass_delta = _float(deltas.get("exact_phrase_pass_rate"))
    overall_pass_delta = _float(deltas.get("overall_pass_rate"))
    if exact_pass_delta < 0.0:
        events.append(
            {
                "severity": "warn",
                "type": "quality",
                "message": "Exact-phrase pass rate regressed",
                "details": {"exact_phrase_pass_delta": exact_pass_delta},
            }
        )
    recall_delta = _float(deltas.get("recall_delta"))
    mrr_delta = _float(deltas.get("mrr_delta"))
    if recall_delta < 0.0:
        events.append(
            {
                "severity": "warn",
                "type": "quality",
                "message": "Recall delta regressed",
                "details": {"recall_delta": recall_delta},
            }
        )
    if mrr_delta < 0.0:
        events.append(
            {
                "severity": "warn",
                "type": "quality",
                "message": "MRR delta regressed",
                "details": {"mrr_delta": mrr_delta},
            }
        )
    if overall_pass_delta < 0.0:
        events.append(
            {
                "severity": "warn",
                "type": "quality",
                "message": "Overall operator pass rate regressed",
                "details": {"overall_pass_delta": overall_pass_delta},
            }
        )

    latency = stress.get("latency_ms", {}) if isinstance(stress.get("latency_ms"), dict) else {}
    throughput = stress.get("throughput", {}) if isinstance(stress.get("throughput"), dict) else {}
    p95_ms = _float(latency.get("p95"))
    p99_ms = _float(latency.get("p99"))
    error_rate = _float(throughput.get("error_rate"))

    if p95_ms > p95_threshold_ms:
        severity = "critical" if p95_ms > (1.25 * p95_threshold_ms) else "warn"
        events.append(
            {
                "severity": severity,
                "type": "latency",
                "message": "Stress p95 latency exceeded threshold",
                "details": {"p95_ms": p95_ms, "threshold_ms": p95_threshold_ms},
            }
        )

    if p99_ms > p99_threshold_ms:
        severity = "critical" if p99_ms > (1.25 * p99_threshold_ms) else "warn"
        events.append(
            {
                "severity": severity,
                "type": "latency",
                "message": "Stress p99 latency exceeded threshold",
                "details": {"p99_ms": p99_ms, "threshold_ms": p99_threshold_ms},
            }
        )

    if error_rate > 0.0:
        events.append(
            {
                "severity": "critical",
                "type": "errors",
                "message": "Stress run reported request errors",
                "details": {"error_rate": error_rate},
            }
        )

    if not events:
        events.append({"severity": "ok", "type": "health", "message": "Nightly checks healthy", "details": {}})
    return events


def _overall_status(events: List[Dict[str, Any]]) -> str:
    max_rank = max((_severity_rank(str(e.get("severity", "ok"))) for e in events), default=1)
    if max_rank >= 3:
        return "critical"
    if max_rank >= 2:
        return "warn"
    return "ok"


def main() -> int:
    parser = argparse.ArgumentParser(description="Send/store BCAS nightly gate + stress notifications.")
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument("--stress", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--webhook-url", type=str, default="")
    parser.add_argument("--webhook-secret-file", type=Path, default=None)
    parser.add_argument("--webhook-timeout-s", type=float, default=6.0)
    parser.add_argument("--p95-threshold-ms", type=float, default=4500.0)
    parser.add_argument("--p99-threshold-ms", type=float, default=5500.0)
    parser.add_argument("--dedupe-key", type=str, default="bcas_phase2_nightly")
    parser.add_argument("--cooldown-seconds", type=int, default=900)
    parser.add_argument(
        "--state-file",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_NOTIFY_STATE.json"),
    )
    parser.add_argument("--fail-on-critical", action="store_true")
    args = parser.parse_args()

    gate = _safe_load(args.gate)
    stress = _safe_load(args.stress)
    events = _build_events(
        gate=gate,
        stress=stress,
        p95_threshold_ms=float(args.p95_threshold_ms),
        p99_threshold_ms=float(args.p99_threshold_ms),
    )
    status = _overall_status(events)

    payload: Dict[str, Any] = {
        "generated_at_unix": time.time(),
        "gate_artifact": str(args.gate),
        "stress_artifact": str(args.stress),
        "thresholds": {
            "p95_ms": float(args.p95_threshold_ms),
            "p99_ms": float(args.p99_threshold_ms),
        },
        "status": status,
        "events": events,
    }

    # v1 routing policy for escalation targets.
    route = "none"
    if status == "critical":
        route = "oncall_pager"
    elif status == "warn":
        route = "ops_warn_channel"
    payload["routing"] = {"policy_version": "v1", "target": route}

    webhook_url, webhook_resolution = _resolve_webhook_url(args.webhook_url, args.webhook_secret_file)
    payload["webhook_resolution"] = webhook_resolution

    allow_send, dedupe_info = _apply_dedupe_throttle(
        dedupe_key=str(args.dedupe_key),
        status=status,
        events=events,
        state_file=args.state_file,
        cooldown_seconds=int(args.cooldown_seconds),
    )
    payload["dedupe"] = dedupe_info
    if status == "ok":
        # Always allow healthy status to reset heartbeat while keeping records.
        allow_send = True

    if webhook_url:
        if allow_send:
            payload["webhook"] = _post_webhook(webhook_url, payload, timeout_s=float(args.webhook_timeout_s))
            if payload["webhook"].get("sent", False):
                _update_dedupe_state(
                    state_file=args.state_file,
                    dedupe_key=str(args.dedupe_key),
                    signature=str(dedupe_info["signature"]),
                )
        else:
            payload["webhook"] = {"sent": False, "status_code": 0, "reason": "throttled"}
    else:
        payload["webhook"] = {"sent": False, "status_code": 0, "reason": "not_configured"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "status": status, "events": len(events), "webhook": payload["webhook"]}, indent=2))

    if args.fail_on_critical and status == "critical":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
