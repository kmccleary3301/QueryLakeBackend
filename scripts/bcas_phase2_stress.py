#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
from concurrent.futures import TimeoutError as FuturesTimeout


def _load_queries(cases_path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    out: List[Dict[str, Any]] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        query = row.get("query")
        dataset = row.get("dataset")
        if isinstance(query, str) and len(query.strip()) > 0 and isinstance(dataset, str):
            out.append({"dataset": dataset, "query": query.strip()})
    return out


def _percentile(values: List[float], p: float) -> float:
    if len(values) == 0:
        return 0.0
    values_sorted = sorted(values)
    idx = int(round((len(values_sorted) - 1) * p))
    return float(values_sorted[max(0, min(idx, len(values_sorted) - 1))])


def _to_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if len(text) == 0:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _is_total_key(key: str) -> bool:
    normalized = "".join(ch for ch in key.lower() if ch.isalnum())
    return normalized in {
        "total",
        "overall",
        "requesttotal",
        "durationtotal",
        "totalms",
        "durationms",
        "latencyms",
        "elapsedms",
    }


def _extract_duration_rows(duration_value: Any, *, seconds_default: bool) -> Dict[str, float]:
    durations: Dict[str, float] = {}
    if isinstance(duration_value, dict):
        for key, value in duration_value.items():
            if not isinstance(key, str):
                continue
            numeric = _to_float(value)
            if numeric is None:
                continue
            if seconds_default:
                numeric *= 1000.0
            durations[key] = numeric
        return durations
    if isinstance(duration_value, list):
        for item in duration_value:
            if not isinstance(item, dict):
                continue
            stage = item.get("stage")
            if not isinstance(stage, str) or len(stage.strip()) == 0:
                stage = item.get("name")
            if not isinstance(stage, str) or len(stage.strip()) == 0:
                continue
            numeric = _to_float(item.get("duration_ms"))
            if numeric is None:
                numeric = _to_float(item.get("duration"))
                if numeric is not None and seconds_default:
                    numeric *= 1000.0
            if numeric is None:
                continue
            durations[stage] = numeric
    return durations


def _extract_server_duration_map(result: Any) -> Dict[str, float]:
    if not isinstance(result, dict):
        return {}
    durations: Dict[str, float] = {}
    for key in ("duration", "durations", "timings"):
        rows = _extract_duration_rows(result.get(key), seconds_default=True)
        if rows:
            durations.update(rows)
    traces = result.get("traces")
    if isinstance(traces, list):
        for item in traces:
            if not isinstance(item, dict):
                continue
            stage = item.get("stage")
            if not isinstance(stage, str) or len(stage.strip()) == 0:
                stage = item.get("name")
            if not isinstance(stage, str) or len(stage.strip()) == 0:
                continue
            numeric = _to_float(item.get("duration_ms"))
            if numeric is None:
                numeric = _to_float(item.get("duration"))
                if numeric is not None:
                    numeric *= 1000.0
            if numeric is None:
                continue
            durations[stage] = numeric
    return durations


def _resolve_server_total_ms(durations: Dict[str, float]) -> Optional[float]:
    if len(durations) == 0:
        return None
    for key, value in durations.items():
        if _is_total_key(key):
            return float(value)
    non_total_values = [float(value) for key, value in durations.items() if not _is_total_key(key)]
    if len(non_total_values) == 0:
        return None
    return float(sum(non_total_values))


def _summary_stats(values: List[float]) -> Dict[str, Any]:
    return {
        "samples": int(len(values)),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "mean": float(statistics.mean(values)) if len(values) > 0 else 0.0,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _compute_stress_gate(
    *,
    candidate_payload: Dict[str, Any],
    baseline_payload: Optional[Dict[str, Any]],
    max_p95_ratio: float,
    max_p99_ratio: float,
    min_success_rps_ratio: float,
    max_error_rate: float,
) -> Dict[str, Any]:
    cand_latency = candidate_payload.get("latency_ms", {}) if isinstance(candidate_payload.get("latency_ms"), dict) else {}
    cand_throughput = candidate_payload.get("throughput", {}) if isinstance(candidate_payload.get("throughput"), dict) else {}
    cand_counts = candidate_payload.get("counts", {}) if isinstance(candidate_payload.get("counts"), dict) else {}
    cand_p95 = _safe_float(cand_latency.get("p95"))
    cand_p99 = _safe_float(cand_latency.get("p99"))
    cand_rps = _safe_float(cand_throughput.get("successful_requests_per_second"))
    cand_error_rate = _safe_float(cand_counts.get("error_rate"))

    base_p95 = 0.0
    base_p99 = 0.0
    base_rps = 0.0
    baseline_available = False
    if isinstance(baseline_payload, dict) and len(baseline_payload) > 0:
        base_latency = baseline_payload.get("latency_ms", {}) if isinstance(baseline_payload.get("latency_ms"), dict) else {}
        base_throughput = baseline_payload.get("throughput", {}) if isinstance(baseline_payload.get("throughput"), dict) else {}
        base_p95 = _safe_float(base_latency.get("p95"))
        base_p99 = _safe_float(base_latency.get("p99"))
        base_rps = _safe_float(base_throughput.get("successful_requests_per_second"))
        baseline_available = True

    p95_ratio = 1.0 if base_p95 <= 0 else (cand_p95 / base_p95)
    p99_ratio = 1.0 if base_p99 <= 0 else (cand_p99 / base_p99)
    success_rps_ratio = 1.0 if base_rps <= 0 else (cand_rps / base_rps)
    checks: Dict[str, Optional[bool]] = {
        "error_rate_ok": cand_error_rate <= float(max_error_rate),
        "p95_ratio_ok": (p95_ratio <= float(max_p95_ratio)) if baseline_available else None,
        "p99_ratio_ok": (p99_ratio <= float(max_p99_ratio)) if baseline_available else None,
        "success_rps_ratio_ok": (success_rps_ratio >= float(min_success_rps_ratio)) if baseline_available else None,
    }
    baseline_checks = [value for key, value in checks.items() if key != "error_rate_ok" and value is not None]
    gate_ok = bool(checks["error_rate_ok"]) and (all(bool(v) for v in baseline_checks) if baseline_checks else True)

    return {
        "baseline_available": baseline_available,
        "gate_ok": gate_ok,
        "thresholds": {
            "max_p95_ratio": float(max_p95_ratio),
            "max_p99_ratio": float(max_p99_ratio),
            "min_success_rps_ratio": float(min_success_rps_ratio),
            "max_error_rate": float(max_error_rate),
        },
        "candidate": {
            "p95_ms": cand_p95,
            "p99_ms": cand_p99,
            "successful_requests_per_second": cand_rps,
            "error_rate": cand_error_rate,
        },
        "baseline": {
            "p95_ms": base_p95,
            "p99_ms": base_p99,
            "successful_requests_per_second": base_rps,
        },
        "deltas": {
            "p95_ms": cand_p95 - base_p95,
            "p99_ms": cand_p99 - base_p99,
            "successful_requests_per_second": cand_rps - base_rps,
            "error_rate": cand_error_rate,
            "p95_ratio": p95_ratio,
            "p99_ratio": p99_ratio,
            "success_rps_ratio": success_rps_ratio,
        },
        "checks": checks,
    }


def _default_gate_out(stress_out: Path) -> Path:
    stem = stress_out.stem
    if stem.endswith(".json"):
        stem = stem[:-5]
    return stress_out.with_name(f"{stem}_GATE.json")


def _default_gate_report_out(stress_out: Path) -> Path:
    stem = stress_out.stem
    if stem.endswith(".json"):
        stem = stem[:-5]
    return stress_out.with_name(f"{stem}_GATE_REPORT.md")


def _render_stress_gate_report(
    *,
    gate_payload: Dict[str, Any],
    candidate_stress: Path,
    baseline_stress: Optional[Path],
    gate_out: Path,
) -> str:
    checks = gate_payload.get("checks", {}) if isinstance(gate_payload.get("checks"), dict) else {}
    candidate = gate_payload.get("candidate", {}) if isinstance(gate_payload.get("candidate"), dict) else {}
    baseline = gate_payload.get("baseline", {}) if isinstance(gate_payload.get("baseline"), dict) else {}
    deltas = gate_payload.get("deltas", {}) if isinstance(gate_payload.get("deltas"), dict) else {}
    thresholds = gate_payload.get("thresholds", {}) if isinstance(gate_payload.get("thresholds"), dict) else {}
    baseline_available = bool(gate_payload.get("baseline_available"))
    gate_ok = gate_payload.get("gate_ok")

    lines = [
        "# BCAS Phase2 Stress Gate Report",
        "",
        f"- `gate_ok`: `{gate_ok}`",
        f"- `baseline_available`: `{baseline_available}`",
        f"- `candidate_stress`: `{candidate_stress}`",
        f"- `baseline_stress`: `{baseline_stress}`",
        f"- `gate_json`: `{gate_out}`",
        "",
        "## Thresholds",
        "",
        f"- `max_p95_ratio`: `{thresholds.get('max_p95_ratio')}`",
        f"- `max_p99_ratio`: `{thresholds.get('max_p99_ratio')}`",
        f"- `min_success_rps_ratio`: `{thresholds.get('min_success_rps_ratio')}`",
        f"- `max_error_rate`: `{thresholds.get('max_error_rate')}`",
        "",
        "## Candidate",
        "",
        f"- `p95_ms`: `{candidate.get('p95_ms')}`",
        f"- `p99_ms`: `{candidate.get('p99_ms')}`",
        f"- `successful_requests_per_second`: `{candidate.get('successful_requests_per_second')}`",
        f"- `error_rate`: `{candidate.get('error_rate')}`",
        "",
        "## Baseline",
        "",
        f"- `p95_ms`: `{baseline.get('p95_ms')}`",
        f"- `p99_ms`: `{baseline.get('p99_ms')}`",
        f"- `successful_requests_per_second`: `{baseline.get('successful_requests_per_second')}`",
        "",
        "## Deltas",
        "",
        f"- `p95_ms`: `{deltas.get('p95_ms')}`",
        f"- `p99_ms`: `{deltas.get('p99_ms')}`",
        f"- `successful_requests_per_second`: `{deltas.get('successful_requests_per_second')}`",
        f"- `error_rate`: `{deltas.get('error_rate')}`",
        f"- `p95_ratio`: `{deltas.get('p95_ratio')}`",
        f"- `p99_ratio`: `{deltas.get('p99_ratio')}`",
        f"- `success_rps_ratio`: `{deltas.get('success_rps_ratio')}`",
        "",
        "## Checks",
        "",
        f"- `error_rate_ok`: `{checks.get('error_rate_ok')}`",
        f"- `p95_ratio_ok`: `{checks.get('p95_ratio_ok')}`",
        f"- `p99_ratio_ok`: `{checks.get('p99_ratio_ok')}`",
        f"- `success_rps_ratio_ok`: `{checks.get('success_rps_ratio_ok')}`",
        "",
    ]
    return "\n".join(lines)


def _one_call(
    *,
    base_url: str,
    api_key: str,
    collection_id: str,
    query: str,
    limit_bm25: int,
    limit_similarity: int,
    limit_sparse: int,
    bm25_weight: float,
    similarity_weight: float,
    sparse_weight: float,
    use_sparse: bool,
    sparse_dimensions: int,
    sparse_embedding_function: str,
    queue_admission_concurrency_limit: int,
    queue_admission_ttl_seconds: int,
    queue_throttle_soft_utilization: float,
    queue_throttle_hard_utilization: float,
    queue_throttle_soft_scale: float,
    queue_throttle_hard_scale: float,
    queue_throttle_disable_sparse_at_hard: bool,
    timeout_s: int,
) -> Dict[str, Any]:
    payload = {
        "auth": {"api_key": api_key},
        "query": {
            "bm25": query,
            "embedding": query,
            "sparse": query,
            "queue_aware_admission": True,
            "queue_admission_concurrency_limit": int(queue_admission_concurrency_limit),
            "queue_admission_ttl_seconds": int(queue_admission_ttl_seconds),
            "queue_throttle_enabled": True,
            "queue_throttle_soft_utilization": float(queue_throttle_soft_utilization),
            "queue_throttle_hard_utilization": float(queue_throttle_hard_utilization),
            "queue_throttle_soft_scale": float(queue_throttle_soft_scale),
            "queue_throttle_hard_scale": float(queue_throttle_hard_scale),
            "queue_throttle_disable_sparse_at_hard": bool(queue_throttle_disable_sparse_at_hard),
        },
        "collection_ids": [collection_id],
        "limit_bm25": int(limit_bm25),
        "limit_similarity": int(limit_similarity),
        "limit_sparse": int(limit_sparse),
        "bm25_weight": float(bm25_weight),
        "similarity_weight": float(similarity_weight),
        "sparse_weight": float(sparse_weight),
        "use_sparse": bool(use_sparse),
        "sparse_dimensions": int(sparse_dimensions),
        "sparse_embedding_function": str(sparse_embedding_function),
    }
    t0 = time.perf_counter()
    try:
        resp = requests.get(f"{base_url.rstrip('/')}/api/search_hybrid", json=payload, timeout=timeout_s)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        body = resp.json()
        if body.get("success") is False:
            return {"ok": False, "latency_ms": latency_ms, "error": str(body.get("error") or body.get("note"))}
        result = body.get("result")
        rows = result.get("rows", []) if isinstance(result, dict) else []
        durations = _extract_server_duration_map(result)
        return {
            "ok": True,
            "latency_ms": latency_ms,
            "rows": len(rows) if isinstance(rows, list) else 0,
            "server_duration_ms": durations,
            "server_total_ms": _resolve_server_total_ms(durations),
        }
    except Exception as exc:
        latency_ms = (time.perf_counter() - t0) * 1000.0
        return {"ok": False, "latency_ms": latency_ms, "error": str(exc)}


def main() -> int:
    parser = argparse.ArgumentParser(description="Concurrent BCAS retrieval stress benchmark.")
    parser.add_argument(
        "--account-config",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json"),
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_LIVE_EVAL_CASES_BASELINE_2026-02-24.json"),
    )
    parser.add_argument("--duration-s", type=int, default=120)
    parser.add_argument("--concurrency", type=int, default=12)
    parser.add_argument("--limit-bm25", type=int, default=12)
    parser.add_argument("--limit-similarity", type=int, default=12)
    parser.add_argument("--limit-sparse", type=int, default=0)
    parser.add_argument("--bm25-weight", type=float, default=0.9)
    parser.add_argument("--similarity-weight", type=float, default=0.1)
    parser.add_argument("--sparse-weight", type=float, default=0.0)
    parser.add_argument("--use-sparse", action="store_true")
    parser.add_argument("--sparse-dimensions", type=int, default=1024)
    parser.add_argument("--sparse-embedding-function", type=str, default="embedding_sparse")
    parser.add_argument("--queue-admission-concurrency-limit", type=int, default=12)
    parser.add_argument("--queue-admission-ttl-seconds", type=int, default=120)
    parser.add_argument("--queue-throttle-soft-utilization", type=float, default=0.75)
    parser.add_argument("--queue-throttle-hard-utilization", type=float, default=0.90)
    parser.add_argument("--queue-throttle-soft-scale", type=float, default=0.75)
    parser.add_argument("--queue-throttle-hard-scale", type=float, default=0.50)
    parser.add_argument("--queue-throttle-disable-sparse-at-hard", action="store_true", dest="queue_throttle_disable_sparse_at_hard")
    parser.add_argument("--no-queue-throttle-disable-sparse-at-hard", action="store_false", dest="queue_throttle_disable_sparse_at_hard")
    parser.set_defaults(queue_throttle_disable_sparse_at_hard=True)
    parser.add_argument("--http-timeout-s", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260224)
    parser.add_argument("--baseline-stress", type=Path, default=None, help="Optional baseline stress JSON for gate deltas.")
    parser.add_argument("--gate-max-p95-ratio", type=float, default=1.10)
    parser.add_argument("--gate-max-p99-ratio", type=float, default=1.15)
    parser.add_argument("--gate-min-success-rps-ratio", type=float, default=0.90)
    parser.add_argument("--gate-max-error-rate", type=float, default=0.0)
    parser.add_argument("--gate-out", type=Path, default=None)
    parser.add_argument("--gate-report-out", type=Path, default=None)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_STRESS_2026-02-24.json"),
    )
    args = parser.parse_args()

    cfg = json.loads(args.account_config.read_text(encoding="utf-8"))
    base_url = str(cfg.get("api_base_url", "http://localhost:8000"))
    api_key = str(cfg.get("api_key", "")).strip()
    collections = cfg.get("collections", {})
    if len(api_key) == 0 or not isinstance(collections, dict):
        raise SystemExit("invalid account config")

    queries = _load_queries(args.cases)
    if len(queries) == 0:
        raise SystemExit("no queries in cases file")

    dataset_to_collection = {
        "hotpotqa": collections.get("hotpotqa"),
        "multihop": collections.get("multihop"),
        "triviaqa": collections.get("triviaqa"),
    }

    rng = random.Random(int(args.seed))
    started = time.perf_counter()
    end_time = started + max(5, int(args.duration_s))
    latencies: List[float] = []
    ok_count = 0
    err_count = 0
    rows_counts: List[int] = []
    server_total_ms: List[float] = []
    queueing_estimate_ms: List[float] = []
    stage_ms: Dict[str, List[float]] = {}
    duration_samples = 0
    errors: Dict[str, int] = {}
    submitted = 0
    inferred_use_sparse = bool(int(args.limit_sparse) > 0 and float(args.sparse_weight) > 0.0)
    use_sparse = bool(args.use_sparse or inferred_use_sparse)

    with ThreadPoolExecutor(max_workers=max(1, int(args.concurrency))) as pool:
        futures = set()
        while time.perf_counter() < end_time or len(futures) > 0:
            while time.perf_counter() < end_time and len(futures) < max(1, int(args.concurrency)):
                row = rng.choice(queries)
                dataset = row["dataset"]
                collection_id = dataset_to_collection.get(dataset)
                if not isinstance(collection_id, str) or len(collection_id) == 0:
                    continue
                fut = pool.submit(
                    _one_call,
                    base_url=base_url,
                    api_key=api_key,
                    collection_id=collection_id,
                    query=row["query"],
                    limit_bm25=int(args.limit_bm25),
                    limit_similarity=int(args.limit_similarity),
                    limit_sparse=int(args.limit_sparse),
                    bm25_weight=float(args.bm25_weight),
                    similarity_weight=float(args.similarity_weight),
                    sparse_weight=float(args.sparse_weight),
                    use_sparse=bool(use_sparse),
                    sparse_dimensions=int(args.sparse_dimensions),
                    sparse_embedding_function=str(args.sparse_embedding_function),
                    queue_admission_concurrency_limit=int(args.queue_admission_concurrency_limit),
                    queue_admission_ttl_seconds=int(args.queue_admission_ttl_seconds),
                    queue_throttle_soft_utilization=float(args.queue_throttle_soft_utilization),
                    queue_throttle_hard_utilization=float(args.queue_throttle_hard_utilization),
                    queue_throttle_soft_scale=float(args.queue_throttle_soft_scale),
                    queue_throttle_hard_scale=float(args.queue_throttle_hard_scale),
                    queue_throttle_disable_sparse_at_hard=bool(args.queue_throttle_disable_sparse_at_hard),
                    timeout_s=int(args.http_timeout_s),
                )
                futures.add(fut)
                submitted += 1
            if len(futures) == 0:
                time.sleep(0.01)
                continue
            done = []
            try:
                for fut in as_completed(futures, timeout=1):
                    done.append(fut)
                    break
            except FuturesTimeout:
                done = []
            for fut in done:
                futures.remove(fut)
                result = fut.result()
                latencies.append(float(result.get("latency_ms", 0.0)))
                if result.get("ok"):
                    ok_count += 1
                    rows_counts.append(int(result.get("rows", 0)))
                    total_ms = _to_float(result.get("server_total_ms"))
                    if total_ms is not None and total_ms >= 0.0:
                        duration_samples += 1
                        server_total_ms.append(total_ms)
                        queueing_estimate_ms.append(max(0.0, float(result.get("latency_ms", 0.0)) - total_ms))
                    duration_map = result.get("server_duration_ms")
                    if isinstance(duration_map, dict):
                        for stage, value in duration_map.items():
                            if not isinstance(stage, str) or _is_total_key(stage):
                                continue
                            numeric = _to_float(value)
                            if numeric is None:
                                continue
                            if stage not in stage_ms:
                                stage_ms[stage] = []
                            stage_ms[stage].append(numeric)
                else:
                    err_count += 1
                    err = str(result.get("error", "error"))
                    errors[err] = int(errors.get(err, 0)) + 1

    runtime_s = max(0.001, time.perf_counter() - started)
    payload = {
        "generated_at_unix": time.time(),
        "params": {
            "duration_s": int(args.duration_s),
            "concurrency": int(args.concurrency),
            "limit_bm25": int(args.limit_bm25),
            "limit_similarity": int(args.limit_similarity),
            "limit_sparse": int(args.limit_sparse),
            "bm25_weight": float(args.bm25_weight),
            "similarity_weight": float(args.similarity_weight),
            "sparse_weight": float(args.sparse_weight),
            "use_sparse": bool(use_sparse),
            "sparse_dimensions": int(args.sparse_dimensions),
            "sparse_embedding_function": str(args.sparse_embedding_function),
            "queue_admission_concurrency_limit": int(args.queue_admission_concurrency_limit),
            "queue_admission_ttl_seconds": int(args.queue_admission_ttl_seconds),
            "queue_throttle_soft_utilization": float(args.queue_throttle_soft_utilization),
            "queue_throttle_hard_utilization": float(args.queue_throttle_hard_utilization),
            "queue_throttle_soft_scale": float(args.queue_throttle_soft_scale),
            "queue_throttle_hard_scale": float(args.queue_throttle_hard_scale),
            "queue_throttle_disable_sparse_at_hard": bool(args.queue_throttle_disable_sparse_at_hard),
            "http_timeout_s": int(args.http_timeout_s),
            "cases": str(args.cases),
        },
        "counts": {
            "submitted": int(submitted),
            "ok": int(ok_count),
            "errors": int(err_count),
            "error_rate": float(err_count / max(1, submitted)),
        },
        "throughput": {
            "requests_per_second": float(submitted / runtime_s),
            "successful_requests_per_second": float(ok_count / runtime_s),
        },
        "latency_ms": {
            "p50": _percentile(latencies, 0.50),
            "p95": _percentile(latencies, 0.95),
            "p99": _percentile(latencies, 0.99),
            "mean": float(statistics.mean(latencies)) if len(latencies) > 0 else 0.0,
        },
        "server_timing": {
            "duration_samples": int(duration_samples),
            "duration_coverage_rate": float(duration_samples / max(1, ok_count)),
            "server_total_ms": _summary_stats(server_total_ms),
            "queueing_estimate_ms": _summary_stats(queueing_estimate_ms),
            "stage_ms": {key: _summary_stats(values) for key, values in sorted(stage_ms.items())},
        },
        "rows_returned": {
            "mean": float(statistics.mean(rows_counts)) if len(rows_counts) > 0 else 0.0,
        },
        "top_errors": sorted(
            [{"error": k, "count": v} for k, v in errors.items()],
            key=lambda x: x["count"],
            reverse=True,
        )[:20],
    }
    baseline_payload = _load_json(args.baseline_stress) if args.baseline_stress is not None and args.baseline_stress.exists() else None
    gate_payload = _compute_stress_gate(
        candidate_payload=payload,
        baseline_payload=baseline_payload,
        max_p95_ratio=float(args.gate_max_p95_ratio),
        max_p99_ratio=float(args.gate_max_p99_ratio),
        min_success_rps_ratio=float(args.gate_min_success_rps_ratio),
        max_error_rate=float(args.gate_max_error_rate),
    )
    gate_payload["generated_at_unix"] = time.time()
    gate_payload["candidate_stress"] = str(args.out)
    gate_payload["baseline_stress"] = str(args.baseline_stress) if args.baseline_stress is not None else None

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    gate_out = args.gate_out if args.gate_out is not None else _default_gate_out(args.out)
    gate_out.parent.mkdir(parents=True, exist_ok=True)
    gate_out.write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
    gate_report_out = args.gate_report_out if args.gate_report_out is not None else _default_gate_report_out(args.out)
    gate_report_out.parent.mkdir(parents=True, exist_ok=True)
    gate_report_out.write_text(
        _render_stress_gate_report(
            gate_payload=gate_payload,
            candidate_stress=args.out,
            baseline_stress=args.baseline_stress,
            gate_out=gate_out,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "out": str(args.out),
                "gate_out": str(gate_out),
                "gate_report_out": str(gate_report_out),
                "counts": payload["counts"],
                "latency_ms": payload["latency_ms"],
                "server_timing": {
                    "duration_coverage_rate": payload["server_timing"]["duration_coverage_rate"],
                    "server_total_p50_ms": payload["server_timing"]["server_total_ms"]["p50"],
                    "queueing_p50_ms": payload["server_timing"]["queueing_estimate_ms"]["p50"],
                },
                "gate_ok": gate_payload.get("gate_ok"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
