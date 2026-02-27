#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return int(proc.returncode), proc.stdout, proc.stderr


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else {}


def _default_gate_report_out(run_out: Path) -> Path:
    stem = run_out.stem
    if stem.endswith(".json"):
        stem = stem[:-5]
    return run_out.with_name(f"{stem}_GATE_REPORT.md")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _extract_eval_overall(metrics_payload: Dict[str, Any]) -> Dict[str, float]:
    overall = (metrics_payload.get("metrics") or {}).get("overall") if isinstance(metrics_payload, dict) else {}
    overall = overall if isinstance(overall, dict) else {}
    return {
        "recall_at_k": _safe_float(overall.get("recall_at_k")),
        "mrr": _safe_float(overall.get("mrr")),
        "avg_response_ms": _safe_float(overall.get("avg_response_ms")),
    }


def _extract_stress_core(stress_payload: Dict[str, Any]) -> Dict[str, float]:
    latency = stress_payload.get("latency_ms", {}) if isinstance(stress_payload.get("latency_ms"), dict) else {}
    throughput = stress_payload.get("throughput", {}) if isinstance(stress_payload.get("throughput"), dict) else {}
    counts = stress_payload.get("counts", {}) if isinstance(stress_payload.get("counts"), dict) else {}
    return {
        "p95_ms": _safe_float(latency.get("p95")),
        "p99_ms": _safe_float(latency.get("p99")),
        "mean_ms": _safe_float(latency.get("mean")),
        "successful_requests_per_second": _safe_float(throughput.get("successful_requests_per_second")),
        "error_rate": _safe_float(counts.get("error_rate")),
    }


def _delta(candidate: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    return {key: _safe_float(candidate.get(key)) - _safe_float(baseline.get(key)) for key in candidate.keys()}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run matched 2-lane baseline vs true 3-lane (BM25+dense+sparse) track with eval/stress gates."
    )
    parser.add_argument(
        "--account-config",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs_tmp/RAG"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_3LANE_TRACK_2026-02-25.json"),
    )
    parser.add_argument("--per-dataset-cases", type=int, default=60)
    parser.add_argument("--max-expected-ids", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260225)
    parser.add_argument("--limit-bm25", type=int, default=15)
    parser.add_argument("--limit-similarity", type=int, default=15)
    parser.add_argument("--bm25-weight", type=float, default=0.70)
    parser.add_argument("--similarity-weight", type=float, default=0.30)
    parser.add_argument("--candidate-limit-sparse", type=int, default=10)
    parser.add_argument("--candidate-sparse-weight", type=float, default=0.15)
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
    parser.add_argument("--stress-duration-s", type=int, default=60)
    parser.add_argument("--stress-concurrency", type=int, default=12)
    parser.add_argument("--http-timeout-s", type=int, default=90)
    parser.add_argument("--gate-min-recall-delta", type=float, default=-0.01)
    parser.add_argument("--gate-min-mrr-delta", type=float, default=-0.01)
    parser.add_argument("--gate-max-latency-ratio", type=float, default=1.10)
    parser.add_argument("--gate-max-p95-ratio", type=float, default=1.10)
    parser.add_argument("--gate-max-p99-ratio", type=float, default=1.15)
    parser.add_argument("--gate-min-success-rps-ratio", type=float, default=0.90)
    args = parser.parse_args()

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())

    baseline_cases = out_dir / f"BCAS_PHASE2_3LANE_BASELINE_CASES_{stamp}.json"
    baseline_eval_metrics = out_dir / f"BCAS_PHASE2_3LANE_BASELINE_EVAL_METRICS_{stamp}.json"
    baseline_eval_gate = out_dir / f"BCAS_PHASE2_3LANE_BASELINE_EVAL_GATE_{stamp}.json"
    candidate_cases = out_dir / f"BCAS_PHASE2_3LANE_CANDIDATE_CASES_{stamp}.json"
    candidate_eval_metrics = out_dir / f"BCAS_PHASE2_3LANE_CANDIDATE_EVAL_METRICS_{stamp}.json"
    candidate_eval_gate = out_dir / f"BCAS_PHASE2_3LANE_CANDIDATE_EVAL_GATE_{stamp}.json"
    baseline_stress = out_dir / f"BCAS_PHASE2_3LANE_BASELINE_STRESS_{stamp}.json"
    baseline_stress_gate = out_dir / f"BCAS_PHASE2_3LANE_BASELINE_STRESS_GATE_{stamp}.json"
    candidate_stress = out_dir / f"BCAS_PHASE2_3LANE_CANDIDATE_STRESS_{stamp}.json"
    candidate_stress_gate = out_dir / f"BCAS_PHASE2_3LANE_CANDIDATE_STRESS_GATE_{stamp}.json"

    commands: Dict[str, Dict[str, Any]] = {}

    eval_baseline_cmd = [
        sys.executable,
        "scripts/bcas_phase2_eval.py",
        "--account-config",
        str(args.account_config),
        "--per-dataset-cases",
        str(int(args.per_dataset_cases)),
        "--max-expected-ids",
        str(int(args.max_expected_ids)),
        "--limit-bm25",
        str(int(args.limit_bm25)),
        "--limit-similarity",
        str(int(args.limit_similarity)),
        "--limit-sparse",
        "0",
        "--top-k",
        str(int(args.top_k)),
        "--seed",
        str(int(args.seed)),
        "--bm25-weight",
        str(float(args.bm25_weight)),
        "--similarity-weight",
        str(float(args.similarity_weight)),
        "--sparse-weight",
        "0.0",
        "--cases-out",
        str(baseline_cases),
        "--metrics-out",
        str(baseline_eval_metrics),
        "--gate-out",
        str(baseline_eval_gate),
        "--queue-admission-concurrency-limit",
        str(int(args.queue_admission_concurrency_limit)),
        "--queue-admission-ttl-seconds",
        str(int(args.queue_admission_ttl_seconds)),
        "--queue-throttle-soft-utilization",
        str(float(args.queue_throttle_soft_utilization)),
        "--queue-throttle-hard-utilization",
        str(float(args.queue_throttle_hard_utilization)),
        "--queue-throttle-soft-scale",
        str(float(args.queue_throttle_soft_scale)),
        "--queue-throttle-hard-scale",
        str(float(args.queue_throttle_hard_scale)),
    ]
    if bool(args.queue_throttle_disable_sparse_at_hard):
        eval_baseline_cmd.append("--queue-throttle-disable-sparse-at-hard")
    else:
        eval_baseline_cmd.append("--no-queue-throttle-disable-sparse-at-hard")
    rc, out, err = _run(eval_baseline_cmd)
    commands["eval_baseline"] = {"cmd": eval_baseline_cmd, "rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
    if rc != 0:
        args.out.write_text(json.dumps({"error": "eval_baseline_failed", "commands": commands}, indent=2), encoding="utf-8")
        return 2

    eval_candidate_cmd = [
        sys.executable,
        "scripts/bcas_phase2_eval.py",
        "--account-config",
        str(args.account_config),
        "--cases-in",
        str(baseline_cases),
        "--limit-bm25",
        str(int(args.limit_bm25)),
        "--limit-similarity",
        str(int(args.limit_similarity)),
        "--limit-sparse",
        str(int(args.candidate_limit_sparse)),
        "--top-k",
        str(int(args.top_k)),
        "--seed",
        str(int(args.seed)),
        "--bm25-weight",
        str(float(args.bm25_weight)),
        "--similarity-weight",
        str(float(args.similarity_weight)),
        "--sparse-weight",
        str(float(args.candidate_sparse_weight)),
        "--use-sparse",
        "--sparse-dimensions",
        str(int(args.sparse_dimensions)),
        "--sparse-embedding-function",
        str(args.sparse_embedding_function),
        "--baseline-metrics",
        str(baseline_eval_metrics),
        "--gate-min-recall-delta",
        str(float(args.gate_min_recall_delta)),
        "--gate-min-mrr-delta",
        str(float(args.gate_min_mrr_delta)),
        "--gate-max-latency-ratio",
        str(float(args.gate_max_latency_ratio)),
        "--cases-out",
        str(candidate_cases),
        "--metrics-out",
        str(candidate_eval_metrics),
        "--gate-out",
        str(candidate_eval_gate),
        "--queue-admission-concurrency-limit",
        str(int(args.queue_admission_concurrency_limit)),
        "--queue-admission-ttl-seconds",
        str(int(args.queue_admission_ttl_seconds)),
        "--queue-throttle-soft-utilization",
        str(float(args.queue_throttle_soft_utilization)),
        "--queue-throttle-hard-utilization",
        str(float(args.queue_throttle_hard_utilization)),
        "--queue-throttle-soft-scale",
        str(float(args.queue_throttle_soft_scale)),
        "--queue-throttle-hard-scale",
        str(float(args.queue_throttle_hard_scale)),
    ]
    if bool(args.queue_throttle_disable_sparse_at_hard):
        eval_candidate_cmd.append("--queue-throttle-disable-sparse-at-hard")
    else:
        eval_candidate_cmd.append("--no-queue-throttle-disable-sparse-at-hard")
    rc, out, err = _run(eval_candidate_cmd)
    commands["eval_candidate"] = {"cmd": eval_candidate_cmd, "rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
    if rc != 0:
        args.out.write_text(json.dumps({"error": "eval_candidate_failed", "commands": commands}, indent=2), encoding="utf-8")
        return 2

    stress_baseline_cmd = [
        sys.executable,
        "scripts/bcas_phase2_stress.py",
        "--account-config",
        str(args.account_config),
        "--cases",
        str(baseline_cases),
        "--duration-s",
        str(int(args.stress_duration_s)),
        "--concurrency",
        str(int(args.stress_concurrency)),
        "--limit-bm25",
        str(int(args.limit_bm25)),
        "--limit-similarity",
        str(int(args.limit_similarity)),
        "--limit-sparse",
        "0",
        "--bm25-weight",
        str(float(args.bm25_weight)),
        "--similarity-weight",
        str(float(args.similarity_weight)),
        "--sparse-weight",
        "0.0",
        "--http-timeout-s",
        str(int(args.http_timeout_s)),
        "--out",
        str(baseline_stress),
        "--gate-out",
        str(baseline_stress_gate),
        "--queue-admission-concurrency-limit",
        str(int(args.queue_admission_concurrency_limit)),
        "--queue-admission-ttl-seconds",
        str(int(args.queue_admission_ttl_seconds)),
        "--queue-throttle-soft-utilization",
        str(float(args.queue_throttle_soft_utilization)),
        "--queue-throttle-hard-utilization",
        str(float(args.queue_throttle_hard_utilization)),
        "--queue-throttle-soft-scale",
        str(float(args.queue_throttle_soft_scale)),
        "--queue-throttle-hard-scale",
        str(float(args.queue_throttle_hard_scale)),
    ]
    if bool(args.queue_throttle_disable_sparse_at_hard):
        stress_baseline_cmd.append("--queue-throttle-disable-sparse-at-hard")
    else:
        stress_baseline_cmd.append("--no-queue-throttle-disable-sparse-at-hard")
    rc, out, err = _run(stress_baseline_cmd)
    commands["stress_baseline"] = {"cmd": stress_baseline_cmd, "rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
    if rc != 0:
        args.out.write_text(json.dumps({"error": "stress_baseline_failed", "commands": commands}, indent=2), encoding="utf-8")
        return 2

    stress_candidate_cmd = [
        sys.executable,
        "scripts/bcas_phase2_stress.py",
        "--account-config",
        str(args.account_config),
        "--cases",
        str(baseline_cases),
        "--duration-s",
        str(int(args.stress_duration_s)),
        "--concurrency",
        str(int(args.stress_concurrency)),
        "--limit-bm25",
        str(int(args.limit_bm25)),
        "--limit-similarity",
        str(int(args.limit_similarity)),
        "--limit-sparse",
        str(int(args.candidate_limit_sparse)),
        "--bm25-weight",
        str(float(args.bm25_weight)),
        "--similarity-weight",
        str(float(args.similarity_weight)),
        "--sparse-weight",
        str(float(args.candidate_sparse_weight)),
        "--use-sparse",
        "--sparse-dimensions",
        str(int(args.sparse_dimensions)),
        "--sparse-embedding-function",
        str(args.sparse_embedding_function),
        "--http-timeout-s",
        str(int(args.http_timeout_s)),
        "--baseline-stress",
        str(baseline_stress),
        "--gate-max-p95-ratio",
        str(float(args.gate_max_p95_ratio)),
        "--gate-max-p99-ratio",
        str(float(args.gate_max_p99_ratio)),
        "--gate-min-success-rps-ratio",
        str(float(args.gate_min_success_rps_ratio)),
        "--out",
        str(candidate_stress),
        "--gate-out",
        str(candidate_stress_gate),
        "--queue-admission-concurrency-limit",
        str(int(args.queue_admission_concurrency_limit)),
        "--queue-admission-ttl-seconds",
        str(int(args.queue_admission_ttl_seconds)),
        "--queue-throttle-soft-utilization",
        str(float(args.queue_throttle_soft_utilization)),
        "--queue-throttle-hard-utilization",
        str(float(args.queue_throttle_hard_utilization)),
        "--queue-throttle-soft-scale",
        str(float(args.queue_throttle_soft_scale)),
        "--queue-throttle-hard-scale",
        str(float(args.queue_throttle_hard_scale)),
    ]
    if bool(args.queue_throttle_disable_sparse_at_hard):
        stress_candidate_cmd.append("--queue-throttle-disable-sparse-at-hard")
    else:
        stress_candidate_cmd.append("--no-queue-throttle-disable-sparse-at-hard")
    rc, out, err = _run(stress_candidate_cmd)
    commands["stress_candidate"] = {"cmd": stress_candidate_cmd, "rc": rc, "stdout_tail": out[-2000:], "stderr_tail": err[-2000:]}
    if rc != 0:
        args.out.write_text(json.dumps({"error": "stress_candidate_failed", "commands": commands}, indent=2), encoding="utf-8")
        return 2

    base_eval_payload = _load_json(baseline_eval_metrics)
    cand_eval_payload = _load_json(candidate_eval_metrics)
    base_eval = _extract_eval_overall(base_eval_payload)
    cand_eval = _extract_eval_overall(cand_eval_payload)
    eval_deltas = _delta(cand_eval, base_eval)

    base_stress_payload = _load_json(baseline_stress)
    cand_stress_payload = _load_json(candidate_stress)
    base_stress = _extract_stress_core(base_stress_payload)
    cand_stress = _extract_stress_core(cand_stress_payload)
    stress_deltas = _delta(cand_stress, base_stress)

    eval_gate_payload = _load_json(candidate_eval_gate)
    stress_gate_payload = _load_json(candidate_stress_gate)
    overall_gate_ok = bool(eval_gate_payload.get("gate_ok") is True and stress_gate_payload.get("gate_ok") is True)

    result = {
        "generated_at_unix": time.time(),
        "track": "bcas_phase2_three_lane",
        "commands": commands,
        "artifacts": {
            "baseline_cases": str(baseline_cases),
            "baseline_eval_metrics": str(baseline_eval_metrics),
            "baseline_eval_gate": str(baseline_eval_gate),
            "baseline_eval_gate_report": str(_default_gate_report_out(baseline_eval_metrics)),
            "candidate_cases": str(candidate_cases),
            "candidate_eval_metrics": str(candidate_eval_metrics),
            "candidate_eval_gate": str(candidate_eval_gate),
            "candidate_eval_gate_report": str(_default_gate_report_out(candidate_eval_metrics)),
            "baseline_stress": str(baseline_stress),
            "baseline_stress_gate": str(baseline_stress_gate),
            "baseline_stress_gate_report": str(_default_gate_report_out(baseline_stress)),
            "candidate_stress": str(candidate_stress),
            "candidate_stress_gate": str(candidate_stress_gate),
            "candidate_stress_gate_report": str(_default_gate_report_out(candidate_stress)),
        },
        "matched_delta_summary": {
            "eval_baseline": base_eval,
            "eval_candidate": cand_eval,
            "eval_deltas": eval_deltas,
            "stress_baseline": base_stress,
            "stress_candidate": cand_stress,
            "stress_deltas": stress_deltas,
        },
        "gates": {
            "eval_gate": eval_gate_payload,
            "stress_gate": stress_gate_payload,
            "overall_gate_ok": overall_gate_ok,
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "overall_gate_ok": overall_gate_ok,
                "eval_gate_ok": eval_gate_payload.get("gate_ok"),
                "stress_gate_ok": stress_gate_payload.get("gate_ok"),
            },
            indent=2,
        )
    )
    return 0 if overall_gate_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
