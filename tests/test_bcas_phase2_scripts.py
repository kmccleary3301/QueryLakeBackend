from __future__ import annotations

import json
import threading
import subprocess
import sys
import importlib.util
from pathlib import Path
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest


ROOT = Path(__file__).resolve().parent.parent


def _load_stress_module():
    module_path = ROOT / "scripts" / "bcas_phase2_stress.py"
    spec = importlib.util.spec_from_file_location("bcas_phase2_stress_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_eval_module():
    module_path = ROOT / "scripts" / "bcas_phase2_eval.py"
    spec = importlib.util.spec_from_file_location("bcas_phase2_eval_module", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_operator_gate_resolves_baseline_from_pointer_fallback(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    pointer = tmp_path / "pointer.json"
    out1 = tmp_path / "gate1.json"
    out2 = tmp_path / "gate2.json"

    _write_json(
        baseline,
        {
            "overall": {"pass_rate": 0.90, "avg_latency_ms": 120.0},
            "by_operator_type": {"exact_phrase": {"pass_rate": 0.80, "avg_latency_ms": 110.0}},
        },
    )
    _write_json(
        candidate,
        {
            "overall": {"pass_rate": 0.93, "avg_latency_ms": 115.0},
            "by_operator_type": {"exact_phrase": {"pass_rate": 0.85, "avg_latency_ms": 100.0}},
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bcas_phase2_operator_gate.py"),
        "--baseline-pointer",
        str(pointer),
        "--baseline-fallback",
        str(baseline),
        "--candidate",
        str(candidate),
        "--out",
        str(out1),
    ]
    run1 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run1.returncode == 0, run1.stderr

    payload1 = json.loads(out1.read_text(encoding="utf-8"))
    assert payload1["baseline_source"] == "fallback"
    assert payload1["baseline"] == str(baseline)
    pointer_payload = json.loads(pointer.read_text(encoding="utf-8"))
    assert pointer_payload["baseline_path"] == str(baseline)

    cmd[-1] = str(out2)
    run2 = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run2.returncode == 0, run2.stderr
    payload2 = json.loads(out2.read_text(encoding="utf-8"))
    assert payload2["baseline_source"] == "pointer"
    assert payload2["baseline"] == str(baseline)


def test_operator_gate_can_enforce_eval_profile_match(tmp_path: Path):
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    out = tmp_path / "gate.json"

    _write_json(
        baseline,
        {
            "params": {"per_dataset": 12, "limit": 8, "http_timeout_s": 60},
            "overall": {"pass_rate": 0.90, "avg_latency_ms": 120.0},
            "by_operator_type": {"exact_phrase": {"pass_rate": 0.80, "avg_latency_ms": 110.0}},
        },
    )
    _write_json(
        candidate,
        {
            "params": {"per_dataset": 8, "limit": 8, "http_timeout_s": 60},
            "overall": {"pass_rate": 0.93, "avg_latency_ms": 115.0},
            "by_operator_type": {"exact_phrase": {"pass_rate": 0.85, "avg_latency_ms": 100.0}},
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bcas_phase2_operator_gate.py"),
        "--baseline",
        str(baseline),
        "--candidate",
        str(candidate),
        "--enforce-profile-match",
        "--out",
        str(out),
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 2
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["gate_ok"] is False
    assert payload["checks"]["eval_profile_match_ok"] is False
    assert payload["eval_profile"]["mismatches"]["per_dataset"] == {"baseline": 12, "candidate": 8}


def test_notify_emits_critical_and_fail_on_critical(tmp_path: Path):
    gate = tmp_path / "gate.json"
    stress = tmp_path / "stress.json"
    out = tmp_path / "notify.json"

    _write_json(
        gate,
        {
            "gate_ok": False,
            "checks": {"overall_pass_delta_ok": False},
            "deltas": {"overall_pass_rate": -0.1, "exact_phrase_pass_rate": -0.2},
        },
    )
    _write_json(
        stress,
        {
            "latency_ms": {"p95": 7000, "p99": 9000},
            "throughput": {"error_rate": 0.12},
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bcas_phase2_notify.py"),
        "--gate",
        str(gate),
        "--stress",
        str(stress),
        "--out",
        str(out),
        "--p95-threshold-ms",
        "4500",
        "--p99-threshold-ms",
        "5500",
        "--fail-on-critical",
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 2

    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == "critical"
    event_types = {event.get("type") for event in payload.get("events", [])}
    assert "gate" in event_types
    assert "latency" in event_types
    assert "errors" in event_types


def test_notify_uses_secret_file_and_throttles_duplicate(tmp_path: Path):
    gate = tmp_path / "gate.json"
    stress = tmp_path / "stress.json"
    out1 = tmp_path / "notify1.json"
    out2 = tmp_path / "notify2.json"
    state_file = tmp_path / "notify_state.json"
    secret_file = tmp_path / "webhook_secret.txt"

    _write_json(
        gate,
        {
            "gate_ok": False,
            "checks": {"overall_pass_delta_ok": False},
            "deltas": {"overall_pass_rate": -0.05, "exact_phrase_pass_rate": -0.1},
        },
    )
    _write_json(
        stress,
        {
            "latency_ms": {"p95": 6000, "p99": 7000},
            "throughput": {"error_rate": 0.05},
        },
    )

    records: dict = {"count": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            records["count"] += 1
            length = int(self.headers.get("Content-Length", "0"))
            _ = self.rfile.read(length)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"ok")

        def log_message(self, format, *args):
            return

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        webhook_url = f"http://127.0.0.1:{server.server_port}/hook"
        secret_file.write_text(webhook_url, encoding="utf-8")

        base_cmd = [
            sys.executable,
            str(ROOT / "scripts" / "bcas_phase2_notify.py"),
            "--gate",
            str(gate),
            "--stress",
            str(stress),
            "--webhook-secret-file",
            str(secret_file),
            "--state-file",
            str(state_file),
            "--cooldown-seconds",
            "3600",
            "--dedupe-key",
            "unit_test_key",
            "--p95-threshold-ms",
            "4500",
            "--p99-threshold-ms",
            "5500",
        ]

        run1 = subprocess.run(base_cmd + ["--out", str(out1)], cwd=ROOT, capture_output=True, text=True)
        assert run1.returncode == 0, run1.stderr
        payload1 = json.loads(out1.read_text(encoding="utf-8"))
        assert payload1["webhook"]["sent"] is True
        assert payload1["webhook_resolution"]["source"] == "secret_file"

        run2 = subprocess.run(base_cmd + ["--out", str(out2)], cwd=ROOT, capture_output=True, text=True)
        assert run2.returncode == 0, run2.stderr
        payload2 = json.loads(out2.read_text(encoding="utf-8"))
        assert payload2["webhook"]["sent"] is False
        assert payload2["webhook"]["reason"] == "throttled"
        assert payload2["dedupe"]["throttled"] is True
        assert records["count"] == 1
    finally:
        server.shutdown()
        server.server_close()


def test_stress_matrix_recommends_best_eligible_profile(tmp_path: Path):
    c8 = tmp_path / "c8.json"
    c10 = tmp_path / "c10.json"
    c12 = tmp_path / "c12.json"
    out_json = tmp_path / "matrix.json"
    out_md = tmp_path / "matrix.md"

    _write_json(
        c8,
        {
            "summary": {
                "requested_runs": 3,
                "successful_runs": 3,
                "median_successful_rps": 2.60,
                "median_latency_p50_ms": 2800.0,
                "median_latency_p95_ms": 3600.0,
                "median_latency_p99_ms": 4500.0,
                "median_latency_mean_ms": 2950.0,
                "max_error_rate": 0.0,
            }
        },
    )
    _write_json(
        c10,
        {
            "summary": {
                "requested_runs": 3,
                "successful_runs": 3,
                "median_successful_rps": 2.55,
                "median_latency_p50_ms": 3800.0,
                "median_latency_p95_ms": 4900.0,
                "median_latency_p99_ms": 5300.0,
                "median_latency_mean_ms": 4100.0,
                "max_error_rate": 0.0,
            }
        },
    )
    _write_json(
        c12,
        {
            "summary": {
                "requested_runs": 3,
                "successful_runs": 3,
                "median_successful_rps": 2.70,
                "median_latency_p50_ms": 4700.0,
                "median_latency_p95_ms": 5600.0,
                "median_latency_p99_ms": 8400.0,
                "median_latency_mean_ms": 5000.0,
                "max_error_rate": 0.0,
            }
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bcas_phase2_stress_matrix.py"),
        "--suite",
        str(c8),
        str(c10),
        str(c12),
        "--p95-budget-ms",
        "5000",
        "--p99-budget-ms",
        "7000",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr

    payload = json.loads(out_json.read_text(encoding="utf-8"))
    rec = payload["recommendation"]["recommended"]
    assert rec["label"] == "c8"
    assert out_md.exists()


def test_stress_duration_extracts_dict_and_prefers_explicit_total():
    module = _load_stress_module()
    result = {"duration": {"retrieve:bm25": 0.120, "fusion": 0.0082, "total": 0.1417}}
    durations = module._extract_server_duration_map(result)
    assert durations["retrieve:bm25"] == pytest.approx(120.0)
    assert durations["fusion"] == pytest.approx(8.2)
    assert module._resolve_server_total_ms(durations) == pytest.approx(141.7)


def test_stress_duration_extracts_trace_list_and_sums_when_total_missing():
    module = _load_stress_module()
    result = {
        "traces": [
            {"stage": "retrieve:dense", "duration_ms": 15.0},
            {"stage": "pack", "duration_ms": "4.5"},
        ]
    }
    durations = module._extract_server_duration_map(result)
    assert durations["retrieve:dense"] == pytest.approx(15.0)
    assert durations["pack"] == pytest.approx(4.5)
    assert module._resolve_server_total_ms(durations) == pytest.approx(19.5)


def test_eval_gate_threshold_checks_with_baseline():
    module = _load_eval_module()
    baseline = {
        "metrics": {
            "overall": {"recall_at_k": 0.20, "mrr": 0.10, "avg_response_ms": 400.0},
        }
    }
    candidate = {
        "metrics": {
            "overall": {"recall_at_k": 0.24, "mrr": 0.11, "avg_response_ms": 430.0},
        }
    }
    gate = module._compute_eval_gate(
        candidate_metrics=candidate,
        baseline_metrics=baseline,
        min_recall_delta=0.0,
        min_mrr_delta=0.0,
        max_latency_ratio=1.10,
    )
    assert gate["baseline_available"] is True
    assert gate["gate_ok"] is True
    assert gate["checks"]["recall_delta_ok"] is True
    assert gate["checks"]["latency_ratio_ok"] is True


def test_stress_gate_threshold_checks_with_baseline():
    module = _load_stress_module()
    baseline = {
        "latency_ms": {"p95": 5000.0, "p99": 7000.0},
        "throughput": {"successful_requests_per_second": 2.0},
        "counts": {"error_rate": 0.0},
    }
    candidate = {
        "latency_ms": {"p95": 5200.0, "p99": 7600.0},
        "throughput": {"successful_requests_per_second": 1.9},
        "counts": {"error_rate": 0.0},
    }
    gate = module._compute_stress_gate(
        candidate_payload=candidate,
        baseline_payload=baseline,
        max_p95_ratio=1.10,
        max_p99_ratio=1.15,
        min_success_rps_ratio=0.90,
        max_error_rate=0.0,
    )
    assert gate["baseline_available"] is True
    assert gate["gate_ok"] is True
    assert gate["checks"]["p95_ratio_ok"] is True
    assert gate["checks"]["success_rps_ratio_ok"] is True


def test_unified_release_gate_passes_on_improved_candidate(tmp_path: Path):
    baseline_eval = tmp_path / "baseline_eval.json"
    candidate_eval = tmp_path / "candidate_eval.json"
    baseline_stress = tmp_path / "baseline_stress.json"
    candidate_stress = tmp_path / "candidate_stress.json"
    out_json = tmp_path / "gate.json"
    out_md = tmp_path / "gate.md"

    _write_json(baseline_eval, {"metrics": {"overall": {"recall_at_k": 0.15, "mrr": 0.12, "avg_response_ms": 410.0}}})
    _write_json(candidate_eval, {"metrics": {"overall": {"recall_at_k": 0.16, "mrr": 0.13, "avg_response_ms": 405.0}}})
    _write_json(
        baseline_stress,
        {
            "latency_ms": {"p95": 5600.0, "p99": 7000.0, "mean": 4800.0},
            "throughput": {"successful_requests_per_second": 2.4, "error_rate": 0.0},
        },
    )
    _write_json(
        candidate_stress,
        {
            "latency_ms": {"p95": 5400.0, "p99": 6800.0, "mean": 4700.0},
            "throughput": {"successful_requests_per_second": 2.5, "error_rate": 0.0},
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bcas_phase2_release_gate.py"),
        "--baseline-eval",
        str(baseline_eval),
        "--candidate-eval",
        str(candidate_eval),
        "--baseline-stress",
        str(baseline_stress),
        "--candidate-stress",
        str(candidate_stress),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--fail-on-gate",
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["gate_ok"] is True
    assert out_md.exists()


def test_unified_release_gate_fails_on_error_rate(tmp_path: Path):
    baseline_eval = tmp_path / "baseline_eval.json"
    candidate_eval = tmp_path / "candidate_eval.json"
    baseline_stress = tmp_path / "baseline_stress.json"
    candidate_stress = tmp_path / "candidate_stress.json"
    out_json = tmp_path / "gate.json"
    out_md = tmp_path / "gate.md"

    _write_json(baseline_eval, {"metrics": {"overall": {"recall_at_k": 0.15, "mrr": 0.12, "avg_response_ms": 410.0}}})
    _write_json(candidate_eval, {"metrics": {"overall": {"recall_at_k": 0.15, "mrr": 0.12, "avg_response_ms": 410.0}}})
    _write_json(
        baseline_stress,
        {
            "latency_ms": {"p95": 5600.0, "p99": 7000.0, "mean": 4800.0},
            "throughput": {"successful_requests_per_second": 2.4, "error_rate": 0.0},
        },
    )
    _write_json(
        candidate_stress,
        {
            "latency_ms": {"p95": 5600.0, "p99": 7000.0, "mean": 4800.0},
            "throughput": {"successful_requests_per_second": 2.4, "error_rate": 0.01},
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bcas_phase2_release_gate.py"),
        "--baseline-eval",
        str(baseline_eval),
        "--candidate-eval",
        str(candidate_eval),
        "--baseline-stress",
        str(baseline_stress),
        "--candidate-stress",
        str(candidate_stress),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--fail-on-gate",
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 2
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["gate_ok"] is False
    assert "stress_error_rate" in payload["diagnostics"]["failed_checks"]


def test_nightly_delta_report_uses_latest_two_days(tmp_path: Path):
    nightly_root = tmp_path / "nightly"
    day1 = nightly_root / "2026-02-24"
    day2 = nightly_root / "2026-02-25"
    day1.mkdir(parents=True)
    day2.mkdir(parents=True)

    _write_json(
        day1 / "BCAS_PHASE2_LIVE_EVAL_METRICS_dense_heavy_augtrivia_2026-02-24.json",
        {"metrics": {"overall": {"recall_at_k": 0.15, "mrr": 0.13, "avg_response_ms": 420.0}}},
    )
    _write_json(
        day2 / "BCAS_PHASE2_LIVE_EVAL_METRICS_dense_heavy_augtrivia_2026-02-25.json",
        {"metrics": {"overall": {"recall_at_k": 0.16, "mrr": 0.14, "avg_response_ms": 410.0}}},
    )
    _write_json(
        day1 / "BCAS_PHASE2_STRESS_c8_2026-02-24.json",
        {"latency_ms": {"p95": 5000.0, "p99": 7000.0}, "throughput": {"successful_requests_per_second": 2.3, "error_rate": 0.0}},
    )
    _write_json(
        day2 / "BCAS_PHASE2_STRESS_c8_2026-02-25.json",
        {"latency_ms": {"p95": 4900.0, "p99": 6900.0}, "throughput": {"successful_requests_per_second": 2.4, "error_rate": 0.0}},
    )

    out_json = tmp_path / "delta.json"
    out_md = tmp_path / "delta.md"
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bcas_phase2_nightly_delta.py"),
        "--nightly-root",
        str(nightly_root),
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["previous_day"] == "2026-02-24"
    assert payload["latest_day"] == "2026-02-25"
    assert out_md.exists()


def test_notify_handles_unified_release_gate_shape(tmp_path: Path):
    gate = tmp_path / "gate.json"
    stress = tmp_path / "stress.json"
    out = tmp_path / "notify.json"

    _write_json(
        gate,
        {
            "gate_ok": False,
            "checks": [
                {"id": "eval_recall_delta", "ok": False},
                {"id": "stress_p95_ratio", "ok": True},
            ],
            "diagnostics": {"recall_delta": -0.02, "mrr_delta": -0.01},
        },
    )
    _write_json(
        stress,
        {
            "latency_ms": {"p95": 4200.0, "p99": 5100.0},
            "throughput": {"error_rate": 0.0},
        },
    )

    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "bcas_phase2_notify.py"),
        "--gate",
        str(gate),
        "--stress",
        str(stress),
        "--out",
        str(out),
    ]
    run = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] in {"warn", "critical"}
    gate_events = [row for row in payload.get("events", []) if row.get("type") == "gate"]
    assert len(gate_events) == 1
    failed_checks = gate_events[0].get("details", {}).get("failed_checks", [])
    assert "eval_recall_delta" in failed_checks
