import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _run_readiness(tmp_path: Path, *, mode: str, counts: dict, env_flags: bool):
    acceptance = tmp_path / "acceptance.json"
    audit = tmp_path / "audit.json"
    output = tmp_path / "readiness.json"

    _write_json(acceptance, {"readiness": {"go_for_canary": True}})
    _write_json(audit, {"counts": counts})

    env = dict(os.environ)
    env["QUERYLAKE_RETRIEVAL_DISABLE_LEGACY_BM25"] = "1" if env_flags else "0"
    env["QUERYLAKE_RETRIEVAL_DISABLE_LEGACY_HYBRID"] = "1" if env_flags else "0"

    proc = subprocess.run(
        [
            "python",
            "scripts/retrieval_retirement_readiness.py",
            "--mode",
            mode,
            "--acceptance-cycle",
            str(acceptance),
            "--legacy-audit",
            str(audit),
            "--output",
            str(output),
        ],
        cwd=str(ROOT),
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(output.read_text(encoding="utf-8"))


def test_retrieval_retirement_readiness_strict_passes_when_markers_zero(tmp_path: Path):
    payload = _run_readiness(
        tmp_path,
        mode="strict",
        counts={
            "_orchestrator_bypass": 0,
            "legacy.search_": 0,
            "QUERYLAKE_RETRIEVAL_ORCHESTRATOR_BM25": 0,
            "QUERYLAKE_RETRIEVAL_ORCHESTRATOR_HYBRID": 0,
        },
        env_flags=True,
    )
    assert payload["ready_to_retire_legacy"] is True
    assert payload["legacy_marker_totals"]["total_markers"] == 0


def test_retrieval_retirement_readiness_strict_fails_when_markers_nonzero(tmp_path: Path):
    payload = _run_readiness(
        tmp_path,
        mode="strict",
        counts={
            "_orchestrator_bypass": 1,
            "legacy.search_": 0,
            "QUERYLAKE_RETRIEVAL_ORCHESTRATOR_BM25": 0,
            "QUERYLAKE_RETRIEVAL_ORCHESTRATOR_HYBRID": 0,
        },
        env_flags=True,
    )
    assert payload["ready_to_retire_legacy"] is False
    assert "legacy markers remain in QueryLake code" in payload["blocking_reasons"][0]

