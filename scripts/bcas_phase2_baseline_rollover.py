#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_POINTER = Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_BASELINE_POINTER.json")
DEFAULT_BASELINE = Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_EVAL_2026-02-24.json")
DEFAULT_PROPOSAL = Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_BASELINE_PROPOSAL.json")
BASELINE_ARCHIVE_DIR = Path("docs_tmp/RAG/baselines")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _ensure_pointer(pointer_path: Path, fallback_baseline: Path) -> Dict[str, Any]:
    pointer_path.parent.mkdir(parents=True, exist_ok=True)
    if pointer_path.exists():
        payload = _load_json(pointer_path)
        if isinstance(payload.get("baseline_path"), str) and len(payload["baseline_path"].strip()) > 0:
            return payload
    payload = {
        "baseline_path": str(fallback_baseline),
        "created_at_unix": time.time(),
        "updated_at_unix": time.time(),
        "source": "bootstrap_default",
    }
    pointer_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _latest_nightly_dirs(nightly_root: Path, max_days: int) -> List[Path]:
    if not nightly_root.exists():
        return []
    dirs = [p for p in nightly_root.iterdir() if p.is_dir()]
    return sorted(dirs)[-max(1, int(max_days)) :]


def _candidate_eval_for_day(day_dir: Path) -> Path:
    return day_dir / f"BCAS_PHASE2_OPERATOR_EVAL_{day_dir.name}.json"


def _gate_for_day(day_dir: Path) -> Path:
    return day_dir / f"BCAS_PHASE2_OPERATOR_GATE_{day_dir.name}.json"


def _metric(payload: Dict[str, Any], *keys: str) -> float:
    node: Any = payload
    for key in keys:
        if not isinstance(node, dict):
            return 0.0
        node = node.get(key)
    try:
        return float(node)
    except Exception:
        return 0.0


def _propose(
    *,
    nightly_root: Path,
    pointer_path: Path,
    proposal_out: Path,
    min_consecutive_days: int,
    min_exact_pass_delta: float,
    max_exact_latency_delta_ms: float,
    max_days_scan: int,
) -> Dict[str, Any]:
    pointer = _ensure_pointer(pointer_path, DEFAULT_BASELINE)
    baseline_path = Path(str(pointer.get("baseline_path", "")))
    if not baseline_path.exists():
        raise SystemExit(f"Baseline file missing: {baseline_path}")
    baseline = _load_json(baseline_path)

    day_dirs = _latest_nightly_dirs(nightly_root, max_days_scan)
    checks: List[Dict[str, Any]] = []
    passing_streak = 0
    latest_candidate: Path | None = None

    for day in day_dirs:
        gate_path = _gate_for_day(day)
        cand_path = _candidate_eval_for_day(day)
        if not gate_path.exists() or not cand_path.exists():
            checks.append(
                {
                    "date": day.name,
                    "present": False,
                    "gate_ok": False,
                    "reason": "missing_gate_or_candidate",
                }
            )
            passing_streak = 0
            continue

        gate = _load_json(gate_path)
        cand = _load_json(cand_path)
        gate_ok = bool(gate.get("gate_ok", False))
        exact_pass_delta = _metric(gate, "deltas", "exact_phrase_pass_rate")
        exact_lat_delta = _metric(gate, "deltas", "exact_phrase_avg_latency_ms")
        exact_pass = _metric(cand, "by_operator_type", "exact_phrase", "pass_rate")
        base_exact_pass = _metric(baseline, "by_operator_type", "exact_phrase", "pass_rate")

        day_ok = (
            gate_ok
            and exact_pass_delta >= float(min_exact_pass_delta)
            and exact_lat_delta <= float(max_exact_latency_delta_ms)
            and exact_pass >= base_exact_pass
        )
        if day_ok:
            passing_streak += 1
            latest_candidate = cand_path
        else:
            passing_streak = 0

        checks.append(
            {
                "date": day.name,
                "present": True,
                "gate_ok": gate_ok,
                "exact_pass_delta": exact_pass_delta,
                "exact_latency_delta_ms": exact_lat_delta,
                "candidate_exact_pass": exact_pass,
                "baseline_exact_pass": base_exact_pass,
                "day_ok": day_ok,
                "candidate_path": str(cand_path),
                "gate_path": str(gate_path),
            }
        )

    approve_ready = passing_streak >= max(1, int(min_consecutive_days)) and latest_candidate is not None
    proposal = {
        "generated_at_unix": time.time(),
        "pointer_path": str(pointer_path),
        "baseline_path": str(baseline_path),
        "nightly_root": str(nightly_root),
        "policy": {
            "min_consecutive_days": int(min_consecutive_days),
            "min_exact_pass_delta": float(min_exact_pass_delta),
            "max_exact_latency_delta_ms": float(max_exact_latency_delta_ms),
            "max_days_scan": int(max_days_scan),
        },
        "checks": checks,
        "passing_streak": int(passing_streak),
        "approve_ready": bool(approve_ready),
        "proposed_candidate_path": str(latest_candidate) if latest_candidate else None,
    }
    proposal_out.parent.mkdir(parents=True, exist_ok=True)
    proposal_out.write_text(json.dumps(proposal, indent=2), encoding="utf-8")
    return proposal


def _approve(*, pointer_path: Path, proposal_path: Path, approve_token: str) -> Dict[str, Any]:
    if approve_token != "APPROVE_BASELINE":
        raise SystemExit("approval rejected: pass --approve-token APPROVE_BASELINE")
    proposal = _load_json(proposal_path)
    if not bool(proposal.get("approve_ready", False)):
        raise SystemExit("proposal is not approve_ready")
    cand = proposal.get("proposed_candidate_path")
    if not isinstance(cand, str) or len(cand.strip()) == 0:
        raise SystemExit("proposal missing proposed_candidate_path")
    cand_path = Path(cand)
    if not cand_path.exists():
        raise SystemExit(f"candidate file missing: {cand_path}")

    BASELINE_ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    archived = BASELINE_ARCHIVE_DIR / f"BCAS_PHASE2_OPERATOR_EVAL_BASELINE_{stamp}.json"
    shutil.copy2(cand_path, archived)

    pointer = _ensure_pointer(pointer_path, DEFAULT_BASELINE)
    pointer["baseline_path"] = str(archived)
    pointer["updated_at_unix"] = time.time()
    pointer["source"] = "manual_approval_rollover"
    pointer["approved_from_proposal"] = str(proposal_path)
    pointer["approved_candidate_path"] = str(cand_path)
    pointer_path.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    return pointer


def main() -> int:
    parser = argparse.ArgumentParser(description="Manual-approval baseline rollover policy for BCAS operator eval.")
    parser.add_argument("--action", choices=["show", "propose", "approve"], default="show")
    parser.add_argument("--pointer", type=Path, default=DEFAULT_POINTER)
    parser.add_argument("--proposal", type=Path, default=DEFAULT_PROPOSAL)
    parser.add_argument("--nightly-root", type=Path, default=Path("docs_tmp/RAG/nightly"))
    parser.add_argument("--min-consecutive-days", type=int, default=3)
    parser.add_argument("--min-exact-pass-delta", type=float, default=0.0)
    parser.add_argument("--max-exact-latency-delta-ms", type=float, default=5.0)
    parser.add_argument("--max-days-scan", type=int, default=14)
    parser.add_argument("--approve-token", type=str, default="")
    args = parser.parse_args()

    if args.action == "show":
        payload = _ensure_pointer(args.pointer, DEFAULT_BASELINE)
        print(json.dumps({"pointer": str(args.pointer), "payload": payload}, indent=2))
        return 0

    if args.action == "propose":
        proposal = _propose(
            nightly_root=args.nightly_root,
            pointer_path=args.pointer,
            proposal_out=args.proposal,
            min_consecutive_days=args.min_consecutive_days,
            min_exact_pass_delta=args.min_exact_pass_delta,
            max_exact_latency_delta_ms=args.max_exact_latency_delta_ms,
            max_days_scan=args.max_days_scan,
        )
        print(json.dumps({"proposal": str(args.proposal), "approve_ready": proposal["approve_ready"], "passing_streak": proposal["passing_streak"]}, indent=2))
        return 0

    if args.action == "approve":
        pointer = _approve(pointer_path=args.pointer, proposal_path=args.proposal, approve_token=args.approve_token)
        print(json.dumps({"pointer": str(args.pointer), "baseline_path": pointer["baseline_path"], "source": pointer.get("source")}, indent=2))
        return 0

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
