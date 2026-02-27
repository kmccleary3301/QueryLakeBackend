#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _float(v: Any, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def _collect_day(day_dir: Path) -> Dict[str, Any]:
    day = day_dir.name
    gate_path = day_dir / f"BCAS_PHASE2_OPERATOR_GATE_{day}.json"
    notify_path = day_dir / f"BCAS_PHASE2_NOTIFICATIONS_{day}.json"
    stress_path = day_dir / f"BCAS_PHASE2_STRESS_c8_{day}.json"
    retrieval_path = day_dir / f"BCAS_PHASE2_LIVE_EVAL_RETRIEVAL_EVAL_dense_heavy_augtrivia_{day}.json"

    gate = _safe_load(gate_path)
    notify = _safe_load(notify_path)
    stress = _safe_load(stress_path)
    retrieval = _safe_load(retrieval_path)

    counts = stress.get("counts", {}) if isinstance(stress.get("counts"), dict) else {}
    throughput = stress.get("throughput", {}) if isinstance(stress.get("throughput"), dict) else {}
    stress_error = _float(throughput.get("error_rate"), _float(counts.get("error_rate"), 1.0))

    required_present = all([gate_path.exists(), notify_path.exists(), stress_path.exists(), retrieval_path.exists()])
    gate_ok = bool(gate.get("gate_ok", False))
    notify_status = str(notify.get("status", "unknown"))
    retrieval_ok = bool(isinstance(retrieval.get("metrics"), dict) and _float((retrieval.get("metrics") or {}).get("case_count"), 0.0) > 0.0)

    day_ok = required_present and gate_ok and (notify_status != "critical") and (stress_error <= 0.0) and retrieval_ok

    return {
        "date": day,
        "required_present": required_present,
        "gate_ok": gate_ok,
        "notify_status": notify_status,
        "stress_error_rate": stress_error,
        "retrieval_ok": retrieval_ok,
        "day_ok": day_ok,
        "artifacts": {
            "gate": str(gate_path),
            "notify": str(notify_path),
            "stress": str(stress_path),
            "retrieval": str(retrieval_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Track BCAS nightly soak readiness over rolling days.")
    parser.add_argument("--nightly-root", type=Path, default=Path("docs_tmp/RAG/nightly"))
    parser.add_argument("--required-days", type=int, default=7)
    parser.add_argument("--max-days", type=int, default=21)
    parser.add_argument("--out", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_SOAK_TRACKER.json"))
    args = parser.parse_args()

    days = sorted([d for d in args.nightly_root.iterdir() if d.is_dir()]) if args.nightly_root.exists() else []
    days = days[-max(1, int(args.max_days)) :]
    rows = [_collect_day(day) for day in days]

    streak = 0
    for row in reversed(rows):
        if bool(row.get("day_ok", False)):
            streak += 1
        else:
            break

    ready = streak >= max(1, int(args.required_days))
    summary = {
        "tracked_days": len(rows),
        "required_days": int(args.required_days),
        "passing_days": int(sum(1 for r in rows if r.get("day_ok", False))),
        "consecutive_passing_streak": int(streak),
        "ready": bool(ready),
    }

    payload = {
        "generated_at_unix": time.time(),
        "nightly_root": str(args.nightly_root),
        "summary": summary,
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "summary": summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
