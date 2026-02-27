#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _env_flag(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def main() -> int:
    parser = argparse.ArgumentParser(description="Assess legacy retirement readiness from artifacts.")
    parser.add_argument("--acceptance-cycle", type=Path, required=True)
    parser.add_argument("--legacy-audit", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=["config", "strict"],
        default="config",
        help="config: require canary readiness + legacy disable controls; strict: additionally require zero functional legacy markers.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    acceptance = _load_json(args.acceptance_cycle)
    audit = _load_json(args.legacy_audit)

    disable_legacy_bm25 = _env_flag("QUERYLAKE_RETRIEVAL_DISABLE_LEGACY_BM25", False)
    disable_legacy_hybrid = _env_flag("QUERYLAKE_RETRIEVAL_DISABLE_LEGACY_HYBRID", False)

    checks = {
        "acceptance_go_for_canary": bool((acceptance.get("readiness") or {}).get("go_for_canary", False)),
        "legacy_disable_controls_present": bool(disable_legacy_bm25 and disable_legacy_hybrid),
        "legacy_disable_bm25": bool(disable_legacy_bm25),
        "legacy_disable_hybrid": bool(disable_legacy_hybrid),
        "mode": args.mode,
    }
    blocking = []
    if not checks["acceptance_go_for_canary"]:
        blocking.append("acceptance cycle is not go-for-canary")
    if not checks["legacy_disable_controls_present"]:
        blocking.append(
            "legacy disable controls are not both enabled; set QUERYLAKE_RETRIEVAL_DISABLE_LEGACY_BM25=1 and QUERYLAKE_RETRIEVAL_DISABLE_LEGACY_HYBRID=1"
        )

    counts = audit.get("counts", {})
    functional_markers = int(counts.get("_orchestrator_bypass", 0)) + int(counts.get("legacy.search_", 0))
    total_markers = sum(int(v) for v in counts.values()) if isinstance(counts, dict) else 0
    if args.mode == "strict" and total_markers > 0:
        blocking.append(f"legacy markers remain in QueryLake code: {total_markers}")

    payload = {
        "ready_to_retire_legacy": len(blocking) == 0,
        "checks": checks,
        "legacy_marker_counts": counts,
        "legacy_marker_totals": {
            "functional_markers": functional_markers,
            "total_markers": total_markers,
        },
        "blocking_reasons": blocking,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "ready_to_retire_legacy": payload["ready_to_retire_legacy"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
