#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _latest(paths: List[Path]) -> Path | None:
    rows = [path for path in paths if path.is_file()]
    if not rows:
        return None
    rows.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return rows[-1]


def _safe_load(path: Path | None) -> Dict[str, Any]:
    if path is None:
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _render_md(payload: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# BCAS Phase2 Signoff Package")
    lines.append("")
    lines.append(f"- generated_at_unix: `{payload.get('generated_at_unix')}`")
    lines.append(f"- signoff_ready: `{payload.get('signoff_ready')}`")
    lines.append("")
    lines.append("## Artifacts")
    for key, value in (payload.get("artifacts") or {}).items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## Checks")
    checks = payload.get("checks", {})
    for key, value in checks.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a consolidated BCAS strict signoff package (json + markdown).")
    parser.add_argument("--ci-root", type=Path, default=Path("docs_tmp/RAG/ci"))
    parser.add_argument("--rag-root", type=Path, default=Path("docs_tmp/RAG"))
    parser.add_argument("--out-json", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_SIGNOFF_PACKAGE_LATEST.json"))
    parser.add_argument("--out-md", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_SIGNOFF_PACKAGE_LATEST.md"))
    args = parser.parse_args()

    latest_track = _latest(list(args.ci_root.glob("**/BCAS_PHASE2_3LANE_TRACK_*_*.json")))
    latest_gate = _latest(list(args.ci_root.glob("**/BCAS_PHASE2_RELEASE_GATE_*_*.json")))
    latest_soak = _latest(list(args.rag_root.glob("soak/**/BCAS_PHASE2_SOAK_SUMMARY_*.json")))
    latest_drill = _latest(list(args.rag_root.glob("drills/BCAS_PHASE2_ROLLBACK_DRILL_SUMMARY_*.json")))
    latest_casepack_registry = args.rag_root / "casepacks/CASEPACK_REGISTRY.json"
    latest_dashboard = args.rag_root / "BCAS_PHASE2_STRICT_DASHBOARD_LATEST.md"

    gate_payload = _safe_load(latest_gate)
    drill_payload = _safe_load(latest_drill)
    soak_payload = _safe_load(latest_soak)
    registry_payload = _safe_load(latest_casepack_registry if latest_casepack_registry.exists() else None)

    checks = {
        "latest_gate_ok": bool(gate_payload.get("gate_ok", False)),
        "rollback_drill_ok": bool(drill_payload.get("gate_after_restore_ok", False)) if len(drill_payload) > 0 else False,
        "soak_ok": bool(soak_payload.get("gate_ok", False)) if len(soak_payload) > 0 else None,
        "casepack_registry_present": latest_casepack_registry.exists(),
        "dashboard_present": latest_dashboard.exists(),
    }
    signoff_ready = bool(checks["latest_gate_ok"]) and bool(checks["rollback_drill_ok"]) and bool(checks["casepack_registry_present"]) and bool(checks["dashboard_present"])

    payload = {
        "generated_at_unix": time.time(),
        "signoff_ready": signoff_ready,
        "checks": checks,
        "artifacts": {
            "latest_track": str(latest_track) if latest_track else None,
            "latest_gate": str(latest_gate) if latest_gate else None,
            "latest_soak": str(latest_soak) if latest_soak else None,
            "latest_rollback_drill": str(latest_drill) if latest_drill else None,
            "casepack_registry": str(latest_casepack_registry) if latest_casepack_registry.exists() else None,
            "strict_dashboard_latest": str(latest_dashboard) if latest_dashboard.exists() else None,
            "casepack_count": len(registry_payload.get("packs", [])) if isinstance(registry_payload.get("packs"), list) else 0,
        },
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.out_md.write_text(_render_md(payload), encoding="utf-8")
    print(json.dumps({"out_json": str(args.out_json), "out_md": str(args.out_md), "signoff_ready": signoff_ready}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
