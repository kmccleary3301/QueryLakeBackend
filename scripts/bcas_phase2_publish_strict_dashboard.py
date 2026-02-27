#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List


def _safe_load(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _latest(paths: List[Path]) -> Path | None:
    rows = [path for path in paths if path.is_file()]
    if not rows:
        return None
    rows.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return rows[-1]


def _find_latest_strict_artifacts(ci_root: Path) -> Dict[str, Path | None]:
    track_rows = list(ci_root.glob("**/BCAS_PHASE2_3LANE_TRACK_*_*.json"))
    gate_rows = list(ci_root.glob("**/BCAS_PHASE2_RELEASE_GATE_*_*.json"))
    return {"track": _latest(track_rows), "gate": _latest(gate_rows)}


def _render(track_path: Path | None, gate_path: Path | None) -> str:
    lines: List[str] = []
    lines.append("# BCAS Phase2 Strict Dashboard (Latest)")
    lines.append("")
    lines.append(f"- Generated: `{time.strftime('%Y-%m-%d %H:%M:%SZ', time.gmtime())}`")
    lines.append("")

    if track_path is not None:
        track = _safe_load(track_path)
        gates = track.get("gates", {}) if isinstance(track.get("gates"), dict) else {}
        delta = track.get("matched_delta_summary", {}) if isinstance(track.get("matched_delta_summary"), dict) else {}
        lines.append("## Latest 3-Lane Track")
        lines.append(f"- Artifact: `{track_path}`")
        lines.append(f"- overall gate: `{gates.get('overall_gate_ok')}`")
        lines.append(f"- eval gate: `{gates.get('eval_gate_ok')}`")
        lines.append(f"- stress gate: `{gates.get('stress_gate_ok')}`")
        lines.append(f"- recall delta: `{delta.get('recall_at_k_delta')}`")
        lines.append(f"- mrr delta: `{delta.get('mrr_delta')}`")
        lines.append(f"- p95 ratio: `{delta.get('stress_p95_ratio')}`")
        lines.append(f"- p99 ratio: `{delta.get('stress_p99_ratio')}`")
        lines.append(f"- success rps ratio: `{delta.get('stress_success_rps_ratio')}`")
        lines.append("")
    else:
        lines.append("## Latest 3-Lane Track")
        lines.append("- No strict track artifact found.")
        lines.append("")

    if gate_path is not None:
        gate = _safe_load(gate_path)
        diagnostics = gate.get("diagnostics", {}) if isinstance(gate.get("diagnostics"), dict) else {}
        lines.append("## Latest Unified Release Gate")
        lines.append(f"- Artifact: `{gate_path}`")
        lines.append(f"- gate_ok: `{gate.get('gate_ok')}`")
        lines.append(f"- policy: `{gate.get('policy_name')}`")
        lines.append(f"- failed checks: `{diagnostics.get('failed_checks')}`")
        lines.append("")
    else:
        lines.append("## Latest Unified Release Gate")
        lines.append("- No unified release-gate artifact found.")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish latest strict dashboard from CI artifacts.")
    parser.add_argument("--ci-root", type=Path, default=Path("docs_tmp/RAG/ci"))
    parser.add_argument("--out", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_STRICT_DASHBOARD_LATEST.md"))
    args = parser.parse_args()

    artifacts = _find_latest_strict_artifacts(args.ci_root)
    text = _render(artifacts.get("track"), artifacts.get("gate"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(json.dumps({"out": str(args.out), "track": str(artifacts.get("track")), "gate": str(artifacts.get("gate"))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
