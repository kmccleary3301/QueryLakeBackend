#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _format_eval_metrics(payload: Dict[str, Any]) -> List[str]:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    recall = float(metrics.get("recall_at_k", 0.0) or 0.0)
    mrr = float(metrics.get("mrr", 0.0) or 0.0)
    cases = int(float(metrics.get("case_count", 0.0) or 0.0))
    return [
        f"- `recall@k`: `{recall:.4f}`",
        f"- `mrr`: `{mrr:.4f}`",
        f"- `cases`: `{cases}`",
    ]


def _format_parity_metrics(payload: Dict[str, Any]) -> List[str]:
    metrics = payload.get("metrics", {}) if isinstance(payload.get("metrics"), dict) else {}
    overlap = float(metrics.get("topk_overlap_mean", metrics.get("topk_overlap", 0.0)) or 0.0)
    ratio = float(metrics.get("latency_ratio", 0.0) or 0.0)
    mrr_delta = float(metrics.get("mrr_delta", 0.0) or 0.0)
    return [
        f"- `topk_overlap`: `{overlap:.4f}`",
        f"- `latency_ratio`: `{ratio:.4f}`",
        f"- `mrr_delta`: `{mrr_delta:.4f}`",
    ]


def _find_latest(files: List[Path], prefix: str) -> Path | None:
    candidates = [path for path in files if path.name.startswith(prefix) and path.suffix == ".json"]
    if not candidates:
        return None
    candidates.sort(key=lambda p: p.name)
    return candidates[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Create a concise markdown summary for retrieval CI artifacts.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--mode", type=str, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--append-step-summary", action="store_true")
    args = parser.parse_args()

    out_dir = args.out_dir
    files = list(out_dir.glob("*.json"))

    eval_file = _find_latest(files, f"retrieval_eval_{args.mode}_")
    parity_file = _find_latest(files, f"retrieval_parity_{args.mode}_")
    track_file = _find_latest(files, "BCAS_PHASE2_3LANE_TRACK_heavy_")

    lines: List[str] = []
    lines.append(f"# Retrieval CI Summary ({args.mode})")
    lines.append("")
    lines.append(f"- Artifact directory: `{out_dir}`")
    lines.append("")

    if eval_file is not None:
        eval_payload = _load_json(eval_file)
        lines.append("## Retrieval Eval")
        lines.append(f"- Source: `{eval_file}`")
        lines.extend(_format_eval_metrics(eval_payload))
        lines.append("")

    if parity_file is not None:
        parity_payload = _load_json(parity_file)
        lines.append("## Retrieval Parity")
        lines.append(f"- Source: `{parity_file}`")
        lines.extend(_format_parity_metrics(parity_payload))
        lines.append("")

    if track_file is not None:
        track_payload = _load_json(track_file)
        gate = track_payload.get("overall_gate")
        lines.append("## 3-Lane Track")
        lines.append(f"- Source: `{track_file}`")
        lines.append(f"- `overall_gate`: `{gate}`")
        lines.append("")

    if len(lines) <= 4:
        lines.append("_No JSON artifacts discovered for summary generation._")

    text = "\n".join(lines).rstrip() + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(text)

    if args.append_step_summary:
        step_summary = Path(".github_step_summary.md")
        step_summary.write_text(text, encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
