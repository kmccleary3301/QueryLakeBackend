#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback BCAS baseline pointer to a previous baseline artifact.")
    parser.add_argument("--pointer", type=Path, default=Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_BASELINE_POINTER.json"))
    parser.add_argument("--baseline-path", type=Path, required=True)
    parser.add_argument("--reason", type=str, default="manual_rollback")
    args = parser.parse_args()

    if not args.baseline_path.exists():
        raise SystemExit(f"Baseline path missing: {args.baseline_path}")

    pointer_payload: Dict[str, Any] = {}
    if args.pointer.exists():
        pointer_payload = _load(args.pointer)
    pointer_payload["baseline_path"] = str(args.baseline_path)
    pointer_payload["updated_at_unix"] = time.time()
    pointer_payload["source"] = "manual_rollback"
    pointer_payload["rollback_reason"] = str(args.reason)
    args.pointer.parent.mkdir(parents=True, exist_ok=True)
    args.pointer.write_text(json.dumps(pointer_payload, indent=2), encoding="utf-8")
    print(json.dumps({"pointer": str(args.pointer), "baseline_path": str(args.baseline_path), "source": "manual_rollback"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
