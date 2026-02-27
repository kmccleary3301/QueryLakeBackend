#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


MARKERS = [
    "_orchestrator_bypass",
    "legacy.search_",
    "QUERYLAKE_RETRIEVAL_ORCHESTRATOR_BM25",
    "QUERYLAKE_RETRIEVAL_ORCHESTRATOR_HYBRID",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit legacy retrieval path markers for deprecation planning.")
    parser.add_argument("--root", type=Path, default=Path("QueryLake"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    hits: Dict[str, List[Dict[str, str]]] = {marker: [] for marker in MARKERS}
    for path in sorted(args.root.rglob("*.py")):
        try:
            text = path.read_text(encoding="utf-8")
        except Exception:
            continue
        lines = text.splitlines()
        for i, line in enumerate(lines, start=1):
            for marker in MARKERS:
                if marker in line:
                    hits[marker].append(
                        {
                            "file": str(path),
                            "line": str(i),
                            "snippet": line.strip(),
                        }
                    )

    payload = {
        "markers": MARKERS,
        "counts": {marker: len(rows) for marker, rows in hits.items()},
        "hits": hits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "counts": payload["counts"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

