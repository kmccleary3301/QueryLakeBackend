#!/usr/bin/env python3
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MATRIX = ROOT / "docs" / "unification" / "compat_matrix.md"


def main() -> int:
    if not MATRIX.exists():
        print(f"Missing compat matrix: {MATRIX}")
        return 1

    text = MATRIX.read_text()
    pins = re.findall(r"\|\s*([A-Za-z0-9_-]+)\s*\|\s*([A-Za-z0-9_-]+)\s*\|\s*git commit\s*\|\s*([^|]+)\s*\|", text)
    missing = []
    for component, repo, pin in pins:
        pin = pin.strip()
        if pin in ("<set>", ""):
            missing.append((component, repo))
    if missing:
        for component, repo in missing:
            print(f"Missing pin for {component} ({repo})")
        return 2

    print("All pins set.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

