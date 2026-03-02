#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile


def _pick_latest_wheel(dist_dir: Path) -> Path:
    wheels = sorted(dist_dir.glob("querylake_sdk-*.whl"))
    if len(wheels) == 0:
        raise FileNotFoundError(f"No SDK wheel found under: {dist_dir}")
    return wheels[-1]


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate querylake-sdk wheel contents.")
    parser.add_argument("--dist-dir", type=Path, default=Path("sdk/python/dist"), help="Directory containing built wheels.")
    parser.add_argument("--wheel", type=Path, default=None, help="Explicit wheel path override.")
    args = parser.parse_args()

    wheel_path = args.wheel if isinstance(args.wheel, Path) else _pick_latest_wheel(args.dist_dir)
    if not wheel_path.exists():
        raise FileNotFoundError(f"Wheel not found: {wheel_path}")

    required_paths = {
        "querylake_sdk/__init__.py",
        "querylake_sdk/client.py",
        "querylake_sdk/cli.py",
        "querylake_sdk/py.typed",
    }

    with zipfile.ZipFile(wheel_path, "r") as zf:
        names = set(zf.namelist())
        missing = sorted(path for path in required_paths if path not in names)
        if missing:
            print(f"FAIL: wheel is missing required files: {missing}")
            return 2

        metadata_files = sorted(name for name in names if name.endswith(".dist-info/METADATA"))
        if len(metadata_files) == 0:
            print("FAIL: wheel metadata file is missing")
            return 2

        metadata_text = zf.read(metadata_files[0]).decode("utf-8", errors="replace")
        required_metadata_tokens = [
            "Name: querylake-sdk",
            "Requires-Python: >=3.10",
        ]
        missing_tokens = [token for token in required_metadata_tokens if token not in metadata_text]
        if missing_tokens:
            print(f"FAIL: wheel metadata missing required tokens: {missing_tokens}")
            return 2

    print(f"OK: validated wheel {wheel_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
