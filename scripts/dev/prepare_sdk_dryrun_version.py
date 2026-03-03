#!/usr/bin/env python3
"""Prepare a unique TestPyPI dry-run version for querylake-sdk."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python <3.11 fallback.
    import tomli as tomllib  # type: ignore[no-redef]
from pathlib import Path

STABLE_SEMVER_RE = re.compile(r"^(\d+\.\d+\.\d+)$")
VERSION_ASSIGNMENT_RE = re.compile(r'(^\s*version\s*=\s*")([^"]+)("\s*$)', re.MULTILINE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create deterministic dry-run SDK version.")
    parser.add_argument("--version-file", default="sdk/python/pyproject.toml")
    parser.add_argument(
        "--token",
        default="",
        help="Numeric uniqueness token for the dry-run suffix. Defaults to UTC timestamp.",
    )
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write dry-run version into the version file in-place.",
    )
    parser.add_argument(
        "--github-output",
        default="",
        help="Optional path to GITHUB_OUTPUT file for workflow outputs.",
    )
    return parser.parse_args()


def load_version(version_file: Path) -> str:
    payload = tomllib.loads(version_file.read_text(encoding="utf-8"))
    return str(payload["project"]["version"]).strip()


def ensure_stable_base(version: str) -> str:
    match = STABLE_SEMVER_RE.fullmatch(version)
    if not match:
        raise ValueError(
            f"Dry-run version preparation requires stable base `X.Y.Z`; got `{version}`."
        )
    return match.group(1)


def sanitize_token(raw_token: str) -> str:
    token = "".join(ch for ch in raw_token if ch.isdigit())
    if not token:
        token = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%d%H%M%S")
    return token


def compute_dryrun_version(base_version: str, token: str) -> str:
    return f"{base_version}.dev{token}"


def write_version_in_place(version_file: Path, dryrun_version: str) -> None:
    source = version_file.read_text(encoding="utf-8")

    def _replace(match: re.Match[str]) -> str:
        return f'{match.group(1)}{dryrun_version}{match.group(3)}'

    updated, count = VERSION_ASSIGNMENT_RE.subn(_replace, source, count=1)
    if count != 1:
        raise ValueError(f"Could not update version assignment in `{version_file}`.")
    version_file.write_text(updated, encoding="utf-8")


def write_github_output(path: Path, dryrun_version: str, base_version: str, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"dryrun_version={dryrun_version}\n")
        handle.write(f"base_version={base_version}\n")
        handle.write(f"dryrun_token={token}\n")


def main() -> int:
    args = parse_args()
    version_file = Path(args.version_file).expanduser().resolve()
    if not version_file.is_file():
        print(f"[sdk-dryrun-version] missing version file: {version_file}", file=sys.stderr)
        return 1

    try:
        current_version = load_version(version_file)
        base_version = ensure_stable_base(current_version)
        token = sanitize_token(str(args.token))
        dryrun_version = compute_dryrun_version(base_version, token)
        if args.write:
            write_version_in_place(version_file, dryrun_version)
        if args.github_output:
            write_github_output(Path(args.github_output), dryrun_version, base_version, token)
    except Exception as exc:  # noqa: BLE001 - CLI boundary.
        print(f"[sdk-dryrun-version] FAILED: {exc}", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "version_file": str(version_file),
                "base_version": base_version,
                "dryrun_version": dryrun_version,
                "dryrun_token": token,
                "written": bool(args.write),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
