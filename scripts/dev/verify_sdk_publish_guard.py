#!/usr/bin/env python3
"""Validate SDK publish safety before uploading to TestPyPI/PyPI."""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

SEMVER_STABLE_RE = re.compile(r"^\d+\.\d+\.\d+$")
PYPI_BASE = "https://pypi.org"
TESTPYPI_BASE = "https://test.pypi.org"


@dataclass(frozen=True)
class PublishGuardInputs:
    target: str
    package_name: str
    version: str
    github_ref: str
    skip_remote_check: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SDK publish guard checks.")
    parser.add_argument("--target", required=True, choices=("testpypi", "pypi"))
    parser.add_argument("--package-name", default="querylake-sdk")
    parser.add_argument("--version-file", default="sdk/python/pyproject.toml")
    parser.add_argument("--github-ref", default="")
    parser.add_argument("--skip-remote-check", action="store_true")
    return parser.parse_args()


def load_sdk_version(version_file: str) -> str:
    payload = tomllib.loads(Path(version_file).read_text(encoding="utf-8"))
    return str(payload["project"]["version"]).strip()


def fetch_release_versions(index_base: str, package_name: str) -> set[str]:
    package_url = f"{index_base.rstrip('/')}/pypi/{package_name}/json"
    request = urllib.request.Request(package_url, headers={"User-Agent": "querylake-sdk-publish-guard/1"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return set()
        raise
    releases = payload.get("releases", {})
    return set(releases.keys())


def ensure_target_constraints(inputs: PublishGuardInputs) -> None:
    if inputs.target == "testpypi":
        return

    if not SEMVER_STABLE_RE.fullmatch(inputs.version):
        raise ValueError(
            f"PyPI target requires stable semver `X.Y.Z`; received version `{inputs.version}`."
        )

    if not inputs.github_ref:
        raise ValueError("PyPI target requires `github_ref` (e.g., refs/heads/main or refs/tags/vX.Y.Z).")

    expected_tag = f"refs/tags/v{inputs.version}"
    allowed_refs = {"refs/heads/main", expected_tag}
    if inputs.github_ref not in allowed_refs:
        raise ValueError(
            "PyPI target requires main branch or matching version tag. "
            f"Expected one of {sorted(allowed_refs)}, got `{inputs.github_ref}`."
        )

    if inputs.github_ref.startswith("refs/tags/") and inputs.github_ref != expected_tag:
        raise ValueError(
            f"PyPI target tag must match version exactly (`{expected_tag}`), got `{inputs.github_ref}`."
        )


def ensure_version_not_already_published(inputs: PublishGuardInputs) -> None:
    if inputs.skip_remote_check:
        return

    index_base = PYPI_BASE if inputs.target == "pypi" else TESTPYPI_BASE
    published_versions = fetch_release_versions(index_base=index_base, package_name=inputs.package_name)
    if inputs.version in published_versions:
        raise ValueError(
            f"Version `{inputs.version}` is already present on {inputs.target} for package `{inputs.package_name}`."
        )


def run_guard(args: argparse.Namespace) -> PublishGuardInputs:
    version = load_sdk_version(args.version_file)
    github_ref = (args.github_ref or "").strip()
    inputs = PublishGuardInputs(
        target=args.target,
        package_name=args.package_name,
        version=version,
        github_ref=github_ref,
        skip_remote_check=bool(args.skip_remote_check),
    )
    ensure_target_constraints(inputs)
    ensure_version_not_already_published(inputs)
    return inputs


def main() -> int:
    args = parse_args()
    try:
        inputs = run_guard(args)
    except Exception as exc:  # noqa: BLE001 - CLI boundary.
        print(f"[sdk-publish-guard] FAILED: {exc}", file=sys.stderr)
        return 1

    remote_status = "skipped" if inputs.skip_remote_check else "checked"
    print(
        "[sdk-publish-guard] OK "
        f"target={inputs.target} version={inputs.version} ref={inputs.github_ref or '(none)'} "
        f"remote={remote_status}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
