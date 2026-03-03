#!/usr/bin/env python3
"""Preflight checks for QueryLake live staging integration tests."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse


SAFE_HOST_HINTS = ("staging", "dev", "localhost", "127.0.0.1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate live integration environment contract.")
    parser.add_argument("--github-output", default="")
    return parser.parse_args()


def fail(message: str) -> int:
    print(f"[live-preflight] FAILED: {message}", file=sys.stderr)
    return 1


def main() -> int:
    _ = parse_args()
    base_url = (os.getenv("QUERYLAKE_LIVE_BASE_URL") or "").strip()
    oauth2 = (os.getenv("QUERYLAKE_LIVE_OAUTH2") or "").strip()
    api_key = (os.getenv("QUERYLAKE_LIVE_API_KEY") or "").strip()
    allow_non_staging = (os.getenv("QUERYLAKE_LIVE_ALLOW_NON_STAGING") or "").strip() == "1"
    allow_write = (os.getenv("QUERYLAKE_LIVE_ALLOW_WRITE") or "").strip() == "1"
    write_collection = (os.getenv("QUERYLAKE_LIVE_TEST_COLLECTION_ID") or "").strip()

    if not base_url:
        return fail("QUERYLAKE_LIVE_BASE_URL is required.")
    if not oauth2 and not api_key:
        return fail("Either QUERYLAKE_LIVE_OAUTH2 or QUERYLAKE_LIVE_API_KEY is required.")

    hostname = (urlparse(base_url).hostname or "").lower()
    if not allow_non_staging and hostname and not any(hint in hostname for hint in SAFE_HOST_HINTS):
        return fail(
            "Refusing to run against non-staging host. "
            "Set QUERYLAKE_LIVE_ALLOW_NON_STAGING=1 to override intentionally."
        )

    if allow_write and not write_collection:
        return fail("QUERYLAKE_LIVE_ALLOW_WRITE=1 requires QUERYLAKE_LIVE_TEST_COLLECTION_ID.")

    run_namespace = f"qlsdk_live_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%d%H%M%S')}"
    payload = {
        "base_url": base_url,
        "hostname": hostname,
        "auth_mode": "oauth2" if oauth2 else "api_key",
        "allow_non_staging": allow_non_staging,
        "allow_write": allow_write,
        "write_collection_id_set": bool(write_collection),
        "run_namespace": run_namespace,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))

    github_output = (os.getenv("GITHUB_OUTPUT") or "").strip()
    if github_output:
        output_path = Path(github_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            handle.write(f"run_namespace={run_namespace}\n")
            handle.write(f"hostname={hostname}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
