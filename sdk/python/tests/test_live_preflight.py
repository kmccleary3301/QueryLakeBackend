from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "live_integration_preflight.py"


def _run(env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    merged = dict(os.environ)
    merged.update(env)
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=False,
        capture_output=True,
        text=True,
        env=merged,
    )


def test_live_preflight_requires_base_url() -> None:
    result = _run(
        {
            "QUERYLAKE_LIVE_BASE_URL": "",
            "QUERYLAKE_LIVE_OAUTH2": "",
            "QUERYLAKE_LIVE_API_KEY": "",
        }
    )
    assert result.returncode == 1
    assert "QUERYLAKE_LIVE_BASE_URL is required" in result.stderr


def test_live_preflight_blocks_non_staging_by_default() -> None:
    result = _run(
        {
            "QUERYLAKE_LIVE_BASE_URL": "https://prod.company.tld",
            "QUERYLAKE_LIVE_OAUTH2": "tok",
            "QUERYLAKE_LIVE_API_KEY": "",
            "QUERYLAKE_LIVE_ALLOW_NON_STAGING": "0",
        }
    )
    assert result.returncode == 1
    assert "Refusing to run against non-staging host" in result.stderr


def test_live_preflight_passes_with_safe_host() -> None:
    result = _run(
        {
            "QUERYLAKE_LIVE_BASE_URL": "https://staging.querylake.local",
            "QUERYLAKE_LIVE_OAUTH2": "tok",
            "QUERYLAKE_LIVE_API_KEY": "",
            "QUERYLAKE_LIVE_ALLOW_NON_STAGING": "0",
            "QUERYLAKE_LIVE_ALLOW_WRITE": "0",
        }
    )
    assert result.returncode == 0, result.stderr
    assert "\"hostname\": \"staging.querylake.local\"" in result.stdout
