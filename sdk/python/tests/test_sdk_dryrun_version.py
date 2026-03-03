from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT = REPO_ROOT / "scripts" / "dev" / "prepare_sdk_dryrun_version.py"


def _write_version_file(path: Path, version: str) -> None:
    path.write_text(
        "\n".join(
            [
                "[project]",
                'name = "querylake-sdk"',
                f'version = "{version}"',
                "",
            ]
        ),
        encoding="utf-8",
    )


def _run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(SCRIPT), *args],
        check=False,
        capture_output=True,
        text=True,
    )


def test_prepare_dryrun_version_write_in_place(tmp_path: Path) -> None:
    version_file = tmp_path / "pyproject.toml"
    _write_version_file(version_file, "0.4.2")
    output_file = tmp_path / "github_output.txt"

    result = _run_script(
        "--version-file",
        str(version_file),
        "--token",
        "12345",
        "--write",
        "--github-output",
        str(output_file),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dryrun_version"] == "0.4.2.dev12345"

    updated = version_file.read_text(encoding="utf-8")
    assert 'version = "0.4.2.dev12345"' in updated
    out = output_file.read_text(encoding="utf-8")
    assert "dryrun_version=0.4.2.dev12345" in out


def test_prepare_dryrun_version_sanitizes_token(tmp_path: Path) -> None:
    version_file = tmp_path / "pyproject.toml"
    _write_version_file(version_file, "1.2.3")
    result = _run_script("--version-file", str(version_file), "--token", "ab-77_42")
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["dryrun_token"] == "7742"
    assert payload["dryrun_version"] == "1.2.3.dev7742"


def test_prepare_dryrun_version_rejects_non_stable_base(tmp_path: Path) -> None:
    version_file = tmp_path / "pyproject.toml"
    _write_version_file(version_file, "1.2.3rc1")
    result = _run_script("--version-file", str(version_file), "--token", "12", "--write")
    assert result.returncode == 1
    assert "requires stable base" in result.stderr
