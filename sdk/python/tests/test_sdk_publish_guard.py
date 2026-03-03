from __future__ import annotations

import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
GUARD_SCRIPT = REPO_ROOT / "scripts" / "dev" / "verify_sdk_publish_guard.py"


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


def _run_guard(*args: str) -> subprocess.CompletedProcess[str]:
    cmd = [sys.executable, str(GUARD_SCRIPT), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def test_publish_guard_testpypi_passes_without_ref(tmp_path: Path) -> None:
    version_file = tmp_path / "pyproject.toml"
    _write_version_file(version_file, "0.2.0rc1")
    result = _run_guard(
        "--target",
        "testpypi",
        "--version-file",
        str(version_file),
        "--skip-remote-check",
    )
    assert result.returncode == 0, result.stderr
    assert "[sdk-publish-guard] OK" in result.stdout


def test_publish_guard_pypi_requires_main_or_matching_tag(tmp_path: Path) -> None:
    version_file = tmp_path / "pyproject.toml"
    _write_version_file(version_file, "0.2.0")
    result = _run_guard(
        "--target",
        "pypi",
        "--version-file",
        str(version_file),
        "--github-ref",
        "refs/heads/feature/some-branch",
        "--skip-remote-check",
    )
    assert result.returncode == 1
    assert "requires main branch or matching version tag" in result.stderr


def test_publish_guard_pypi_rejects_prerelease(tmp_path: Path) -> None:
    version_file = tmp_path / "pyproject.toml"
    _write_version_file(version_file, "0.2.0rc1")
    result = _run_guard(
        "--target",
        "pypi",
        "--version-file",
        str(version_file),
        "--github-ref",
        "refs/heads/main",
        "--skip-remote-check",
    )
    assert result.returncode == 1
    assert "requires stable semver" in result.stderr


def test_publish_guard_pypi_accepts_matching_tag(tmp_path: Path) -> None:
    version_file = tmp_path / "pyproject.toml"
    _write_version_file(version_file, "0.2.0")
    result = _run_guard(
        "--target",
        "pypi",
        "--version-file",
        str(version_file),
        "--github-ref",
        "refs/tags/v0.2.0",
        "--skip-remote-check",
    )
    assert result.returncode == 0, result.stderr
    assert "target=pypi version=0.2.0 ref=refs/tags/v0.2.0" in result.stdout
