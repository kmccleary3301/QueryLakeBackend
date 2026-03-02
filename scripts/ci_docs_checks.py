#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple
from urllib.parse import unquote


LINK_PATTERN = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
CODE_FENCE_PATTERN = re.compile(r"^\s*```")


def _normalize_target(raw: str) -> str:
    value = raw.strip()
    if value.startswith("<") and value.endswith(">"):
        value = value[1:-1].strip()
    # Drop optional title part: (path "title")
    if " " in value and not value.startswith("#") and not value.startswith(("http://", "https://", "mailto:", "data:")):
        value = value.split(" ", 1)[0].strip()
    return unquote(value)


def _github_slug(text: str) -> str:
    slug = text.strip().lower()
    slug = slug.replace("`", "")
    slug = re.sub(r"[^\w\- ]", "", slug)
    slug = slug.replace(" ", "-")
    slug = re.sub(r"-{2,}", "-", slug)
    return slug.strip("-")


def _extract_headings(md_path: Path, cache: Dict[Path, Set[str]]) -> Set[str]:
    cached = cache.get(md_path)
    if cached is not None:
        return cached

    if not md_path.exists():
        cache[md_path] = set()
        return cache[md_path]

    text = md_path.read_text(encoding="utf-8", errors="ignore")
    anchors: Set[str] = set()
    seen: Dict[str, int] = {}
    in_code = False
    for line in text.splitlines():
        if CODE_FENCE_PATTERN.match(line):
            in_code = not in_code
            continue
        if in_code:
            continue
        match = HEADING_PATTERN.match(line)
        if match is None:
            continue
        base = _github_slug(match.group(2))
        if not base:
            continue
        idx = seen.get(base, 0)
        seen[base] = idx + 1
        anchor = base if idx == 0 else f"{base}-{idx}"
        anchors.add(anchor)

    cache[md_path] = anchors
    return anchors


def _iter_markdown_files(repo_root: Path, paths: Iterable[str]) -> List[Path]:
    files: List[Path] = []
    for rel in paths:
        base = repo_root / rel
        if base.is_file() and base.suffix.lower() == ".md":
            files.append(base)
        elif base.is_dir():
            files.extend(sorted(base.rglob("*.md")))
    out = []
    for path in files:
        rel = path.relative_to(repo_root).as_posix()
        if rel.startswith("docs_tmp/"):
            continue
        out.append(path)
    return out


def _validate_file(repo_root: Path, md_path: Path, heading_cache: Dict[Path, Set[str]]) -> List[str]:
    errors: List[str] = []
    content = md_path.read_text(encoding="utf-8", errors="ignore")
    for match in LINK_PATTERN.finditer(content):
        raw_target = match.group(1)
        target = _normalize_target(raw_target)
        if not target:
            continue
        if target.startswith(("http://", "https://", "mailto:", "data:")):
            continue
        if "${{" in target or target.startswith("{{"):
            continue

        if target.startswith("#"):
            anchor = target[1:].strip().lower()
            anchors = _extract_headings(md_path, heading_cache)
            if anchor and anchor not in anchors:
                errors.append(
                    f"{md_path.relative_to(repo_root)}: missing local anchor '#{anchor}'"
                )
            continue

        path_part, _, anchor_part = target.partition("#")
        target_path = (md_path.parent / path_part).resolve()
        if not target_path.exists():
            errors.append(
                f"{md_path.relative_to(repo_root)}: missing path '{path_part}'"
            )
            continue

        if anchor_part and target_path.suffix.lower() == ".md":
            anchors = _extract_headings(target_path, heading_cache)
            anchor = anchor_part.strip().lower()
            if anchor and anchor not in anchors:
                errors.append(
                    f"{md_path.relative_to(repo_root)}: missing anchor '#{anchor}' in '{path_part}'"
                )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate local markdown links and anchors.")
    parser.add_argument(
        "--path",
        action="append",
        default=[],
        help="File or directory to scan. Defaults to README.md, CONTRIBUTING.md, docs/.",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    scan_paths = args.path or ["README.md", "CONTRIBUTING.md", "docs"]
    files = _iter_markdown_files(repo_root, scan_paths)
    heading_cache: Dict[Path, Set[str]] = {}

    all_errors: List[str] = []
    for file_path in files:
        all_errors.extend(_validate_file(repo_root, file_path, heading_cache))

    if all_errors:
        print("Docs checks failed:")
        for err in all_errors:
            print(f"- {err}")
        return 2

    print(f"Docs checks passed for {len(files)} markdown files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
