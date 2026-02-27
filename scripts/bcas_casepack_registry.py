#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_INCLUDE_PATTERNS = [
    "BCAS_PHASE2_3LANE_TRACK_*.json",
    "BCAS_PHASE2_*_EVAL_METRICS_*.json",
    "BCAS_PHASE2_*_STRESS_*.json",
    "BCAS_PHASE2_*_GATE*.json",
    "BCAS_PHASE2_*_GATE_REPORT.md",
    "BCAS_PHASE2_STRICT_QUEUE_SWEEP_*.json",
    "BCAS_PHASE2_STRICT_QUEUE_SWEEP_*.md",
]


@dataclass
class PackedFile:
    source: Path
    source_rel: str
    packed_rel: str
    sha256: str
    bytes_size: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _latest_by_mtime(paths: Iterable[Path]) -> Path | None:
    rows = [path for path in paths if path.is_file()]
    if not rows:
        return None
    rows.sort(key=lambda p: (p.stat().st_mtime, str(p)))
    return rows[-1]


def _collect_files(
    *,
    source_root: Path,
    include_patterns: List[str],
    latest_per_pattern: bool,
) -> List[Path]:
    selected: List[Path] = []
    for pattern in include_patterns:
        matches = sorted(source_root.glob(pattern))
        if not matches:
            continue
        if latest_per_pattern:
            latest = _latest_by_mtime(matches)
            if latest is not None:
                selected.append(latest)
        else:
            selected.extend([path for path in matches if path.is_file()])
    # stable de-dup
    seen = set()
    out: List[Path] = []
    for path in selected:
        key = str(path.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(path)
    out.sort(key=lambda p: p.name)
    return out


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return path.name


def _ensure_registry(path: Path) -> Dict[str, object]:
    if path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                payload.setdefault("version", "v1")
                payload.setdefault("packs", [])
                return payload
        except Exception:
            pass
    return {"version": "v1", "packs": []}


def _export(
    *,
    label: str,
    source_root: Path,
    out_root: Path,
    include_patterns: List[str],
    latest_per_pattern: bool,
    make_zip: bool,
    registry_path: Path,
) -> Dict[str, object]:
    source_root = source_root.resolve()
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    pack_id = f"{label}_{stamp}"
    pack_dir = out_root / pack_id
    files_dir = pack_dir / "files"
    files_dir.mkdir(parents=True, exist_ok=True)

    selected = _collect_files(
        source_root=source_root,
        include_patterns=include_patterns,
        latest_per_pattern=latest_per_pattern,
    )
    if len(selected) == 0:
        raise SystemExit(f"No files matched include patterns under {source_root}")

    packed: List[PackedFile] = []
    for idx, source in enumerate(selected):
        packed_name = f"{idx+1:03d}_{source.name}"
        packed_path = files_dir / packed_name
        shutil.copy2(source, packed_path)
        packed.append(
            PackedFile(
                source=source,
                source_rel=_safe_rel(source, source_root),
                packed_rel=str(Path("files") / packed_name),
                sha256=_sha256(packed_path),
                bytes_size=packed_path.stat().st_size,
            )
        )

    manifest = {
        "version": "v1",
        "pack_id": pack_id,
        "label": label,
        "created_at_unix": time.time(),
        "source_root": str(source_root),
        "include_patterns": include_patterns,
        "latest_per_pattern": bool(latest_per_pattern),
        "file_count": len(packed),
        "files": [
            {
                "source": str(row.source),
                "source_rel": row.source_rel,
                "packed_rel": row.packed_rel,
                "sha256": row.sha256,
                "bytes": row.bytes_size,
            }
            for row in packed
        ],
    }

    manifest_path = pack_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    zip_path = None
    if make_zip:
        archive_base = pack_dir / pack_id
        zip_path = Path(shutil.make_archive(str(archive_base), "zip", root_dir=pack_dir))

    registry = _ensure_registry(registry_path)
    packs = registry.get("packs")
    if not isinstance(packs, list):
        packs = []
    packs.append(
        {
            "pack_id": pack_id,
            "label": label,
            "created_at_unix": manifest["created_at_unix"],
            "manifest": str(manifest_path),
            "zip": str(zip_path) if zip_path is not None else None,
            "file_count": len(packed),
        }
    )
    registry["packs"] = packs
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(json.dumps(registry, indent=2), encoding="utf-8")

    return {
        "ok": True,
        "action": "export",
        "pack_id": pack_id,
        "pack_dir": str(pack_dir),
        "manifest": str(manifest_path),
        "zip": str(zip_path) if zip_path is not None else None,
        "registry": str(registry_path),
        "file_count": len(packed),
    }


def _restore(
    *,
    manifest_path: Path,
    restore_root: Path,
    flatten: bool,
) -> Dict[str, object]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SystemExit(f"Invalid manifest: {manifest_path}")
    files = payload.get("files")
    if not isinstance(files, list) or len(files) == 0:
        raise SystemExit(f"Manifest has no files: {manifest_path}")

    pack_dir = manifest_path.parent
    restore_root.mkdir(parents=True, exist_ok=True)
    restored: List[Tuple[str, str]] = []

    for row in files:
        if not isinstance(row, dict):
            continue
        packed_rel = str(row.get("packed_rel", "")).strip()
        if len(packed_rel) == 0:
            continue
        packed_path = (pack_dir / packed_rel).resolve()
        if not packed_path.exists():
            raise SystemExit(f"Packed file missing: {packed_path}")
        source_rel = str(row.get("source_rel", packed_path.name))
        dest_rel = Path(source_rel).name if flatten else Path(source_rel)
        dest_path = (restore_root / dest_rel).resolve()
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(packed_path, dest_path)
        restored.append((str(packed_path), str(dest_path)))

    out = {
        "ok": True,
        "action": "restore",
        "manifest": str(manifest_path),
        "restore_root": str(restore_root),
        "restored_files": len(restored),
        "rows": [{"from": src, "to": dst} for src, dst in restored],
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Export/restore BCAS retrieval casepacks with registry tracking.")
    sub = parser.add_subparsers(dest="action", required=True)

    export = sub.add_parser("export", help="Export latest retrieval artifacts into a versioned casepack.")
    export.add_argument("--label", type=str, required=True)
    export.add_argument("--source-root", type=Path, default=Path("docs_tmp/RAG"))
    export.add_argument("--out-root", type=Path, default=Path("docs_tmp/RAG/casepacks"))
    export.add_argument("--registry", type=Path, default=Path("docs_tmp/RAG/casepacks/CASEPACK_REGISTRY.json"))
    export.add_argument("--include-glob", action="append", default=[])
    export.add_argument("--all-matches", action="store_true")
    export.add_argument("--zip", action="store_true")

    restore = sub.add_parser("restore", help="Restore files from a casepack manifest.")
    restore.add_argument("--manifest", type=Path, required=True)
    restore.add_argument("--restore-root", type=Path, default=Path("docs_tmp/RAG/restored_casepacks"))
    restore.add_argument("--flatten", action="store_true")

    args = parser.parse_args()
    if args.action == "export":
        include_patterns = args.include_glob if len(args.include_glob) > 0 else list(DEFAULT_INCLUDE_PATTERNS)
        payload = _export(
            label=str(args.label),
            source_root=args.source_root,
            out_root=args.out_root,
            include_patterns=include_patterns,
            latest_per_pattern=not bool(args.all_matches),
            make_zip=bool(args.zip),
            registry_path=args.registry,
        )
    else:
        payload = _restore(
            manifest_path=args.manifest,
            restore_root=args.restore_root,
            flatten=bool(args.flatten),
        )

    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
