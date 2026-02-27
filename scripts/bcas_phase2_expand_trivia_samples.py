#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List

from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.create_db_session import initialize_database_engine


TRIVIA_NAME_RE = re.compile(r"triviaqa__triviaqa_hf_train_(\d+)_(\d+)__")


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _row_index_from_document_name(name: str) -> int | None:
    m = TRIVIA_NAME_RE.search(name or "")
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Expand TriviaQA sample rows for BCAS phase-2 eval.")
    parser.add_argument(
        "--account-config",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json"),
    )
    parser.add_argument(
        "--samples-root-in",
        type=Path,
        default=Path("docs_tmp/RAG/ingest_logs_2026-02-23"),
    )
    parser.add_argument(
        "--samples-root-out",
        type=Path,
        default=Path("docs_tmp/RAG/ingest_logs_2026-02-24_aug"),
    )
    parser.add_argument("--target-trivia-rows", type=int, default=240)
    parser.add_argument("--db-limit", type=int, default=4000)
    args = parser.parse_args()

    cfg = _load_json(args.account_config)
    collections = cfg.get("collections", {})
    if not isinstance(collections, dict):
        raise SystemExit("invalid account config: missing collections")
    trivia_collection_id = collections.get("triviaqa")
    if not isinstance(trivia_collection_id, str) or len(trivia_collection_id) == 0:
        raise SystemExit("invalid account config: missing triviaqa collection")

    # Copy over non-trivia sample logs unchanged.
    args.samples_root_out.mkdir(parents=True, exist_ok=True)
    for dataset in ("hotpotqa", "multihop"):
        src = args.samples_root_in / f"{dataset}.samples.jsonl"
        dst = args.samples_root_out / f"{dataset}.samples.jsonl"
        if src.exists():
            dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    db, _ = initialize_database_engine()
    stmt = text(
        """
        SELECT document_name, LEFT(text, 1200) AS preview
        FROM DocumentChunk
        WHERE collection_id = :cid
          AND document_name LIKE 'triviaqa__triviaqa_hf_train_%'
        ORDER BY creation_timestamp ASC
        LIMIT :lim
        """
    ).bindparams(cid=trivia_collection_id, lim=max(1, int(args.db_limit)))
    rows = db.exec(stmt).all()

    by_row_index: Dict[int, Dict[str, Any]] = {}
    for document_name, preview in rows:
        if not isinstance(document_name, str):
            continue
        row_index = _row_index_from_document_name(document_name)
        if row_index is None:
            continue
        if row_index not in by_row_index:
            by_row_index[row_index] = {
                "document_name": document_name,
                "preview": str(preview or ""),
            }

    # Build row-index -> question map using HF dataset (same source used for ingestion).
    from datasets import load_dataset  # lazy import to avoid startup cost when unused

    ds = load_dataset("trivia_qa", "rc.wikipedia", split="train")
    max_needed = max(by_row_index.keys()) if by_row_index else -1
    if max_needed < 0:
        raise SystemExit("no TriviaQA document_name rows discovered in collection")
    if max_needed >= len(ds):
        max_needed = len(ds) - 1

    # Start with existing trivia rows to preserve deterministic seed order where possible.
    existing = _iter_jsonl(args.samples_root_in / "triviaqa.samples.jsonl")
    existing_by_doc_name = {
        str(row.get("file_name", "")): row
        for row in existing
        if isinstance(row.get("file_name"), str)
    }

    out_rows: List[Dict[str, Any]] = []
    target = max(1, int(args.target_trivia_rows))
    for row_index in sorted(by_row_index.keys()):
        entry = by_row_index[row_index]
        file_name = entry["document_name"]
        if file_name in existing_by_doc_name:
            out_rows.append(existing_by_doc_name[file_name])
            if len(out_rows) >= target:
                break
            continue
        if row_index > max_needed:
            continue
        ds_row = ds[int(row_index)]
        question = ds_row.get("question")
        qid = ds_row.get("question_id")
        if not isinstance(question, str) or len(question.strip()) == 0:
            continue
        out_rows.append(
            {
                "dataset": "triviaqa",
                "file_name": file_name,
                "content_hash": None,
                "title": file_name.rsplit("__", 1)[0],
                "source": "hf://trivia_qa/rc.wikipedia/train",
                "preview": entry["preview"],
                "metadata": {
                    "dataset": "triviaqa",
                    "source": "hf://trivia_qa/rc.wikipedia/train",
                    "split": "train",
                    "row_index": int(row_index),
                    "question": question,
                    "question_id": qid,
                },
            }
        )
        if len(out_rows) >= target:
            break

    out_path = args.samples_root_out / "triviaqa.samples.jsonl"
    with out_path.open("w", encoding="utf-8") as handle:
        for row in out_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    result = {
        "samples_root_out": str(args.samples_root_out),
        "trivia_rows_out": len(out_rows),
        "target_trivia_rows": target,
        "db_rows_scanned": len(rows),
        "unique_row_indices_in_db": len(by_row_index),
        "output_file": str(out_path),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
