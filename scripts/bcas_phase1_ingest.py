#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import sys
import time
import zipfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlencode

import requests

try:
    import pandas as pd
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency `pandas`: {exc}")


def detect_default_source_root() -> Path:
    env_value = os.environ.get("BCAS_SOURCE_ROOT", "").strip()
    if env_value:
        return Path(env_value).expanduser()

    script_path = Path(__file__).resolve()
    candidate_paths = [
        script_path.parents[2] / "rag_project",
        Path("/shared_folders/querylake_server/rag_project"),
    ]
    for candidate in candidate_paths:
        if candidate.exists():
            return candidate
    return candidate_paths[-1]


DEFAULT_SOURCE_ROOT = detect_default_source_root()
DEFAULT_MANIFEST_OUT = Path("docs_tmp/RAG/BCAS_PHASE1_INGESTION_MANIFEST.json")
DEFAULT_SAMPLES_OUT = Path("docs_tmp/RAG/BCAS_PHASE1_INGESTION_SAMPLES.jsonl")


@dataclass
class DatasetStats:
    dataset: str
    source_files: List[str] = field(default_factory=list)
    missing_files: List[str] = field(default_factory=list)
    raw_records_seen: int = 0
    unique_content_records: int = 0
    duplicate_within_dataset: int = 0
    duplicate_across_selected_datasets: int = 0
    skipped_existing_in_collection: int = 0
    candidate_upload_records: int = 0
    uploaded_records: int = 0
    upload_batches: int = 0
    parse_errors: int = 0
    limit_hit: bool = False


@dataclass
class UploadStats:
    attempted: bool = False
    scan_existing_enabled: bool = False
    existing_integrity_hashes_found: int = 0
    upload_batches_ok: int = 0
    upload_batches_failed: int = 0
    uploaded_records: int = 0
    last_error: Optional[str] = None


def resolve_default_embedding_model(
    *,
    api_base_url: str,
    api_key: str,
    timeout_s: int,
) -> str:
    resp = requests.get(
        f"{api_base_url.rstrip('/')}/api/get_available_models",
        json={"auth": {"api_key": api_key}},
        timeout=timeout_s,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("success") is False:
        raise RuntimeError(body.get("note") or body.get("error") or "get_available_models failed")
    result = body.get("result") or {}
    available_models = result.get("available_models") if isinstance(result, dict) else None
    default_models = available_models.get("default_models") if isinstance(available_models, dict) else None
    embedding_model = default_models.get("embedding") if isinstance(default_models, dict) else None
    if not isinstance(embedding_model, str) or len(embedding_model.strip()) == 0:
        raise RuntimeError("No default embedding model configured in get_available_models.")
    return embedding_model.strip()


def run_embedding_preflight(
    *,
    api_base_url: str,
    api_key: str,
    timeout_s: int,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    model = model_override.strip() if isinstance(model_override, str) and len(model_override.strip()) > 0 else None
    if model is None:
        model = resolve_default_embedding_model(
            api_base_url=api_base_url,
            api_key=api_key,
            timeout_s=timeout_s,
        )

    resp = requests.post(
        f"{api_base_url.rstrip('/')}/v1/embeddings",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={"model": model, "input": ["querylake embedding preflight"]},
        timeout=timeout_s,
    )
    if resp.status_code != 200:
        raise RuntimeError(f"/v1/embeddings returned HTTP {resp.status_code}: {resp.text[:500]}")

    body = resp.json()
    if isinstance(body, dict) and ("error" in body or "message" in body and "data" not in body):
        raise RuntimeError(f"/v1/embeddings returned error payload: {body}")
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, list) or len(data) == 0:
        raise RuntimeError(f"/v1/embeddings returned no vectors: {body}")
    first = data[0] if isinstance(data[0], dict) else {}
    vector = first.get("embedding") if isinstance(first, dict) else None
    if not isinstance(vector, list) or len(vector) == 0:
        raise RuntimeError(f"/v1/embeddings returned malformed vector payload: {body}")
    return {
        "ok": True,
        "model": model,
        "dimensions": len(vector),
    }


def parse_csv(value: str) -> List[str]:
    return [piece.strip() for piece in value.split(",") if piece.strip()]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sanitize_filename(value: str, *, max_len: int = 80) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._ -]+", "_", value).strip().strip(".")
    cleaned = re.sub(r"\s+", " ", cleaned)
    if len(cleaned) == 0:
        cleaned = "doc"
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip(" ._")
    return cleaned or "doc"


def make_doc_filename(dataset: str, title: str, content_hash: str) -> str:
    prefix = sanitize_filename(dataset.replace(" ", "_"), max_len=20)
    stem = sanitize_filename(title, max_len=70)
    return f"{prefix}__{stem}__{content_hash[:12]}.md"


def iter_hotpot_documents(hotpot_file: Path) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    with hotpot_file.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    for entry_idx, entry in enumerate(rows):
        context = entry.get("context", [])
        if not isinstance(context, list):
            continue
        for article_idx, article in enumerate(context):
            if not isinstance(article, (list, tuple)) or len(article) != 2:
                continue
            title, sentences = article
            if not isinstance(title, str) or not isinstance(sentences, list):
                continue
            text = "\n".join([str(s) for s in sentences if isinstance(s, str)]).strip()
            if len(text) == 0:
                continue
            metadata = {
                "dataset": "hotpotqa",
                "source": str(hotpot_file),
                "entry_index": entry_idx,
                "article_index": article_idx,
                "question": entry.get("question"),
                "level": entry.get("level"),
            }
            yield title, text, metadata


def _parse_2wiki_context(raw_context: Any) -> List[Any]:
    if isinstance(raw_context, str):
        return json.loads(raw_context)
    if isinstance(raw_context, list):
        return raw_context
    return []


def iter_multihop_documents(multihop_files: Sequence[Path]) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    for parquet_path in multihop_files:
        df = pd.read_parquet(parquet_path)
        for row_idx, (_, row) in enumerate(df.iterrows()):
            context_articles = _parse_2wiki_context(row.get("context"))
            if not isinstance(context_articles, list):
                continue
            for article_idx, article in enumerate(context_articles):
                if not isinstance(article, (list, tuple)) or len(article) != 2:
                    continue
                title, paragraphs = article
                if not isinstance(title, str) or not isinstance(paragraphs, list):
                    continue
                text = "\n".join([str(s) for s in paragraphs if isinstance(s, str)]).strip()
                if len(text) == 0:
                    continue
                metadata = {
                    "dataset": "multihop",
                    "source": str(parquet_path),
                    "row_index": int(row_idx),
                    "article_index": int(article_idx),
                    "question": row.get("question"),
                }
                yield title, text, metadata


def iter_triviaqa_documents(trivia_files: Sequence[Path]) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    for parquet_path in trivia_files:
        df = pd.read_parquet(parquet_path)
        for row_idx, (_, row) in enumerate(df.iterrows()):
            entity_pages = row.get("entity_pages")
            wiki_context: List[str] = []
            if isinstance(entity_pages, dict):
                context_value = entity_pages.get("wiki_context", [])
                if isinstance(context_value, list):
                    wiki_context = [str(x) for x in context_value if isinstance(x, str)]
            for page_idx, page_text in enumerate(wiki_context):
                content = page_text.strip()
                if len(content) == 0:
                    continue
                title = f"triviaqa_page_{row_idx}_{page_idx}"
                metadata = {
                    "dataset": "triviaqa",
                    "source": str(parquet_path),
                    "row_index": int(row_idx),
                    "page_index": int(page_idx),
                    "question": row.get("question"),
                }
                yield title, content, metadata


def iter_triviaqa_documents_hf(
    splits: Sequence[str],
) -> Iterator[Tuple[str, str, Dict[str, Any]]]:
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(f"HuggingFace datasets loader unavailable: {exc}")

    for split in splits:
        ds = load_dataset("trivia_qa", "rc.wikipedia", split=split, streaming=True)
        for row_idx, row in enumerate(ds):
            entity_pages = row.get("entity_pages")
            wiki_context: List[str] = []
            if isinstance(entity_pages, dict):
                context_value = entity_pages.get("wiki_context", [])
                if isinstance(context_value, list):
                    wiki_context = [str(x) for x in context_value if isinstance(x, str)]

            for page_idx, page_text in enumerate(wiki_context):
                content = page_text.strip()
                if len(content) == 0:
                    continue
                title = f"triviaqa_hf_{split}_{row_idx}_{page_idx}"
                metadata = {
                    "dataset": "triviaqa",
                    "source": f"hf://trivia_qa/rc.wikipedia/{split}",
                    "split": split,
                    "row_index": int(row_idx),
                    "page_index": int(page_idx),
                    "question": row.get("question"),
                    "question_id": row.get("question_id"),
                }
                yield title, content, metadata


def resolve_dataset_files(
    source_root: Path,
    dataset: str,
    hotpot_file: str,
    multihop_files: Sequence[str],
    trivia_files: Sequence[str],
) -> Tuple[List[Path], List[Path]]:
    if dataset == "hotpotqa":
        candidate = source_root / "hotpotqa" / hotpot_file
        return ([candidate] if candidate.exists() else [], [candidate] if not candidate.exists() else [])
    if dataset == "multihop":
        base = source_root / "2WikiMultihopQA"
        resolved = [base / rel for rel in multihop_files]
        existing = [p for p in resolved if p.exists()]
        missing = [p for p in resolved if not p.exists()]
        return existing, missing
    if dataset == "triviaqa":
        base = source_root / "triviaqa_wikipedia"
        resolved = [base / rel for rel in trivia_files]
        existing = [p for p in resolved if p.exists()]
        missing = [p for p in resolved if not p.exists()]
        return existing, missing
    raise ValueError(f"Unsupported dataset `{dataset}`")


def scan_existing_integrity_hashes(
    *,
    api_base_url: str,
    api_key: str,
    collection_id: str,
    page_size: int = 200,
    timeout_s: int = 120,
) -> set[str]:
    integrities: set[str] = set()
    offset = 0
    while True:
        payload = {
            "auth": {"api_key": api_key},
            "query": "",
            "collection_ids": [collection_id],
            "limit": max(1, min(200, int(page_size))),
            "offset": int(offset),
            "sort_by": "integrity_sha256",
            "table": "document",
        }
        resp = requests.get(
            f"{api_base_url.rstrip('/')}/api/search_bm25",
            json=payload,
            timeout=timeout_s,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("success") is False:
            raise RuntimeError(body.get("note") or body.get("error") or "search_bm25 failed")
        rows = body.get("result") or []
        if not isinstance(rows, list):
            raise RuntimeError("search_bm25 returned non-list result")
        if len(rows) == 0:
            break
        for row in rows:
            digest = row.get("integrity_sha256")
            if isinstance(digest, str) and len(digest) > 0:
                integrities.add(digest)
        offset += len(rows)
    return integrities


def upload_zip_batch(
    *,
    api_base_url: str,
    api_key: str,
    collection_id: str,
    file_entries: Dict[str, bytes],
    create_embeddings: bool,
    await_embedding: bool,
    timeout_s: int,
) -> Dict[str, Any]:
    if len(file_entries) == 0:
        return {"document_results": [], "time_log": {}}

    archive = io.BytesIO()
    with zipfile.ZipFile(archive, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_name, file_bytes in file_entries.items():
            zf.writestr(file_name, file_bytes)
    archive.seek(0)

    args_post_call = {
        "auth": {"api_key": api_key},
        "collection_hash_id": collection_id,
        "create_embeddings": bool(create_embeddings),
        "await_embedding": bool(await_embedding),
    }
    encoded = urlencode({"parameters": json.dumps(args_post_call)})

    files = {
        "file": ("bcas_batch.zip", archive.getvalue(), "application/zip"),
    }
    resp = requests.post(
        f"{api_base_url.rstrip('/')}/upload_document?{encoded}",
        files=files,
        timeout=timeout_s,
    )
    resp.raise_for_status()
    payload = resp.json()
    if payload.get("success") is False:
        raise RuntimeError(payload.get("note") or payload.get("error") or "upload failed")
    result = payload.get("result")
    if not isinstance(result, dict):
        return {"document_results": [], "time_log": {}}
    return result


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Phase-1 BCAS ingestion runner for QueryLake. Reads source datasets in place "
            "from rag_project and supports dry-run manifests plus optional upload batching."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=DEFAULT_SOURCE_ROOT,
        help="Path to BCAS source repository root.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="hotpotqa,multihop,triviaqa",
        help="Comma-separated datasets to process.",
    )
    parser.add_argument(
        "--hotpot-file",
        type=str,
        default="hotpot_train_v1.1.json",
        help="HotpotQA source filename under hotpotqa/.",
    )
    parser.add_argument(
        "--multihop-files",
        type=str,
        default="train.parquet,dev.parquet,test.parquet",
        help="Comma-separated 2Wiki parquet filenames under 2WikiMultihopQA/.",
    )
    parser.add_argument(
        "--triviaqa-files",
        type=str,
        default="trivia_qa_wiki_1.parquet,trivia_qa_wiki_2.parquet,trivia_qa_wiki_3.parquet,trivia_qa_wiki_4.parquet,trivia_qa_wiki_5.parquet,trivia_qa_wiki_6.parquet,trivia_qa_wiki_7.parquet,trivia_qa_wiki_8.parquet",
        help="Comma-separated TriviaQA parquet filenames under triviaqa_wikipedia/.",
    )
    parser.add_argument(
        "--triviaqa-hf-fallback",
        action="store_true",
        default=True,
        help="If local TriviaQA parquet files are missing, stream from HF trivia_qa rc.wikipedia.",
    )
    parser.add_argument(
        "--no-triviaqa-hf-fallback",
        dest="triviaqa_hf_fallback",
        action="store_false",
        help="Disable HF fallback for TriviaQA.",
    )
    parser.add_argument(
        "--triviaqa-hf-splits",
        type=str,
        default="train,validation,test",
        help="Comma-separated HF splits for TriviaQA fallback.",
    )
    parser.add_argument(
        "--limit-per-dataset",
        type=int,
        default=0,
        help="Optional cap on raw records per dataset (0 means no cap).",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Number of candidate docs per dataset to emit into samples output.",
    )
    parser.add_argument(
        "--manifest-out",
        type=Path,
        default=DEFAULT_MANIFEST_OUT,
        help="Manifest output path.",
    )
    parser.add_argument(
        "--samples-out",
        type=Path,
        default=DEFAULT_SAMPLES_OUT,
        help="JSONL output path for sample candidate docs.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Upload candidate docs to QueryLake using /upload_document archive flow.",
    )
    parser.add_argument("--api-base-url", type=str, default="http://localhost:8000")
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("QUERYLAKE_API_KEY", "").strip(),
        help="QueryLake API key (or set QUERYLAKE_API_KEY).",
    )
    parser.add_argument(
        "--collection-id",
        type=str,
        default=os.getenv("QUERYLAKE_COLLECTION_ID", "").strip(),
        help="Target collection id (or set QUERYLAKE_COLLECTION_ID).",
    )
    parser.add_argument("--batch-size", type=int, default=400, help="Upload archive batch size.")
    parser.add_argument("--http-timeout-s", type=int, default=180, help="HTTP timeout per request.")
    parser.add_argument(
        "--create-embeddings",
        action="store_true",
        default=True,
        help="Request embedding creation during upload.",
    )
    parser.add_argument(
        "--no-create-embeddings",
        dest="create_embeddings",
        action="store_false",
        help="Disable embedding creation during upload.",
    )
    parser.add_argument(
        "--await-embedding",
        action="store_true",
        default=True,
        help="Wait for embedding/chunk processing per uploaded batch.",
    )
    parser.add_argument(
        "--no-await-embedding",
        dest="await_embedding",
        action="store_false",
        help="Do not wait for embedding/chunk completion.",
    )
    parser.add_argument(
        "--strict-embedding-preflight",
        action="store_true",
        default=True,
        help="Hard-fail before ingestion if /v1/embeddings probe fails.",
    )
    parser.add_argument(
        "--no-strict-embedding-preflight",
        dest="strict_embedding_preflight",
        action="store_false",
        help="Disable strict embedding preflight.",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="",
        help="Optional embedding model override for preflight probe.",
    )
    parser.add_argument(
        "--preflight-timeout-s",
        type=int,
        default=30,
        help="HTTP timeout for strict preflight checks.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip docs whose integrity hash already exists in target collection.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Do not rescan and skip existing collection documents.",
    )
    return parser


def main() -> int:
    parser = build_arg_parser()
    args = parser.parse_args()

    started_at = time.time()
    source_root = args.source_root.resolve()
    datasets = parse_csv(args.datasets)
    multihop_files = parse_csv(args.multihop_files)
    triviaqa_files = parse_csv(args.triviaqa_files)
    triviaqa_hf_splits = parse_csv(args.triviaqa_hf_splits)

    if len(datasets) == 0:
        raise SystemExit("No datasets selected.")
    for dataset_name in datasets:
        if dataset_name not in {"hotpotqa", "multihop", "triviaqa"}:
            raise SystemExit(f"Unsupported dataset: {dataset_name}")

    upload_stats = UploadStats(
        attempted=bool(args.upload),
        scan_existing_enabled=bool(args.upload and args.skip_existing),
    )
    embedding_preflight: Dict[str, Any] = {"enabled": False}

    existing_integrities: set[str] = set()
    if args.upload:
        if len(args.api_key) == 0 or len(args.collection_id) == 0:
            raise SystemExit("Upload mode requires --api-key and --collection-id (or env vars).")
        if args.create_embeddings and args.strict_embedding_preflight:
            try:
                embedding_preflight = {
                    "enabled": True,
                    **run_embedding_preflight(
                        api_base_url=args.api_base_url,
                        api_key=args.api_key,
                        timeout_s=int(args.preflight_timeout_s),
                        model_override=args.embedding_model,
                    ),
                }
                print(
                    f"[preflight] embedding probe OK model={embedding_preflight['model']} "
                    f"dimensions={embedding_preflight['dimensions']}"
                )
            except Exception as exc:
                raise SystemExit(
                    "Embedding preflight failed. Refusing to ingest with --create-embeddings enabled.\n"
                    f"Reason: {exc}\n"
                    "If you intentionally want non-embedding ingest, use --no-create-embeddings.\n"
                    "If you intentionally want to bypass this gate, use --no-strict-embedding-preflight."
                )
        if args.skip_existing:
            try:
                existing_integrities = scan_existing_integrity_hashes(
                    api_base_url=args.api_base_url,
                    api_key=args.api_key,
                    collection_id=args.collection_id,
                    timeout_s=args.http_timeout_s,
                )
                upload_stats.existing_integrity_hashes_found = len(existing_integrities)
                print(f"[scan] existing integrity hashes in collection: {len(existing_integrities)}")
            except Exception as exc:
                upload_stats.last_error = f"existing scan failed: {exc}"
                raise

    args.samples_out.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_out.parent.mkdir(parents=True, exist_ok=True)

    sample_handles: Dict[str, int] = {name: 0 for name in datasets}
    sample_fp = args.samples_out.open("w", encoding="utf-8")

    global_seen_hashes: set[str] = set()
    per_dataset_stats: Dict[str, DatasetStats] = {name: DatasetStats(dataset=name) for name in datasets}

    try:
        for dataset_name in datasets:
            stats = per_dataset_stats[dataset_name]
            existing_paths, missing_paths = resolve_dataset_files(
                source_root=source_root,
                dataset=dataset_name,
                hotpot_file=args.hotpot_file,
                multihop_files=multihop_files,
                trivia_files=triviaqa_files,
            )
            stats.source_files = [str(p) for p in existing_paths]
            stats.missing_files = [str(p) for p in missing_paths]

            if len(existing_paths) == 0:
                if dataset_name == "triviaqa" and args.triviaqa_hf_fallback:
                    stats.source_files = [f"hf://trivia_qa/rc.wikipedia/{s}" for s in triviaqa_hf_splits]
                    stats.missing_files = [str(p) for p in missing_paths]
                else:
                    print(f"[{dataset_name}] no source files found, skipping dataset.")
                    continue

            local_seen_hashes: set[str] = set()
            batch_files: Dict[str, bytes] = {}

            if dataset_name == "hotpotqa":
                doc_iter = iter_hotpot_documents(existing_paths[0])
            elif dataset_name == "multihop":
                doc_iter = iter_multihop_documents(existing_paths)
            elif dataset_name == "triviaqa":
                if len(existing_paths) > 0:
                    doc_iter = iter_triviaqa_documents(existing_paths)
                else:
                    print(
                        f"[{dataset_name}] local files missing; falling back to HF splits={triviaqa_hf_splits}"
                    )
                    doc_iter = iter_triviaqa_documents_hf(triviaqa_hf_splits)
            else:
                raise RuntimeError(f"Unhandled dataset: {dataset_name}")

            for title, text, metadata in doc_iter:
                if args.limit_per_dataset and stats.raw_records_seen >= int(args.limit_per_dataset):
                    stats.limit_hit = True
                    break

                stats.raw_records_seen += 1
                text_bytes = text.encode("utf-8", errors="ignore")
                content_hash = sha256_bytes(text_bytes)

                if content_hash in local_seen_hashes:
                    stats.duplicate_within_dataset += 1
                    continue
                local_seen_hashes.add(content_hash)

                if content_hash in global_seen_hashes:
                    stats.duplicate_across_selected_datasets += 1
                    continue
                global_seen_hashes.add(content_hash)
                stats.unique_content_records += 1

                if args.upload and args.skip_existing and content_hash in existing_integrities:
                    stats.skipped_existing_in_collection += 1
                    continue

                stats.candidate_upload_records += 1
                file_name = make_doc_filename(dataset_name, title, content_hash)

                if sample_handles[dataset_name] < max(0, int(args.sample_size)):
                    sample_handles[dataset_name] += 1
                    sample_fp.write(
                        json.dumps(
                            {
                                "dataset": dataset_name,
                                "file_name": file_name,
                                "content_hash": content_hash,
                                "title": title,
                                "source": metadata.get("source"),
                                "preview": text[:700],
                                "metadata": metadata,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                if not args.upload:
                    continue

                batch_files[file_name] = text_bytes
                if len(batch_files) >= int(args.batch_size):
                    try:
                        upload_result = upload_zip_batch(
                            api_base_url=args.api_base_url,
                            api_key=args.api_key,
                            collection_id=args.collection_id,
                            file_entries=batch_files,
                            create_embeddings=args.create_embeddings,
                            await_embedding=args.await_embedding,
                            timeout_s=args.http_timeout_s,
                        )
                        batch_uploaded_count = len(upload_result.get("document_results", []) or [])
                        stats.upload_batches += 1
                        stats.uploaded_records += batch_uploaded_count
                        upload_stats.upload_batches_ok += 1
                        upload_stats.uploaded_records += batch_uploaded_count
                    except Exception as exc:
                        upload_stats.upload_batches_failed += 1
                        upload_stats.last_error = str(exc)
                        raise
                    finally:
                        batch_files = {}

            if args.upload and len(batch_files) > 0:
                try:
                    upload_result = upload_zip_batch(
                        api_base_url=args.api_base_url,
                        api_key=args.api_key,
                        collection_id=args.collection_id,
                        file_entries=batch_files,
                        create_embeddings=args.create_embeddings,
                        await_embedding=args.await_embedding,
                        timeout_s=args.http_timeout_s,
                    )
                    batch_uploaded_count = len(upload_result.get("document_results", []) or [])
                    stats.upload_batches += 1
                    stats.uploaded_records += batch_uploaded_count
                    upload_stats.upload_batches_ok += 1
                    upload_stats.uploaded_records += batch_uploaded_count
                except Exception as exc:
                    upload_stats.upload_batches_failed += 1
                    upload_stats.last_error = str(exc)
                    raise

            print(
                f"[{dataset_name}] raw={stats.raw_records_seen} unique={stats.unique_content_records} "
                f"candidate={stats.candidate_upload_records} uploaded={stats.uploaded_records} "
                f"missing_files={len(stats.missing_files)}"
            )
    finally:
        sample_fp.close()

    manifest = {
        "generated_at_unix": time.time(),
        "generated_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started_at, 3),
        "source_root": str(source_root),
        "datasets_selected": datasets,
        "args": {
            "limit_per_dataset": args.limit_per_dataset,
            "batch_size": args.batch_size,
            "upload": bool(args.upload),
            "skip_existing": bool(args.skip_existing),
            "create_embeddings": bool(args.create_embeddings),
            "await_embedding": bool(args.await_embedding),
            "api_base_url": args.api_base_url,
            "strict_embedding_preflight": bool(args.strict_embedding_preflight),
            "embedding_model_override": args.embedding_model,
        },
        "preflight": embedding_preflight,
        "upload": asdict(upload_stats),
        "datasets": {name: asdict(stats) for name, stats in per_dataset_stats.items()},
        "global": {
            "datasets_processed": len(datasets),
            "global_unique_content_records": len(global_seen_hashes),
            "global_candidate_upload_records": int(
                sum(s.candidate_upload_records for s in per_dataset_stats.values())
            ),
            "global_uploaded_records": int(sum(s.uploaded_records for s in per_dataset_stats.values())),
            "missing_files_total": int(sum(len(s.missing_files) for s in per_dataset_stats.values())),
        },
    }
    args.manifest_out.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"[done] manifest -> {args.manifest_out}")
    print(f"[done] samples  -> {args.samples_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
