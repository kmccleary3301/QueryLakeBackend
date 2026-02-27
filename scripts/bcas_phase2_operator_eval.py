#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import requests


TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]{2,}")


@dataclass
class OperatorCase:
    dataset: str
    operator_type: str
    query: str
    must_terms: List[str]
    must_phrase: str | None
    top_text_preview: str
    passed: bool
    result_count: int
    latency_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "operator_type": self.operator_type,
            "query": self.query,
            "must_terms": self.must_terms,
            "must_phrase": self.must_phrase,
            "top_text_preview": self.top_text_preview,
            "passed": self.passed,
            "result_count": self.result_count,
            "latency_ms": round(self.latency_ms, 3),
        }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _iter_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = line.strip()
            if len(raw) == 0:
                continue
            try:
                obj = json.loads(raw)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
    return rows


def _tokenize(text: str) -> List[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


def _select_terms(preview: str) -> List[str]:
    tokens = _tokenize(preview)
    unique: List[str] = []
    for token in tokens:
        if token not in unique:
            unique.append(token)
        if len(unique) >= 10:
            break
    return unique


def _call_search_bm25(
    *,
    api_base_url: str,
    api_key: str,
    collection_id: str,
    query: str,
    limit: int,
    timeout_s: int,
) -> tuple[List[Dict[str, Any]], float]:
    payload = {
        "auth": {"api_key": api_key},
        "query": query,
        "collection_ids": [collection_id],
        "limit": int(limit),
        "table": "document_chunk",
    }
    t0 = time.perf_counter()
    resp = requests.get(f"{api_base_url.rstrip('/')}/api/search_bm25", json=payload, timeout=timeout_s)
    elapsed = (time.perf_counter() - t0) * 1000.0
    resp.raise_for_status()
    body = resp.json()
    if body.get("success") is False:
        raise RuntimeError(body.get("error") or body.get("note") or "search_bm25 failed")
    result = body.get("result", [])
    if not isinstance(result, list):
        result = []
    return result, elapsed


def _top_text(rows: List[Dict[str, Any]]) -> str:
    if len(rows) == 0:
        return ""
    top = rows[0]
    if not isinstance(top, dict):
        return ""
    return str(top.get("text") or "")


def _contains_phrase(text: str, phrase: str) -> bool:
    return phrase.lower() in (text or "").lower()


def _contains_terms(text: str, terms: List[str]) -> bool:
    lowered = (text or "").lower()
    return all(term.lower() in lowered for term in terms)


def main() -> int:
    parser = argparse.ArgumentParser(description="BCAS operator-constraint eval for live /api/search_bm25 behavior.")
    parser.add_argument(
        "--account-config",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json"),
    )
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("docs_tmp/RAG/ingest_logs_2026-02-23"),
    )
    parser.add_argument("--per-dataset", type=int, default=12)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--http-timeout-s", type=int, default=60)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_OPERATOR_EVAL_2026-02-24.json"),
    )
    args = parser.parse_args()

    cfg = _load_json(args.account_config)
    api_base_url = str(cfg.get("api_base_url", "http://localhost:8000"))
    api_key = str(cfg.get("api_key", "")).strip()
    collections = cfg.get("collections", {})
    if len(api_key) == 0 or not isinstance(collections, dict):
        raise SystemExit("Invalid account config.")

    datasets = ["hotpotqa", "multihop", "triviaqa"]
    cases: List[OperatorCase] = []
    for dataset in datasets:
        collection_id = collections.get(dataset)
        if not isinstance(collection_id, str) or len(collection_id) == 0:
            continue
        rows = _iter_rows(args.samples_root / f"{dataset}.samples.jsonl")
        if len(rows) == 0:
            continue
        for row in rows[: max(1, int(args.per_dataset))]:
            preview = str(row.get("preview") or "")
            terms = _select_terms(preview)
            if len(terms) < 4:
                continue
            phrase = f"{terms[0]} {terms[1]}"
            required_terms = [terms[2], terms[3]]

            q_phrase = f"\"{phrase}\""
            phrase_rows, phrase_ms = _call_search_bm25(
                api_base_url=api_base_url,
                api_key=api_key,
                collection_id=collection_id,
                query=q_phrase,
                limit=int(args.limit),
                timeout_s=int(args.http_timeout_s),
            )
            phrase_top = _top_text(phrase_rows)
            cases.append(
                OperatorCase(
                    dataset=dataset,
                    operator_type="exact_phrase",
                    query=q_phrase,
                    must_terms=[],
                    must_phrase=phrase,
                    top_text_preview=phrase_top[:220],
                    passed=_contains_phrase(phrase_top, phrase),
                    result_count=len(phrase_rows),
                    latency_ms=phrase_ms,
                )
            )

            q_required = f"+{required_terms[0]} +{required_terms[1]}"
            req_rows, req_ms = _call_search_bm25(
                api_base_url=api_base_url,
                api_key=api_key,
                collection_id=collection_id,
                query=q_required,
                limit=int(args.limit),
                timeout_s=int(args.http_timeout_s),
            )
            req_top = _top_text(req_rows)
            cases.append(
                OperatorCase(
                    dataset=dataset,
                    operator_type="required_terms",
                    query=q_required,
                    must_terms=required_terms,
                    must_phrase=None,
                    top_text_preview=req_top[:220],
                    passed=_contains_terms(req_top, required_terms),
                    result_count=len(req_rows),
                    latency_ms=req_ms,
                )
            )

    by_type: Dict[str, List[OperatorCase]] = {}
    by_dataset: Dict[str, List[OperatorCase]] = {}
    for case in cases:
        by_type.setdefault(case.operator_type, []).append(case)
        by_dataset.setdefault(case.dataset, []).append(case)

    def _summary(rows: List[OperatorCase]) -> Dict[str, float]:
        if len(rows) == 0:
            return {"cases": 0.0, "pass_rate": 0.0, "avg_latency_ms": 0.0}
        return {
            "cases": float(len(rows)),
            "pass_rate": float(sum(1.0 for r in rows if r.passed) / len(rows)),
            "avg_latency_ms": float(sum(r.latency_ms for r in rows) / len(rows)),
        }

    payload = {
        "generated_at_unix": time.time(),
        "params": {
            "per_dataset": int(args.per_dataset),
            "limit": int(args.limit),
            "http_timeout_s": int(args.http_timeout_s),
        },
        "overall": _summary(cases),
        "by_operator_type": {k: _summary(v) for k, v in sorted(by_type.items())},
        "by_dataset": {k: _summary(v) for k, v in sorted(by_dataset.items())},
        "cases": [c.as_dict() for c in cases],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"out": str(args.out), "cases": len(cases), "overall": payload["overall"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
