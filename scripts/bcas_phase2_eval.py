#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import requests
from sqlalchemy import text

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.create_db_session import initialize_database_engine


@dataclass
class EvalCase:
    dataset: str
    query: str
    expected_ids: List[str]
    retrieved_ids: List[str]
    allowed_collection_ids: List[str]
    retrieved_collection_ids: List[str]
    response_ms: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "dataset": self.dataset,
            "scenario": self.dataset,
            "query": self.query,
            "expected_ids": self.expected_ids,
            "retrieved_ids": self.retrieved_ids,
            "allowed_collection_ids": self.allowed_collection_ids,
            "retrieved_collection_ids": self.retrieved_collection_ids,
            "response_ms": round(self.response_ms, 3),
        }


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cases(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        return []
    return [row for row in payload if isinstance(row, dict)]


def _sample_questions(
    db,
    *,
    collection_id: str,
    sample_size: int,
    seed: int,
) -> List[str]:
    # Pull a broad random sample from DB then deterministic-shuffle to make runs reproducible.
    stmt = text(
        """
        SELECT question
        FROM (
            SELECT DISTINCT COALESCE(md->>'question', '') AS question
            FROM DocumentChunk
            WHERE collection_id = :cid
              AND COALESCE(md->>'question', '') <> ''
        ) q
        ORDER BY random()
        LIMIT :lim
        """
    ).bindparams(cid=collection_id, lim=max(1, int(sample_size * 3)))
    rows = [str(row[0]).strip() for row in db.exec(stmt).all() if isinstance(row[0], str) and len(str(row[0]).strip()) > 0]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:sample_size]


def _expected_ids_for_question(
    db,
    *,
    collection_id: str,
    question: str,
    max_expected: int,
) -> List[str]:
    stmt = text(
        """
        SELECT id
        FROM DocumentChunk
        WHERE collection_id = :cid
          AND COALESCE(md->>'question', '') = :question
        ORDER BY creation_timestamp ASC
        LIMIT :lim
        """
    ).bindparams(cid=collection_id, question=question, lim=max(1, int(max_expected)))
    return [str(row[0]) for row in db.exec(stmt).all() if row[0] is not None]


def _expected_ids_for_document_name(
    db,
    *,
    collection_id: str,
    document_name: str,
    max_expected: int,
) -> List[str]:
    stmt = text(
        """
        SELECT id
        FROM DocumentChunk
        WHERE collection_id = :cid
          AND document_name = :document_name
        ORDER BY creation_timestamp ASC
        LIMIT :lim
        """
    ).bindparams(cid=collection_id, document_name=document_name, lim=max(1, int(max_expected)))
    return [str(row[0]) for row in db.exec(stmt).all() if row[0] is not None]


def _iter_sample_log_rows(path: Path) -> List[Dict[str, Any]]:
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


def _iter_jsonl_rows(path: Path) -> List[Dict[str, Any]]:
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


def _query_live_hybrid(
    *,
    api_base_url: str,
    api_key: str,
    query: str,
    collection_id: str,
    limit_bm25: int,
    limit_similarity: int,
    limit_sparse: int,
    timeout_s: int,
    max_retries: int,
    bm25_weight: float,
    similarity_weight: float,
    sparse_weight: float,
    use_sparse: bool,
    sparse_embedding_function: str,
    sparse_dimensions: int,
    queue_admission_concurrency_limit: int,
    queue_admission_ttl_seconds: int,
    queue_throttle_soft_utilization: float,
    queue_throttle_hard_utilization: float,
    queue_throttle_soft_scale: float,
    queue_throttle_hard_scale: float,
    queue_throttle_disable_sparse_at_hard: bool,
) -> tuple[List[str], List[str], float]:
    payload = {
        "auth": {"api_key": api_key},
        "query": {
            "bm25": query,
            "embedding": query,
            "sparse": query,
            "queue_aware_admission": True,
            "queue_admission_concurrency_limit": int(queue_admission_concurrency_limit),
            "queue_admission_ttl_seconds": int(queue_admission_ttl_seconds),
            "queue_throttle_enabled": True,
            "queue_throttle_soft_utilization": float(queue_throttle_soft_utilization),
            "queue_throttle_hard_utilization": float(queue_throttle_hard_utilization),
            "queue_throttle_soft_scale": float(queue_throttle_soft_scale),
            "queue_throttle_hard_scale": float(queue_throttle_hard_scale),
            "queue_throttle_disable_sparse_at_hard": bool(queue_throttle_disable_sparse_at_hard),
        },
        "collection_ids": [collection_id],
        "limit_bm25": int(limit_bm25),
        "limit_similarity": int(limit_similarity),
        "limit_sparse": int(limit_sparse),
        "bm25_weight": float(bm25_weight),
        "similarity_weight": float(similarity_weight),
        "sparse_weight": float(sparse_weight),
        "use_sparse": bool(use_sparse),
        "sparse_embedding_function": str(sparse_embedding_function),
        "sparse_dimensions": int(sparse_dimensions),
    }
    t0 = time.perf_counter()
    last_error: Exception | None = None
    body: Dict[str, Any] | None = None
    elapsed_ms = 0.0
    for attempt in range(max(1, int(max_retries))):
        try:
            t0 = time.perf_counter()
            resp = requests.get(
                f"{api_base_url.rstrip('/')}/api/search_hybrid",
                json=payload,
                timeout=timeout_s,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            resp.raise_for_status()
            body = resp.json()
            break
        except Exception as exc:
            last_error = exc
            if attempt >= max(1, int(max_retries)) - 1:
                raise
            time.sleep(min(3.0, 0.2 * (2**attempt)))
    if body is None:
        raise RuntimeError(f"search_hybrid failed without JSON response: {last_error}")
    if body.get("success") is False:
        raise RuntimeError(f"search_hybrid failed: {body.get('error') or body.get('note')}")
    result = body.get("result")
    rows = result.get("rows", []) if isinstance(result, dict) else []
    if not isinstance(rows, list):
        rows = []
    retrieved_ids: List[str] = []
    retrieved_collection_ids: List[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        row_id = row.get("id")
        if isinstance(row_id, list) and len(row_id) > 0:
            retrieved_ids.extend([str(v) for v in row_id if v is not None])
        elif row_id is not None:
            retrieved_ids.append(str(row_id))
        collection_id_row = row.get("collection_id")
        if isinstance(collection_id_row, str):
            retrieved_collection_ids.append(collection_id_row)
    return retrieved_ids, retrieved_collection_ids, elapsed_ms


def _evaluate(cases: Sequence[EvalCase], *, top_k: int) -> Dict[str, Any]:
    def recall_at_k(expected: List[str], retrieved: List[str], k: int) -> float:
        if len(expected) == 0:
            return 0.0
        return len(set(expected) & set(retrieved[:k])) / float(len(set(expected)))

    def mrr(expected: List[str], retrieved: List[str]) -> float:
        expected_set = set(expected)
        for idx, rid in enumerate(retrieved, start=1):
            if rid in expected_set:
                return 1.0 / float(idx)
        return 0.0

    by_dataset: Dict[str, List[EvalCase]] = {}
    for case in cases:
        by_dataset.setdefault(case.dataset, []).append(case)

    def summary(rows: List[EvalCase]) -> Dict[str, float]:
        if len(rows) == 0:
            return {"case_count": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "avg_response_ms": 0.0}
        recalls = [recall_at_k(c.expected_ids, c.retrieved_ids, top_k) for c in rows]
        mrrs = [mrr(c.expected_ids, c.retrieved_ids) for c in rows]
        latencies = [c.response_ms for c in rows]
        return {
            "case_count": float(len(rows)),
            "recall_at_k": float(sum(recalls) / len(recalls)),
            "mrr": float(sum(mrrs) / len(mrrs)),
            "avg_response_ms": float(sum(latencies) / len(latencies)),
        }

    return {
        "overall": summary(list(cases)),
        "by_dataset": {name: summary(rows) for name, rows in sorted(by_dataset.items())},
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return float(default)
        return float(value)
    except Exception:
        return float(default)


def _compute_eval_gate(
    *,
    candidate_metrics: Dict[str, Any],
    baseline_metrics: Optional[Dict[str, Any]],
    min_recall_delta: float,
    min_mrr_delta: float,
    max_latency_ratio: float,
) -> Dict[str, Any]:
    cand_overall = (candidate_metrics.get("metrics") or {}).get("overall") if isinstance(candidate_metrics, dict) else None
    cand_overall = cand_overall if isinstance(cand_overall, dict) else {}

    base_overall: Dict[str, Any] = {}
    if isinstance(baseline_metrics, dict):
        maybe_metrics = baseline_metrics.get("metrics")
        if isinstance(maybe_metrics, dict):
            maybe_overall = maybe_metrics.get("overall")
            if isinstance(maybe_overall, dict):
                base_overall = maybe_overall

    cand_recall = _safe_float(cand_overall.get("recall_at_k"))
    cand_mrr = _safe_float(cand_overall.get("mrr"))
    cand_latency = _safe_float(cand_overall.get("avg_response_ms"))
    base_recall = _safe_float(base_overall.get("recall_at_k"))
    base_mrr = _safe_float(base_overall.get("mrr"))
    base_latency = _safe_float(base_overall.get("avg_response_ms"))

    baseline_available = len(base_overall) > 0
    recall_delta = cand_recall - base_recall
    mrr_delta = cand_mrr - base_mrr
    latency_ratio = 1.0 if base_latency <= 0 else (cand_latency / base_latency)

    checks: Dict[str, Optional[bool]] = {
        "recall_delta_ok": (recall_delta >= float(min_recall_delta)) if baseline_available else None,
        "mrr_delta_ok": (mrr_delta >= float(min_mrr_delta)) if baseline_available else None,
        "latency_ratio_ok": (latency_ratio <= float(max_latency_ratio)) if baseline_available else None,
    }
    gate_ok: Optional[bool]
    if baseline_available:
        gate_ok = bool(all(bool(v) for v in checks.values()))
    else:
        gate_ok = None

    return {
        "baseline_available": baseline_available,
        "gate_ok": gate_ok,
        "thresholds": {
            "min_recall_delta": float(min_recall_delta),
            "min_mrr_delta": float(min_mrr_delta),
            "max_latency_ratio": float(max_latency_ratio),
        },
        "candidate": {
            "recall_at_k": cand_recall,
            "mrr": cand_mrr,
            "avg_response_ms": cand_latency,
        },
        "baseline": {
            "recall_at_k": base_recall,
            "mrr": base_mrr,
            "avg_response_ms": base_latency,
        },
        "deltas": {
            "recall_at_k": recall_delta,
            "mrr": mrr_delta,
            "avg_response_ms": cand_latency - base_latency,
            "latency_ratio": latency_ratio,
        },
        "checks": checks,
    }


def _default_gate_out(metrics_out: Path) -> Path:
    stem = metrics_out.stem
    if stem.endswith(".json"):
        stem = stem[:-5]
    return metrics_out.with_name(f"{stem}_GATE.json")


def _default_gate_report_out(metrics_out: Path) -> Path:
    stem = metrics_out.stem
    if stem.endswith(".json"):
        stem = stem[:-5]
    return metrics_out.with_name(f"{stem}_GATE_REPORT.md")


def _render_eval_gate_report(
    *,
    gate_payload: Dict[str, Any],
    candidate_metrics: Path,
    baseline_metrics: Optional[Path],
    gate_out: Path,
) -> str:
    checks = gate_payload.get("checks", {}) if isinstance(gate_payload.get("checks"), dict) else {}
    candidate = gate_payload.get("candidate", {}) if isinstance(gate_payload.get("candidate"), dict) else {}
    baseline = gate_payload.get("baseline", {}) if isinstance(gate_payload.get("baseline"), dict) else {}
    deltas = gate_payload.get("deltas", {}) if isinstance(gate_payload.get("deltas"), dict) else {}
    thresholds = gate_payload.get("thresholds", {}) if isinstance(gate_payload.get("thresholds"), dict) else {}
    baseline_available = bool(gate_payload.get("baseline_available"))
    gate_ok = gate_payload.get("gate_ok")

    lines = [
        "# BCAS Phase2 Eval Gate Report",
        "",
        f"- `gate_ok`: `{gate_ok}`",
        f"- `baseline_available`: `{baseline_available}`",
        f"- `candidate_metrics`: `{candidate_metrics}`",
        f"- `baseline_metrics`: `{baseline_metrics}`",
        f"- `gate_json`: `{gate_out}`",
        "",
        "## Thresholds",
        "",
        f"- `min_recall_delta`: `{thresholds.get('min_recall_delta')}`",
        f"- `min_mrr_delta`: `{thresholds.get('min_mrr_delta')}`",
        f"- `max_latency_ratio`: `{thresholds.get('max_latency_ratio')}`",
        "",
        "## Candidate",
        "",
        f"- `recall_at_k`: `{candidate.get('recall_at_k')}`",
        f"- `mrr`: `{candidate.get('mrr')}`",
        f"- `avg_response_ms`: `{candidate.get('avg_response_ms')}`",
        "",
        "## Baseline",
        "",
        f"- `recall_at_k`: `{baseline.get('recall_at_k')}`",
        f"- `mrr`: `{baseline.get('mrr')}`",
        f"- `avg_response_ms`: `{baseline.get('avg_response_ms')}`",
        "",
        "## Deltas",
        "",
        f"- `recall_at_k`: `{deltas.get('recall_at_k')}`",
        f"- `mrr`: `{deltas.get('mrr')}`",
        f"- `avg_response_ms`: `{deltas.get('avg_response_ms')}`",
        f"- `latency_ratio`: `{deltas.get('latency_ratio')}`",
        "",
        "## Checks",
        "",
        f"- `recall_delta_ok`: `{checks.get('recall_delta_ok')}`",
        f"- `mrr_delta_ok`: `{checks.get('mrr_delta_ok')}`",
        f"- `latency_ratio_ok`: `{checks.get('latency_ratio_ok')}`",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run live BCAS retrieval eval against ingested collections.")
    parser.add_argument(
        "--account-config",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE1_ACCOUNT_AND_COLLECTIONS_2026-02-23.json"),
    )
    parser.add_argument("--api-base-url", type=str, default=None)
    parser.add_argument("--per-dataset-cases", type=int, default=80)
    parser.add_argument("--max-expected-ids", type=int, default=8)
    parser.add_argument("--limit-bm25", type=int, default=20)
    parser.add_argument("--limit-similarity", type=int, default=20)
    parser.add_argument("--limit-sparse", type=int, default=0)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260224)
    parser.add_argument("--http-timeout-s", type=int, default=90)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--bm25-weight", type=float, default=0.9)
    parser.add_argument("--similarity-weight", type=float, default=0.1)
    parser.add_argument("--sparse-weight", type=float, default=0.0)
    parser.add_argument("--sparse-embedding-function", type=str, default="embedding_sparse")
    parser.add_argument("--sparse-dimensions", type=int, default=1024)
    parser.add_argument("--queue-admission-concurrency-limit", type=int, default=12)
    parser.add_argument("--queue-admission-ttl-seconds", type=int, default=120)
    parser.add_argument("--queue-throttle-soft-utilization", type=float, default=0.75)
    parser.add_argument("--queue-throttle-hard-utilization", type=float, default=0.90)
    parser.add_argument("--queue-throttle-soft-scale", type=float, default=0.75)
    parser.add_argument("--queue-throttle-hard-scale", type=float, default=0.50)
    parser.add_argument("--queue-throttle-disable-sparse-at-hard", action="store_true", dest="queue_throttle_disable_sparse_at_hard")
    parser.add_argument("--no-queue-throttle-disable-sparse-at-hard", action="store_false", dest="queue_throttle_disable_sparse_at_hard")
    parser.set_defaults(queue_throttle_disable_sparse_at_hard=True)
    parser.add_argument(
        "--use-sparse",
        action="store_true",
        help="Force sparse lane on (otherwise inferred from limit/weight).",
    )
    parser.add_argument(
        "--cases-in",
        type=Path,
        default=None,
        help="Optional fixed evaluation cases JSON (from a prior run) for matched-delta replay.",
    )
    parser.add_argument("--baseline-metrics", type=Path, default=None, help="Optional baseline metrics JSON for gate deltas.")
    parser.add_argument("--gate-min-recall-delta", type=float, default=-0.01)
    parser.add_argument("--gate-min-mrr-delta", type=float, default=-0.01)
    parser.add_argument("--gate-max-latency-ratio", type=float, default=1.10)
    parser.add_argument("--gate-out", type=Path, default=None)
    parser.add_argument("--gate-report-out", type=Path, default=None)
    parser.add_argument(
        "--samples-root",
        type=Path,
        default=Path("docs_tmp/RAG/ingest_logs_2026-02-23"),
        help="Fallback per-dataset JSONL sample logs when question metadata is unavailable in DB chunks.",
    )
    parser.add_argument(
        "--fallback-samples-jsonl",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE1_INGESTION_SAMPLES_FULL_HOTPOT_2WIKI_2026-02-23.jsonl"),
        help="Additional JSONL source used when per-dataset sample logs are too small.",
    )
    parser.add_argument(
        "--cases-out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_LIVE_EVAL_CASES_2026-02-24.json"),
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=Path("docs_tmp/RAG/BCAS_PHASE2_LIVE_EVAL_METRICS_2026-02-24.json"),
    )
    args = parser.parse_args()

    cfg = _load_json(args.account_config)
    api_key = str(cfg.get("api_key", "")).strip()
    if len(api_key) == 0:
        raise SystemExit("account config missing api_key")
    api_base_url = str(args.api_base_url or cfg.get("api_base_url", "http://localhost:8000")).strip()
    collections = cfg.get("collections", {})
    if not isinstance(collections, dict):
        raise SystemExit("account config missing collections mapping")

    db, _ = initialize_database_engine()
    all_cases: List[EvalCase] = []
    stats: Dict[str, Any] = {"datasets": {}}
    inferred_use_sparse = bool(int(args.limit_sparse) > 0 and float(args.sparse_weight) > 0.0)
    use_sparse = bool(args.use_sparse or inferred_use_sparse)

    datasets = [
        ("hotpotqa", collections.get("hotpotqa")),
        ("multihop", collections.get("multihop")),
        ("triviaqa", collections.get("triviaqa")),
    ]
    dataset_collection_lookup = {name: cid for name, cid in datasets if isinstance(cid, str) and len(cid.strip()) > 0}

    if args.cases_in is not None:
        replay_rows = _load_cases(args.cases_in)
        per_dataset_counter: Dict[str, int] = {}
        for row in replay_rows:
            dataset = str(row.get("dataset", "")).strip()
            query = str(row.get("query", "")).strip()
            if len(dataset) == 0 or len(query) == 0:
                continue
            collection_id = dataset_collection_lookup.get(dataset)
            if not isinstance(collection_id, str) or len(collection_id) == 0:
                continue
            expected_ids = [str(v) for v in (row.get("expected_ids") or []) if isinstance(v, str) and len(v) > 0]
            if len(expected_ids) == 0:
                continue
            allowed_collection_ids = [collection_id]
            retrieved_ids, retrieved_collection_ids, elapsed_ms = _query_live_hybrid(
                api_base_url=api_base_url,
                api_key=api_key,
                query=query,
                collection_id=collection_id,
                limit_bm25=max(0, int(args.limit_bm25)),
                limit_similarity=max(0, int(args.limit_similarity)),
                limit_sparse=max(0, int(args.limit_sparse)),
                timeout_s=max(10, int(args.http_timeout_s)),
                max_retries=max(1, int(args.max_retries)),
                bm25_weight=float(args.bm25_weight),
                similarity_weight=float(args.similarity_weight),
                sparse_weight=float(args.sparse_weight),
                use_sparse=bool(use_sparse),
                sparse_embedding_function=str(args.sparse_embedding_function),
                sparse_dimensions=int(args.sparse_dimensions),
                queue_admission_concurrency_limit=int(args.queue_admission_concurrency_limit),
                queue_admission_ttl_seconds=int(args.queue_admission_ttl_seconds),
                queue_throttle_soft_utilization=float(args.queue_throttle_soft_utilization),
                queue_throttle_hard_utilization=float(args.queue_throttle_hard_utilization),
                queue_throttle_soft_scale=float(args.queue_throttle_soft_scale),
                queue_throttle_hard_scale=float(args.queue_throttle_hard_scale),
                queue_throttle_disable_sparse_at_hard=bool(args.queue_throttle_disable_sparse_at_hard),
            )
            all_cases.append(
                EvalCase(
                    dataset=dataset,
                    query=query,
                    expected_ids=expected_ids,
                    retrieved_ids=retrieved_ids,
                    allowed_collection_ids=allowed_collection_ids,
                    retrieved_collection_ids=retrieved_collection_ids,
                    response_ms=elapsed_ms,
                )
            )
            per_dataset_counter[dataset] = int(per_dataset_counter.get(dataset, 0)) + 1
        for dataset, collection_id in datasets:
            if not isinstance(collection_id, str) or len(collection_id.strip()) == 0:
                continue
            stats["datasets"][dataset] = {
                "collection_id": collection_id,
                "mode": "cases_in",
                "question_cases": int(per_dataset_counter.get(dataset, 0)),
            }
            print(f"[{dataset}] cases={int(per_dataset_counter.get(dataset, 0))}")
    else:
        for i, (dataset, collection_id) in enumerate(datasets):
            if not isinstance(collection_id, str) or len(collection_id.strip()) == 0:
                continue
            dataset_seed = int(args.seed) + (i * 1009)
            questions = _sample_questions(
                db,
                collection_id=collection_id,
                sample_size=max(1, int(args.per_dataset_cases)),
                seed=dataset_seed,
            )
            dataset_cases: List[EvalCase] = []
            if len(questions) > 0:
                for question in questions:
                    expected_ids = _expected_ids_for_question(
                        db,
                        collection_id=collection_id,
                        question=question,
                        max_expected=max(1, int(args.max_expected_ids)),
                    )
                    if len(expected_ids) == 0:
                        continue
                    retrieved_ids, retrieved_collection_ids, elapsed_ms = _query_live_hybrid(
                        api_base_url=api_base_url,
                        api_key=api_key,
                        query=question,
                        collection_id=collection_id,
                        limit_bm25=max(0, int(args.limit_bm25)),
                        limit_similarity=max(0, int(args.limit_similarity)),
                        limit_sparse=max(0, int(args.limit_sparse)),
                        timeout_s=max(10, int(args.http_timeout_s)),
                        max_retries=max(1, int(args.max_retries)),
                        bm25_weight=float(args.bm25_weight),
                        similarity_weight=float(args.similarity_weight),
                        sparse_weight=float(args.sparse_weight),
                        use_sparse=bool(use_sparse),
                        sparse_embedding_function=str(args.sparse_embedding_function),
                        sparse_dimensions=int(args.sparse_dimensions),
                        queue_admission_concurrency_limit=int(args.queue_admission_concurrency_limit),
                        queue_admission_ttl_seconds=int(args.queue_admission_ttl_seconds),
                        queue_throttle_soft_utilization=float(args.queue_throttle_soft_utilization),
                        queue_throttle_hard_utilization=float(args.queue_throttle_hard_utilization),
                        queue_throttle_soft_scale=float(args.queue_throttle_soft_scale),
                        queue_throttle_hard_scale=float(args.queue_throttle_hard_scale),
                        queue_throttle_disable_sparse_at_hard=bool(args.queue_throttle_disable_sparse_at_hard),
                    )
                    dataset_cases.append(
                        EvalCase(
                            dataset=dataset,
                            query=question,
                            expected_ids=expected_ids,
                            retrieved_ids=retrieved_ids,
                            allowed_collection_ids=[collection_id],
                            retrieved_collection_ids=retrieved_collection_ids,
                            response_ms=elapsed_ms,
                        )
                    )
            else:
                sample_log = args.samples_root / f"{dataset}.samples.jsonl"
                rows = _iter_sample_log_rows(sample_log)
                if args.fallback_samples_jsonl.exists():
                    rows.extend(
                        [
                            row
                            for row in _iter_jsonl_rows(args.fallback_samples_jsonl)
                            if str(row.get("dataset", "")).strip() == dataset
                        ]
                    )
                rng = random.Random(dataset_seed)
                rng.shuffle(rows)
                for row in rows[: max(1, int(args.per_dataset_cases))]:
                    if not isinstance(row, dict):
                        continue
                    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
                    question = metadata.get("question")
                    file_name = row.get("file_name")
                    if not isinstance(question, str) or len(question.strip()) == 0:
                        continue
                    if not isinstance(file_name, str) or len(file_name.strip()) == 0:
                        continue
                    expected_ids = _expected_ids_for_document_name(
                        db,
                        collection_id=collection_id,
                        document_name=file_name.strip(),
                        max_expected=max(1, int(args.max_expected_ids)),
                    )
                    if len(expected_ids) == 0:
                        continue
                    retrieved_ids, retrieved_collection_ids, elapsed_ms = _query_live_hybrid(
                        api_base_url=api_base_url,
                        api_key=api_key,
                        query=question.strip(),
                        collection_id=collection_id,
                        limit_bm25=max(0, int(args.limit_bm25)),
                        limit_similarity=max(0, int(args.limit_similarity)),
                        limit_sparse=max(0, int(args.limit_sparse)),
                        timeout_s=max(10, int(args.http_timeout_s)),
                        max_retries=max(1, int(args.max_retries)),
                        bm25_weight=float(args.bm25_weight),
                        similarity_weight=float(args.similarity_weight),
                        sparse_weight=float(args.sparse_weight),
                        use_sparse=bool(use_sparse),
                        sparse_embedding_function=str(args.sparse_embedding_function),
                        sparse_dimensions=int(args.sparse_dimensions),
                        queue_admission_concurrency_limit=int(args.queue_admission_concurrency_limit),
                        queue_admission_ttl_seconds=int(args.queue_admission_ttl_seconds),
                        queue_throttle_soft_utilization=float(args.queue_throttle_soft_utilization),
                        queue_throttle_hard_utilization=float(args.queue_throttle_hard_utilization),
                        queue_throttle_soft_scale=float(args.queue_throttle_soft_scale),
                        queue_throttle_hard_scale=float(args.queue_throttle_hard_scale),
                        queue_throttle_disable_sparse_at_hard=bool(args.queue_throttle_disable_sparse_at_hard),
                    )
                    dataset_cases.append(
                        EvalCase(
                            dataset=dataset,
                            query=question.strip(),
                            expected_ids=expected_ids,
                            retrieved_ids=retrieved_ids,
                            allowed_collection_ids=[collection_id],
                            retrieved_collection_ids=retrieved_collection_ids,
                            response_ms=elapsed_ms,
                        )
                    )
            all_cases.extend(dataset_cases)
            stats["datasets"][dataset] = {
                "collection_id": collection_id,
                "mode": "question_md" if len(questions) > 0 else "sample_log",
                "question_cases": len(dataset_cases),
            }
            print(f"[{dataset}] cases={len(dataset_cases)}")

    metrics = _evaluate(all_cases, top_k=max(1, int(args.top_k)))
    payload_metrics = {
        "generated_at_unix": time.time(),
        "seed": int(args.seed),
        "params": {
            "per_dataset_cases": int(args.per_dataset_cases),
            "max_expected_ids": int(args.max_expected_ids),
            "limit_bm25": int(args.limit_bm25),
            "limit_similarity": int(args.limit_similarity),
            "limit_sparse": int(args.limit_sparse),
            "top_k": int(args.top_k),
            "bm25_weight": float(args.bm25_weight),
            "similarity_weight": float(args.similarity_weight),
            "sparse_weight": float(args.sparse_weight),
            "use_sparse": bool(use_sparse),
            "sparse_embedding_function": str(args.sparse_embedding_function),
            "sparse_dimensions": int(args.sparse_dimensions),
            "cases_in": str(args.cases_in) if args.cases_in is not None else None,
        },
        "stats": stats,
        "metrics": metrics,
    }
    baseline_metrics = _load_json(args.baseline_metrics) if args.baseline_metrics is not None and args.baseline_metrics.exists() else None
    gate_payload = _compute_eval_gate(
        candidate_metrics=payload_metrics,
        baseline_metrics=baseline_metrics,
        min_recall_delta=float(args.gate_min_recall_delta),
        min_mrr_delta=float(args.gate_min_mrr_delta),
        max_latency_ratio=float(args.gate_max_latency_ratio),
    )
    gate_payload["generated_at_unix"] = time.time()
    gate_payload["candidate_metrics"] = str(args.metrics_out)
    gate_payload["baseline_metrics"] = str(args.baseline_metrics) if args.baseline_metrics is not None else None

    args.cases_out.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    args.cases_out.write_text(json.dumps([c.as_dict() for c in all_cases], indent=2), encoding="utf-8")
    args.metrics_out.write_text(json.dumps(payload_metrics, indent=2), encoding="utf-8")
    gate_out = args.gate_out if args.gate_out is not None else _default_gate_out(args.metrics_out)
    gate_out.parent.mkdir(parents=True, exist_ok=True)
    gate_out.write_text(json.dumps(gate_payload, indent=2), encoding="utf-8")
    gate_report_out = args.gate_report_out if args.gate_report_out is not None else _default_gate_report_out(args.metrics_out)
    gate_report_out.parent.mkdir(parents=True, exist_ok=True)
    gate_report_out.write_text(
        _render_eval_gate_report(
            gate_payload=gate_payload,
            candidate_metrics=args.metrics_out,
            baseline_metrics=args.baseline_metrics,
            gate_out=gate_out,
        ),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases_out": str(args.cases_out),
                "metrics_out": str(args.metrics_out),
                "gate_out": str(gate_out),
                "gate_report_out": str(gate_report_out),
                "case_count": len(all_cases),
                "gate_ok": gate_payload.get("gate_ok"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
