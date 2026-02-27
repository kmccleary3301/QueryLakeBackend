#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Dict, Iterable, List, Mapping


def topk_overlap_ratio(a: List[str], b: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    sa = set(a[:k])
    sb = set(b[:k])
    denom = max(1, len(sa))
    return len(sa & sb) / float(denom)


def reciprocal_rank(result_ids: List[str], expected_ids: Iterable[str]) -> float:
    expected = {str(v) for v in expected_ids}
    if len(expected) == 0:
        return 0.0
    for idx, result_id in enumerate(result_ids):
        if str(result_id) in expected:
            return 1.0 / float(idx + 1)
    return 0.0


def mean_reciprocal_rank(
    *,
    cases: List[Dict[str, object]],
    retrievals_by_query: Mapping[str, List[str]],
) -> float:
    if len(cases) == 0:
        return 0.0
    values: List[float] = []
    for case in cases:
        query = str(case.get("query", ""))
        expected_ids = case.get("expected_ids", [])
        if not isinstance(expected_ids, list):
            expected_ids = []
        values.append(reciprocal_rank(retrievals_by_query.get(query, []), expected_ids))
    return mean(values) if len(values) > 0 else 0.0


def _normalize_runs_payload(payload: object) -> Dict[str, List[str]]:
    if isinstance(payload, dict):
        out: Dict[str, List[str]] = {}
        for key, value in payload.items():
            if isinstance(value, list):
                out[str(key)] = [str(v) for v in value]
        return out
    if isinstance(payload, list):
        out = {}
        for entry in payload:
            if not isinstance(entry, dict):
                continue
            query = entry.get("query")
            retrieved_ids = entry.get("retrieved_ids")
            if isinstance(query, str) and isinstance(retrieved_ids, list):
                out[query] = [str(v) for v in retrieved_ids]
        return out
    return {}


def parity_metrics(
    *,
    chunk_results: List[str],
    segment_results: List[str],
    k: int,
    chunk_latency_ms: float,
    segment_latency_ms: float,
) -> Dict[str, float]:
    overlap = topk_overlap_ratio(chunk_results, segment_results, k)
    latency_ratio = 1.0 if chunk_latency_ms <= 0 else float(segment_latency_ms) / float(chunk_latency_ms)
    return {
        "topk_overlap": overlap,
        "chunk_count": float(len(chunk_results)),
        "segment_count": float(len(segment_results)),
        "latency_ratio": latency_ratio,
    }


def parity_metrics_from_cases(
    *,
    cases: List[Dict[str, object]],
    chunk_retrievals_by_query: Mapping[str, List[str]],
    segment_retrievals_by_query: Mapping[str, List[str]],
    k: int,
    chunk_latency_ms: float,
    segment_latency_ms: float,
) -> Dict[str, object]:
    per_query: List[Dict[str, object]] = []
    overlap_scores: List[float] = []
    for case in cases:
        query = str(case.get("query", ""))
        expected_ids = case.get("expected_ids", [])
        if not isinstance(expected_ids, list):
            expected_ids = []
        chunk_results = chunk_retrievals_by_query.get(query, [])
        segment_results = segment_retrievals_by_query.get(query, [])
        overlap_k = topk_overlap_ratio(chunk_results, segment_results, k)
        rr_chunk = reciprocal_rank(chunk_results, expected_ids)
        rr_segment = reciprocal_rank(segment_results, expected_ids)
        overlap_scores.append(overlap_k)
        per_query.append(
            {
                "query": query,
                "expected_count": len(expected_ids),
                "chunk_topk": chunk_results[:k],
                "segment_topk": segment_results[:k],
                "overlap_at_k": overlap_k,
                "chunk_rr": rr_chunk,
                "segment_rr": rr_segment,
            }
        )
    chunk_mrr = mean_reciprocal_rank(cases=cases, retrievals_by_query=chunk_retrievals_by_query)
    segment_mrr = mean_reciprocal_rank(cases=cases, retrievals_by_query=segment_retrievals_by_query)
    latency_ratio = 1.0 if chunk_latency_ms <= 0 else float(segment_latency_ms) / float(chunk_latency_ms)
    return {
        "mode": "query_set",
        "query_count": len(cases),
        "topk_overlap_mean": (mean(overlap_scores) if len(overlap_scores) > 0 else 0.0),
        "chunk_mrr": chunk_mrr,
        "segment_mrr": segment_mrr,
        "mrr_delta": segment_mrr - chunk_mrr,
        "latency_ratio": latency_ratio,
        "per_query": per_query,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Chunk-vs-segment retrieval parity harness.")
    parser.add_argument("--chunk-json", help="JSON list of result ids from chunk retrieval")
    parser.add_argument("--segment-json", help="JSON list of result ids from segment retrieval")
    parser.add_argument("--cases-json", help="JSON list of benchmark cases with query + expected_ids")
    parser.add_argument("--chunk-runs-json", help="JSON mapping/list of query->retrieved_ids for chunk retrieval")
    parser.add_argument("--segment-runs-json", help="JSON mapping/list of query->retrieved_ids for segment retrieval")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--chunk-latency-ms", type=float, default=0.0)
    parser.add_argument("--segment-latency-ms", type=float, default=0.0)
    parser.add_argument("--min-overlap", type=float, default=0.6)
    parser.add_argument("--max-latency-ratio", type=float, default=1.5)
    parser.add_argument("--max-mrr-drop", type=float, default=0.05)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    k = max(1, int(args.k))
    thresholds = {
        "min_overlap": float(args.min_overlap),
        "max_latency_ratio": float(args.max_latency_ratio),
        "max_mrr_drop": float(args.max_mrr_drop),
    }

    query_set_mode = bool(args.cases_json and args.chunk_runs_json and args.segment_runs_json)
    if query_set_mode:
        cases_payload = json.loads(Path(args.cases_json).read_text(encoding="utf-8"))
        chunk_runs_payload = json.loads(Path(args.chunk_runs_json).read_text(encoding="utf-8"))
        segment_runs_payload = json.loads(Path(args.segment_runs_json).read_text(encoding="utf-8"))
        if not isinstance(cases_payload, list):
            raise ValueError("cases-json must contain a JSON list")
        cases = [case for case in cases_payload if isinstance(case, dict)]
        metrics = parity_metrics_from_cases(
            cases=cases,
            chunk_retrievals_by_query=_normalize_runs_payload(chunk_runs_payload),
            segment_retrievals_by_query=_normalize_runs_payload(segment_runs_payload),
            k=k,
            chunk_latency_ms=float(args.chunk_latency_ms),
            segment_latency_ms=float(args.segment_latency_ms),
        )
    else:
        if not args.chunk_json or not args.segment_json:
            raise ValueError("Either provide --chunk-json/--segment-json or full query-set inputs")
        chunk_results = json.loads(Path(args.chunk_json).read_text(encoding="utf-8"))
        segment_results = json.loads(Path(args.segment_json).read_text(encoding="utf-8"))
        if not isinstance(chunk_results, list) or not isinstance(segment_results, list):
            raise ValueError("chunk-json and segment-json must contain JSON lists")
        metrics = parity_metrics(
            chunk_results=[str(v) for v in chunk_results],
            segment_results=[str(v) for v in segment_results],
            k=k,
            chunk_latency_ms=float(args.chunk_latency_ms),
            segment_latency_ms=float(args.segment_latency_ms),
        )
        metrics["mode"] = "single_query"

    payload = {"metrics": metrics, "thresholds": thresholds}
    print(json.dumps(payload, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    overlap_value = float(metrics.get("topk_overlap_mean", metrics.get("topk_overlap", 0.0)))
    if overlap_value < thresholds["min_overlap"]:
        return 2
    if float(metrics.get("latency_ratio", 1.0)) > thresholds["max_latency_ratio"]:
        return 2
    mrr_delta = float(metrics.get("mrr_delta", 0.0))
    if mrr_delta < -thresholds["max_mrr_drop"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
