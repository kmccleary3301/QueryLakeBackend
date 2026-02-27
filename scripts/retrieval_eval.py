#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class RetrievalCase:
    query: str
    expected_ids: List[str]
    retrieved_ids: List[str]
    dataset: str = "default"
    scenario: str = "default"
    allowed_collection_ids: List[str] = field(default_factory=list)
    retrieved_collection_ids: List[str] = field(default_factory=list)


def _validate_case_row(row: dict, *, source: str, row_idx: int) -> None:
    assert isinstance(row, dict), f"{source}[{row_idx}] must be an object"
    assert isinstance(row.get("query"), str) and len(row.get("query").strip()) > 0, f"{source}[{row_idx}].query must be a non-empty string"
    assert isinstance(row.get("expected_ids"), list), f"{source}[{row_idx}].expected_ids must be a list"
    assert isinstance(row.get("retrieved_ids"), list), f"{source}[{row_idx}].retrieved_ids must be a list"
    assert len(row["expected_ids"]) > 0, f"{source}[{row_idx}].expected_ids must be non-empty"
    assert len(row["retrieved_ids"]) > 0, f"{source}[{row_idx}].retrieved_ids must be non-empty"
    if "scenario" in row:
        assert isinstance(row.get("scenario"), str) and len(row.get("scenario").strip()) > 0, (
            f"{source}[{row_idx}].scenario must be a non-empty string when provided"
        )
    if "allowed_collection_ids" in row:
        assert isinstance(row.get("allowed_collection_ids"), list), (
            f"{source}[{row_idx}].allowed_collection_ids must be a list when provided"
        )
    if "retrieved_collection_ids" in row:
        assert isinstance(row.get("retrieved_collection_ids"), list), (
            f"{source}[{row_idx}].retrieved_collection_ids must be a list when provided"
        )


def _load_case_rows(path: Path) -> List[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, list), f"{path} must contain a top-level list of retrieval cases"
    for i, row in enumerate(payload):
        _validate_case_row(row, source=str(path), row_idx=i)
    return payload


def _load_cases(path: Path) -> Tuple[List[RetrievalCase], List[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    out: List[RetrievalCase] = []
    datasets_used: List[str] = []

    if isinstance(payload, dict) and isinstance(payload.get("datasets"), list):
        for dataset_row in payload["datasets"]:
            assert isinstance(dataset_row, dict), f"{path}: dataset entry must be an object"
            name = str(dataset_row.get("name", "")).strip()
            rel_file = str(dataset_row.get("file", "")).strip()
            assert len(name) > 0, f"{path}: dataset entry requires non-empty name"
            assert len(rel_file) > 0, f"{path}: dataset entry requires non-empty file"
            dataset_path = (path.parent / rel_file).resolve()
            rows = _load_case_rows(dataset_path)
            datasets_used.append(name)
            for row in rows:
                out.append(
                    RetrievalCase(
                        query=str(row["query"]),
                        expected_ids=[str(v) for v in row.get("expected_ids", [])],
                        retrieved_ids=[str(v) for v in row.get("retrieved_ids", [])],
                        dataset=name,
                        scenario=str(row.get("scenario", name)),
                        allowed_collection_ids=[str(v) for v in row.get("allowed_collection_ids", [])],
                        retrieved_collection_ids=[str(v) for v in row.get("retrieved_collection_ids", [])],
                    )
                )
        return out, datasets_used

    assert isinstance(payload, list), f"{path} must contain either a list or a manifest object with datasets"
    datasets_used = [path.stem]
    for i, row in enumerate(payload):
        _validate_case_row(row, source=str(path), row_idx=i)
        out.append(
            RetrievalCase(
                query=str(row["query"]),
                expected_ids=[str(v) for v in row.get("expected_ids", [])],
                retrieved_ids=[str(v) for v in row.get("retrieved_ids", [])],
                dataset=path.stem,
                scenario=str(row.get("scenario", path.stem)),
                allowed_collection_ids=[str(v) for v in row.get("allowed_collection_ids", [])],
                retrieved_collection_ids=[str(v) for v in row.get("retrieved_collection_ids", [])],
            )
        )
    return out, datasets_used


def recall_at_k(expected_ids: List[str], retrieved_ids: List[str], k: int) -> float:
    if len(expected_ids) == 0:
        return 0.0
    expected = set(expected_ids)
    found = set(retrieved_ids[:k])
    return len(expected & found) / float(len(expected))


def mrr(expected_ids: List[str], retrieved_ids: List[str]) -> float:
    expected = set(expected_ids)
    for idx, content_id in enumerate(retrieved_ids, start=1):
        if content_id in expected:
            return 1.0 / float(idx)
    return 0.0


def tenant_isolation_violation_rate(cases: List[RetrievalCase], k: int) -> float:
    evaluable = 0
    violations = 0
    for case in cases:
        if len(case.allowed_collection_ids) == 0 or len(case.retrieved_collection_ids) == 0:
            continue
        evaluable += 1
        allowed = set(case.allowed_collection_ids)
        top_k = case.retrieved_collection_ids[:k]
        if any(collection_id not in allowed for collection_id in top_k):
            violations += 1
    if evaluable == 0:
        return 0.0
    return float(violations) / float(evaluable)


def evaluate_cases(cases: List[RetrievalCase], k: int = 5) -> Dict[str, float]:
    if len(cases) == 0:
        return {"case_count": 0.0, "recall_at_k": 0.0, "mrr": 0.0, "tenant_isolation_violation_rate": 0.0}
    recall_scores = [recall_at_k(case.expected_ids, case.retrieved_ids, k) for case in cases]
    mrr_scores = [mrr(case.expected_ids, case.retrieved_ids) for case in cases]
    return {
        "case_count": float(len(cases)),
        "recall_at_k": float(sum(recall_scores) / len(recall_scores)),
        "mrr": float(sum(mrr_scores) / len(mrr_scores)),
        "tenant_isolation_violation_rate": tenant_isolation_violation_rate(cases, k=max(1, int(k))),
    }


def evaluate_cases_by_dataset(cases: List[RetrievalCase], k: int = 5) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[RetrievalCase]] = {}
    for case in cases:
        grouped.setdefault(case.dataset, []).append(case)
    return {name: evaluate_cases(rows, k=k) for name, rows in sorted(grouped.items())}


def evaluate_cases_by_scenario(cases: List[RetrievalCase], k: int = 5) -> Dict[str, Dict[str, float]]:
    grouped: Dict[str, List[RetrievalCase]] = {}
    for case in cases:
        grouped.setdefault(case.scenario, []).append(case)
    return {name: evaluate_cases(rows, k=k) for name, rows in sorted(grouped.items())}


def _scenario_depth(label: str) -> int:
    match = re.search(r"(?:depth|d)\D*?(\d+)", label.lower())
    if match is None:
        return -1
    return int(match.group(1))


def build_tradeoff_curve(metrics_by_scenario: Dict[str, Dict[str, float]]) -> List[Dict[str, float]]:
    rows = []
    for scenario, metrics in metrics_by_scenario.items():
        rows.append(
            {
                "scenario": scenario,
                "depth": float(_scenario_depth(scenario)),
                "case_count": float(metrics.get("case_count", 0.0)),
                "recall_at_k": float(metrics.get("recall_at_k", 0.0)),
                "mrr": float(metrics.get("mrr", 0.0)),
            }
        )
    rows.sort(key=lambda row: (row["depth"], row["scenario"]))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrieval eval harness scaffold for QueryLake.")
    parser.add_argument("--mode", choices=["smoke", "nightly", "heavy"], default="smoke")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--input", type=str, default=None, help="Optional input fixture JSON path.")
    parser.add_argument("--output", type=str, default=None, help="Optional output JSON path.")
    parser.add_argument("--min-recall", type=float, default=0.5)
    parser.add_argument("--min-mrr", type=float, default=0.4)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    default_smoke = repo_root / "tests" / "fixtures" / "retrieval_eval_smoke.json"
    default_nightly = repo_root / "tests" / "fixtures" / "retrieval_eval_nightly.json"
    default_heavy = repo_root / "tests" / "fixtures" / "retrieval_eval_agentic_depth.json"
    if args.input:
        input_path = Path(args.input)
    elif args.mode == "smoke":
        input_path = default_smoke
    elif args.mode == "nightly":
        input_path = default_nightly
    else:
        input_path = default_heavy
    if not input_path.exists() and args.mode == "heavy":
        input_path = default_nightly
    if not input_path.exists() and args.mode in {"nightly", "heavy"}:
        input_path = default_smoke
    if not input_path.exists():
        raise FileNotFoundError(f"Retrieval eval input file not found: {input_path}")

    cases, datasets = _load_cases(input_path)
    metrics = evaluate_cases(cases, k=max(1, int(args.k)))
    metrics_by_dataset = evaluate_cases_by_dataset(cases, k=max(1, int(args.k)))
    metrics_by_scenario = evaluate_cases_by_scenario(cases, k=max(1, int(args.k)))
    tradeoff_curve = build_tradeoff_curve(metrics_by_scenario)
    payload = {
        "mode": args.mode,
        "input": str(input_path),
        "datasets": datasets,
        "metrics": metrics,
        "metrics_by_dataset": metrics_by_dataset,
        "metrics_by_scenario": metrics_by_scenario,
        "tradeoff_curve": tradeoff_curve,
        "thresholds": {"min_recall": float(args.min_recall), "min_mrr": float(args.min_mrr)},
    }

    print(json.dumps(payload, indent=2))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if metrics["recall_at_k"] < float(args.min_recall):
        print(f"FAIL: recall_at_k {metrics['recall_at_k']:.4f} < threshold {float(args.min_recall):.4f}")
        return 2
    if metrics["mrr"] < float(args.min_mrr):
        print(f"FAIL: mrr {metrics['mrr']:.4f} < threshold {float(args.min_mrr):.4f}")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
