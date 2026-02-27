from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.retrieval_eval import (
    RetrievalCase,
    _load_cases,
    build_tradeoff_curve,
    evaluate_cases,
    evaluate_cases_by_dataset,
    evaluate_cases_by_scenario,
    mrr,
    recall_at_k,
    tenant_isolation_violation_rate,
)


def test_recall_and_mrr_metrics():
    expected = ["a", "b"]
    retrieved = ["x", "a", "y", "b"]
    assert recall_at_k(expected, retrieved, 3) == 0.5
    assert abs(mrr(expected, retrieved) - 0.5) < 1e-9


def test_evaluate_cases_aggregates_scores():
    cases = [
        RetrievalCase(query="q1", expected_ids=["a"], retrieved_ids=["a", "b"]),
        RetrievalCase(query="q2", expected_ids=["z"], retrieved_ids=["x", "z"]),
    ]
    metrics = evaluate_cases(cases, k=2)
    assert metrics["case_count"] == 2.0
    assert metrics["recall_at_k"] == 1.0
    assert abs(metrics["mrr"] - 0.75) < 1e-9


def test_manifest_fixture_loads_multiple_datasets():
    root = Path(__file__).resolve().parent / "fixtures"
    cases, datasets = _load_cases(root / "retrieval_eval_nightly.json")
    assert set(datasets) == {"enterprise_docs", "multihop", "code_search"}
    assert len(cases) >= 6
    by_dataset = evaluate_cases_by_dataset(cases, k=3)
    assert "enterprise_docs" in by_dataset


def test_scenario_metrics_and_tradeoff_curve():
    cases = [
        RetrievalCase(query="q1", expected_ids=["a"], retrieved_ids=["a"], dataset="multihop", scenario="depth_1_single"),
        RetrievalCase(query="q2", expected_ids=["b"], retrieved_ids=["x", "b"], dataset="multihop", scenario="depth_2_multi"),
        RetrievalCase(query="q3", expected_ids=["c"], retrieved_ids=["x", "y", "c"], dataset="multihop", scenario="depth_3_multi"),
    ]
    by_scenario = evaluate_cases_by_scenario(cases, k=3)
    assert set(by_scenario.keys()) == {"depth_1_single", "depth_2_multi", "depth_3_multi"}
    curve = build_tradeoff_curve(by_scenario)
    assert [row["scenario"] for row in curve] == ["depth_1_single", "depth_2_multi", "depth_3_multi"]
    assert [row["depth"] for row in curve] == [1.0, 2.0, 3.0]


def test_tenant_isolation_violation_metric():
    cases = [
        RetrievalCase(
            query="q1",
            expected_ids=["a"],
            retrieved_ids=["a"],
            allowed_collection_ids=["c1"],
            retrieved_collection_ids=["c1"],
        ),
        RetrievalCase(
            query="q2",
            expected_ids=["b"],
            retrieved_ids=["b"],
            allowed_collection_ids=["c1"],
            retrieved_collection_ids=["c2"],
        ),
    ]
    assert abs(tenant_isolation_violation_rate(cases, k=3) - 0.5) < 1e-9
    metrics = evaluate_cases(cases, k=3)
    assert abs(metrics["tenant_isolation_violation_rate"] - 0.5) < 1e-9
