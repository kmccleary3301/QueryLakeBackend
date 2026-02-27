import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime import retrieval_shadow
from QueryLake.typing.retrieval_primitives import (
    RetrievalCandidate,
    RetrievalExecutionResult,
    RetrievalStageTrace,
)


async def _baseline_result():
    return RetrievalExecutionResult(
        pipeline_id="baseline",
        pipeline_version="v1",
        candidates=[RetrievalCandidate(content_id="a", text="A")],
        traces=[RetrievalStageTrace(stage="retrieve", duration_ms=10.0)],
        metadata={},
    )


async def _candidate_result():
    return RetrievalExecutionResult(
        pipeline_id="candidate",
        pipeline_version="v2",
        candidates=[RetrievalCandidate(content_id="b", text="B"), RetrievalCandidate(content_id="c", text="C")],
        traces=[RetrievalStageTrace(stage="retrieve", duration_ms=12.0), RetrievalStageTrace(stage="rerank", duration_ms=8.0)],
        metadata={},
    )


def test_shadow_mode_publishes_baseline_by_default():
    result = asyncio.run(
        retrieval_shadow.run_shadow_mode(
            baseline_executor=_baseline_result,
            candidate_executor=_candidate_result,
        )
    )
    assert result["published"].pipeline_id == "baseline"
    assert result["baseline_metrics"]["results_count"] == 1.0
    assert result["candidate_metrics"]["results_count"] == 2.0


def test_shadow_mode_logs_experiment_when_database_and_id_provided(monkeypatch):
    captured = {"called": False, "kwargs": None}

    def _fake_log_experiment_run(database, **kwargs):
        captured["called"] = True
        captured["kwargs"] = kwargs

    monkeypatch.setattr(retrieval_shadow, "log_experiment_run", _fake_log_experiment_run)
    result = asyncio.run(
        retrieval_shadow.run_shadow_mode(
            baseline_executor=_baseline_result,
            candidate_executor=_candidate_result,
            publish_result="candidate",
            database=object(),
            experiment_id="exp_1",
            query_text="boiler pressure",
            query_hash="abc",
            baseline_run_id="run_b",
            candidate_run_id="run_c",
        )
    )
    assert result["published"].pipeline_id == "candidate"
    assert captured["called"] is True
    assert captured["kwargs"]["experiment_id"] == "exp_1"
    assert captured["kwargs"]["publish_mode"] == "candidate"
    assert captured["kwargs"]["published_pipeline_id"] == "candidate"
