from __future__ import annotations

from typing import Awaitable, Callable, Dict, Optional

from sqlmodel import Session

from QueryLake.runtime.retrieval_experiments import log_experiment_run
from QueryLake.typing.retrieval_primitives import RetrievalExecutionResult


def summarize_execution(result: RetrievalExecutionResult) -> Dict[str, float]:
    total_latency_seconds = sum((trace.duration_ms or 0.0) for trace in result.traces) / 1000.0
    return {
        "results_count": float(len(result.candidates)),
        "latency_seconds": float(total_latency_seconds),
        "trace_count": float(len(result.traces)),
    }


async def run_shadow_mode(
    *,
    baseline_executor: Callable[[], Awaitable[RetrievalExecutionResult]],
    candidate_executor: Callable[[], Awaitable[RetrievalExecutionResult]],
    publish_result: str = "baseline",
    database: Optional[Session] = None,
    experiment_id: Optional[str] = None,
    query_text: str = "",
    query_hash: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    candidate_run_id: Optional[str] = None,
) -> Dict[str, object]:
    assert publish_result in {"baseline", "candidate"}, "publish_result must be baseline or candidate"
    baseline = await baseline_executor()
    candidate = await candidate_executor()

    baseline_metrics = summarize_execution(baseline)
    candidate_metrics = summarize_execution(candidate)

    if database is not None and experiment_id is not None:
        published_pipeline_id = baseline.pipeline_id if publish_result == "baseline" else candidate.pipeline_id
        published_pipeline_version = baseline.pipeline_version if publish_result == "baseline" else candidate.pipeline_version
        log_experiment_run(
            database,
            experiment_id=experiment_id,
            query_text=query_text,
            query_hash=query_hash,
            baseline_run_id=baseline_run_id,
            candidate_run_id=candidate_run_id,
            baseline_metrics=baseline_metrics,
            candidate_metrics=candidate_metrics,
            publish_mode=publish_result,
            published_pipeline_id=published_pipeline_id,
            published_pipeline_version=published_pipeline_version,
        )

    published = baseline if publish_result == "baseline" else candidate
    return {
        "published": published,
        "baseline": baseline,
        "candidate": candidate,
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
    }
