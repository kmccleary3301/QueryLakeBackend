from __future__ import annotations

from time import time
from typing import Any, Dict, List, Optional

from sqlmodel import Session, select

from QueryLake.database.sql_db_tables import (
    retrieval_experiment as RetrievalExperiment,
    retrieval_experiment_run as RetrievalExperimentRun,
)


def _numeric_delta(baseline: Dict[str, Any], candidate: Dict[str, Any]) -> Dict[str, Any]:
    keys = set(baseline.keys()) | set(candidate.keys())
    out: Dict[str, Any] = {}
    for key in sorted(keys):
        base_val = baseline.get(key)
        cand_val = candidate.get(key)
        if isinstance(base_val, (int, float)) and isinstance(cand_val, (int, float)):
            out[key] = float(cand_val) - float(base_val)
    return out


def create_experiment(
    database: Session,
    *,
    title: str,
    baseline_pipeline_id: str,
    baseline_pipeline_version: str,
    candidate_pipeline_id: str,
    candidate_pipeline_version: str,
    owner: Optional[str] = None,
    status: str = "draft",
    md: Optional[Dict[str, Any]] = None,
) -> RetrievalExperiment:
    assert isinstance(title, str) and len(title.strip()) > 0, "title must be non-empty"
    row = RetrievalExperiment(
        title=title.strip(),
        owner=owner,
        status=status,
        baseline_pipeline_id=baseline_pipeline_id,
        baseline_pipeline_version=baseline_pipeline_version,
        candidate_pipeline_id=candidate_pipeline_id,
        candidate_pipeline_version=candidate_pipeline_version,
        md=md or {},
    )
    database.add(row)
    database.commit()
    database.refresh(row)
    return row


def list_experiments(
    database: Session,
    *,
    status: Optional[str] = None,
    owner: Optional[str] = None,
    limit: int = 100,
) -> List[RetrievalExperiment]:
    assert isinstance(limit, int) and 1 <= limit <= 1000, "limit must be in [1, 1000]"
    stmt = select(RetrievalExperiment)
    if status is not None:
        stmt = stmt.where(RetrievalExperiment.status == status)
    if owner is not None:
        stmt = stmt.where(RetrievalExperiment.owner == owner)
    stmt = stmt.order_by(RetrievalExperiment.updated_at.desc()).limit(limit)
    return list(database.exec(stmt).all())


def log_experiment_run(
    database: Session,
    *,
    experiment_id: str,
    query_text: str,
    query_hash: Optional[str] = None,
    baseline_run_id: Optional[str] = None,
    candidate_run_id: Optional[str] = None,
    baseline_metrics: Optional[Dict[str, Any]] = None,
    candidate_metrics: Optional[Dict[str, Any]] = None,
    publish_mode: str = "baseline",
    published_pipeline_id: Optional[str] = None,
    published_pipeline_version: Optional[str] = None,
) -> RetrievalExperimentRun:
    baseline_metrics = baseline_metrics or {}
    candidate_metrics = candidate_metrics or {}
    row = RetrievalExperimentRun(
        experiment_id=experiment_id,
        query_hash=query_hash,
        query_text=query_text,
        baseline_run_id=baseline_run_id,
        candidate_run_id=candidate_run_id,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        delta_metrics=_numeric_delta(baseline_metrics, candidate_metrics),
        publish_mode=publish_mode,
        published_pipeline_id=published_pipeline_id,
        published_pipeline_version=published_pipeline_version,
    )
    database.add(row)
    # Keep parent experiment "warm" in listings.
    exp = database.exec(
        select(RetrievalExperiment).where(RetrievalExperiment.experiment_id == experiment_id)
    ).first()
    if exp is not None:
        exp.updated_at = time()
        database.add(exp)
    database.commit()
    database.refresh(row)
    return row


def audit_experiment_runs(
    database: Session,
    *,
    experiment_id: str,
) -> Dict[str, Any]:
    rows = list(
        database.exec(
            select(RetrievalExperimentRun).where(
                RetrievalExperimentRun.experiment_id == experiment_id
            )
        ).all()
    )
    missing_link_rows = 0
    publish_drift_rows = 0
    for row in rows:
        if not row.baseline_run_id or not row.candidate_run_id:
            missing_link_rows += 1
        if row.publish_mode == "baseline" and row.published_pipeline_id and row.published_pipeline_id != "baseline":
            publish_drift_rows += 1
    return {
        "experiment_id": experiment_id,
        "row_count": len(rows),
        "missing_link_rows": missing_link_rows,
        "publish_drift_rows": publish_drift_rows,
        "ok": missing_link_rows == 0 and publish_drift_rows == 0,
    }
