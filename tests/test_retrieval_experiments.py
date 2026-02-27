from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.database.sql_db_tables import (
    retrieval_experiment as RetrievalExperiment,
    retrieval_experiment_run as RetrievalExperimentRun,
)
from QueryLake.runtime.retrieval_experiments import (
    audit_experiment_runs,
    create_experiment,
    list_experiments,
    log_experiment_run,
)


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def first(self):
        return self._rows[0] if len(self._rows) > 0 else None

    def all(self):
        return self._rows


class _DummySession:
    def __init__(self):
        self.experiments = []
        self.runs = []

    def add(self, row):
        if isinstance(row, RetrievalExperiment):
            if all(existing.experiment_id != row.experiment_id for existing in self.experiments):
                self.experiments.append(row)
        elif isinstance(row, RetrievalExperimentRun):
            self.runs.append(row)

    def commit(self):
        return None

    def refresh(self, row):
        return None

    def exec(self, stmt):
        sql = str(stmt)
        params = stmt.compile().params
        if "FROM retrieval_experiment_run" in sql:
            rows = self.runs
            exp_id = params.get("experiment_id_1")
            if exp_id is not None:
                rows = [row for row in rows if row.experiment_id == exp_id]
            return _Result(rows)

        rows = self.experiments
        status = params.get("status_1")
        owner = params.get("owner_1")
        exp_id = params.get("experiment_id_1")
        if status is not None:
            rows = [row for row in rows if row.status == status]
        if owner is not None:
            rows = [row for row in rows if row.owner == owner]
        if exp_id is not None:
            rows = [row for row in rows if row.experiment_id == exp_id]
        return _Result(rows)


def test_create_and_list_experiments():
    db = _DummySession()
    row = create_experiment(
        db,
        title="Hybrid experiment",
        baseline_pipeline_id="legacy.search_hybrid",
        baseline_pipeline_version="v1",
        candidate_pipeline_id="orchestrated.search_hybrid",
        candidate_pipeline_version="v1",
        owner="tester",
        status="running",
    )
    assert row.title == "Hybrid experiment"
    rows = list_experiments(db, status="running", owner="tester")
    assert len(rows) == 1
    assert rows[0].experiment_id == row.experiment_id


def test_log_experiment_run_computes_numeric_deltas():
    db = _DummySession()
    exp = create_experiment(
        db,
        title="Delta test",
        baseline_pipeline_id="a",
        baseline_pipeline_version="v1",
        candidate_pipeline_id="b",
        candidate_pipeline_version="v1",
    )
    run = log_experiment_run(
        db,
        experiment_id=exp.experiment_id,
        query_text="boiler pressure",
        baseline_metrics={"latency": 0.8, "n": 12, "label": "base"},
        candidate_metrics={"latency": 0.6, "n": 14, "label": "cand"},
    )
    assert abs(run.delta_metrics["latency"] - (-0.2)) < 1e-9
    assert run.delta_metrics["n"] == 2.0
    assert "label" not in run.delta_metrics


def test_audit_experiment_runs_detects_link_issues():
    db = _DummySession()
    exp = create_experiment(
        db,
        title="Audit",
        baseline_pipeline_id="a",
        baseline_pipeline_version="v1",
        candidate_pipeline_id="b",
        candidate_pipeline_version="v1",
    )
    log_experiment_run(
        db,
        experiment_id=exp.experiment_id,
        query_text="q1",
        baseline_run_id="r1",
        candidate_run_id="r2",
        baseline_metrics={"mrr": 0.6},
        candidate_metrics={"mrr": 0.7},
        publish_mode="baseline",
        published_pipeline_id="baseline",
        published_pipeline_version="v1",
    )
    ok_audit = audit_experiment_runs(db, experiment_id=exp.experiment_id)
    assert ok_audit["ok"] is True

    log_experiment_run(
        db,
        experiment_id=exp.experiment_id,
        query_text="q2",
        baseline_run_id=None,
        candidate_run_id="r3",
        baseline_metrics={"mrr": 0.6},
        candidate_metrics={"mrr": 0.5},
        publish_mode="baseline",
        published_pipeline_id="candidate",
        published_pipeline_version="v2",
    )
    bad_audit = audit_experiment_runs(db, experiment_id=exp.experiment_id)
    assert bad_audit["ok"] is False
    assert bad_audit["missing_link_rows"] >= 1
    assert bad_audit["publish_drift_rows"] >= 1
