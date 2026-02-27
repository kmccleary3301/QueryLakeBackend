from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_readiness import evaluate_rollout_readiness


def test_evaluate_rollout_readiness_blocks_without_latency():
    report = evaluate_rollout_readiness(
        tests_green=True,
        eval_smoke_pass=True,
        eval_nightly_pass=True,
        g0_1={"meets_gate_g0_1": True},
        g0_3=None,
    )
    assert report["go_for_canary"] is False
    assert any("G0.3" in reason for reason in report["blocking_reasons"])


def test_evaluate_rollout_readiness_go_when_all_checks_pass():
    report = evaluate_rollout_readiness(
        tests_green=True,
        eval_smoke_pass=True,
        eval_nightly_pass=True,
        g0_1={"meets_gate_g0_1": True},
        g0_3={"meets_gate_g0_3": True},
    )
    assert report["go_for_canary"] is True
    assert report["blocking_reasons"] == []

