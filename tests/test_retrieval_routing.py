from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_routing import CanaryRoutingPolicy, choose_pipeline


def test_choose_pipeline_uses_allowlist_first():
    policy = CanaryRoutingPolicy(
        baseline_pipeline_id="baseline",
        baseline_pipeline_version="v1",
        candidate_pipeline_id="candidate",
        candidate_pipeline_version="v2",
        tenant_allowlist={"tenant_a"},
    )
    decision = choose_pipeline(policy=policy, tenant_scope="tenant_a")
    assert decision["pipeline_id"] == "candidate"
    assert decision["reason"] == "allowlist"


def test_choose_pipeline_uses_percent_canary_deterministically():
    policy = CanaryRoutingPolicy(
        baseline_pipeline_id="baseline",
        baseline_pipeline_version="v1",
        candidate_pipeline_id="candidate",
        candidate_pipeline_version="v2",
        candidate_percent=1.0,
    )
    decision = choose_pipeline(policy=policy, tenant_scope="tenant_b", query_fingerprint="abc")
    assert decision["pipeline_id"] == "candidate"
    assert decision["reason"] == "percent"
