from __future__ import annotations

import hashlib
from typing import Dict, Optional, Set

from pydantic import BaseModel, Field


class CanaryRoutingPolicy(BaseModel):
    baseline_pipeline_id: str
    baseline_pipeline_version: str
    candidate_pipeline_id: str
    candidate_pipeline_version: str
    tenant_allowlist: Set[str] = Field(default_factory=set)
    workspace_allowlist: Set[str] = Field(default_factory=set)
    candidate_percent: float = Field(default=0.0, ge=0.0, le=1.0)


def _deterministic_bucket(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    # 0..9999 bucket.
    bucket = int(digest[:8], 16) % 10000
    return float(bucket) / 10000.0


def choose_pipeline(
    *,
    policy: CanaryRoutingPolicy,
    tenant_scope: Optional[str] = None,
    workspace_id: Optional[str] = None,
    query_fingerprint: Optional[str] = None,
) -> Dict[str, str]:
    allowlist_hit = False
    if tenant_scope and tenant_scope in policy.tenant_allowlist:
        allowlist_hit = True
    if workspace_id and workspace_id in policy.workspace_allowlist:
        allowlist_hit = True

    if allowlist_hit:
        return {
            "pipeline_id": policy.candidate_pipeline_id,
            "pipeline_version": policy.candidate_pipeline_version,
            "reason": "allowlist",
        }

    if policy.candidate_percent > 0:
        key = query_fingerprint or f"{tenant_scope or 'none'}:{workspace_id or 'none'}"
        if _deterministic_bucket(key) < policy.candidate_percent:
            return {
                "pipeline_id": policy.candidate_pipeline_id,
                "pipeline_version": policy.candidate_pipeline_version,
                "reason": "percent",
            }

    return {
        "pipeline_id": policy.baseline_pipeline_id,
        "pipeline_version": policy.baseline_pipeline_version,
        "reason": "baseline",
    }
