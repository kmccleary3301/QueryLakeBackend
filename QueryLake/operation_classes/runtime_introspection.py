"""Utilities for inspecting runtime placement details from Ray Serve replicas."""
from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional

import ray

from QueryLake.observability import metrics as observability_metrics

LOGGER = logging.getLogger(__name__)


def collect_runtime_metadata(
    role: str,
    model_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Gather a consistent snapshot of runtime placement details."""
    ctx = ray.get_runtime_context()
    metadata: Dict[str, Any] = {
        "role": role,
        "model_id": model_id,
        "node_id": ctx.get_node_id(),
        "worker_id": ctx.get_worker_id(),
        "job_id": ctx.get_job_id(),
        "task_id": ctx.get_task_id(),
        "placement_group_id": ctx.get_placement_group_id(),
        "gpu_ids": ray.get_gpu_ids(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if extra:
        metadata.update(extra)
    return metadata


class RuntimeIntrospectionMixin:
    """Mixin providing a standard `describe_runtime` hook for Serve deployments."""

    _runtime_role: str = "deployment"
    _runtime_model_id: str = "unknown"
    _runtime_extra_metadata: Dict[str, Any]
    _last_runtime_metadata: Optional[Dict[str, Any]] = None

    def _collect_and_record_runtime_metadata(self) -> Dict[str, Any]:
        """Collect placement metadata and mirror it into observability metrics."""
        extra = getattr(self, "_runtime_extra_metadata", {})
        metadata = collect_runtime_metadata(
            role=getattr(self, "_runtime_role", "deployment"),
            model_id=getattr(self, "_runtime_model_id", "unknown"),
            extra=extra,
        )
        try:
            observability_metrics.record_gpu_runtime_metadata(
                role=str(metadata.get("role", "unknown")),
                model_id=str(metadata.get("model_id", "unknown")),
                node_id=str(metadata.get("node_id") or "unknown"),
                worker_id=str(metadata.get("worker_id") or "unknown"),
                placement_group_id=str(metadata.get("placement_group_id") or "unknown"),
                gpu_ids=",".join(str(gpu_id) for gpu_id in metadata.get("gpu_ids") or []) or "none",
                cuda_visible=str(metadata.get("cuda_visible_devices") or "none"),
            )
        except Exception:  # pragma: no cover - best effort metrics
            LOGGER.debug("Unable to record GPU runtime metadata", exc_info=True)
        self._last_runtime_metadata = metadata
        return metadata

    def _publish_runtime_metadata(self) -> None:
        """Best-effort snapshot + metrics emission invoked during replica startup."""
        try:
            self._collect_and_record_runtime_metadata()
        except Exception:  # pragma: no cover - defensive
            LOGGER.debug("Runtime metadata publish failed during init", exc_info=True)

    async def describe_runtime(self) -> Dict[str, Any]:
        """Return placement and environment metadata for diagnostics."""
        return self._collect_and_record_runtime_metadata()
