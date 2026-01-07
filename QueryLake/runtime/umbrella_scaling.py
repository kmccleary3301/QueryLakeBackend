from __future__ import annotations

import json
import os
from typing import Any, Dict


def get_umbrella_deployment_options(env: Dict[str, str] | None = None) -> Dict[str, Any]:
    """Build Umbrella autoscaling options from env without changing defaults."""
    env = env or os.environ
    max_ongoing = int(env.get("QL_UMBRELLA_MAX_ONGOING_REQUESTS", "100"))
    min_replicas = int(env.get("QL_UMBRELLA_MIN_REPLICAS", "1"))
    max_replicas = int(env.get("QL_UMBRELLA_MAX_REPLICAS", "1"))

    options: Dict[str, Any] = {"max_ongoing_requests": max_ongoing}
    if max_replicas != min_replicas or min_replicas != 1:
        options["autoscaling_config"] = {
            "min_replicas": min_replicas,
            "max_replicas": max_replicas,
            "target_ongoing_requests": int(env.get("QL_UMBRELLA_TARGET_ONGOING_REQUESTS", "32")),
            "upscale_delay_s": float(env.get("QL_UMBRELLA_UPSCALE_DELAY_S", "5")),
            "downscale_delay_s": float(env.get("QL_UMBRELLA_DOWNSCALE_DELAY_S", "30")),
        }

    resources_json = env.get("QL_UMBRELLA_RESOURCES_JSON")
    if resources_json:
        try:
            resources = json.loads(resources_json)
        except json.JSONDecodeError:
            resources = None
        if isinstance(resources, dict) and resources:
            options["ray_actor_options"] = {"resources": resources}
    return options
