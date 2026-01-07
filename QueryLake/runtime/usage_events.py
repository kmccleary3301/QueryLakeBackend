from __future__ import annotations

import time
from typing import Any, Dict
import logging


def build_usage_event(
    kind: str,
    request_id: str,
    route: str,
    model: str | None = None,
    principal_id: str | None = None,
    provider: str | None = None,
    usage: Dict[str, Any] | None = None,
    status: str | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    return {
        "kind": kind,
        "ts": time.time(),
        "request_id": request_id,
        "route": route,
        "model": model,
        "principal_id": principal_id,
        "provider": provider,
        "usage": usage or {},
        "status": status,
        "error": error,
    }


def log_usage_event(event: Dict[str, Any]) -> None:
    logger = logging.getLogger("querylake.usage")
    logger.info("usage_event=%s", event)
