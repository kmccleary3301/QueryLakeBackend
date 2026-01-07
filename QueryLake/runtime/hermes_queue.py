from __future__ import annotations

import json
import time
from typing import Any, Optional

from QueryLake.runtime.redis_client import get_redis_client


class HermesQueue:
    """Minimal Redis-backed queue for Hermes crawl jobs."""

    def __init__(self, prefix: str = "hermes"):
        self.prefix = prefix

    @property
    def pending_key(self) -> str:
        return f"{self.prefix}:queue:pending"

    @property
    def retry_key(self) -> str:
        return f"{self.prefix}:queue:retry"

    def enqueue(self, job_id: str, payload: dict) -> None:
        client = get_redis_client()
        if client is None:
            raise RuntimeError("Redis not configured")
        job_key = f"{self.prefix}:job:{job_id}"
        client.hset(job_key, mapping={"payload": json.dumps(payload), "state": "pending"})
        client.rpush(self.pending_key, job_id)

    def dequeue(self, timeout: int = 1) -> Optional[str]:
        client = get_redis_client()
        if client is None:
            raise RuntimeError("Redis not configured")
        result = client.blpop(self.pending_key, timeout=timeout)
        if result is None:
            return None
        job_id = result[1]
        if isinstance(job_id, bytes):
            job_id = job_id.decode()
        return job_id

    def mark_failed(self, job_id: str, backoff_seconds: int = 30, *, now: Optional[float] = None) -> None:
        client = get_redis_client()
        if client is None:
            raise RuntimeError("Redis not configured")
        job_key = f"{self.prefix}:job:{job_id}"
        client.hset(job_key, mapping={"state": "retry"})
        score = (now if now is not None else time.time()) + backoff_seconds
        client.zadd(self.retry_key, {job_id: score})

    def requeue_due(self, *, now: Optional[float] = None, limit: int = 100) -> int:
        client = get_redis_client()
        if client is None:
            raise RuntimeError("Redis not configured")
        now_ts = now if now is not None else time.time()
        due = client.zrangebyscore(self.retry_key, 0, now_ts, start=0, num=limit)
        moved = 0
        for job_id in due:
            if isinstance(job_id, bytes):
                job_id = job_id.decode()
            client.zrem(self.retry_key, job_id)
            client.rpush(self.pending_key, job_id)
            moved += 1
        return moved


def try_enqueue_job(job_id: str, payload: dict, prefix: str = "hermes") -> bool:
    """Best-effort enqueue that no-ops when Redis is unavailable."""
    try:
        HermesQueue(prefix=prefix).enqueue(job_id, payload)
        return True
    except Exception:
        return False
