from __future__ import annotations

import os
from typing import Optional

import redis


def get_redis_client() -> Optional[redis.Redis]:
    url = os.getenv("QUERYLAKE_REDIS_URL") or os.getenv("REDIS_URL")
    if not url:
        return None
    return redis.Redis.from_url(url)

