from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from QueryLake.runtime.redis_client import get_redis_client


@dataclass(frozen=True)
class RateLimitResult:
    allowed: bool
    retry_after_seconds: Optional[int] = None
    remaining: Optional[int] = None


_RATE_LIMIT_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])

redis.call('ZREMRANGEBYSCORE', key, 0, now - window)
local count = redis.call('ZCARD', key)
if count >= limit then
  return {0, count}
end
redis.call('ZADD', key, now, tostring(now) .. '-' .. tostring(math.random()))
redis.call('EXPIRE', key, window + 5)
return {1, count + 1}
"""

_CONCURRENCY_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local ttl = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local token = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - ttl)
local count = redis.call('ZCARD', key)
if count >= limit then
  return {0, count}
end
redis.call('ZADD', key, now, token)
redis.call('EXPIRE', key, ttl + 5)
return {1, count + 1}
"""


def _now() -> int:
    return int(time.time())


def acquire_rate_limit(key: str, window_seconds: int, limit: int) -> RateLimitResult:
    client = get_redis_client()
    if client is None:
        return RateLimitResult(allowed=True)
    now = _now()
    allowed, count = client.eval(_RATE_LIMIT_SCRIPT, 1, key, now, window_seconds, limit)
    if allowed == 1:
        return RateLimitResult(allowed=True, remaining=max(0, limit - count))
    retry_after = window_seconds
    return RateLimitResult(allowed=False, retry_after_seconds=retry_after)


def acquire_concurrency(key: str, limit: int, ttl_seconds: int, token: str) -> RateLimitResult:
    client = get_redis_client()
    if client is None:
        return RateLimitResult(allowed=True)
    now = _now()
    allowed, count = client.eval(_CONCURRENCY_SCRIPT, 1, key, now, ttl_seconds, limit, token)
    if allowed == 1:
        return RateLimitResult(allowed=True, remaining=max(0, limit - count))
    return RateLimitResult(allowed=False, retry_after_seconds=ttl_seconds)


def release_concurrency(key: str, token: str) -> None:
    client = get_redis_client()
    if client is None:
        return
    client.zrem(key, token)

