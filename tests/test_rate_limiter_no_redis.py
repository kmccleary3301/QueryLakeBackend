from QueryLake.runtime.rate_limiter import (
    acquire_concurrency,
    acquire_rate_limit,
    release_concurrency,
)


def test_rate_limiter_allows_without_redis(monkeypatch):
    # Ensure no Redis URL is present
    monkeypatch.delenv("QUERYLAKE_REDIS_URL", raising=False)
    monkeypatch.delenv("REDIS_URL", raising=False)

    rate = acquire_rate_limit("test:key", window_seconds=10, limit=1)
    assert rate.allowed is True

    conc = acquire_concurrency("test:conc", limit=1, ttl_seconds=5, token="tok")
    assert conc.allowed is True

    release_concurrency("test:conc", "tok")
