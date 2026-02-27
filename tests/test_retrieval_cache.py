import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.runtime.retrieval_cache import (
    TTLCache,
    build_embedding_cache_key,
    build_rerank_cache_key,
)


def test_ttl_cache_roundtrip_and_expiry():
    cache = TTLCache(max_entries=4, ttl_seconds=0.01)
    cache.set("k1", [1, 2, 3])
    assert cache.get("k1") == [1, 2, 3]
    time.sleep(0.02)
    assert cache.get("k1") is None


def test_cache_keys_are_deterministic():
    e1 = build_embedding_cache_key("alice", "what is reflux?")
    e2 = build_embedding_cache_key("alice", "what is reflux?")
    r1 = build_rerank_cache_key("alice", "q", "doc text")
    r2 = build_rerank_cache_key("alice", "q", "doc text")
    assert e1 == e2
    assert r1 == r2
    assert e1 != r1
