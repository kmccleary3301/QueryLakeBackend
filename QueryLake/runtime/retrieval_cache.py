from __future__ import annotations

import hashlib
import os
import threading
import time
from collections import OrderedDict
from typing import Generic, Optional, Tuple, TypeVar

from QueryLake.observability import metrics

T = TypeVar("T")


class TTLCache(Generic[T]):
    def __init__(self, max_entries: int, ttl_seconds: float) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_seconds = max(0.0, float(ttl_seconds))
        self._store: "OrderedDict[str, Tuple[float, T]]" = OrderedDict()
        self._lock = threading.Lock()

    def _is_expired(self, ts: float) -> bool:
        if self.ttl_seconds <= 0:
            return True
        return (time.time() - ts) > self.ttl_seconds

    def get(self, key: str) -> Optional[T]:
        with self._lock:
            value = self._store.get(key)
            if value is None:
                return None
            ts, payload = value
            if self._is_expired(ts):
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return payload

    def set(self, key: str, value: T) -> None:
        with self._lock:
            self._store[key] = (time.time(), value)
            self._store.move_to_end(key)
            while len(self._store) > self.max_entries:
                self._store.popitem(last=False)


def _hash_payload(payload: str) -> str:
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_embedding_cache_key(actor_user: str, query_text: str) -> str:
    return _hash_payload(f"embed::{actor_user}::{query_text}")


def build_rerank_cache_key(actor_user: str, query_text: str, candidate_text: str) -> str:
    return _hash_payload(f"rerank::{actor_user}::{query_text}::{candidate_text}")


_default_ttl = float(os.getenv("QUERYLAKE_RETRIEVAL_CACHE_TTL_SECONDS", "300") or 300)
_embedding_max = int(os.getenv("QUERYLAKE_RETRIEVAL_CACHE_MAX_EMBEDDINGS", "2048") or 2048)
_rerank_max = int(os.getenv("QUERYLAKE_RETRIEVAL_CACHE_MAX_RERANK", "8192") or 8192)

_embedding_cache: TTLCache[list] = TTLCache(max_entries=_embedding_max, ttl_seconds=_default_ttl)
_rerank_cache: TTLCache[float] = TTLCache(max_entries=_rerank_max, ttl_seconds=_default_ttl)


def get_embedding(key: str) -> Optional[list]:
    value = _embedding_cache.get(key)
    metrics.record_retrieval_cache("embedding", "hit" if value is not None else "miss")
    return value


def set_embedding(key: str, value: list) -> None:
    _embedding_cache.set(key, value)
    metrics.record_retrieval_cache("embedding", "set")


def get_rerank_score(key: str) -> Optional[float]:
    value = _rerank_cache.get(key)
    metrics.record_retrieval_cache("rerank", "hit" if value is not None else "miss")
    return value


def set_rerank_score(key: str, value: float) -> None:
    _rerank_cache.set(key, float(value))
    metrics.record_retrieval_cache("rerank", "set")

