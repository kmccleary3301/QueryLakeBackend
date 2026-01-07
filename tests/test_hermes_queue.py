from QueryLake.runtime import hermes_queue


class _FakeRedis:
    def __init__(self) -> None:
        self.hashes = {}
        self.lists = {}
        self.zsets = {}

    def hset(self, key, mapping):
        self.hashes.setdefault(key, {}).update(mapping)

    def rpush(self, key, value):
        self.lists.setdefault(key, []).append(value)

    def blpop(self, key, timeout=1):
        items = self.lists.get(key, [])
        if not items:
            return None
        value = items.pop(0)
        return (key, value)

    def zadd(self, key, mapping):
        zset = self.zsets.setdefault(key, {})
        for member, score in mapping.items():
            zset[member] = score

    def zrangebyscore(self, key, min_score, max_score, start=0, num=100):
        zset = self.zsets.get(key, {})
        members = [m for m, score in sorted(zset.items(), key=lambda kv: kv[1]) if min_score <= score <= max_score]
        return members[start : start + num]

    def zrem(self, key, member):
        self.zsets.get(key, {}).pop(member, None)


def test_hermes_queue_retry_requeue(monkeypatch):
    fake = _FakeRedis()

    monkeypatch.setattr(hermes_queue, "get_redis_client", lambda: fake)
    queue = hermes_queue.HermesQueue(prefix="test")

    queue.enqueue("job1", {"url": "https://example.com"})
    assert fake.hashes["test:job:job1"]["state"] == "pending"
    assert queue.dequeue(timeout=0) == "job1"

    queue.mark_failed("job1", backoff_seconds=10, now=100)
    assert fake.hashes["test:job:job1"]["state"] == "retry"
    assert queue.requeue_due(now=105) == 0
    assert queue.requeue_due(now=111) == 1
    assert queue.dequeue(timeout=0) == "job1"


def test_try_enqueue_job_no_redis(monkeypatch):
    monkeypatch.setattr(hermes_queue, "get_redis_client", lambda: None)
    assert hermes_queue.try_enqueue_job("job2", {"x": 1}, prefix="test") is False
