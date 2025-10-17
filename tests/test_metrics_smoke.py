import asyncio
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from QueryLake.observability import metrics
from QueryLake.runtime.sse import SSESubscriber


def test_expose_metrics_has_querylake_series():
    body, content_type = metrics.expose_metrics()
    assert isinstance(body, (bytes, bytearray))
    assert b"querylake_" in body  # namespace present
    assert ";" in content_type  # content type shape


def test_events_counter_increments():
    # Take a snapshot
    before, _ = metrics.expose_metrics()
    metrics.inc_event("NODE_STARTED")
    after, _ = metrics.expose_metrics()
    # Ensure the events counter series is present
    assert b"querylake_events_total" in after
    # We can't reliably assert absolute counts, but after should differ
    assert after != before


def test_jobs_counter_increments():
    before, _ = metrics.expose_metrics()
    metrics.job_transition("COMPLETED", node_id="compose")
    after, _ = metrics.expose_metrics()
    assert b"querylake_jobs_total" in after
    assert after != before
    # If labels are rendered, we should see node_id as well
    if b"node_id" in after:
        assert b"compose" in after


def test_sse_drop_counter_increments_event_loop():
    async def _run():
        sub = SSESubscriber(session_id="sess-test", queue_size=1)
        # First push fits
        await sub.push({"event": "message", "data": "{}", "id": 1})
        # Second push causes a drop and increments metric
        await sub.push({"event": "message", "data": "{}", "id": 2})

    asyncio.run(_run())
    body, _ = metrics.expose_metrics()
    assert b"querylake_sse_drops_total" in body
