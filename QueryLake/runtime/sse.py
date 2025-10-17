from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator, Dict, List, Optional


class SSESubscriber:
    def __init__(self, session_id: str, queue_size: int = 100) -> None:
        self.session_id = session_id
        self.queue: asyncio.Queue[Dict[str, Any]] = asyncio.Queue(maxsize=queue_size)
        self.closed = False
        self._logger = logging.getLogger(__name__)

    async def push(self, message: Dict[str, Any]) -> None:
        if self.closed:
            return
        message_copy = dict(message)
        event_type = message_copy.pop("event", "message")
        payload = json.dumps(message_copy)
        queue_item = {
            "event": event_type,
            "data": payload,
            "id": message_copy.get("rev"),
        }
        try:
            self.queue.put_nowait(queue_item)
        except asyncio.QueueFull:
            metrics.inc_sse_drop(self.session_id, 1)
            dropped = 0
            try:
                _ = self.queue.get_nowait()
                self.queue.task_done()
                dropped = 1
            except asyncio.QueueEmpty:
                dropped = 0

            meta = message_copy.get("meta", {})
            meta = dict(meta)
            meta["backpressure_drop"] = {
                "dropped": dropped,
                "latest_rev": message_copy.get("rev"),
            }
            message_copy["meta"] = meta
            payload = json.dumps(message_copy)
            queue_item = {
                "event": event_type,
                "data": payload,
                "id": message_copy.get("rev"),
            }
            await self.queue.put(queue_item)
            self._logger.warning(
                "sse.drop",
                extra={
                    "session_id": self.session_id,
                    "dropped": dropped,
                    "latest_rev": message_copy.get("rev"),
                },
            )

    async def stream(self) -> AsyncIterator[Dict[str, Any]]:
        try:
            while not self.closed:
                payload = await self.queue.get()
                yield payload
                self.queue.task_done()
        finally:
            self.closed = True

    def close(self) -> None:
        self.closed = True


class SessionStreamHub:
    def __init__(self) -> None:
        self._subscribers: Dict[str, List[SSESubscriber]] = {}
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def subscribe(self, session_id: str) -> SSESubscriber:
        subscriber = SSESubscriber(session_id)
        async with self._lock:
            self._subscribers.setdefault(session_id, []).append(subscriber)
            metrics.set_sse_subscribers(session_id, len(self._subscribers[session_id]))
            self._logger.debug(
                "sse.subscribe",
                extra={
                    "session_id": session_id,
                    "subscribers": len(self._subscribers[session_id]),
                },
            )
        return subscriber

    async def unsubscribe(self, session_id: str, subscriber: SSESubscriber) -> None:
        async with self._lock:
            subs = self._subscribers.get(session_id, [])
            if subscriber in subs:
                subs.remove(subscriber)
            if not subs:
                self._subscribers.pop(session_id, None)
            metrics.set_sse_subscribers(session_id, len(self._subscribers.get(session_id, [])))
        subscriber.close()
        self._logger.debug("sse.unsubscribe", extra={"session_id": session_id})

    async def publish(self, session_id: str, message: Dict[str, Any]) -> None:
        async with self._lock:
            subscribers = list(self._subscribers.get(session_id, []))
        for subscriber in subscribers:
            await subscriber.push(message)
        if subscribers:
            self._logger.debug(
                "sse.publish",
                extra={"session_id": session_id, "subscriber_count": len(subscribers)},
            )
from QueryLake.observability import metrics
