from __future__ import annotations

import asyncio
import inspect
from typing import Awaitable, Callable, Dict, List, Optional


AsyncCallback = Callable[[], Optional[Awaitable[None]]]


async def _maybe_await(callback: AsyncCallback) -> None:
    """Invoke ``callback`` and await it when it returns an awaitable."""
    if callback is None:
        return
    result = callback()
    if inspect.isawaitable(result):
        await result  # type: ignore[misc]


class JobSignal:
    """Per-job cancellation/event signal used by long-running operations."""

    def __init__(self, job_id: str) -> None:
        self.job_id = job_id
        self._event = asyncio.Event()
        self._callbacks: List[AsyncCallback] = []
        self._lock = asyncio.Lock()
        self._registry = None
        self._session_id: Optional[str] = None
        self._node_id: Optional[str] = None

    def triggered(self) -> bool:
        return self._event.is_set()

    async def wait(self) -> None:
        await self._event.wait()

    def configure(self, registry, session_id: str, node_id: str) -> None:
        self._registry = registry
        self._session_id = session_id
        self._node_id = node_id

    async def on_stop(self, callback: AsyncCallback) -> None:
        """Register ``callback`` to run exactly once when the signal fires."""

        if callback is None:
            return
        async with self._lock:
            if self._event.is_set():
                # Already triggered; run immediately outside the lock.
                pass
            else:
                self._callbacks.append(callback)
                return
        await _maybe_await(callback)

    async def _trigger(self) -> None:
        async with self._lock:
            if self._event.is_set():
                callbacks = self._callbacks
                self._callbacks = []
            else:
                self._event.set()
                callbacks = self._callbacks
                self._callbacks = []
        for callback in callbacks:
            await _maybe_await(callback)

    async def set_request_id(self, request_id: str) -> None:
        if not request_id:
            return
        if self._registry and self._session_id and self._node_id:
            await self._registry.attach_metadata(
                self.job_id,
                self._session_id,
                self._node_id,
                request_id=request_id,
            )

    async def set_progress(self, progress: Dict[str, Any]) -> None:
        if self._registry and self._session_id and self._node_id:
            await self._registry.attach_metadata(
                self.job_id,
                self._session_id,
                self._node_id,
                progress=progress,
            )


class JobSignalBus:
    """Lightweight registry of :class:`JobSignal` instances."""

    def __init__(self) -> None:
        self._signals: Dict[str, JobSignal] = {}
        self._lock = asyncio.Lock()

    async def create(self, job_id: str) -> JobSignal:
        async with self._lock:
            signal = self._signals.get(job_id)
            if signal is None:
                signal = JobSignal(job_id)
                self._signals[job_id] = signal
            return signal

    async def get(self, job_id: str) -> Optional[JobSignal]:
        async with self._lock:
            return self._signals.get(job_id)

    async def trigger(self, job_id: str) -> bool:
        async with self._lock:
            signal = self._signals.get(job_id)
        if signal is None:
            return False
        await signal._trigger()
        return True

    async def discard(self, job_id: str) -> None:
        async with self._lock:
            self._signals.pop(job_id, None)
