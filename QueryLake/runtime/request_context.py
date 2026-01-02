from __future__ import annotations

import contextvars
import uuid
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class RequestContext:
    request_id: str
    actor_id: Optional[str] = None
    api_key_id: Optional[str] = None
    plugin_id: Optional[str] = None
    route: Optional[str] = None
    model: Optional[str] = None
    session_id: Optional[str] = None
    run_id: Optional[str] = None
    idempotency_key: Optional[str] = None


_REQUEST_CONTEXT: contextvars.ContextVar[Optional[RequestContext]] = contextvars.ContextVar(
    "querylake_request_context",
    default=None,
)


def set_request_context(ctx: RequestContext) -> None:
    _REQUEST_CONTEXT.set(ctx)


def get_request_context() -> Optional[RequestContext]:
    return _REQUEST_CONTEXT.get()


def ensure_request_id(existing: Optional[str] = None) -> str:
    return existing or f"req_{uuid.uuid4().hex}"


def set_request_id(request_id: Optional[str] = None) -> str:
    current = _REQUEST_CONTEXT.get()
    req_id = ensure_request_id(request_id or (current.request_id if current else None))
    if current is None:
        _REQUEST_CONTEXT.set(RequestContext(request_id=req_id))
    elif current.request_id != req_id:
        _REQUEST_CONTEXT.set(RequestContext(**{**current.__dict__, "request_id": req_id}))
    return req_id


def get_request_id() -> Optional[str]:
    ctx = _REQUEST_CONTEXT.get()
    return ctx.request_id if ctx else None

