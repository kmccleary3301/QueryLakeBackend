from __future__ import annotations

from typing import Any, Dict, Tuple

try:
    from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST
    _PROM = True
except Exception:  # pragma: no cover
    _PROM = False


if _PROM:
    # Prometheus-backed metrics
    EVENTS_TOTAL = Counter(
        "querylake_events_total",
        "Total number of toolchain runtime events emitted",
        labelnames=("kind",),
    )
    REQUESTS_TOTAL = Counter(
        "querylake_requests_total",
        "Total HTTP requests",
        labelnames=("route", "status"),
    )
    REQUEST_LATENCY_SECONDS = Histogram(
        "querylake_request_latency_seconds",
        "HTTP request latency in seconds",
        labelnames=("route",),
        buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    SESSIONS_CREATED_TOTAL = Counter("querylake_sessions_created_total", "Sessions created")
    SESSIONS_DELETED_TOTAL = Counter("querylake_sessions_deleted_total", "Sessions deleted")

    SSE_SUBSCRIBERS = Gauge(
        "querylake_sse_subscribers",
        "Current number of SSE subscribers per session",
        labelnames=("session_id",),
    )
    SSE_DROPS_TOTAL = Counter(
        "querylake_sse_drops_total",
        "Total number of SSE backpressure drops",
        labelnames=("session_id",),
    )

    JOBS_TOTAL = Counter(
        "querylake_jobs_total",
        "Total number of job state transitions",
        labelnames=("status", "node_id"),
    )

    CANCEL_REQUESTS_TOTAL = Counter("querylake_cancel_requests_total", "Total cancel requests")
    CANCEL_SUCCESS_TOTAL = Counter("querylake_cancel_success_total", "Successful cancels")
    CANCEL_LATENCY_SECONDS = Histogram(
        "querylake_cancel_latency_seconds",
        "Latency for cancel requests (seconds)",
        buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )

    RATE_LIMIT_DENIED_TOTAL = Counter(
        "querylake_rate_limit_denied_total",
        "Total number of rate limit denials",
        labelnames=("route",),
    )

    GPU_REPLICA_RESIDENCY = Gauge(
        "querylake_gpu_replica_resident",
        "Indicator gauge for GPU-backed replicas; value is always 1 while the replica is alive.",
        labelnames=(
            "role",
            "model_id",
            "node_id",
            "worker_id",
            "placement_group_id",
            "gpu_ids",
            "cuda_visible",
        ),
    )

    def inc_event(kind: str) -> None:
        EVENTS_TOTAL.labels(kind=kind).inc()
    def record_request(route: str, status: int, latency_seconds: float) -> None:
        REQUESTS_TOTAL.labels(route=route, status=str(status)).inc()
        REQUEST_LATENCY_SECONDS.labels(route=route).observe(latency_seconds)

    def session_created() -> None:
        SESSIONS_CREATED_TOTAL.inc()

    def session_deleted() -> None:
        SESSIONS_DELETED_TOTAL.inc()

    def set_sse_subscribers(session_id: str, count: int) -> None:
        SSE_SUBSCRIBERS.labels(session_id=session_id).set(count)

    def inc_sse_drop(session_id: str, dropped: int = 1) -> None:
        SSE_DROPS_TOTAL.labels(session_id=session_id).inc(dropped)

    def job_transition(status: str, node_id: str | None = None) -> None:
        JOBS_TOTAL.labels(status=status, node_id=node_id or "unknown").inc()

    def cancel_requested() -> None:
        CANCEL_REQUESTS_TOTAL.inc()

    def cancel_succeeded(latency_seconds: float) -> None:
        CANCEL_SUCCESS_TOTAL.inc()
        CANCEL_LATENCY_SECONDS.observe(latency_seconds)

    def rate_limit_denied(route: str) -> None:
        RATE_LIMIT_DENIED_TOTAL.labels(route=route).inc()

    def record_gpu_runtime_metadata(
        role: str,
        model_id: str,
        node_id: str,
        worker_id: str,
        placement_group_id: str,
        gpu_ids: str,
        cuda_visible: str,
    ) -> None:
        GPU_REPLICA_RESIDENCY.labels(
            role=role,
            model_id=model_id,
            node_id=node_id,
            worker_id=worker_id,
            placement_group_id=placement_group_id,
            gpu_ids=gpu_ids,
            cuda_visible=cuda_visible,
        ).set(1.0)

    def expose_metrics() -> Tuple[bytes, str]:
        return generate_latest(), CONTENT_TYPE_LATEST

else:
    # Lightweight fallback without prometheus_client
    _counters: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {
        "querylake_events_total": {},
        "querylake_requests_total": {},
        "querylake_sessions_created_total": {(): 0.0},
        "querylake_sessions_deleted_total": {(): 0.0},
        "querylake_sse_drops_total": {},
        "querylake_jobs_total": {},
        "querylake_cancel_requests_total": {(): 0.0},
        "querylake_cancel_success_total": {(): 0.0},
        # summary-like values for cancel latency
        "querylake_cancel_latency_seconds_sum": {(): 0.0},
        "querylake_cancel_latency_seconds_count": {(): 0.0},
        "querylake_request_latency_seconds_sum": {(): 0.0},
        "querylake_request_latency_seconds_count": {(): 0.0},
        "querylake_rate_limit_denied_total": {},
    }
    _gauges: Dict[str, Dict[Tuple[Tuple[str, str], ...], float]] = {
        "querylake_sse_subscribers": {},
        "querylake_gpu_replica_resident": {},
    }

    def _inc(name: str, labels: Dict[str, str] | None = None, value: float = 1.0) -> None:
        key = tuple(sorted((labels or {}).items()))
        _counters.setdefault(name, {})
        _counters[name][key] = _counters[name].get(key, 0.0) + value

    def _set_gauge(name: str, labels: Dict[str, str] | None, value: float) -> None:
        key = tuple(sorted((labels or {}).items()))
        _gauges.setdefault(name, {})
        _gauges[name][key] = value

    def inc_event(kind: str) -> None:
        _inc("querylake_events_total", {"kind": kind})
    def record_request(route: str, status: int, latency_seconds: float) -> None:
        _inc("querylake_requests_total", {"route": route, "status": str(status)})
        _inc("querylake_request_latency_seconds_sum", value=float(latency_seconds))
        _inc("querylake_request_latency_seconds_count")

    def session_created() -> None:
        _inc("querylake_sessions_created_total")

    def session_deleted() -> None:
        _inc("querylake_sessions_deleted_total")

    def set_sse_subscribers(session_id: str, count: int) -> None:
        _set_gauge("querylake_sse_subscribers", {"session_id": session_id}, float(count))

    def inc_sse_drop(session_id: str, dropped: int = 1) -> None:
        _inc("querylake_sse_drops_total", {"session_id": session_id}, float(dropped))

    def job_transition(status: str, node_id: str | None = None) -> None:
        labels = {"status": status}
        if node_id:
            labels["node_id"] = node_id
        _inc("querylake_jobs_total", labels)

    def cancel_requested() -> None:
        _inc("querylake_cancel_requests_total")

    def cancel_succeeded(latency_seconds: float) -> None:
        _inc("querylake_cancel_success_total")
        _inc("querylake_cancel_latency_seconds_sum", value=float(latency_seconds))
        _inc("querylake_cancel_latency_seconds_count")

    def rate_limit_denied(route: str) -> None:
        _inc("querylake_rate_limit_denied_total", {"route": route})

    def record_gpu_runtime_metadata(
        role: str,
        model_id: str,
        node_id: str,
        worker_id: str,
        placement_group_id: str,
        gpu_ids: str,
        cuda_visible: str,
    ) -> None:
        labels = {
            "role": role,
            "model_id": model_id,
            "node_id": node_id,
            "worker_id": worker_id,
            "placement_group_id": placement_group_id,
            "gpu_ids": gpu_ids,
            "cuda_visible": cuda_visible,
        }
        _set_gauge("querylake_gpu_replica_resident", labels, 1.0)

    def expose_metrics() -> Tuple[bytes, str]:
        # Render minimal Prometheus exposition format
        lines: list[str] = []
        for name, series in _counters.items():
            lines.append(f"# TYPE {name} counter")
            for labels_tuple, val in series.items():
                if labels_tuple:
                    labels = ",".join([f"{k}\"{v}\"".replace("\\", "\\\\").replace("\"", "\\\"") for k, v in labels_tuple])
                    lines.append(f"{name}{{{labels}}} {val}")
                else:
                    lines.append(f"{name} {val}")
        for name, series in _gauges.items():
            lines.append(f"# TYPE {name} gauge")
            for labels_tuple, val in series.items():
                if labels_tuple:
                    labels = ",".join([f"{k}\"{v}\"".replace("\\", "\\\\").replace("\"", "\\\"") for k, v in labels_tuple])
                    lines.append(f"{name}{{{labels}}} {val}")
                else:
                    lines.append(f"{name} {val}")
        body = ("\n".join(lines) + "\n").encode("utf-8")
        return body, "text/plain; version=0.0.4; charset=utf-8"
