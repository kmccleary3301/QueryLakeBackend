from QueryLake.observability import metrics


def test_metrics_smoke() -> None:
    metrics.record_request("/v1/chat/completions", 200, 0.1)
    metrics.rate_limit_denied("/v1/chat/completions")
    metrics.session_created()
    metrics.session_deleted()
    metrics.job_transition("queued", node_id="n1")
    metrics.cancel_requested()
    metrics.cancel_succeeded(0.2)
