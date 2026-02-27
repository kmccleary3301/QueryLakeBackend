from QueryLake.observability import metrics


def test_metrics_smoke() -> None:
    metrics.record_request("/v1/chat/completions", 200, 0.1)
    metrics.record_retrieval("search_hybrid", "ok", 0.05, 8)
    metrics.record_retrieval_cache("embedding", "hit")
    metrics.rate_limit_denied("/v1/chat/completions")
    metrics.session_created()
    metrics.session_deleted()
    metrics.job_transition("queued", node_id="n1")
    metrics.cancel_requested()
    metrics.cancel_succeeded(0.2)
