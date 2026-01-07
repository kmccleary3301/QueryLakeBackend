# Observability v1 (Draft)

## Metrics
- requests_total{route,status}
- request_latency_seconds_bucket{route}
- rate_limit_denied_total{route}
- queue_depth{service}
- request_id present in logs

## Logs
- request_id in every log line
- structured JSON for policy denials

## Tracing / Events
- RequestContext includes request_id + start time
- Usage events emitted for inference and retrieval paths
- Session events stored in EventStore for toolchain runs
- Minimal event fields: event_type, request_id, route, model, session_id, run_id, actor_id, status, ts

## Dashboards
- Inference latency + error rate
- Crawl throughput + failure rate
- Rate-limit denials by route

## Alerts (minimal)
- 5xx error rate > 2% (5 min)
- P95 latency regression > 2x (5 min)
- Crawl failures > 5% (15 min)
- Rate-limit denials spike > 3x baseline

## Validation
- Unit smoke test calls metrics API paths (record_request, rate_limit_denied, etc)
- RequestContext middleware emits request metrics for all HTTP routes
