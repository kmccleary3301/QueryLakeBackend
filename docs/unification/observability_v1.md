# Observability v1 (Draft)

## Metrics
- requests_total{route,status}
- request_latency_ms_bucket{route}
- rate_limit_denied_total{route}
- queue_depth{service}

## Logs
- request_id in every log line
- structured JSON for policy denials

## Dashboards
- Inference latency + error rate
- Crawl throughput + failure rate

