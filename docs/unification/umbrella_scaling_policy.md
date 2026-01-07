# Umbrella Scaling Policy (Draft)

## Goals
- Support multi‑replica Umbrella deployments without starving GPU workloads.
- Preserve VRAM‑aware scheduling and avoid overcommit.

## Placement Policy
- Default: Umbrella replicas on CPU nodes (if available), not GPU nodes.
- If CPU nodes are not available, allow GPU nodes but reserve minimal GPU fraction (0.01).

## Replica Policy
- Use `min_replicas=1`, `max_replicas=N` based on concurrency metrics.
- Cap replicas per node to avoid resource contention.

## Scaling Model Decision
- Per‑node replicas for Umbrella (control plane), per‑model replicas for inference.
- Umbrella should scale independently of model deployments.

## Serve Configuration (example)
```python
@serve.deployment(
  autoscaling_config={
    "min_replicas": 1,
    "max_replicas": 4,
    "target_num_ongoing_requests_per_replica": 32,
    "upscale_delay_s": 5,
    "downscale_delay_s": 30,
  },
  max_ongoing_requests=100,
)
```

## Environment Knobs (QueryLake)
- `QL_UMBRELLA_MIN_REPLICAS` / `QL_UMBRELLA_MAX_REPLICAS`
- `QL_UMBRELLA_TARGET_ONGOING_REQUESTS`
- `QL_UMBRELLA_UPSCALE_DELAY_S` / `QL_UMBRELLA_DOWNSCALE_DELAY_S`
- `QL_UMBRELLA_MAX_ONGOING_REQUESTS`
- `QL_UMBRELLA_RESOURCES_JSON` (optional Ray resource labels; JSON dict)

## Load Balancing
- Use Ray Serve built‑in request routing; target queue length per replica.
- If latency spikes, scale up within limits.

## Operational Playbook (minimal)
- Rollback: set `QL_UMBRELLA_MIN_REPLICAS=1` and `QL_UMBRELLA_MAX_REPLICAS=1`
- Failure mode: surge in 429 → verify rate limit config + autoscaling bounds
- Observation: watch request latency and queue depth metrics

## VRAM Policy
- Umbrella replicas do not reserve VRAM resources.
- Inference deployments continue to reserve VRAM_MB custom resources.

## Design Notes (for 75% milestone)
- Prefer PACK for inference deployments on GPU nodes; SPREAD for Umbrella replicas on CPU nodes.
- Use placement groups with `STRICT_PACK` for per‑model replicas when possible.
- Keep Umbrella autoscaling independent of inference autoscaling.

## Failure Modes
- Replica crash: auto restart by Serve.
- Overload: 429 responses when queue depth is exceeded.
