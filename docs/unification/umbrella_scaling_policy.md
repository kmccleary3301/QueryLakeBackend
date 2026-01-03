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

## Load Balancing
- Use Ray Serve built‑in request routing; target queue length per replica.
- If latency spikes, scale up within limits.

## VRAM Policy
- Umbrella replicas do not reserve VRAM resources.
- Inference deployments continue to reserve VRAM_MB custom resources.

## Failure Modes
- Replica crash: auto restart by Serve.
- Overload: 429 responses when queue depth is exceeded.
