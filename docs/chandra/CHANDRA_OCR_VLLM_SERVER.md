# Chandra OCR via External vLLM Server (Recommended)

This doc describes the **recommended** way to run QueryLake's Chandra OCR: keep the OCR model in an
**external vLLM OpenAI-compatible server**, and configure QueryLake to proxy OCR requests to it.

Why this is the best fit for QueryLake patterns:
- You get **vLLM continuous batching + high throughput** for OCR pages.
- You can run **1 GPU (default)** or **2 GPU striped** topologies without needing Ray-managed DP=2.
- QueryLake stays responsible for **routing, batching, caching, files pipeline**, and toolchain integration.
- Ray PACK/SPREAD/autoscaling remains clean because QueryLake's Chandra deployment becomes **CPU-only** in this mode.

---

## Topologies

### 1) Single (default)

One vLLM server on one GPU:
- best for day-to-day QueryLake usage (keeps a GPU free for local LLMs/embeddings/rerankers)
- good enough throughput for most OCR workloads

### 2) Striped (max throughput)

Two vLLM servers, each pinned to a different GPU, both serving the same model.
QueryLake is configured with both endpoints and stripes page requests across them.

This effectively dedicates **two GPUs** to OCR throughput.

---

## vLLM Server: start/stop

Use:
- `scripts/chandra_vllm_server.sh`

For striped:

```bash
# GPU0
CHANDRA_VLLM_RUNTIME_DIR=/tmp/querylake_chandra_vllm_server_gpu0 \
CHANDRA_VLLM_PORT=8022 \
CHANDRA_VLLM_CUDA_VISIBLE_DEVICES=0 \
./scripts/chandra_vllm_server.sh start

# GPU1
CHANDRA_VLLM_RUNTIME_DIR=/tmp/querylake_chandra_vllm_server_gpu1 \
CHANDRA_VLLM_PORT=8023 \
CHANDRA_VLLM_CUDA_VISIBLE_DEVICES=1 \
./scripts/chandra_vllm_server.sh start
```

---

## QueryLake Configuration (JSON patch)

QueryLake configs (`config.json`) are git-ignored in this repo. Treat the snippets below as patches.

### Enable Chandra (external vLLM server mode)

```json
{
  "enabled_model_classes": {
    "chandra": true
  },
  "other_local_models": {
    "chandra_models": [
      {
        "name": "Chandra OCR",
        "id": "chandra",
        "source": "models/chandra",
        "deployment_config": {
          "runtime_args": {
            "runtime_backend": "vllm_server"
          }
        }
      }
    ]
  }
}
```

### Point QueryLake at your vLLM server(s)

Single:

```json
{
  "chandra_vllm_server_base_urls": ["http://127.0.0.1:8022/v1"],
  "chandra_vllm_server_model": "chandra",
  "chandra_vllm_server_topology": "single"
}
```

Striped:

```json
{
  "chandra_vllm_server_base_urls": [
    "http://127.0.0.1:8022/v1",
    "http://127.0.0.1:8023/v1"
  ],
  "chandra_vllm_server_model": "chandra",
  "chandra_vllm_server_topology": "striped"
}
```

### Optional: let QueryLake autostart the vLLM servers (local dev convenience)

Autostart is intentionally **local-only** (it is not a multi-node supervisor).

```json
{
  "chandra_vllm_server_autostart": true,
  "chandra_vllm_server_autostart_topology": "single",
  "chandra_vllm_server_autostart_port_base": 8022,
  "chandra_vllm_server_autostart_gpu_memory_utilization": 0.90
}
```

Notes:
- Default autostart topology is **single**.
- If autostart is enabled, QueryLake will **reserve** the GPU(s) used by the external server by
  **not starting Ray worker nodes** on those GPU indices.

---

## Ray Scheduling Integration (important)

In `runtime_backend="vllm_server"` mode:
- QueryLake's Chandra Serve deployment is **CPU-only**
- GPU scheduling is done by how you provision the external vLLM servers

If you supervise vLLM externally (systemd/docker/k8s), explicitly reserve those GPU indices from Ray:
- Config: `ray_cluster.worker_gpu_exclude_indices=[0]` (or `[0,1]` for striped)
- Env: `QUERYLAKE_RAY_WORKER_GPU_EXCLUDE=0` (or `0,1`)

---

## Environment Variables (alternative to config)

```bash
export QUERYLAKE_CHANDRA_RUNTIME_BACKEND=vllm_server
export QUERYLAKE_CHANDRA_VLLM_SERVER_BASE_URLS=http://127.0.0.1:8022/v1,http://127.0.0.1:8023/v1
export QUERYLAKE_CHANDRA_VLLM_SERVER_MODEL=chandra
export QUERYLAKE_CHANDRA_VLLM_SERVER_API_KEY=chandra-local-key

# Reserve GPUs from Ray workers when the vLLM servers are external.
export QUERYLAKE_RAY_WORKER_GPU_EXCLUDE=0,1
```

---

## Failure Mode: HF fallback in vLLM-server mode

By default, QueryLake does **not** fall back to HF inside the vLLM-server proxy actor.
This is intentional: HF fallback can silently use GPUs outside Ray scheduling, which is unsafe.

If you want fallback anyway (not recommended), set:
- `QUERYLAKE_CHANDRA_VLLM_SERVER_FALLBACK_TO_HF=1`

