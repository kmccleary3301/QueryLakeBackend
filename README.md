# QueryLake Backend

QueryLake is a **self-hosted AI platform backend** that combines:
- **OpenAI-compatible APIs** (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`)
- A rich **native function API** (`/api/<function_name>`) for documents, search, toolchains, auth, files, etc.
- **GPU-aware inference orchestration** via **Ray Serve** (one Ray worker per GPU, plus a `VRAM_MB` custom resource)
- **RAG storage** in Postgres/ParadeDB (**pgvector** + **BM25**)
- Optional **Redis** (runtime coordination / queues / future throughput gates)
- **Toolchains**: JSON-defined workflow graphs + interface schemas

This repository is the **backend half** of QueryLake. The UI lives in a separate repo (often checked out at
`/shared_folders/querylake_server/QueryLake/` on the dev machine).

---

## Why QueryLake is different

- **Dual API layer**: OpenAI-compatible endpoints *and* a function-style API for full platform control.
- **Toolchains**: Persistent agentic workflows stored as JSON and editable in UI.
- **Interface schemas**: Toolchains can define their **own UI** (V1 legacy or V2 schema).
- **GPU scheduling that respects VRAM**: models request `vram_required`, Ray schedules by `VRAM_MB`.
- **API-only mode**: Run without local GPU inference and still get full doc + toolchain functionality.

---

## What’s new (recent upgrades)

- **OAuth secret key support** via `QUERYLAKE_OAUTH_SECRET_KEY` (stable OAuth tokens across restarts).
- **Embedding failure tolerance**: chunking now proceeds even if embeddings fail.
- **7z encryption fixes** for file-like objects.
- **Toolchains V2 UI spec** support (bindings/actions/modals) for BASF Self-Guided RAG.
- **Hardened local Redis** compose + restart script.
- **uv-based dependency management** + slim **API-only** requirements file.

---

## Architecture (high level)

```mermaid
flowchart LR
    UI[QueryLake UI] -->|HTTP + WebSocket| API[FastAPI + Ray Serve]
    API -->|SQLModel| DB[(Postgres + ParadeDB)]
    API -->|Optional| Redis[(Redis)]
    API -->|Ray Serve| Ray[Ray head/workers]
    Ray -->|vLLM / HF / OCR| Models[Local model deployments]
    API -->|File blobs| Store[(object_store)]
```

---

## Table of contents

- [Quickstart (API-only)](#quickstart-api-only)
- [Install](#install)
- [Run](#run)
- [Configuration](#configuration)
- [Model serving & GPU scheduling](#model-serving--gpu-scheduling)
- [Auth & API keys](#auth--api-keys)
- [Documents & RAG pipeline](#documents--rag-pipeline)
- [Toolchains (execution + interface)](#toolchains-execution--interface)
- [Database (ParadeDB/Postgres)](#database-paradedbpostgres)
- [Redis (optional)](#redis-optional)
- [Scripts & utilities](#scripts--utilities)
- [Development](#development)
- [Troubleshooting / quirks](#troubleshooting--quirks)
- [Security notes](#security-notes)

---

## Quickstart (API-only)

Fastest path to a working backend (no local GPU inference):

### 1) Install deps (uv)

```bash
uv venv --python 3.12
uv sync
```

### 2) Start the database (docker)

```bash
docker compose -f docker-compose-only-db.yml up -d
```

Default local DB URL (hard-coded in `QueryLake/database/create_db_session.py`):

```
postgresql://querylake_access:querylake_access_password@localhost:5444/querylake_database
```

### 3) Start QueryLake (API-only)

```bash
export QUERYLAKE_API_ONLY=1
python start_querylake.py
```

Verify:

```bash
curl -sS http://127.0.0.1:8000/healthz | jq .
curl -sS http://127.0.0.1:8000/readyz | jq .
curl -sS http://127.0.0.1:8000/v1/models | jq .
```

---

## Install

### Recommended: uv + extras

Base dependencies are **API-only**. Heavy stacks live in extras:

```bash
uv venv --python 3.12
uv sync
```

Extras:
- `uv sync --extra cli` (helper CLI tools)
- `uv sync --extra inference-hf` (HF/torch: embeddings/rerank/etc.)
- `uv sync --extra ocr` (Marker/Surya + OCRmyPDF)
- `uv sync --extra vllm` (local vLLM experiments)
- `uv sync --extra dev` (pytest tooling)

### Legacy: conda + requirements

```bash
conda create --name QL_1 python=3.10
conda activate QL_1
pip install -r requirements.txt
```

Split requirement files exist for older flows:
- `requirements_api.txt`
- `requirements_inference.txt`
- `requirements_ocr.txt`

---

## Run

### Primary entrypoint: `start_querylake.py`

`start_querylake.py` is the “one-node-per-GPU” Ray starter + QueryLake deployer:
- Starts a Ray head + N worker processes (1 per GPU)
- Registers per-worker VRAM as `VRAM_MB`
- Deploys QueryLake via Ray Serve (placement strategy controls scheduling)

Default:

```bash
python start_querylake.py
```

Placement strategy:

```bash
python start_querylake.py --strategy SPREAD
```

Worker-only (connect to existing head):

```bash
python start_querylake.py --workers --head-node 192.168.1.100:6394
```

Safety flags:
- `--allow-ray-stop` → permits `ray stop --force` (off by default)
- `--with-gpu-workers` → start GPU workers even if `QUERYLAKE_API_ONLY=1`
- `--no-monitor` → disable Ray monitor loop

### API-only mode

```bash
export QUERYLAKE_API_ONLY=1
python start_querylake.py
```

This disables local inference deployments (llm / embedding / rerank / surya) but keeps
all document + toolchain APIs active.

### Skip vLLM if not installed

```bash
export QUERYLAKE_SKIP_VLLM=1
python start_querylake.py
```

---

## Configuration

### `config.json`

Primary config file for:
- Ray ports + default placement strategy
- enabled model classes (llm / embedding / rerank / surya)
- local model definitions and external provider catalogs
- default model selection

Model entries typically include:
- `engine` (`vllm`, `hf`, etc.)
- `system_path` or `source`
- `engine_args` (e.g., `gpu_memory_utilization`)
- `deployment_config` (`vram_required`, `ray_actor_options`, autoscaling config)

### Key environment flags

- `QUERYLAKE_API_ONLY=1` → skip local model deployments
- `QUERYLAKE_SKIP_VLLM=1` → skip vLLM deployments if requested by config
- `QL_TOOLCHAINS_SEED_ONLY=1` → seed toolchains only if missing (no overwrite)
- `QUERYLAKE_DB_CONNECT_TIMEOUT=5` → Postgres connect timeout
- `QUERYLAKE_OAUTH_SECRET_KEY=...` → stable OAuth token signing
- `QUERYLAKE_REDIS_URL=...` → enable Redis integration
- `QUERYLAKE_RAY_TMPDIR=/tmp/querylake_ray` → Ray temp/log dir override

---

## Model serving & GPU scheduling

QueryLake uses Ray Serve to deploy inference runtimes:

- **1 worker per GPU** (`CUDA_VISIBLE_DEVICES` set per worker)
- **`VRAM_MB` custom resource** to prevent over-commit
- **Placement strategies** (`PACK`, `SPREAD`, `STRICT_*`) to control scheduling
- **Autoscaling** per model (`min_replicas`, `max_replicas`, `target_ongoing_requests`, etc.)

This enables mixed workloads (e.g., vLLM + embeddings + rerank) without starving large VRAM models.

---

## Auth & API keys

### OAuth tokens

OAuth tokens are signed using `QUERYLAKE_OAUTH_SECRET_KEY`. If this variable is **not set**,
QueryLake generates an ephemeral key and **tokens will invalidate on restart**.

Recommended:

```bash
export QUERYLAKE_OAUTH_SECRET_KEY="your-long-random-secret"
```

### API keys

API keys are used with `Authorization: Bearer sk-...`.
Other bearer tokens are interpreted as OAuth2 tokens.

Example: create a user

```bash
curl -sS -X POST http://127.0.0.1:8000/api/add_user \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' | jq .
```

Example: create an API key (requires OAuth token)

```bash
curl -sS -X POST http://127.0.0.1:8000/api/create_api_key \
  -H 'Content-Type: application/json' \
  -d '{"auth":"<oauth_token>","title":"dev key"}' | jq .
```

---

## Documents & RAG pipeline

Core ingestion flow:
1. **Upload / register** document
2. **Parse** into raw text (OCR optional)
3. **Chunk** into `documentchunk`
4. **Embed** chunks (pgvector)
5. **Index** text for BM25 (ParadeDB)

QueryLake supports:
- file uploads
- OCR pipelines (Marker/Surya)
- hybrid BM25 + vector search

Search + retrieval are used by Toolchains (RAG) and the native API.

---

## Toolchains (execution + interface)

Toolchains are **workflow graphs** with state, node execution, and events.

Two layers:
1. **Execution** (backend): graph runtime, sessions, node events, output artifacts
2. **Interface** (frontend): UI schemas defined inside toolchain JSON

Toolchains are loaded from `toolchains/*.json` at startup and seeded into the DB.

### V1 vs V2 UI specs

- **V1**: `display_configuration` inside toolchain JSON (legacy)
- **V2**: `ui_spec_v2` inside toolchain JSON (modern)

`toolchains/self_guided_rag.json` contains:
- legacy V1 display
- V2 UI spec for the BASF Self-Guided RAG experience

### Toolchain seeding behavior

To prevent JSON files from overwriting DB-edited toolchains:

```bash
export QL_TOOLCHAINS_SEED_ONLY=1
```

---

## Database (ParadeDB/Postgres)

QueryLake uses Postgres with ParadeDB:
- **pgvector** + HNSW index for embeddings
- **BM25** indexes for text search

Start DB:

```bash
docker compose -f docker-compose-only-db.yml up -d
```

### Initialization quirks

`QueryLake/database/create_db_session.py`:
- creates tables and indexes at startup
- includes a first-boot workaround for index creation
- uses a hard-coded local DB URL today (change the file to override)

### Full reset (destructive)

`./restart_database.sh` removes DB volumes and data. Use with care.

---

## Redis (optional)

Enable Redis by setting `QUERYLAKE_REDIS_URL` (or `REDIS_URL`).

This repo ships a **hardened local Redis** compose:
- `docker-compose-redis.yml` (binds to `127.0.0.1`, requires password)
- `scripts/restart_querylake_redis.sh`

Example:

```bash
export QUERYLAKE_REDIS_PASSWORD='change-me'
./scripts/restart_querylake_redis.sh

export QUERYLAKE_REDIS_URL="redis://:${QUERYLAKE_REDIS_PASSWORD}@127.0.0.1:6393/0"
```

---

## Scripts & utilities

- `restart_database.sh` → destructive Postgres reset
- `scripts/restart_querylake_redis.sh` → local Redis restart
- `scripts/demo_toolchains_v2.sh` → load / run V2 toolchain demo
- `scripts/verify_placement.py` → sanity check placement strategy logic
- `scripts/driver_upgrade_580.sh` → driver upgrade helper (ops-only)

---

## Development

```bash
uv sync --extra dev
pytest -q
```

`pytest.ini` excludes large folders (`models`, `docs_tmp`, `object_store`, etc.).

---

## Troubleshooting / quirks

### “AsyncEngineDeadError” (vLLM)

If you see:
`vllm.engine.async_llm_engine.AsyncEngineDeadError: Background loop is not running`

Common causes:
- GPU OOM / KV cache pressure
- vLLM/torch/CUDA version mismatch
- model max length too high

For API-only runs:

```bash
export QUERYLAKE_API_ONLY=1
export QUERYLAKE_SKIP_VLLM=1
python start_querylake.py
```

### NVML / driver mismatch

If Ray reports:
`pynvml error: RM has detected an NVML/RM version mismatch`

You likely need to reconcile driver / CUDA / container versions
or restart after a driver upgrade.

### Ray log bloat (/tmp/ray/session_*)

`start_querylake.py` defaults `RAY_TMPDIR` to `/tmp/querylake_ray` and
sets conservative rotation limits. You can override with:

```bash
export QUERYLAKE_RAY_TMPDIR=/path/to/fast/disk
```

---

## Security notes

- Docker compose files are **local dev defaults**. For production:
  - change DB creds
  - bind services to private networks
  - use TLS at ingress
- Redis is **password-protected** and bound to localhost in the provided compose.
  Do not expose publicly without ACLs / firewalling.
