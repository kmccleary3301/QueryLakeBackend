# QueryLake Backend

QueryLake is a self-hosted AI platform backend built around:

- **OpenAI-compatible APIs** (`/v1/chat/completions`, `/v1/embeddings`, `/v1/models`)
- **A “native” function API** (`/api/<function_name>`) for documents, search, toolchains, auth, etc.
- **GPU-aware model serving** via **Ray Serve** (one Ray worker per GPU, plus a VRAM_MB custom resource)
- **RAG storage** in Postgres/ParadeDB (**pgvector** + **BM25** indexes)
- Optional **Redis** (runtime coordination / future queuing use-cases)
- Optional local inference stacks (HF/torch, OCR, vLLM), kept in **uv extras** so “API-only” installs stay lean

This repository is the **backend half** of QueryLake. The UI lives in a separate repo (commonly checked out as
`/shared_folders/querylake_server/QueryLake/` on the dev machine).

## Table of contents

- [Quickstart (API-only)](#quickstart-api-only)
- [Install](#install)
- [Run](#run)
- [Database (ParadeDB/Postgres)](#database-paradedbpostgres)
- [Redis (optional)](#redis-optional)
- [API overview](#api-overview)
- [Toolchains overview](#toolchains-overview)
- [Configuration](#configuration)
- [Development](#development)
- [Troubleshooting / quirks](#troubleshooting--quirks)
- [Security notes](#security-notes)

## Quickstart (API-only)

This is the fastest path to a working backend that can:
- start Ray Serve + FastAPI
- connect to the database
- serve `/healthz`, `/readyz`, `/v1/models`, and `/api/*`

It intentionally **does not** start any local GPU model deployments.

### 1) Install deps (uv)

```bash
uv venv --python 3.12
uv sync
```

### 2) Start the database (docker)

```bash
docker compose -f docker-compose-only-db.yml up -d
```

Default local DB URL (currently hard-coded in `QueryLake/database/create_db_session.py`):

```
postgresql://querylake_access:querylake_access_password@localhost:5444/querylake_database
```

### 3) Start QueryLake (API-only)

```bash
export QUERYLAKE_API_ONLY=1
python start_querylake.py
```

Then verify:

```bash
curl -sS http://127.0.0.1:8000/healthz | jq .
curl -sS http://127.0.0.1:8000/readyz | jq .
curl -sS http://127.0.0.1:8000/v1/models | jq .
```

> Ray Serve’s HTTP port is controlled by Ray Serve configuration; in a default local run it is typically `8000`.

## Install

### Recommended: uv + extras

Base dependencies are intended for an **API-only** footprint.
Heavier stacks live in optional extras in `pyproject.toml`.

```bash
uv venv --python 3.12
uv sync
```

Optional extras:
- `uv sync --extra cli` (enables interactive helpers like `setup.py` downloads)
- `uv sync --extra inference-hf` (local HF/torch inference helpers: embeddings/rerank/etc.)
- `uv sync --extra ocr` (Marker/Surya + OCRmyPDF)
- `uv sync --extra vllm` (local vLLM experiments; production usually runs vLLM upstream)
- `uv sync --extra dev` (pytest tooling)

### Legacy: conda + requirements

This is kept for compatibility with older deployments / environments:

```bash
conda create --name QL_1 python=3.10
conda activate QL_1
pip install -r requirements.txt
```

There are also split requirement files (`requirements_api.txt`, `requirements_inference.txt`, etc.) for older flows.

## Run

### Recommended local entrypoint: `start_querylake.py`

`start_querylake.py` is the “one-node-per-GPU” Ray starter + QueryLake deployer:

- Starts a Ray head node + N GPU worker processes (1 worker per GPU via `CUDA_VISIBLE_DEVICES`)
- Registers per-worker VRAM as a Ray custom resource (`VRAM_MB`)
- Deploys QueryLake via Ray Serve with a placement strategy (`PACK`, `SPREAD`, etc.)

Default (head + workers on this machine, using `config.json`):

```bash
python start_querylake.py
```

Override placement strategy:

```bash
python start_querylake.py --strategy SPREAD
```

Worker-only mode (connect to an existing head node):

```bash
python start_querylake.py --workers --head-node 192.168.1.100:6394
```

Important safety flags:
- By default QueryLake **does not** run `ray stop --force` (to avoid killing unrelated Ray workloads).
  Use `--allow-ray-stop` only when you are sure it’s safe.
- If `QUERYLAKE_API_ONLY=1`, GPU worker nodes are skipped unless you pass `--with-gpu-workers`.

### API-only mode

`QUERYLAKE_API_ONLY=1` disables local model deployments inside `server.py`:
- `llm`, `embedding`, `rerank`, `surya` are forced off for this run
- You can still use the backend for documents, toolchains, and external providers

### Running without vLLM installed

If your config requests local vLLM deployments but your environment does not have vLLM installed, you can start with:

```bash
export QUERYLAKE_SKIP_VLLM=1
python start_querylake.py
```

This is meant for “API-only / bring-your-own-inference” workflows.

## Database (ParadeDB/Postgres)

QueryLake uses Postgres (ParadeDB image) with:
- **pgvector** for embeddings (with an **HNSW** vector index)
- **BM25** indexes (ParadeDB) for text search

Start the DB:

```bash
docker compose -f docker-compose-only-db.yml up -d
```

### DB initialization quirks

`QueryLake/database/create_db_session.py`:
- Creates tables and indices on startup if they don’t exist.
- Contains a known “first boot” quirk around index creation that is worked around by creating a fresh SQLModel session.
- Uses a hard-coded local DB URL today (see the Quickstart section). Changing the DB host/port currently requires editing that file.

### Full reset script (destructive)

`./restart_database.sh` performs a destructive reset using Docker (including volume removal).
Read it before running on shared machines.

## Redis (optional)

QueryLake can use Redis when `QUERYLAKE_REDIS_URL` (or `REDIS_URL`) is set.

This repo ships a hardened local Redis compose:

- `docker-compose-redis.yml` (binds to `127.0.0.1`, requires a password)
- `scripts/restart_querylake_redis.sh` (helper wrapper; reads `.env` if present)

Example:

```bash
export QUERYLAKE_REDIS_PASSWORD='change-me'
./scripts/restart_querylake_redis.sh

export QUERYLAKE_REDIS_URL="redis://:${QUERYLAKE_REDIS_PASSWORD}@127.0.0.1:6393/0"
```

## API overview

### Health / readiness / metrics

- `GET /healthz` → `{ "ok": true }`
- `GET /readyz` → checks DB and reports configured local model IDs
- `GET /metrics` → Prometheus-style metrics

### OpenAI-compatible endpoints

- `POST /v1/chat/completions`
- `POST /v1/embeddings`
- `GET /v1/models`

Auth is via `Authorization: Bearer ...`:
- `Bearer sk-...` is treated as a QueryLake API key
- Any other bearer token is treated as an OAuth2 token

### Native function API (`/api/*`)

QueryLake exposes many backend capabilities as “functions” under:

- `POST /api/<function_name>`
- `GET /api/<function_name>?parameters=<json>`

Examples:

Create a user:

```bash
curl -sS -X POST http://127.0.0.1:8000/api/add_user \
  -H 'Content-Type: application/json' \
  -d '{"username":"admin","password":"admin"}' | jq .
```

Create an API key (note: uses OAuth2 token, not an API key):

```bash
curl -sS -X POST http://127.0.0.1:8000/api/create_api_key \
  -H 'Content-Type: application/json' \
  -d '{"auth":"<oauth_token>","title":"dev key"}' | jq .
```

Function discovery:

- `GET /api/help` (all available functions)
- `GET /api/help/<function_name>` (signature + docs preview)

## Toolchains overview

Toolchains are QueryLake’s “workflow graphs” (agentic/automation pipelines).
They exist in two layers:

1. **Execution** (backend): stateful sessions, node graphs, event routing, outputs/files.
2. **Interface** (frontend): interactive UIs that can be generated from a toolchain config.

The backend loads toolchains from `toolchains/*.json` at startup and seeds them into the DB.

### Toolchain UI specs (V1 vs V2)

- **V1**: `display_configuration` inside a toolchain JSON (legacy interface schema).
- **V2**: `ui_spec_v2` inside a toolchain JSON (new interface schema).

`toolchains/self_guided_rag.json` now contains both:
- legacy V1 display configuration
- a V2 UI spec for the BASF-style “Self Guided RAG” experience

There is also an extracted example V2 spec in:
- `toolchains_v2_examples/self_guided_rag_v2_ui.json`

### Toolchain seeding behavior (important)

On startup, QueryLake seeds toolchains into the SQL database.

If you do **not** want local JSON files to overwrite DB-edited toolchains, set:

```bash
export QL_TOOLCHAINS_SEED_ONLY=1
```

## Configuration

### `config.json`

`config.json` is the primary configuration source for:
- Ray cluster ports + default placement strategy
- enabled model classes (llm/embedding/rerank/surya)
- local model definitions and external provider model catalogs
- default model selection

### Key environment flags

- `QUERYLAKE_API_ONLY=1` → skip local model deployments
- `QUERYLAKE_SKIP_VLLM=1` → if config requests vLLM but it’s not installed, skip vLLM deployments
- `QL_TOOLCHAINS_SEED_ONLY=1` → seed toolchains only if missing (do not overwrite existing DB rows)
- `QUERYLAKE_DB_CONNECT_TIMEOUT=5` → Postgres connect timeout in seconds
- `QUERYLAKE_REDIS_URL=...` → enable Redis client usage
- `RAY_TMPDIR=/tmp/querylake_ray` → Ray temp/log directory (QueryLake defaults this in `start_querylake.py`)

## Development

Run tests:

```bash
uv sync --extra dev
pytest -q
```

Notes:
- `pytest.ini` excludes heavy directories (models, docs_tmp, object_store, etc.).

## Troubleshooting / quirks

### “RayTaskError(AsyncEngineDeadError)” (vLLM)

If you see errors like:
- `vllm.engine.async_llm_engine.AsyncEngineDeadError: Background loop is not running`

It usually indicates the vLLM engine actor crashed. Inspect Ray logs for the replica and check:
- GPU memory pressure / OOM
- incompatible vLLM/torch/CUDA stack
- model max length / KV cache sizing issues

For an API-only run (no local vLLM), use:

```bash
export QUERYLAKE_API_ONLY=1
export QUERYLAKE_SKIP_VLLM=1
python start_querylake.py
```

### Ray log bloat (/tmp/ray/session_*)

`start_querylake.py` sets conservative Ray log rotation defaults (100 MiB * 10 backups) and defaults `RAY_TMPDIR`
to `/tmp/querylake_ray` to keep session artifacts contained.

### Toolchain JSON evolves over time

Toolchain configs are treated as editor state as well as runtime config.
Expect forward-compatible “extra fields” (e.g. `ui_spec_v2`) in JSON stored in the DB.

## Security notes

- The included docker-compose files are intended for **local development** defaults.
  For real deployments you should:
  - change DB creds
  - bind services to private networks
  - use TLS at the ingress layer
- Redis is shipped with:
  - a required password
  - localhost-only host binding (`127.0.0.1:<port>`)
  Do not expose it publicly without ACLs / firewalling.
