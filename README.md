# QueryLake

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![SDK Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml)
[![Retrieval Eval](https://github.com/kmccleary3301/QueryLake/actions/workflows/retrieval_eval.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/retrieval_eval.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)
[![PyPI](https://img.shields.io/pypi/v/querylake-sdk?logo=pypi&color=F9A03C)](https://pypi.org/project/querylake-sdk/)
[![Python](https://img.shields.io/pypi/pyversions/querylake-sdk?logo=python&color=3776AB)](https://pypi.org/project/querylake-sdk/)
[![License](https://img.shields.io/github/license/kmccleary3301/QueryLake?color=2ea44f)](./LICENSE)
[![Last Commit](https://img.shields.io/github/last-commit/kmccleary3301/QueryLake)](https://github.com/kmccleary3301/QueryLake/commits/main)
[![Backend](https://img.shields.io/badge/backend-FastAPI%20%2B%20Ray%20Serve-0A7BBB)](#repository-layout)
[![Studio](https://img.shields.io/badge/studio-Next.js%20included-111827)](#repository-layout)

QueryLake is a production-oriented, research-friendly platform for self-hosted AI, retrieval, document ingestion, and toolchain-driven application runtimes.

It combines four things that are usually scattered across separate repos and operational stacks:

- a **FastAPI + Ray Serve backend** with OpenAI-compatible and platform-native APIs,
- a **hybrid retrieval platform** built around BM25, dense, and sparse search lanes,
- a **toolchain runtime** for graph execution and app-like interfaces,
- and a **Python SDK + CLI** intended for developers, operators, and RAG researchers.

> QueryLake is opinionated where it matters: ParadeDB/Postgres is the recommended gold-stack for full retrieval semantics, the SDK is the preferred integration surface, and compatibility/degradation behavior is treated as a product contract rather than an implementation detail.

## Table of contents

- [What this repository is](#what-this-repository-is)
- [Why QueryLake exists](#why-querylake-exists)
- [Capability snapshot](#capability-snapshot)
- [Repository layout](#repository-layout)
- [Quickstart](#quickstart)
- [Install and setup](#install-and-setup)
- [Usage examples](#usage-examples)
- [Retrieval and RAG model](#retrieval-and-rag-model)
- [Toolchains and studio](#toolchains-and-studio)
- [Project status and support matrix](#project-status-and-support-matrix)
- [Documentation map](#documentation-map)
- [CI, release, and packaging](#ci-release-and-packaging)
- [Contributing](#contributing)
- [License](#license)

## What this repository is

This is the **canonical QueryLake repository**.

It contains:

| Surface | What lives here | Intended consumer |
|---|---|---|
| Backend runtime | `QueryLake/`, `server.py`, deployment/config files | Backend deployers, infra engineers |
| Studio frontend | `apps/studio/` | UI engineers, toolchain/app authors |
| Python SDK + CLI | `sdk/python/` | App developers, researchers, automation users |
| Retrieval harnesses | `scripts/`, `tests/` | RAG researchers, backend engineers |
| Docs and runbooks | `docs/` | Anyone touching setup, release, migration, or runtime ops |

There is still some historical naming/migration context:

- the old standalone frontend repo was renamed to [`QueryLakeStudio`](https://github.com/kmccleary3301/QueryLakeStudio),
- the current frontend code now lives under `apps/studio/` in this repository,
- and the old local filesystem alias `QueryLakeBackend` is deprecated in favor of the canonical local path `QueryLake`.

See [`docs/unification/repo_migration.md`](docs/unification/repo_migration.md) and [`docs/unification/symlink_retirement_runbook.md`](docs/unification/symlink_retirement_runbook.md).

## Why QueryLake exists

QueryLake is built around a practical observation: most self-hosted RAG/application stacks fail because the retrieval layer, ingestion layer, UI surface, and runtime orchestration all evolve independently and end up disagreeing with each other.

This repo exists to keep those pieces aligned:

- **API contract stability**
  - OpenAI-compatible routes for common client compatibility.
  - Platform-native `/api/<function_name>` routes for the actual control plane.
- **Research velocity**
  - retrieval experiments, rollout gates, parity harnesses, and stress suites live in the same repo as the runtime they are evaluating.
- **Developer ergonomics**
  - `querylake-sdk` and the `querylake` CLI are the first-class integration surfaces.
- **Operational honesty**
  - feature gating, compatibility checks, and structured failures matter more than pretending every deployment profile behaves identically.
- **App/runtime coherence**
  - the Toolchains V2 designer, V2 app surface, and backend runtime are increasingly sharing the same spine instead of drifting apart.

## Capability snapshot

| Area | Scope | Status |
|---|---|---|
| OpenAI-compatible API | `/v1/chat/completions`, `/v1/embeddings`, `/v1/models` | 🟢 |
| Platform API | `/api/*` function routes for auth, collections, search, ingestion, toolchains | 🟢 |
| Hybrid retrieval | BM25 + dense + sparse fusion | 🟢 |
| Advanced lexical controls | constraint-aware parsing, operator-aware retrieval, hard prefilters | 🟢 |
| ParadeDB-first gold stack | best-supported retrieval semantics | 🟢 |
| Toolchains V2 editor | graph editor, interface designer, settings, V2 app surface | 🟡 |
| Shared V2 runtime spine | designer preview aligned with V2 app route | 🟡 |
| Legacy runner retirement | compatibility shims in place, full retirement still staged | 🟡 |
| Multi-backend DB compatibility | planned, architecture work staged, not yet implemented | 🔴 |

### What QueryLake is good at right now

| Use case | Fit |
|---|---|
| Local/self-hosted RAG backend with strong retrieval controls | 🟢 Strong |
| Retrieval experiments with measurable eval/stress outputs | 🟢 Strong |
| Document ingestion + OCR + searchable corpora | 🟢 Strong |
| Toolchain-style internal apps and workflows | 🟡 Working, still maturing |
| “Drop-in portability to any DB/search backend” | 🔴 Not the current promise |

## Repository layout

```text
QueryLake/
├── QueryLake/                         Core backend package
│   ├── api/                           Function-style API handlers
│   ├── auth/                          Auth providers, token/session logic
│   ├── database/                      SQLModel / Postgres / ParadeDB integration
│   ├── document_parse/                Parsing, OCR, ingestion helpers
│   ├── files/                         File/object handling primitives
│   ├── kernel/                        V2 kernel session plumbing
│   ├── observability/                 Metrics, tracing, diagnostics
│   ├── operation_classes/             Runtime operation classes
│   ├── runtime/                       Retrieval runtime, toolchain execution, events
│   ├── toolchains/                    Toolchain/runtime support code
│   ├── typing/                        Shared typed payloads/contracts
│   ├── vector_database/               Vector index support and helpers
│   └── web/                           HTTP/web-facing helpers
├── apps/
│   └── studio/                        Next.js studio frontend and Toolchains V2 UI
├── sdk/
│   └── python/                        `querylake-sdk` package and CLI
├── examples/
│   └── sdk/                           Runnable SDK examples and offline demos
├── scripts/                           CI, eval, rollout, BCAS, Chandra, ops tooling
├── tests/                             Retrieval, ingestion, auth, toolchains, runtime tests
├── toolchains/                        Runtime toolchain definitions (.json)
├── toolchains_v2_examples/            V2 UI-spec examples and demo toolchains
├── docs/
│   ├── setup/                         Setup and developer experience docs
│   ├── sdk/                           SDK reference, release, CI, integration docs
│   ├── unification/                   Repo/API/path topology and migration docs
│   └── chandra/                       Chandra OCR/runtime notes
├── docker-compose.yml                 Full local stack
├── docker-compose-only-db.yml         DB-only local stack
├── Makefile                           Common dev/CI/release entry points
├── pyproject.toml                     Backend package metadata (`querylake-backend`)
├── server.py                          FastAPI + Ray Serve entrypoint
└── README.md                          You are here
```

<details>
<summary>Additional topology notes</summary>

- `sdk/python/` is intentionally a standalone packaging lane. Application developers should usually start there, not by importing backend internals.
- `apps/studio/` is now part of the canonical repo. `QueryLakeStudio` is historical/deprecated as a standalone development target.
- `scripts/` is not just miscellaneous tooling; it contains much of the retrieval-eval, BCAS, rollout-gate, and operations surface that keeps retrieval changes honest.

</details>

## Quickstart

### Fastest path: local API-only backend + SDK

This is the best starting point for most developers and RAG researchers.

```bash
cp .env.example .env
make bootstrap
make up-db
make run-api-only
make health
make sdk-smoke
```

At that point you have:

- a local QueryLake backend running,
- ParadeDB/Postgres available,
- the API surface responding,
- and the SDK smoke path validated.

### If you want the full local stack

```bash
cp .env.example .env
make bootstrap
make up-db
make up-redis
make run
```

### If you only need the Python SDK

```bash
pip install querylake-sdk
querylake --url http://127.0.0.1:8000 doctor
```

## Choose your starting path

| If you are... | Start here | Why |
|---|---|---|
| building an app against QueryLake | [`querylake-sdk`](https://pypi.org/project/querylake-sdk/) + [`docs/sdk/SDK_QUICKSTART.md`](docs/sdk/SDK_QUICKSTART.md) | smallest surface area, fastest path to value |
| bringing up a local backend | [`docs/setup/DEVELOPER_SETUP.md`](docs/setup/DEVELOPER_SETUP.md) | canonical local environment instructions |
| working on retrieval quality/perf | [`scripts/`](scripts/) + [`tests/test_retrieval_*.py`](tests/) + [`docs/sdk/RAG_RESEARCH_PLAYBOOK.md`](docs/sdk/RAG_RESEARCH_PLAYBOOK.md) | eval, parity, gates, and reproducible harnesses already exist |
| working on toolchains/studio | [`apps/studio/`](apps/studio/) + [`toolchains/`](toolchains/) | editor/runtime/UI surfaces live in-repo |
| trying to understand repo topology first | [`docs/README.md`](docs/README.md) + [`docs/unification/`](docs/unification/) | architecture and migration context |

## Install and setup

### Prerequisites

| Requirement | Why it matters |
|---|---|
| Python `3.12` | primary local development target |
| `uv` | fastest and cleanest local package workflow |
| Docker | local ParadeDB/Postgres and optional Redis |
| NVIDIA stack (optional) | local embedding/rerank/LLM deployment work |

### Local backend setup

```bash
git clone https://github.com/kmccleary3301/QueryLake.git
cd QueryLake
cp .env.example .env
make bootstrap
make up-db
make run-api-only
```

### Local studio setup

```bash
cd apps/studio
npm install
npm run dev
```

By default the studio expects the backend to be running locally. The backend/studio are developed in the same repo now; this is the intended topology.

### Common backend start modes

| Mode | Command | Uses local models? | Intended use |
|---|---|---|---|
| API-only | `make run-api-only` | No | 🟢 local dev, SDK work, retrieval API integration |
| Full local runtime | `make run` | Yes, if configured | 🟡 heavier local runtime experiments |
| DB-only infra | `make up-db` | N/A | 🟢 local backend dependency bring-up |
| Redis add-on | `make up-redis` | N/A | 🟡 session/cache/event features |

### Environment variables you will touch first

| Variable | Meaning |
|---|---|
| `QUERYLAKE_API_ONLY=1` | disable local LLM deployments while keeping API/RAG/toolchains online |
| `QUERYLAKE_OAUTH_SECRET_KEY=<secret>` | stable local auth tokens |
| `QUERYLAKE_REDIS_URL=redis://localhost:6379/0` | enable Redis-backed runtime surfaces |
| `QUERYLAKE_DB_CONNECT_TIMEOUT=5` | DB connection timeout control |

For the concrete local setup path, read [`docs/setup/DEVELOPER_SETUP.md`](docs/setup/DEVELOPER_SETUP.md).

## Health and API surface

| Endpoint / command | Role | Notes |
|---|---|---|
| `GET /healthz` | liveness check | lightweight backend reachability |
| `GET /readyz` | readiness check | useful for deployment orchestration |
| `GET /api/ping` | platform API probe | confirms function API surface is online |
| `GET /v1/models` | OpenAI-compatible probe | quick compatibility smoke |
| `querylake doctor` | SDK/CLI-side health check | best developer-first smoke command |

## Usage examples

### Python SDK: create a collection, ingest files, run hybrid retrieval

```python
from querylake_sdk import QueryLakeClient

client = QueryLakeClient(base_url="http://127.0.0.1:8000")
client.login(username="demo", password="demo-pass")

collection = client.create_collection(name="papers")
collection_id = collection["hash_id"]

client.upload_document(
    file_path="paper.pdf",
    collection_hash_id=collection_id,
    await_embedding=True,
    create_sparse_embeddings=True,
)

results = client.search_hybrid_chunks(
    query="What is the main contribution?",
    collection_ids=[collection_id],
    limit_bm25=12,
    limit_similarity=12,
    limit_sparse=12,
    bm25_weight=0.4,
    similarity_weight=0.4,
    sparse_weight=0.2,
)

for row in results[:5]:
    print(row.document_name, row.hybrid_score)
```

### CLI: health, auth, ingest, search

```bash
querylake --url http://127.0.0.1:8000 doctor
querylake setup --url http://127.0.0.1:8000 --profile local --username demo --password demo-pass --non-interactive
querylake login --url http://127.0.0.1:8000 --profile local --username demo --password demo-pass
querylake --profile local rag create-collection --name "papers"
querylake --profile local rag upload --collection-id <id> --file ./paper.pdf --await-embedding
querylake --profile local rag search --collection-id <id> --query "hybrid retrieval design" --preset tri-lane --with-metrics
```

### OpenAI-compatible endpoints

```bash
curl http://127.0.0.1:8000/v1/models

curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{
    "model": "openai_compatible",
    "messages": [
      {"role": "user", "content": "Summarize the retrieval stack in one paragraph."}
    ]
  }'
```

### Platform-native API route

```bash
curl http://127.0.0.1:8000/api/search_hybrid \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer <token>' \
  -d '{
    "query": "vapor recovery",
    "collection_ids": ["<collection-id>"],
    "limit_bm25": 12,
    "limit_similarity": 12,
    "limit_sparse": 12
  }'
```

### Runnable example from this repo

```bash
python examples/sdk/rag_bulk_ingest_and_search.py \
  --base-url http://127.0.0.1:8000 \
  --username demo \
  --password demo-pass \
  --collection sdk-bulk-demo \
  --dir ./docs \
  --pattern "*.md" \
  --recursive \
  --query "hybrid retrieval"
```

### Offline example mode

```bash
python examples/sdk/rag_bulk_ingest_and_search.py \
  --offline-demo \
  --dir ./docs \
  --pattern "*.md" \
  --recursive \
  --query "hybrid retrieval"
```

<details>
<summary>More CLI and SDK examples</summary>

#### Batch retrieval benchmark via SDK example

```bash
python examples/sdk/rag_search_batch_benchmark.py \
  --offline-demo \
  --queries-file ./examples/sdk/fixtures/offline_queries.txt \
  --output-file ./artifacts/benchmark_offline.json
```

#### Bulk upload planning mode

```bash
querylake --profile local rag upload-dir \
  --collection-id <id> \
  --dir ./docs \
  --pattern "*" \
  --recursive \
  --extensions ".pdf,.md" \
  --exclude-glob "archive/*" \
  --dry-run \
  --list-files \
  --selection-output ./artifacts/selected_files.json \
  --report-file ./artifacts/upload_dry_run.json
```

#### Async SDK client

```python
import asyncio
from querylake_sdk import AsyncQueryLakeClient

async def run():
    async with AsyncQueryLakeClient(base_url="http://127.0.0.1:8000") as client:
        await client.login(username="demo", password="demo-pass")
        print(await client.healthz())

asyncio.run(run())
```

</details>

## Retrieval and RAG model

QueryLake’s retrieval surface is not “just vector search.” The current gold-stack is designed around lane composition and measurable rollout discipline.

### Retrieval lanes

| Lane | Purpose | Current posture |
|---|---|---|
| BM25 / lexical | keyword precision, constraints, operator-aware retrieval | 🟢 first-class |
| Dense vectors | semantic similarity | 🟢 first-class |
| Sparse vectors | semantic lexical expansion / hybrid support | 🟢 first-class |
| Fusion | weighted or orchestrated hybrid ranking | 🟢 first-class |
| Graph / agentic layers | higher-order retrieval traversal/orchestration | 🟡 evolving |

### What is emphasized in this repo

- retrieval eval and parity harnesses live alongside the runtime,
- rollout gates and strict-check scripts exist for retrieval changes,
- dense/sparse/BM25 ingestion controls are request-level concerns,
- the recommended stack is still **ParadeDB + Postgres**, because that is where the strongest semantics currently exist.

### Representative scripts

| Script | Purpose |
|---|---|
| `scripts/retrieval_eval.py` | retrieval evaluation harness |
| `scripts/retrieval_parity.py` | parity checks against expected retrieval behavior |
| `scripts/retrieval_gate_report.py` | rollout/report generation |
| `scripts/bcas_phase2_eval.py` | BCAS-oriented eval workflow |
| `scripts/bcas_phase2_stress.py` | stress and throughput characterization |
| `scripts/bcas_phase2_three_lane_track.py` | explicit tri-lane track work |

### Representative tests

| Test area | Files |
|---|---|
| Retrieval contracts and runtime | `tests/test_retrieval_*.py` |
| Sparse dimension checks | `tests/test_sparse_dimension_consistency.py` |
| ParadeDB query parsing | `tests/test_paradedb_query_parser.py` |
| Search budgeting | `tests/test_search_budgeting.py` |
| Toolchains V2 retrieval/runtime integration | `tests/test_toolchains_v2*.py` |

If you want the dense local deep-dive on BM25 behavior, operator constraints, and QueryLake retrieval semantics, see `docs_tmp/RAG/QL_BM25_AND_SPECIFICS_MEGA_GUIDE.md` in a working checkout.

## Toolchains and studio

QueryLake includes an application/runtime layer, not just an API server.

That surface is now split across:

| Path | Role |
|---|---|
| `toolchains/` | runtime toolchain definitions |
| `toolchains_v2_examples/` | V2 interface examples and demo specs |
| `apps/studio/` | studio UI, Toolchains V2 editor, V2 app surface |
| `QueryLake/runtime/` | backend execution, event streams, kernel/runtime state |

Recent direction of travel:

- the Toolchains V2 graph editor and interface designer have been recovered into the canonical repo,
- the V2 interface designer preview and V2 app surface now share the same rendering spine,
- legacy runner shims still exist, but the V2 app route is the intended direction.

<details>
<summary>Frontend / Toolchains notes</summary>

- Studio entrypoint lives under `apps/studio/` and uses Next.js.
- Toolchains V2 work is still active; expect some surfaces to continue changing.
- Legacy runtime compatibility is still present in places, but the current direction is toward V2-first routes and a shared UI runtime surface.

</details>

## Project status and support matrix

### Runtime and packaging status

| Surface | Status | Notes |
|---|---|---|
| Backend package (`querylake-backend`) | 🟡 | package metadata exists; backend is primarily consumed from source/deployment repo |
| Python SDK (`querylake-sdk`) | 🟢 | installable, tested, PyPI-facing |
| Studio frontend | 🟡 | active, canonical location is `apps/studio/` |
| Retrieval platform | 🟢 | heavily instrumented, eval/stress tooling present |
| Chandra OCR/runtime | 🟡 | supported, specialized runtime path |

### Platform assumptions

| Platform | Current stance |
|---|---|
| Linux + Docker | 🟢 primary development path |
| ParadeDB/Postgres | 🟢 recommended retrieval/data stack |
| Redis | 🟡 optional but useful for some runtime features |
| Remote vLLM / model services | 🟡 supported in practice, deployment-specific |
| Broad multi-backend DB/search portability | 🔴 planned, not current promise |

## Documentation map

### Start here

| If you want to… | Read this |
|---|---|
| bring up a local backend | [`docs/setup/DEVELOPER_SETUP.md`](docs/setup/DEVELOPER_SETUP.md) |
| understand the docs surface | [`docs/README.md`](docs/README.md) |
| use the Python SDK | [`docs/sdk/SDK_QUICKSTART.md`](docs/sdk/SDK_QUICKSTART.md) |
| do RAG/retrieval work with the SDK | [`docs/sdk/RAG_RESEARCH_PLAYBOOK.md`](docs/sdk/RAG_RESEARCH_PLAYBOOK.md) |
| run bulk ingestion carefully | [`docs/sdk/BULK_INGEST_REFERENCE.md`](docs/sdk/BULK_INGEST_REFERENCE.md) |
| inspect SDK methods/endpoints | [`docs/sdk/API_REFERENCE.md`](docs/sdk/API_REFERENCE.md) |
| understand publishing/release policy | [`docs/sdk/PYPI_RELEASE.md`](docs/sdk/PYPI_RELEASE.md) |
| understand CI profiles | [`docs/sdk/CI_PROFILES.md`](docs/sdk/CI_PROFILES.md) |
| understand repo/API/path migration | [`docs/unification/`](docs/unification/) |
| work on Chandra | [`docs/chandra/CHANDRA_OCR_VLLM_SERVER.md`](docs/chandra/CHANDRA_OCR_VLLM_SERVER.md) |

### Useful local-only design/history artifacts

Some of the most detailed project notes live in `docs_tmp/`. Those are intentionally not treated as polished user docs, but they are useful if you are working deeply on retrieval, runtime cutovers, or frontend/toolchain redesign work.

## CI, release, and packaging

### Main workflows

| Workflow | Purpose |
|---|---|
| [`docs_checks.yml`](.github/workflows/docs_checks.yml) | README/docs consistency and docs checks |
| [`sdk_checks.yml`](.github/workflows/sdk_checks.yml) | SDK tests, lint, typecheck, release guard |
| [`retrieval_eval.yml`](.github/workflows/retrieval_eval.yml) | smoke/nightly/heavy retrieval eval paths |
| [`sdk_live_integration.yml`](.github/workflows/sdk_live_integration.yml) | live staging SDK integration |
| [`sdk_publish.yml`](.github/workflows/sdk_publish.yml) | official SDK publish workflow |
| [`sdk_publish_dryrun.yml`](.github/workflows/sdk_publish_dryrun.yml) | scheduled/manual TestPyPI dry-run |
| [`unification_checks.yml`](.github/workflows/unification_checks.yml) | repo/path unification guardrails |
| [`legacy_path_guard.yml`](.github/workflows/legacy_path_guard.yml) | canonical naming enforcement |
| [`ci_runtime_profiler.yml`](.github/workflows/ci_runtime_profiler.yml) | CI runtime profiling snapshots |

### Local commands you will actually use

```bash
# Core local bring-up
make bootstrap
make up-db
make up-redis
make run-api-only
make run
make health

# Repo hygiene
make ci-docs
make ci-unification
make ci-legacy-path-guard

# SDK
make sdk-install-dev
make sdk-precommit-install
make sdk-lint
make sdk-type
make sdk-test
make sdk-build
make sdk-ci
make sdk-smoke

# Retrieval
make ci-retrieval-smoke
```

### Packaging lanes

| Package | Location | Intended audience |
|---|---|---|
| `querylake-backend` | root `pyproject.toml` | deployers, backend packagers |
| `querylake-sdk` | `sdk/python/pyproject.toml` | developers, researchers, automation users |

Practical guidance:

- if you are building applications against QueryLake, start with **`querylake-sdk`**;
- if you are deploying or changing the runtime itself, work from this repository directly.

## Contributing

Read [`CONTRIBUTING.md`](CONTRIBUTING.md) before opening a PR.

A few practical expectations matter here:

- retrieval changes should come with evidence, not just intuition,
- SDK changes are external API contract changes and should be treated conservatively,
- setup/runtime changes should update the relevant docs in the same PR,
- if a deployment profile or compatibility claim changes, it should be written down explicitly.

> This repository is optimized for engineers and researchers who are willing to read real documentation, run concrete commands, and evaluate runtime behavior empirically. That is deliberate.

## License

QueryLake is licensed under Apache-2.0. See [`LICENSE`](LICENSE).
