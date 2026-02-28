# QueryLake Developer Setup

This setup path is optimized for backend contributors and RAG researchers.

## Prerequisites

- Python 3.12
- `uv` (recommended package manager)
- Docker (for Postgres/ParadeDB, optional Redis)

## 1) Clone and bootstrap

```bash
git clone <repo-url>
cd QueryLakeBackend
cp .env.example .env
make bootstrap
```

`make bootstrap` installs API-only runtime + dev tooling.

## 2) Start infrastructure

```bash
make up-db
make up-redis
```

## 3) Start backend

```bash
make run-api-only
```

or full runtime:

```bash
make run
```

## 4) Smoke checks

```bash
make health
```

## 5) SDK local development

```bash
make sdk-install-dev
make sdk-test
make sdk-smoke
make sdk-release-check
```

## Common workflows

- Retrieval eval harness:
  - `python scripts/retrieval_eval.py --help`
- BCAS phase2 controls:
  - `python scripts/bcas_phase2_eval.py --help`
  - `python scripts/bcas_phase2_stress.py --help`

## Environment variables (common)

- `QUERYLAKE_API_ONLY=1` for API-only runtime.
- `QUERYLAKE_OAUTH_SECRET_KEY=<secret>` for stable OAuth tokens.
- `QUERYLAKE_REDIS_URL=redis://localhost:6379/0` to enable Redis-backed components.
- `QUERYLAKE_DB_CONNECT_TIMEOUT=5` for DB connection timeout control.

## Notes

- Local `config.json` is ignored by git (machine-specific).
- Use `config.json.bak` as baseline when recreating local config.
