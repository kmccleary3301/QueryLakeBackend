SHELL := /bin/bash

.PHONY: bootstrap up-db down-db up-redis down-redis run run-api-only health test sdk-install-dev sdk-test sdk-build sdk-smoke

bootstrap:
	./scripts/dev/bootstrap.sh

up-db:
	docker compose -f docker-compose-only-db.yml up -d

down-db:
	docker compose -f docker-compose-only-db.yml down

up-redis:
	docker compose -f docker-compose-redis.yml up -d

down-redis:
	docker compose -f docker-compose-redis.yml down

run:
	uv run start_querylake.py

run-api-only:
	QUERYLAKE_API_ONLY=1 uv run start_querylake.py

health:
	@echo "==> /healthz"
	@curl -sS http://127.0.0.1:8000/healthz || true
	@echo ""
	@echo "==> /readyz"
	@curl -sS http://127.0.0.1:8000/readyz || true
	@echo ""
	@echo "==> /v1/models"
	@curl -sS http://127.0.0.1:8000/v1/models || true
	@echo ""

test:
	uv run pytest

sdk-install-dev:
	uv run --project sdk/python pip install -e sdk/python

sdk-test:
	uv run --project sdk/python pytest sdk/python/tests

sdk-build:
	uv run --project sdk/python --with build python -m build sdk/python

sdk-smoke:
	./scripts/dev/smoke_sdk_local.sh
