SHELL := /bin/bash

.PHONY: bootstrap up-db down-db up-redis down-redis run run-api-only health test ci-docs ci-unification ci-retrieval-smoke ci-runtime-profile ci-runtime-delta sdk-install-dev sdk-precommit-install sdk-precommit-run sdk-lint sdk-type sdk-test sdk-build sdk-ci sdk-smoke sdk-release-check sdk-release-testpypi sdk-release-pypi sdk-publish-guard sdk-dryrun-version

bootstrap:
	./scripts/dev/bootstrap.sh

up-db:
	@if docker ps --format '{{.Names}}' | grep -qx 'querylake_db'; then \
		echo "[up-db] querylake_db already running; reusing existing container."; \
	elif docker ps -a --format '{{.Names}}' | grep -qx 'querylake_db'; then \
		echo "[up-db] querylake_db exists but is stopped; starting container."; \
		docker start querylake_db >/dev/null; \
	else \
		docker compose -f docker-compose-only-db.yml up -d; \
	fi

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

ci-docs:
	uv run --no-project bash scripts/ci_docs_checks.sh

ci-unification:
	uv run --no-project bash scripts/ci_unification_checks.sh

ci-retrieval-smoke:
	CI_RETRIEVAL_OUT_DIR=docs_tmp/RAG/ci/local/make_smoke uv run --no-project bash scripts/ci_retrieval_preflight.sh smoke
	CI_RETRIEVAL_OUT_DIR=docs_tmp/RAG/ci/local/make_smoke uv run --no-project bash scripts/ci_retrieval_eval.sh smoke
	CI_RETRIEVAL_OUT_DIR=docs_tmp/RAG/ci/local/make_smoke uv run --no-project bash scripts/ci_retrieval_parity.sh smoke

ci-runtime-profile:
	@if [[ -z "$(REPO)" ]]; then \
		echo "usage: make ci-runtime-profile REPO=owner/repo [DAYS=7] [MAX_RUNS=1000] [OUT_DIR=docs_tmp/RAG/ci/runtime_profile/local]"; \
		exit 2; \
	fi
	@OUT_DIR="$(if $(OUT_DIR),$(OUT_DIR),docs_tmp/RAG/ci/runtime_profile/local)"; \
	DAYS="$(if $(DAYS),$(DAYS),7)"; \
	MAX_RUNS="$(if $(MAX_RUNS),$(MAX_RUNS),1000)"; \
	mkdir -p "$$OUT_DIR"; \
	uv run --no-project python scripts/dev/ci_runtime_profile.py \
		--repo "$(REPO)" \
		--days "$$DAYS" \
		--max-runs "$$MAX_RUNS" \
		--out-json "$$OUT_DIR/profile.json" \
		--out-md "$$OUT_DIR/profile.md"

ci-runtime-delta:
	@if [[ -z "$(BEFORE)" || -z "$(AFTER)" ]]; then \
		echo "usage: make ci-runtime-delta BEFORE=path/to/before.json AFTER=path/to/after.json [OUT=docs_tmp/RAG/ci/runtime_profile/local/delta.md]"; \
		exit 2; \
	fi
	@OUT="$(if $(OUT),$(OUT),docs_tmp/RAG/ci/runtime_profile/local/delta.md)"; \
	uv run --no-project python scripts/dev/ci_runtime_delta.py \
		--before-json "$(BEFORE)" \
		--after-json "$(AFTER)" \
		--out-md "$$OUT"

sdk-install-dev:
	uv run --project sdk/python pip install -e sdk/python

sdk-precommit-install:
	uv run --project sdk/python --extra dev pre-commit install --hook-type pre-commit --hook-type pre-push

sdk-precommit-run:
	uv run --project sdk/python --extra dev pre-commit run --all-files --hook-stage pre-commit
	uv run --project sdk/python --extra dev pre-commit run --all-files --hook-stage pre-push

sdk-lint:
	bash scripts/dev/sdk_quality_gate.sh lint

sdk-type:
	bash scripts/dev/sdk_quality_gate.sh type

sdk-test:
	bash scripts/dev/sdk_quality_gate.sh test

sdk-build:
	uv run --project sdk/python --with build python -m build sdk/python

sdk-ci:
	bash scripts/ci_sdk_checks.sh

sdk-smoke:
	./scripts/dev/smoke_sdk_local.sh

sdk-release-check:
	./scripts/dev/release_sdk.sh check

sdk-release-testpypi:
	./scripts/dev/release_sdk.sh testpypi

sdk-release-pypi:
	./scripts/dev/release_sdk.sh pypi

sdk-publish-guard:
	@if [[ -z "$(TARGET)" ]]; then \
		echo "usage: make sdk-publish-guard TARGET=testpypi|pypi [GITHUB_REF=refs/heads/main] [SKIP_REMOTE_CHECK=1]"; \
		exit 2; \
	fi
	@GUARD_ARGS="--target $(TARGET) --package-name querylake-sdk"; \
	if [[ -n "$(GITHUB_REF)" ]]; then GUARD_ARGS="$$GUARD_ARGS --github-ref $(GITHUB_REF)"; fi; \
	if [[ "$(SKIP_REMOTE_CHECK)" == "1" ]]; then GUARD_ARGS="$$GUARD_ARGS --skip-remote-check"; fi; \
	uv run --no-project python scripts/dev/verify_sdk_publish_guard.py $$GUARD_ARGS

sdk-dryrun-version:
	@TOKEN_VALUE="$(TOKEN)"; \
	if [[ -z "$$TOKEN_VALUE" ]]; then TOKEN_VALUE="$$(date -u +%Y%m%d%H%M%S)"; fi; \
	uv run --no-project python scripts/dev/prepare_sdk_dryrun_version.py \
		--version-file sdk/python/pyproject.toml \
		--token "$$TOKEN_VALUE" \
		$(if $(WRITE),--write,)
