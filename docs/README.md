# QueryLake Documentation

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![SDK Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml)
[![Retrieval Eval](https://github.com/kmccleary3301/QueryLake/actions/workflows/retrieval_eval.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/retrieval_eval.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

This directory is the durable documentation surface for QueryLake.

If the root [`README.md`](../README.md) is the front page, this file is the navigator for setup, SDK usage, CI/release policy, architecture, migration, and specialized runtime docs.

## Table of contents

- [Start here](#start-here)
- [Choose your path](#choose-your-path)
- [Documentation map](#documentation-map)
- [Setup and developer experience](#setup-and-developer-experience)
- [SDK and application integration](#sdk-and-application-integration)
- [Architecture, topology, and migration](#architecture-topology-and-migration)
- [Specialized runtime docs](#specialized-runtime-docs)
- [How to use this docs tree](#how-to-use-this-docs-tree)

## Start here

| If you need to... | Read this first | Why |
|---|---|---|
| bring up a local backend | [`setup/DEVELOPER_SETUP.md`](setup/DEVELOPER_SETUP.md) | canonical local environment instructions |
| understand the current developer experience direction | [`setup/DEVELOPER_EXPERIENCE_PLAN.md`](setup/DEVELOPER_EXPERIENCE_PLAN.md) | roadmap and rationale for DX work |
| integrate from Python | [`sdk/SDK_QUICKSTART.md`](sdk/SDK_QUICKSTART.md) | shortest path to useful SDK usage |
| do retrieval/RAG work with the SDK | [`sdk/RAG_RESEARCH_PLAYBOOK.md`](sdk/RAG_RESEARCH_PLAYBOOK.md) | practical ingestion/search research workflows |
| understand repo/API/path migration | [`unification/`](unification/) | canonical naming, routing, and topology |
| understand Chandra runtime notes | [`chandra/CHANDRA_OCR_VLLM_SERVER.md`](chandra/CHANDRA_OCR_VLLM_SERVER.md) | specialized OCR/runtime setup |

## Choose your path

| Audience | Best entry docs |
|---|---|
| Backend contributor | [`setup/DEVELOPER_SETUP.md`](setup/DEVELOPER_SETUP.md), [`unification/program_control.md`](unification/program_control.md) |
| SDK consumer | [`sdk/SDK_QUICKSTART.md`](sdk/SDK_QUICKSTART.md), [`sdk/API_REFERENCE.md`](sdk/API_REFERENCE.md) |
| RAG researcher | [`sdk/RAG_RESEARCH_PLAYBOOK.md`](sdk/RAG_RESEARCH_PLAYBOOK.md), [`sdk/BULK_INGEST_REFERENCE.md`](sdk/BULK_INGEST_REFERENCE.md) |
| Release / package maintainer | [`sdk/PYPI_RELEASE.md`](sdk/PYPI_RELEASE.md), [`sdk/TESTPYPI_DRYRUN.md`](sdk/TESTPYPI_DRYRUN.md) |
| CI / staging operator | [`sdk/CI_PROFILES.md`](sdk/CI_PROFILES.md), [`sdk/LIVE_STAGING_INTEGRATION.md`](sdk/LIVE_STAGING_INTEGRATION.md) |
| Repo topology / migration reviewer | [`unification/repo_migration.md`](unification/repo_migration.md), [`unification/symlink_retirement_runbook.md`](unification/symlink_retirement_runbook.md) |

## Documentation map

```text
docs/
├── README.md                           This index / landing page
├── setup/
│   ├── DEVELOPER_SETUP.md             Local backend + environment bring-up
│   └── DEVELOPER_EXPERIENCE_PLAN.md   DX planning and standardization work
├── sdk/
│   ├── SDK_QUICKSTART.md              First SDK usage
│   ├── RAG_RESEARCH_PLAYBOOK.md       Retrieval/RAG workflows through the SDK
│   ├── BULK_INGEST_REFERENCE.md       Upload-dir, dry-run, checkpoints, dedupe
│   ├── API_REFERENCE.md               SDK method and contract reference
│   ├── PYPI_RELEASE.md                Publish/release runbook
│   ├── CI_PROFILES.md                 CI matrix and release policy
│   ├── TESTPYPI_DRYRUN.md             Dry-run release workflow
│   ├── CI_PERFORMANCE_POLICY.md       Runtime profiling and CI cost controls
│   └── LIVE_STAGING_INTEGRATION.md    Live environment integration contract
├── unification/
│   ├── api_strategy.md                API direction and route strategy
│   ├── auth_provider_interface.md     Auth abstraction contracts
│   ├── compat_matrix.md               Compatibility/program control notes
│   ├── observability_v1.md            Observability direction
│   ├── program_control.md             Rollout/control guidance
│   ├── repo_migration.md              Canonical repo/path migration state
│   ├── route_prefixes.md              Route layout guidance
│   ├── repo_pinning_playbook.md       Downstream pinning policy
│   ├── symlink_retirement_runbook.md  Legacy path retirement schedule
│   └── unification_done_bar.md        Status tracker
├── chandra/
│   └── CHANDRA_OCR_VLLM_SERVER.md     Chandra OCR/vLLM runtime notes
└── deps_upgrade/
    ├── UPGRADE_GATES.md               Dependency upgrade gates
    └── UPGRADE_MATRIX_CHOOSER.md      Upgrade path/matrix guidance
```

## Setup and developer experience

| Doc | What it covers |
|---|---|
| [`setup/DEVELOPER_SETUP.md`](setup/DEVELOPER_SETUP.md) | local bring-up, Docker services, backend run modes, SDK smoke path |
| [`setup/DEVELOPER_EXPERIENCE_PLAN.md`](setup/DEVELOPER_EXPERIENCE_PLAN.md) | standardizing setup, SDK-first usage, and docs/packaging polish |

Practical recommendation:

- if you are new to the repo, start with `DEVELOPER_SETUP.md`,
- if you are changing how people install or use QueryLake, also read `DEVELOPER_EXPERIENCE_PLAN.md`.

## SDK and application integration

This is the most active and most externally relevant documentation area.

| Doc | Use it when... |
|---|---|
| [`sdk/SDK_QUICKSTART.md`](sdk/SDK_QUICKSTART.md) | you want the shortest path to a working client |
| [`sdk/RAG_RESEARCH_PLAYBOOK.md`](sdk/RAG_RESEARCH_PLAYBOOK.md) | you want retrieval and ingestion workflows, not just auth and health |
| [`sdk/BULK_INGEST_REFERENCE.md`](sdk/BULK_INGEST_REFERENCE.md) | you need dry-run planning, checkpointing, resume, dedupe, and large ingest ergonomics |
| [`sdk/API_REFERENCE.md`](sdk/API_REFERENCE.md) | you need method-level reference material |
| [`sdk/LIVE_STAGING_INTEGRATION.md`](sdk/LIVE_STAGING_INTEGRATION.md) | you are validating against a staging deployment |

### Suggested reading order for new SDK users

1. [`sdk/SDK_QUICKSTART.md`](sdk/SDK_QUICKSTART.md)
2. [`sdk/API_REFERENCE.md`](sdk/API_REFERENCE.md)
3. [`sdk/RAG_RESEARCH_PLAYBOOK.md`](sdk/RAG_RESEARCH_PLAYBOOK.md)
4. [`sdk/BULK_INGEST_REFERENCE.md`](sdk/BULK_INGEST_REFERENCE.md)

## Architecture, topology, and migration

These docs matter if you are modifying backend structure, route layout, auth abstractions, or repository naming/layout assumptions.

| Doc | Focus |
|---|---|
| [`unification/api_strategy.md`](unification/api_strategy.md) | API shape and platform direction |
| [`unification/auth_provider_interface.md`](unification/auth_provider_interface.md) | auth provider abstraction boundaries |
| [`unification/route_prefixes.md`](unification/route_prefixes.md) | route organization and naming |
| [`unification/observability_v1.md`](unification/observability_v1.md) | observability guidance |
| [`unification/program_control.md`](unification/program_control.md) | compatibility and rollout control |
| [`unification/compat_matrix.md`](unification/compat_matrix.md) | supported combinations / compatibility notes |
| [`unification/repo_migration.md`](unification/repo_migration.md) | repo/path migration history and policy |
| [`unification/repo_pinning_playbook.md`](unification/repo_pinning_playbook.md) | downstream compatibility pinning |
| [`unification/symlink_retirement_runbook.md`](unification/symlink_retirement_runbook.md) | staged retirement of legacy local alias |
| [`unification/unification_done_bar.md`](unification/unification_done_bar.md) | status tracking |

### Read these if you are touching naming or compatibility

- do not change canonical pathing or repo naming assumptions blindly,
- read the migration/runbook docs first,
- and treat compatibility as a product contract, not an afterthought.

## Specialized runtime docs

| Area | Doc | Notes |
|---|---|---|
| Chandra OCR/runtime | [`chandra/CHANDRA_OCR_VLLM_SERVER.md`](chandra/CHANDRA_OCR_VLLM_SERVER.md) | specialized OCR and model-serving notes |
| Dependency upgrades | [`deps_upgrade/UPGRADE_GATES.md`](deps_upgrade/UPGRADE_GATES.md) | what has to be true before dependency upgrades land |
| Upgrade matrix | [`deps_upgrade/UPGRADE_MATRIX_CHOOSER.md`](deps_upgrade/UPGRADE_MATRIX_CHOOSER.md) | selecting safe upgrade paths |

## How to use this docs tree

A few practical rules make this easier:

- use the root [`README.md`](../README.md) as the project front page,
- use this file as the stable navigation layer,
- use `docs_tmp/` for working notes, experiments, reports, and temporary artifacts,
- and promote material into `docs/` only when it is durable enough to be part of the maintained surface.

> `docs/` is for maintained documentation. `docs_tmp/` is for active work, scans, reports, design notes, and transient planning artifacts.

### Repo-adjacent surfaces worth knowing about

| Surface | Location | Why it matters |
|---|---|---|
| Root repo front page | [`../README.md`](../README.md) | high-level overview, quickstart, repo map |
| SDK package page | [`../sdk/python/README.md`](../sdk/python/README.md) | package-specific install/usage docs |
| Runnable SDK examples | [`../examples/sdk/`](../examples/sdk/) | practical examples and offline demos |
| Contributor guide | [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | repo expectations and contribution workflow |

If you are not sure where to start, use this sequence:

1. [`../README.md`](../README.md)
2. [`setup/DEVELOPER_SETUP.md`](setup/DEVELOPER_SETUP.md)
3. [`sdk/SDK_QUICKSTART.md`](sdk/SDK_QUICKSTART.md)
