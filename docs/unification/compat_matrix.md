# Unification Compatibility Matrix

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Single-source compatibility pin record for QueryLake and key downstream repos during the unification program.

| Field | Value |
|---|---|
| Audience | Release maintainers, integration owners, downstream repo maintainers |
| Use this when | Use this when you need the current pinned commits or when updating validated downstream compatibility references. |
| Prerequisites | Access to the relevant repos and the unification release/check process. |
| Related docs | [`repo_pinning_playbook.md`](repo_pinning_playbook.md), [`program_control.md`](program_control.md), [`unification_done_bar.md`](unification_done_bar.md) |
| Status | 🟢 active source of truth |

| Component | Repo | Pin Type | Current Pin | Notes |
|---|---|---|---|---|
| QueryLake | QueryLake | git commit | 2d17442aa8cae9c2df0a55b37690b7608ec396f6 | canonical monorepo |
| BreadBoard | ray_testing/ray_SCE | git commit | 10a097254224fa520434c934d8c0e63b6f4c8d60 | local dev snapshot |
| Hermes | ray_testing/hermes | git commit | 70861b36ccbe3ac310427793bff59bba404558bf | local dev snapshot |

## Guardrails
- CI should verify pins exist and are fetchable
- CI should run contract surface tests against pinned versions

## Pin Format
- Primary: commit hash
- Optional: tag mirror (e.g., `unification-v1`) that points at the commit

## How to set pins
- Replace `<set>` with a commit hash for each repo.
- Store the pins in `docs/unification/compat_matrix.md` only (single source of truth).
