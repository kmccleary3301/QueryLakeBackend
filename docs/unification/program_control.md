# Unification Program Control

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Owner, cadence, escalation, and release-gating control surface for the broader unification program.

| Field | Value |
|---|---|
| Audience | Project leads, release maintainers, downstream integration owners |
| Use this when | Use this when deciding who owns a unification decision, what gates must pass, or how issues should escalate. |
| Prerequisites | Awareness of the unification workstream and downstream repo dependencies. |
| Related docs | [`compat_matrix.md`](compat_matrix.md), [`repo_pinning_playbook.md`](repo_pinning_playbook.md), [`unification_done_bar.md`](unification_done_bar.md) |
| Status | 🟢 active control document |

## Owner + Cadence
- Owner: QueryLake lead (designated by project lead)
- Cadence: weekly sync + monthly milestone review

## Escalation Path
- Auth provider strategy → QueryLake lead + security owner
- Umbrella scaling policy → infra owner + QueryLake lead
- Repo strategy / pin updates → release manager

## Release Gating Criteria (v1)
- All contract docs A–F published and implemented
- Umbrella scaling tested in staging (multi‑replica)
- Repo pins set and CI guardrail running
- API strategy stabilized with v2 shims
- Observability v1 dashboards live
