# Unification Done Bar

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Milestone bar and completion criteria for the v1 unification release.

| Field | Value |
|---|---|
| Audience | Project leads, release maintainers, downstream integration owners |
| Use this when | Use this when checking what counts as 75 percent complete vs the actual v1 done bar. |
| Prerequisites | General awareness of the unification program workstreams. |
| Related docs | [`program_control.md`](program_control.md), [`compat_matrix.md`](compat_matrix.md), [`repo_pinning_playbook.md`](repo_pinning_playbook.md) |
| Status | 🟢 active status tracker |

## 75% Milestone (foundation complete)
- Contract docs A–F updated with schemas + semantics
- API strategy + v2 compatibility shims present
- Auth provider abstraction + OAuth placeholder in place
- Hermes Redis queue model + retry/requeue + tests
- Observability v1 baseline + alerts documented
- Billing/usage accounting stub + provider fields
- Node/cloud early plan documented
- Repo pinning playbook + pin format documented

## 100% Done Bar (v1 unification release)
- All contract surfaces implemented and validated against BreadBoard + Hermes
- Umbrella scaling (multi-replica) implemented + tested in staging
- Repo pins finalized with real commits + CI guardrails enforced
- API strategy enforced with published SDKs and deprecation plan
- Auth provider abstraction wired to external provider(s)
- Hermes reliability validated under failure scenarios
- Observability dashboards + alerts operational
- Billing pipeline ready for optional enablement

## Implementation Notes (current)
- Model catalog endpoints (`/v1/models`, `/v1/models/{id}`) implemented in QueryLake.
- Umbrella autoscaling knobs exposed via env vars and unit-tested.
- CI guardrail workflow added for compat pins.
- Legacy path alias retirement runbook documented:
  - `docs/unification/symlink_retirement_runbook.md`
