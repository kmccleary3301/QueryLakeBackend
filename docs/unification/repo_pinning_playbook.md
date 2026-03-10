# Repo Pinning Playbook

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Operational playbook for keeping QueryLake and downstream repos aligned by commit-pinned compatibility checks instead of a forced monorepo.

| Field | Value |
|---|---|
| Audience | Release maintainers, CI owners, downstream integration maintainers |
| Use this when | Use this when updating downstream pins or validating that a QueryLake change remains compatible with Breadboard and Hermes. |
| Prerequisites | Access to downstream repos and the compatibility matrix. |
| Related docs | [`compat_matrix.md`](compat_matrix.md), [`program_control.md`](program_control.md), [`symlink_retirement_runbook.md`](symlink_retirement_runbook.md) |
| Status | 🟢 active playbook |

## Goal
Keep QueryLake, BreadBoard, and Hermes aligned without forcing a monorepo.

## Rules
- Always pin by commit hash.
- Optional: add a tag mirror for human readability (tag points to commit).
- Update pins in a single source of truth: `docs/unification/compat_matrix.md`.
- When updating a pin, run contract tests and record results.

## Procedure
1. Update `compat_matrix.md` with new commit hash.
2. Run `scripts/verify_compat_pins.py` to ensure pins are set.
3. Run contract checks (A–F) against the pinned versions.
4. Record outcome in the change log.

## CI Guardrail (lightweight)
- Add a CI step that runs `scripts/verify_compat_pins.py`
- Fail CI if any pin is `<set>` or empty
- Optional helper: `scripts/ci_unification_checks.sh`
- GitHub Actions workflow: `.github/workflows/unification_checks.yml`

## Related Migration Control
- Legacy local-path alias retirement:
  - `docs/unification/symlink_retirement_runbook.md`
