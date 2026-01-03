# Repo Pinning Playbook

## Goal
Keep QueryLake, BreadBoard, and Hermes aligned without forcing a monorepo.

## Rules
- Always pin by commit hash.
- Update pins in a single source of truth: `docs/unification/compat_matrix.md`.
- When updating a pin, run contract tests and record results.

## Procedure
1. Update `compat_matrix.md` with new commit hash.
2. Run `scripts/verify_compat_pins.py` to ensure pins are set.
3. Run contract checks (A–F) against the pinned versions.
4. Record outcome in the change log.

