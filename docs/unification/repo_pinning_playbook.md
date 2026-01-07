# Repo Pinning Playbook

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
