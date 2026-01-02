# Unification Compatibility Matrix

| Component | Repo | Pin Type | Current Pin | Notes |
|----------|------|----------|-------------|-------|
| QueryLake | QueryLakeBackend | git commit | <set> | master integration point |
| BreadBoard | BreadBoard | git commit | <set> | active dev — keep adapters loose |
| Hermes | Hermes | git commit | <set> | integration via web tooling contract |

## Guardrails
- CI should verify pins exist and are fetchable
- CI should run contract surface tests against pinned versions

