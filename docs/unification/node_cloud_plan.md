# Node / Cloud Plan (Draft)

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![Unification Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/unification_checks.yml)

Early deployment-shape and cloud-readiness notes for moving QueryLake from single-node development to multi-node operation.

| Field | Value |
|---|---|
| Audience | Infra owners, backend maintainers, deployment reviewers |
| Use this when | Use this when planning node roles, resource boundaries, and migration assumptions for nontrivial deployments. |
| Prerequisites | Understanding of current local deployment, Redis/Postgres dependencies, and Ray/Serve usage. |
| Related docs | [`umbrella_scaling_policy.md`](umbrella_scaling_policy.md), [`observability_v1.md`](observability_v1.md), [`program_control.md`](program_control.md) |
| Status | 🔵 draft deployment plan |

## Deployment Shapes
- Single node (dev)
- Multi node (prod)
- Hybrid (dedicated GPU nodes + API nodes)

## Prereqs
- Redis
- Postgres
- Metrics/alerts

## Migration Notes
- Document how to move from single node to multi node

## Early Design Decisions
- Separate control plane (API + scheduler) from GPU inference nodes
- Pin Redis/Postgres to stable host for consistency
- Prefer static IPs for worker nodes in early stages

## Cloud Readiness Checklist
- Object storage for large artifacts
- Automated node join (Ray + SSH/bootstrap)
- Secret management (API keys + provider creds)
