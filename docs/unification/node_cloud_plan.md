# Node / Cloud Plan (Draft)

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
