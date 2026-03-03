# CI Runtime Profile

- Generated UTC: `2026-03-03T20:19:11.816128+00:00`
- Repository: `kmccleary3301/QueryLakeBackend`
- Lookback days: `7`
- Sampled runs: `174`

## Overall

- Success rate: `0.8678`
- Error rate: `0.1322`
- Rerun rate: `0.0`
- Duration median/p95/mean (s): `14.0` / `28.0` / `16.3`
- Queue median/p95/mean (s): `0.0` / `0.0` / `0.0`
- Compute minutes total: `47.27`

## By Workflow

| Workflow | Runs | Success | Error | Rerun | Dur p50 (s) | Dur p95 (s) | Queue p50 (s) | Queue p95 (s) | Compute min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Docs Checks | 30 | 1.0 | 0.0 | 0.0 | 10.5 | 13.55 | 0.0 | 0.0 | 5.38 |
| Retrieval Eval | 51 | 0.8235 | 0.1765 | 0.0 | 18.0 | 22.0 | 0.0 | 0.0 | 14.77 |
| SDK Checks | 42 | 0.7619 | 0.2381 | 0.0 | 24.0 | 28.95 | 0.0 | 0.0 | 16.03 |
| SDK Publish Dry-Run (TestPyPI) | 4 | 0.0 | 1.0 | 0.0 | 35.0 | 47.75 | 0.0 | 0.0 | 2.53 |
| Unification Checks | 47 | 1.0 | 0.0 | 0.0 | 11.0 | 14.7 | 0.0 | 0.0 | 8.55 |
