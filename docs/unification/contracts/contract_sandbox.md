# Contract E — Sandbox / Code Execution

## Purpose
Define a safe execution contract for toolchains.

## Input
- code / command
- resource limits (cpu, memory, timeout)
- env vars (optional)
- input files (optional)

## Output
- stdout/stderr
- exit code
- artifact references

## Resource Limits / Policy Hooks
- `limits.cpu` (cores), `limits.memory_mb`, `limits.timeout_s` are enforced per run
- Optional `limits.disk_mb` and `limits.network` flags for future isolation
- Policy hook: per-tenant overrides based on `principal_id` or `api_key_id`

## Contract Shape (suggested)
```json
{
  "command": "python main.py",
  "env": {"KEY": "value"},
  "limits": {"cpu": 2, "memory_mb": 2048, "timeout_s": 30},
  "files": [{"path": "input.txt", "bytes_cas": "sha256:..."}]
}
```

```json
{
  "stdout": "...",
  "stderr": "...",
  "exit_code": 0,
  "artifacts": [{"path": "output.txt", "bytes_cas": "sha256:..."}]
}
```
