# QueryLake Bulk Ingest Reference

This reference covers the two standardized bulk-ingest paths:

- CLI: `querylake rag upload-dir`
- Python SDK: `QueryLakeClient.upload_directory(...)`

Use this when you need repeatable ingestion runs with explicit file-selection artifacts.

## CLI: `querylake rag upload-dir`

### Core usage

```bash
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --dir ./dataset \
  --pattern "*.txt" \
  --recursive
```

### Selection controls

| Flag | Purpose |
|---|---|
| `--dir` | Source directory (required unless `--from-selection`) |
| `--from-selection` | Replay from prior selection artifact JSON (`selected_files`) |
| `--pattern` | Glob pattern for file selection |
| `--recursive` | Recurse into subdirectories |
| `--max-files` | Cap selected file count after sorting |
| `--extensions` | Comma-separated extension filter (e.g. `.pdf,.md`) |
| `--exclude-glob` | Repeatable exclude pattern relative to `--dir` |
| `--list-files` | Include selected file paths in CLI output |

### Execution controls

| Flag | Purpose |
|---|---|
| `--dry-run` | Selection/planning only; no uploads |
| `--fail-fast` | Stop on first upload error |
| `--await-embedding` | Wait for embedding completion during upload |
| `--no-scan` | Disable text scan on ingest |
| `--no-embeddings` | Disable dense embeddings |
| `--sparse-embeddings` | Enable sparse embeddings |
| `--sparse-dimensions` | Sparse vector dimensions |

### Artifact controls

| Flag | Purpose |
|---|---|
| `--selection-output` | Write selected-file artifact JSON |
| `--report-file` | Write final run payload JSON |

### Planning and replay workflow

```bash
# 1) Plan and persist selection
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --dir ./dataset \
  --pattern "*" \
  --recursive \
  --extensions ".txt,.md" \
  --exclude-glob "archive/*" \
  --dry-run \
  --selection-output ./artifacts/selected_files.json \
  --report-file ./artifacts/upload_dry_run.json

# 2) Replay exact file set
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --from-selection ./artifacts/selected_files.json \
  --report-file ./artifacts/upload_run.json
```

## Python SDK: `QueryLakeClient.upload_directory(...)`

### Directory scan mode

```python
from querylake_sdk import QueryLakeClient

client = QueryLakeClient(base_url="http://127.0.0.1:8000", oauth2="<token>")
report = client.upload_directory(
    collection_hash_id="col_123",
    directory="./dataset",
    pattern="*",
    recursive=True,
    include_extensions=[".txt", ".md"],
    exclude_globs=["archive/*", "*.tmp"],
    fail_fast=True,
    await_embedding=False,
    create_sparse_embeddings=True,
)
```

### Explicit file-list mode (replay)

```python
import json
from pathlib import Path
from querylake_sdk import QueryLakeClient

selection = json.loads(Path("./artifacts/selected_files.json").read_text(encoding="utf-8"))
selected_files = selection["selected_files"]

client = QueryLakeClient(base_url="http://127.0.0.1:8000", oauth2="<token>")
report = client.upload_directory(
    collection_hash_id="col_123",
    file_paths=selected_files,
    fail_fast=False,
)
```

### Returned payload fields

| Field | Meaning |
|---|---|
| `directory` | Directory context (`<explicit-file-list>` in list mode) |
| `selection_mode` | `directory-scan` or `explicit-file-list` |
| `requested_files` | Number of selected files |
| `selected_files` | Selected file path list |
| `uploaded` | Upload success count |
| `failed` | Upload failure count |
| `errors` | Optional list of file-level error objects |
| `dry_run` | Whether run was planning-only |
| `fail_fast` | Whether run stops on first error |

## Recommended defaults

- Start with `dry-run` + artifact output on new datasets.
- Replay with `--from-selection` for deterministic comparison runs.
- Keep `fail_fast=false` for broad diagnostics, `true` for CI gating.
- Enable sparse embeddings only when your retrieval profile actually consumes sparse lanes.
