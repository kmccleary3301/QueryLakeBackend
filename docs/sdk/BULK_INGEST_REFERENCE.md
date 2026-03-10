# QueryLake Bulk Ingest Reference

[![Docs Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/docs_checks.yml)
[![SDK Checks](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml/badge.svg)](https://github.com/kmccleary3301/QueryLake/actions/workflows/sdk_checks.yml)

Reference for repeatable large-scale ingestion through the CLI and Python SDK.

| Field | Value |
|---|---|
| Audience | RAG researchers, operators, and engineers ingesting non-trivial corpora |
| Use this when | Use this when you need selection artifacts, dry-run planning, checkpoints, resumable uploads, and dedupe/idempotency controls. |
| Prerequisites | A working QueryLake environment and familiarity with the basic `rag upload` / collection workflow. |
| Related docs | [`SDK_QUICKSTART.md`](SDK_QUICKSTART.md), [`RAG_RESEARCH_PLAYBOOK.md`](RAG_RESEARCH_PLAYBOOK.md) |
| Status | 🟢 maintained operational reference |

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
| `--checkpoint-file` | Persist resumable progress checkpoint JSON |
| `--resume` | Resume from checkpoint uploaded set |
| `--checkpoint-save-every` | Save checkpoint every N processed files |
| `--no-checkpoint-strict` | Allow resume despite selection hash mismatch |
| `--ingest-profile` | Apply named baseline ingest profile (`dense-fast`, `dense-blocking`, `tri-lane-fast`, `tri-lane-blocking`) |
| `--ingest-profile-file` | Load JSON ingest profile and merge over baseline |
| `--dedupe-content-hash` | Skip duplicate files by SHA-256 content hash |
| `--no-dedupe-content-hash` | Force disable dedupe when enabled by profile |
| `--dedupe-scope` | Dedupe scope: `run-local`, `checkpoint-resume`, or `all` |
| `--idempotency-strategy` | Inject idempotency key in metadata (`none`, `content-hash`, `path-hash`) |
| `--idempotency-prefix` | Prefix for generated idempotency keys |
| `--no-sparse-embeddings` | Force disable sparse lane when enabled by profile |

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
  --report-file ./artifacts/upload_run.json \
  --checkpoint-file ./artifacts/upload_checkpoint.json

# 3) Resume interrupted run from checkpoint
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --from-selection ./artifacts/selected_files.json \
  --ingest-profile tri-lane-fast \
  --resume \
  --checkpoint-file ./artifacts/upload_checkpoint.json \
  --checkpoint-save-every 10 \
  --dedupe-content-hash \
  --dedupe-scope all \
  --idempotency-strategy content-hash \
  --idempotency-prefix qlsdk \
  --report-file ./artifacts/upload_resume.json
```

### Custom ingest profile file (JSON)

```json
{
  "scan_text": true,
  "create_embeddings": true,
  "create_sparse_embeddings": true,
  "await_embedding": false,
  "sparse_embedding_dimensions": 1024,
  "dedupe_by_content_hash": true,
  "dedupe_scope": "all",
  "idempotency_strategy": "content-hash",
  "idempotency_prefix": "qlsdk",
  "fail_fast": false,
  "checkpoint_save_every": 10
}
```

Use with:

```bash
querylake --profile local rag upload-dir \
  --collection-id <collection_id> \
  --dir ./dataset \
  --pattern "*.txt" \
  --recursive \
  --ingest-profile-file ./profiles/tri_lane_fast.json
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
    checkpoint_file="./artifacts/upload_checkpoint.json",
    checkpoint_save_every=10,
    dedupe_by_content_hash=True,
    dedupe_scope="all",
    idempotency_strategy="content-hash",
    idempotency_prefix="qlsdk",
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
| `selection_sha256` | Deterministic hash of selected file set |
| `ingest_profile` | Optional named ingest profile used for the run |
| `ingest_profile_file` | Optional ingest profile JSON path used for the run |
| `ingest_controls` | Effective merged ingest controls applied by CLI/SDK |
| `resumed_from_checkpoint` | Whether run resumed from checkpoint |
| `skipped_already_uploaded` | Count skipped because checkpoint marked uploaded |
| `dedupe_by_content_hash` | Whether content-hash dedupe was enabled |
| `dedupe_scope` | Dedupe scope (`run-local`, `checkpoint-resume`, `all`, or `none`) |
| `dedupe_skipped` | Count skipped by dedupe filter |
| `pending_files` | Number of files queued for this run after resume filtering |
| `uploaded` | Upload success count |
| `failed` | Upload failure count |
| `errors` | Optional list of file-level error objects |
| `dry_run` | Whether run was planning-only |
| `fail_fast` | Whether run stops on first error |
| `_meta` | Artifact provenance metadata (timestamp, sdk version, profile, base URL, cwd) |

## Recommended defaults

- Start with `dry-run` + artifact output on new datasets.
- Replay with `--from-selection` for deterministic comparison runs.
- Keep `fail_fast=false` for broad diagnostics, `true` for CI gating.
- Enable sparse embeddings only when your retrieval profile actually consumes sparse lanes.