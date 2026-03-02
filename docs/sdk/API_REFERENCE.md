# QueryLake SDK API Reference (Python)

## Client classes

| Class | Use case |
|---|---|
| `QueryLakeClient` | Synchronous scripts, notebooks, backend services |
| `AsyncQueryLakeClient` | Async services and task runners |

## Authentication model

SDK auth payloads map directly to QueryLake backend auth contracts:

- OAuth2 token: `{"oauth2": "<token>"}`
- API key: `{"api_key": "sk-..."}` (if enabled by backend policy)

Typical bootstrap:

1. `client.login(username=..., password=...)`
2. SDK stores returned OAuth2 token in-memory for subsequent calls

## QueryLakeClient methods

### Core connectivity

- `healthz() -> dict`
- `readyz() -> dict`
- `ping() -> Any`
- `list_models() -> dict`
- `chat_completions(payload: dict) -> dict`
- `embeddings(payload: dict) -> dict`

### Auth/session

- `login(username, password) -> dict`
- `add_user(username, password) -> dict`
- `set_auth(auth=..., oauth2=..., api_key=...)`

### Collections + documents

- `create_collection(name, description=None, public=False, organization_id=None) -> dict`
- `list_collections(organization_id=None, global_collections=False) -> dict`
- `fetch_collection(collection_hash_id) -> dict`
- `modify_collection(collection_hash_id, title=None, description=None) -> Any`
- `list_collection_documents(collection_hash_id, limit=100, offset=0) -> list[dict]`
- `delete_document(document_hash_id) -> Any`
- `upload_document(file_path, collection_hash_id, ...) -> dict`
- `upload_directory(collection_hash_id, directory=..., file_paths=..., checkpoint_file=..., resume=False, checkpoint_save_every=1, strict_checkpoint_match=True, ...) -> dict`

### Retrieval

- `search_hybrid(query, collection_ids, ..., **kwargs) -> list[dict]`
- `search_hybrid_with_metrics(query, collection_ids, ..., **kwargs) -> dict`
- `search_hybrid_chunks(...) -> list[SearchResultChunk]`
- `count_chunks(collection_ids=None) -> dict`
- `get_random_chunks(limit=5, collection_ids=None) -> list[dict]`

`search_hybrid_with_metrics` returns the orchestrated backend payload shape (`rows`, `duration`, optional plan/queue metadata) when available.

### Escape hatch for all backend functions

- `api(function_name, payload=None, method=\"POST\", auth=None) -> Any`

This keeps the SDK forward-compatible with new backend API functions without waiting for SDK helper wrappers.

## CLI commands

- `querylake doctor`
- `querylake login`
- `querylake models`
- `querylake profile list`
- `querylake profile show`
- `querylake profile set-default`
- `querylake profile set-url`
- `querylake profile delete`
- `querylake rag create-collection`
- `querylake rag list-collections`
- `querylake rag get-collection`
- `querylake rag update-collection`
- `querylake rag list-documents`
- `querylake rag count-chunks`
- `querylake rag random-chunks`
- `querylake rag delete-document`
- `querylake rag upload`
- `querylake rag upload-dir`
  - source selection: `--dir` (required unless `--from-selection` is provided)
  - bulk selection controls: `--pattern`, `--recursive`, `--max-files`
  - replay exact file sets: `--from-selection` (JSON artifact with `selected_files`)
  - filter controls: `--extensions`, `--exclude-glob` (repeatable)
  - planning mode: `--dry-run` + `--list-files` for selection preview without upload
  - artifact output: `--selection-output` (selected file list), `--report-file` (final payload JSON)
  - resumable mode: `--checkpoint-file`, `--resume`, `--checkpoint-save-every`, `--no-checkpoint-strict`
  - ingest controls: `--await-embedding`, `--no-scan`, `--no-embeddings`, `--sparse-embeddings`, `--sparse-dimensions`, `--fail-fast`
- `querylake rag search`
  - add `--with-metrics` in hybrid mode to include duration/profile fields
  - add `--preset` for common retrieval profiles (`balanced`, `tri-lane`, `lexical-heavy`, `semantic-heavy`, `sparse-heavy`)
  - optional gates: `--min-total-results`, `--fail-on-empty` (exit code `2` on gate failure)
- `querylake rag search-batch`
  - run newline-delimited query files with same mode/preset/metrics options as `rag search`
  - supports `--output-file` to persist full batch JSON report
  - optional gates: `--min-total-results`, `--fail-on-empty` (exit code `2` on gate failure)

## Error model

| Exception | Meaning |
|---|---|
| `QueryLakeTransportError` | Network/transport-level failure |
| `QueryLakeHTTPStatusError` | Non-2xx HTTP status from backend |
| `QueryLakeAPIError` | `/api/*` response returned `success=false` |

## Design notes

- SDK is intentionally thin around query shaping and retrieval strategy so research teams can pass lane/fusion options directly.
- Helper methods standardize common workflows; low-level `api()` keeps full backend reach.
- `upload_document` passes parameters via query string JSON to mirror backend multipart contract exactly.

## SDK quality gate (contributors)

From repo root:

```bash
make sdk-precommit-install
make sdk-precommit-run
make sdk-lint
make sdk-type
make sdk-ci
```

Single-source quality script (used by Make, hooks, and CI):

```bash
bash scripts/dev/sdk_quality_gate.sh all
```
