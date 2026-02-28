#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is required"
  exit 1
fi

echo "[sdk-smoke] base_url=${BASE_URL}"
echo "[sdk-smoke] checking backend health..."
curl -fsS "${BASE_URL}/healthz" >/dev/null
curl -fsS "${BASE_URL}/readyz" >/dev/null

TMP_HOME="$(mktemp -d)"
USERNAME="sdk_smoke_$(date +%s)"
PASSWORD="SmokePass123!"
DOC_FILE="/tmp/querylake_sdk_smoke_${USERNAME}.txt"

cat > "${DOC_FILE}" <<'EOF'
QueryLake SDK smoke document.
This line should be retrievable using BM25 and hybrid search.
EOF

echo "[sdk-smoke] creating temporary user ${USERNAME}"
USERNAME="${USERNAME}" PASSWORD="${PASSWORD}" BASE_URL="${BASE_URL}" \
uv run --project sdk/python python - <<'PY'
import os
from querylake_sdk import QueryLakeClient
client = QueryLakeClient(base_url=os.environ["BASE_URL"])
try:
    out = client.add_user(username=os.environ["USERNAME"], password=os.environ["PASSWORD"])
    assert isinstance(out, dict), "add_user did not return a dictionary"
finally:
    client.close()
PY

echo "[sdk-smoke] logging in via CLI profile"
HOME="${TMP_HOME}" uv run --project sdk/python querylake \
  login \
  --url "${BASE_URL}" \
  --profile local \
  --username "${USERNAME}" \
  --password "${PASSWORD}" >/tmp/querylake_sdk_smoke_login.json

echo "[sdk-smoke] creating collection"
HOME="${TMP_HOME}" uv run --project sdk/python querylake \
  --profile local \
  --url "${BASE_URL}" \
  rag create-collection \
  --name "SDK Smoke Collection" >/tmp/querylake_sdk_smoke_collection.json

COLLECTION_ID="$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('/tmp/querylake_sdk_smoke_collection.json').read_text())
print(payload['hash_id'])
PY
)"

echo "[sdk-smoke] collection_id=${COLLECTION_ID}"
echo "[sdk-smoke] uploading test document"
HOME="${TMP_HOME}" uv run --project sdk/python querylake \
  --profile local \
  --url "${BASE_URL}" \
  rag upload \
  --collection-id "${COLLECTION_ID}" \
  --file "${DOC_FILE}" \
  --await-embedding \
  --no-embeddings >/tmp/querylake_sdk_smoke_upload.json

echo "[sdk-smoke] verifying lexical retrieval path"
HOME="${TMP_HOME}" uv run --project sdk/python querylake \
  --profile local \
  --url "${BASE_URL}" \
  rag search \
  --mode bm25 \
  --collection-id "${COLLECTION_ID}" \
  --query "BM25 and hybrid search" \
  --top-k 3 >/tmp/querylake_sdk_smoke_search_bm25.json

BM25_TOTAL="$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('/tmp/querylake_sdk_smoke_search_bm25.json').read_text())
print(int(payload.get('total', 0)))
PY
)"

if [[ "${BM25_TOTAL}" -le 0 ]]; then
  echo "[sdk-smoke] BM25 search returned zero results"
  exit 2
fi

echo "[sdk-smoke] verifying hybrid retrieval path"
HOME="${TMP_HOME}" uv run --project sdk/python querylake \
  --profile local \
  --url "${BASE_URL}" \
  rag search \
  --mode hybrid \
  --collection-id "${COLLECTION_ID}" \
  --query "BM25 and hybrid search" \
  --top-k 3 \
  --limit-bm25 10 \
  --limit-similarity 0 \
  --limit-sparse 0 \
  --bm25-weight 1.0 \
  --similarity-weight 0.0 \
  --sparse-weight 0.0 >/tmp/querylake_sdk_smoke_search_hybrid.json

HYBRID_TOTAL="$(python - <<'PY'
import json
from pathlib import Path
payload = json.loads(Path('/tmp/querylake_sdk_smoke_search_hybrid.json').read_text())
print(int(payload.get('total', 0)))
PY
)"

if [[ "${HYBRID_TOTAL}" -le 0 ]]; then
  echo "[sdk-smoke] hybrid search returned zero results"
  exit 3
fi

echo "[sdk-smoke] success"
echo "[sdk-smoke] tmp_home=${TMP_HOME}"
echo "[sdk-smoke] artifacts:"
echo "  /tmp/querylake_sdk_smoke_login.json"
echo "  /tmp/querylake_sdk_smoke_collection.json"
echo "  /tmp/querylake_sdk_smoke_upload.json"
echo "  /tmp/querylake_sdk_smoke_search_bm25.json"
echo "  /tmp/querylake_sdk_smoke_search_hybrid.json"
