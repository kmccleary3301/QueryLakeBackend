#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

ACTION="${1:-start}"

CHANDRA_VLLM_VENV="${CHANDRA_VLLM_VENV:-$REPO_ROOT/.venv_vllm_0_13}"
CHANDRA_VLLM_MODEL="${CHANDRA_VLLM_MODEL:-models/chandra}"
CHANDRA_VLLM_SERVED_MODEL_NAME="${CHANDRA_VLLM_SERVED_MODEL_NAME:-chandra}"
CHANDRA_VLLM_HOST="${CHANDRA_VLLM_HOST:-127.0.0.1}"
CHANDRA_VLLM_PORT="${CHANDRA_VLLM_PORT:-8022}"
CHANDRA_VLLM_API_KEY="${CHANDRA_VLLM_API_KEY:-chandra-local-key}"
CHANDRA_VLLM_TENSOR_PARALLEL_SIZE="${CHANDRA_VLLM_TENSOR_PARALLEL_SIZE:-1}"
CHANDRA_VLLM_DATA_PARALLEL_SIZE="${CHANDRA_VLLM_DATA_PARALLEL_SIZE:-1}"
CHANDRA_VLLM_GPU_MEMORY_UTILIZATION="${CHANDRA_VLLM_GPU_MEMORY_UTILIZATION:-0.95}"
CHANDRA_VLLM_MAX_MODEL_LEN="${CHANDRA_VLLM_MAX_MODEL_LEN:-131072}"
CHANDRA_VLLM_MAX_NUM_SEQS="${CHANDRA_VLLM_MAX_NUM_SEQS:-8}"
CHANDRA_VLLM_DTYPE="${CHANDRA_VLLM_DTYPE:-auto}"
CHANDRA_VLLM_RUNTIME_DIR="${CHANDRA_VLLM_RUNTIME_DIR:-/tmp/querylake_chandra_vllm_server}"
CHANDRA_VLLM_STARTUP_TIMEOUT_SECONDS="${CHANDRA_VLLM_STARTUP_TIMEOUT_SECONDS:-180}"
CHANDRA_VLLM_WARMUP="${CHANDRA_VLLM_WARMUP:-1}"
CHANDRA_VLLM_CUDA_VISIBLE_DEVICES="${CHANDRA_VLLM_CUDA_VISIBLE_DEVICES:-}"

PID_FILE="$CHANDRA_VLLM_RUNTIME_DIR/chandra_vllm_server.pid"
LOG_FILE="$CHANDRA_VLLM_RUNTIME_DIR/chandra_vllm_server.log"
HEALTH_URL="http://${CHANDRA_VLLM_HOST}:${CHANDRA_VLLM_PORT}/health"
BASE_URL="http://${CHANDRA_VLLM_HOST}:${CHANDRA_VLLM_PORT}/v1"

mkdir -p "$CHANDRA_VLLM_RUNTIME_DIR"

is_running() {
  if [[ ! -f "$PID_FILE" ]]; then
    return 1
  fi
  local pid
  pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -z "$pid" ]]; then
    return 1
  fi
  if ps -p "$pid" >/dev/null 2>&1; then
    return 0
  fi
  return 1
}

wait_for_health() {
  local deadline
  deadline=$((SECONDS + CHANDRA_VLLM_STARTUP_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
  done
  return 1
}

warmup_request() {
  if [[ "$CHANDRA_VLLM_WARMUP" != "1" ]]; then
    return 0
  fi
  local tiny_png_data_url
  tiny_png_data_url="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO8Ok9kAAAAASUVORK5CYII="
  curl -fsS "${BASE_URL}/chat/completions" \
    -H "Content-Type: application/json" \
    -H "Authorization: Bearer ${CHANDRA_VLLM_API_KEY}" \
    -d @- >/dev/null <<JSON
{
  "model": "${CHANDRA_VLLM_SERVED_MODEL_NAME}",
  "messages": [
    {
      "role": "user",
      "content": [
        {"type": "text", "text": "OCR warmup"},
        {"type": "image_url", "image_url": {"url": "${tiny_png_data_url}"}}
      ]
    }
  ],
  "max_tokens": 16,
  "temperature": 0.0,
  "top_p": 1.0,
  "stream": false
}
JSON
}

start_server() {
  if is_running; then
    echo "[chandra_vllm_server] already running (pid=$(cat "$PID_FILE"))."
    return 0
  fi

  if [[ ! -d "$CHANDRA_VLLM_VENV" ]]; then
    echo "[chandra_vllm_server] missing venv: $CHANDRA_VLLM_VENV"
    exit 1
  fi

  # shellcheck disable=SC1090
  source "$CHANDRA_VLLM_VENV/bin/activate"

  local -a cmd
  cmd=(
    python -m vllm.entrypoints.openai.api_server
    --host "$CHANDRA_VLLM_HOST"
    --port "$CHANDRA_VLLM_PORT"
    --model "$CHANDRA_VLLM_MODEL"
    --served-model-name "$CHANDRA_VLLM_SERVED_MODEL_NAME"
    --api-key "$CHANDRA_VLLM_API_KEY"
    --tensor-parallel-size "$CHANDRA_VLLM_TENSOR_PARALLEL_SIZE"
    --data-parallel-size "$CHANDRA_VLLM_DATA_PARALLEL_SIZE"
    --gpu-memory-utilization "$CHANDRA_VLLM_GPU_MEMORY_UTILIZATION"
    --max-model-len "$CHANDRA_VLLM_MAX_MODEL_LEN"
    --max-num-seqs "$CHANDRA_VLLM_MAX_NUM_SEQS"
    --dtype "$CHANDRA_VLLM_DTYPE"
    --disable-log-stats
    --enable-chunked-prefill
  )

  if [[ -n "$CHANDRA_VLLM_CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$CHANDRA_VLLM_CUDA_VISIBLE_DEVICES"
  fi

  (
    cd "$REPO_ROOT"
    nohup "${cmd[@]}" >>"$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
  )

  if ! wait_for_health; then
    echo "[chandra_vllm_server] health check failed. Check log: $LOG_FILE"
    exit 1
  fi

  warmup_request || true
  echo "[chandra_vllm_server] started. pid=$(cat "$PID_FILE") health=${HEALTH_URL}"
}

stop_server() {
  if ! is_running; then
    echo "[chandra_vllm_server] not running."
    rm -f "$PID_FILE"
    return 0
  fi
  local pid
  pid="$(cat "$PID_FILE")"
  kill "$pid" >/dev/null 2>&1 || true
  for _ in $(seq 1 20); do
    if ! ps -p "$pid" >/dev/null 2>&1; then
      rm -f "$PID_FILE"
      echo "[chandra_vllm_server] stopped."
      return 0
    fi
    sleep 1
  done
  kill -9 "$pid" >/dev/null 2>&1 || true
  rm -f "$PID_FILE"
  echo "[chandra_vllm_server] killed."
}

status_server() {
  if is_running; then
    echo "[chandra_vllm_server] running (pid=$(cat "$PID_FILE"))."
    if curl -fsS "$HEALTH_URL" >/dev/null 2>&1; then
      echo "[chandra_vllm_server] health=ok (${HEALTH_URL})"
    else
      echo "[chandra_vllm_server] health=failed (${HEALTH_URL})"
    fi
  else
    echo "[chandra_vllm_server] stopped."
    return 1
  fi
}

case "$ACTION" in
  start)
    start_server
    ;;
  stop)
    stop_server
    ;;
  restart)
    stop_server
    start_server
    ;;
  status)
    status_server
    ;;
  health)
    curl -fsS "$HEALTH_URL"
    ;;
  logs)
    tail -n 200 "$LOG_FILE"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|health|logs}"
    exit 1
    ;;
esac
