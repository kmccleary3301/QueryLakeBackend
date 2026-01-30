#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
COMPOSE_FILE="$REPO_ROOT/docker-compose-redis.yml"
DOTENV_FILE="$REPO_ROOT/.env"

if [ -f "$DOTENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$DOTENV_FILE"
  set +a
fi

# --- Config (overridable via env / .env) ---
REDIS_CONTAINER_NAME="${QUERYLAKE_REDIS_CONTAINER_NAME:-querylake_redis}"
REDIS_HOST_PORT="${QUERYLAKE_REDIS_HOST_PORT:-6393}"
REDIS_PASSWORD="${QUERYLAKE_REDIS_PASSWORD:-}"
REDIS_COMPOSE_PROJECT="${QUERYLAKE_REDIS_COMPOSE_PROJECT:-querylake_redis}"

if [ -z "$REDIS_PASSWORD" ]; then
  echo "[restart_querylake_redis] ERROR: QUERYLAKE_REDIS_PASSWORD is required (no default)."
  echo "[restart_querylake_redis] Example:"
  echo "  export QUERYLAKE_REDIS_PASSWORD='...'"
  echo "  export QUERYLAKE_REDIS_HOST_PORT=6393   # optional"
  echo "  docker compose -f \"$COMPOSE_FILE\" up -d redis"
  exit 1
fi

# Detect whether the script itself was launched via sudo.
DOCKER=(docker)
if [ -n "${SUDO_USER:-}" ]; then
  if command -v sudo >/dev/null 2>&1; then
    DOCKER=(sudo docker)
  else
    echo "[restart_querylake_redis] Detected sudo invocation (SUDO_USER set) but sudo is not available."
    exit 1
  fi
else
  if ! docker ps >/dev/null 2>&1; then
    echo "[restart_querylake_redis] docker is not usable without sudo."
    echo "[restart_querylake_redis] Fix: add your user to the docker group, or run this script with sudo."
    exit 1
  fi
fi

echo "[restart_querylake_redis] Ensuring no legacy container blocks the name (${REDIS_CONTAINER_NAME})..."
"${DOCKER[@]}" rm -f "$REDIS_CONTAINER_NAME" >/dev/null 2>&1 || true

echo "[restart_querylake_redis] Starting QueryLake Redis (localhost:${REDIS_HOST_PORT})..."
if "${DOCKER[@]}" compose version >/dev/null 2>&1; then
  QUERYLAKE_REDIS_PASSWORD="$REDIS_PASSWORD" QUERYLAKE_REDIS_HOST_PORT="$REDIS_HOST_PORT" \
    "${DOCKER[@]}" compose -p "$REDIS_COMPOSE_PROJECT" -f "$COMPOSE_FILE" up -d redis
elif command -v docker-compose >/dev/null 2>&1; then
  DOCKER_COMPOSE=(docker-compose)
  if [ "${DOCKER[0]}" = "sudo" ]; then
    DOCKER_COMPOSE=(sudo docker-compose)
  fi
  QUERYLAKE_REDIS_PASSWORD="$REDIS_PASSWORD" QUERYLAKE_REDIS_HOST_PORT="$REDIS_HOST_PORT" \
    "${DOCKER_COMPOSE[@]}" -p "$REDIS_COMPOSE_PROJECT" -f "$COMPOSE_FILE" up -d redis
else
  echo "[restart_querylake_redis] docker compose not found; running standalone Redis container..."
  tmp_conf=$(mktemp)
  {
    echo "bind 0.0.0.0"
    echo "protected-mode yes"
    echo "appendonly yes"
    echo "requirepass $REDIS_PASSWORD"
  } > "$tmp_conf"
  "${DOCKER[@]}" run \
    --name "$REDIS_CONTAINER_NAME" \
    --restart unless-stopped \
    -p "127.0.0.1:${REDIS_HOST_PORT}:6379" \
    -v "querylake_redis_data:/data" \
    -v "$tmp_conf:/usr/local/etc/redis/redis.conf:ro" \
    -d \
    redis:7-alpine \
    redis-server /usr/local/etc/redis/redis.conf
  rm -f "$tmp_conf"
fi

echo "[restart_querylake_redis] Done."
