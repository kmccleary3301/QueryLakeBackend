#!/usr/bin/env bash
set -euo pipefail

BASE_URL="${1:-http://127.0.0.1:8000}"
USER="${QL_DEMO_USER:-demo}"
PASS="${QL_DEMO_PASS:-demo}"

echo "== QueryLake v2 demo against ${BASE_URL} =="

echo "-- Health/Ready --"
curl -sS "${BASE_URL}/healthz" || true; echo
curl -sS "${BASE_URL}/readyz" || true; echo

echo "-- API ping --"
curl -sS "${BASE_URL}/api/ping" || true
echo

echo "-- Add user (best-effort) --"
curl -sS -X POST "${BASE_URL}/api/add_user" -H 'Content-Type: application/json' \
  -d "{\"username\":\"${USER}\",\"password\":\"${PASS}\"}" || true
echo

echo "-- Login --"
LOGIN_JSON=$(curl -sS -X POST "${BASE_URL}/api/login" -H 'Content-Type: application/json' \
  -d "{\"auth\":{\"username\":\"${USER}\",\"password\":\"${PASS}\"}}")
echo "$LOGIN_JSON" | jq . >/dev/null 2>&1 || echo "$LOGIN_JSON"

echo "-- Create API key --"
API_KEY=$(curl -sS -X POST "${BASE_URL}/api/create_api_key" -H 'Content-Type: application/json' \
  -d "{\"auth\":{\"username\":\"${USER}\",\"password\":\"${PASS}\"}}" | jq -r '.result.api_key')
echo "API_KEY: ${API_KEY:0:10}..."

echo "-- Create v2 session --"
SESSION_JSON=$(curl -sS -X POST "${BASE_URL}/sessions" -H 'Content-Type: application/json' \
  -d "{\"toolchain_id\":\"demo_streaming_coauthor\",\"auth\":{\"username\":\"${USER}\",\"password\":\"${PASS}\"}}")
SID=$(echo "$SESSION_JSON" | jq -r .session_id)
echo "SESSION: $SID"

echo "-- Start SSE stream (background, 30s) --"
SSE_LOG="/tmp/querylake_sse_${SID}.log"
( curl -sS -N -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/sessions/${SID}/stream" --max-time 30 > "${SSE_LOG}" ) &
SSE_PID=$!
sleep 2

echo "-- Post event --"
curl -sS -X POST "${BASE_URL}/sessions/${SID}/event" -H 'Content-Type: application/json' \
  -d "{\"auth\":{\"username\":\"${USER}\",\"password\":\"${PASS}\"},\"node_id\":\"compose\",\"inputs\":{\"topic\":\"Write a short summary about IoT devices in healthcare.\"}}"
echo

sleep 5
echo "-- Jobs --"
curl -sS -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/sessions/${SID}/jobs" | jq . || true

echo "-- Events Backlog (first 3) --"
curl -sS -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/sessions/${SID}/events?since=0" \
  | jq '.events | .[:3]' || true

echo "-- Stop SSE and show last lines --"
wait ${SSE_PID} || true
tail -n 20 "${SSE_LOG}" || true
echo "Saved SSE log: ${SSE_LOG}"

echo "-- Metrics (first 50 lines) --"
curl -sS "${BASE_URL}/metrics" | head -n 50 || true
