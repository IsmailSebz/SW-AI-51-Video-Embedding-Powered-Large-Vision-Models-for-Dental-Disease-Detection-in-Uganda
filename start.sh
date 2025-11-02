#!/usr/bin/env bash
set -euo pipefail

# start.sh - start backend (uvicorn) and frontend (streamlit) together
# This script starts both processes in the background and forwards signals to them.

UVICORN_MODULE="backend.main:app"
UVICORN_HOST="0.0.0.0"
UVICORN_PORT="8000"

STREAMLIT_SCRIPT="frontend/website.py"
STREAMLIT_PORT="8501"

LOGDIR="/tmp/service_logs"
mkdir -p "$LOGDIR"

pids=()

function _term() {
  echo "[start.sh] received termination signal, stopping children..."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" || true
    fi
  done
  wait
  exit 0
}

trap _term SIGTERM SIGINT

echo "[start.sh] Starting uvicorn on ${UVICORN_HOST}:${UVICORN_PORT}"
uvicorn ${UVICORN_MODULE} --host ${UVICORN_HOST} --port ${UVICORN_PORT} 2>&1 &
pids+=("$!")

echo "[start.sh] Starting streamlit on port ${STREAMLIT_PORT}"
streamlit run ${STREAMLIT_SCRIPT} --server.port ${STREAMLIT_PORT} --server.address 0.0.0.0 --server.enableCORS false >"${LOGDIR}/streamlit.log" 2>&1 &
pids+=("$!")

echo "[start.sh] processes started; PIDs: ${pids[*]}"

# wait for any child to exit; then forward its exit status
wait -n
EXITCODE=$?
echo "[start.sh] one of the processes exited (code=${EXITCODE}), shutting down others..."
_term
