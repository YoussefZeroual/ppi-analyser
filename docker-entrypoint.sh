#!/bin/bash
# docker-entrypoint.sh
# Starts the Stanza API server in the background, waits for it,
# then starts the FastAPI/uvicorn server in the foreground.
set -e

STANZA_PORT=${STANZA_PORT:-5000}
UVICORN_PORT=${UVICORN_PORT:-8000}
UVICORN_HOST=${UVICORN_HOST:-0.0.0.0}
UVICORN_WORKERS=${UVICORN_WORKERS:-1}

echo "═══════════════════════════════════════════════"
echo "  PPI Analyser"
echo "═══════════════════════════════════════════════"

# ── 1. Start Stanza API server ───────────────────────────────────────────────
echo "[entrypoint] Starting Stanza API server on port ${STANZA_PORT}..."
python /app/ppi_analyser/stanza/stanza_api.py &
STANZA_PID=$!

# Wait until the Stanza server is ready (max 10 s)
echo "[entrypoint] Waiting for Stanza server..."
for i in $(seq 1 10); do
    if curl -sf "http://localhost:${STANZA_PORT}/health" > /dev/null 2>&1; then
        echo "[entrypoint] Stanza server ready (${i}s)."
        break
    fi
    sleep 1
    if [ $i -eq 10 ]; then
        echo "[entrypoint] WARNING: Stanza server did not respond after 60s — continuing anyway."
    fi
done

# ── 2. Start FastAPI / uvicorn ───────────────────────────────────────────────
echo "[entrypoint] Starting uvicorn on ${UVICORN_HOST}:${UVICORN_PORT}..."

# --reload is handy in dev (bind-mount mode); in prod you can unset RELOAD.
RELOAD_FLAG=""
if [ "${UVICORN_RELOAD:-0}" = "1" ]; then
    RELOAD_FLAG="--reload"
fi

exec uvicorn ppi_analyser.server:app \
    --host "${UVICORN_HOST}" \
    --port "${UVICORN_PORT}" \
    --workers "${UVICORN_WORKERS}" \
    $RELOAD_FLAG
