#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# start.sh — container entrypoint
#
# 1. Wait until PostgreSQL is reachable via pg_isready.
# 2. Seed / refresh the dataset by running src.data.download.
# 3. Start the Uvicorn API server.
# ---------------------------------------------------------------------------

set -euo pipefail

# ---------------------------------------------------------------------------
# 1. Wait for PostgreSQL
# ---------------------------------------------------------------------------
# DATABASE_URL is expected to follow the format:
#   postgresql://user:password@host:port/dbname
# We extract host and port so pg_isready can probe exactly that endpoint.

DB_HOST="${PGHOST:-db}"
DB_PORT="${PGPORT:-5432}"
DB_USER="${PGUSER:-postgres}"

echo "[start.sh] Waiting for PostgreSQL at ${DB_HOST}:${DB_PORT} ..."

MAX_ATTEMPTS=30
ATTEMPT=0
until pg_isready --host="${DB_HOST}" --port="${DB_PORT}" --username="${DB_USER}" --quiet; do
    ATTEMPT=$(( ATTEMPT + 1 ))
    if [ "${ATTEMPT}" -ge "${MAX_ATTEMPTS}" ]; then
        echo "[start.sh] ERROR: PostgreSQL did not become ready after ${MAX_ATTEMPTS} attempts. Aborting."
        exit 1
    fi
    echo "[start.sh] Attempt ${ATTEMPT}/${MAX_ATTEMPTS} — PostgreSQL not ready yet. Retrying in 2 s ..."
    sleep 2
done

echo "[start.sh] PostgreSQL is ready."

# ---------------------------------------------------------------------------
# 2. Seed the database / download dataset
# ---------------------------------------------------------------------------
echo "[start.sh] Running data download (src.data.download) ..."
python -m src.data.download || {
    echo "[start.sh] WARNING: Data download step failed — continuing anyway."
}

# ---------------------------------------------------------------------------
# 3. Start Uvicorn
# ---------------------------------------------------------------------------
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "[start.sh] Starting Uvicorn on ${HOST}:${PORT} ..."
exec uvicorn src.api.main:app --host "${HOST}" --port "${PORT}"
