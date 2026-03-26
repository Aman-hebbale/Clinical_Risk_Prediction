# ---------------------------------------------------------------------------
# Stage 1 — Builder
# Install all runtime dependencies into an isolated prefix so the runtime
# stage receives only pre-built wheels with no build tooling.
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /build

# Upgrade pip once in the builder; no need to carry this into the runtime.
RUN pip install --upgrade pip

# libpq-dev + gcc are required to compile psycopg2 from source.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy only the project manifest so Docker caches the dep-install layer
# separately from source changes.
COPY pyproject.toml ./

# Minimal stub so hatchling can resolve package metadata without the full src.
RUN mkdir -p src && touch src/__init__.py

# Install runtime deps into /install (no dev extras in the shipped image).
RUN pip install --prefix=/install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2 — Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# curl for the HEALTHCHECK; libpq for psycopg2 at runtime;
# postgresql-client for pg_isready in the start script.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        libpq5 \
        postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Non-root user — created before any files are copied in.
RUN useradd --create-home --shell /bin/bash app

WORKDIR /app

# Pull the pre-built package tree from the builder stage.
COPY --from=builder /install /usr/local

# Copy only the application source (not tests, notebooks, etc.).
COPY src/ ./src/

# Copy startup script and make it executable.
COPY scripts/start.sh ./scripts/start.sh
RUN chmod +x ./scripts/start.sh

# Transfer ownership so the non-root user can write runtime artefacts.
RUN chown -R app:app /app

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["./scripts/start.sh"]
