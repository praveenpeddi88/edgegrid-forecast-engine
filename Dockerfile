# syntax=docker/dockerfile:1.6
#
# EdgeGrid Forecast Engine — production image
# ────────────────────────────────────────────
# Multi-stage build:
#   stage 1 (builder): installs build-tooling, compiles wheels for native deps
#   stage 2 (runtime): slim image with only the runtime artefacts + model bundles
#
# Expected runtime layout (bind-mount or COPY at deploy):
#   /app/models/v5/*           — v5 LightGBM bundles + manifest
#   /app/data/                 — (optional) parquet frames for hot-path caching
#   /app/benchmarks/results/   — benchmark artefacts served by /v5/healthz
#
# Build:
#   docker build -t edgegrid-forecast:v5 .
# Run:
#   docker run --rm -p 8000:8000 \
#     -v $(pwd)/models:/app/models:ro \
#     -v $(pwd)/data:/app/data:ro \
#     edgegrid-forecast:v5

# ─── Stage 1 — Builder ────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Build-time system deps: LightGBM / Prophet / pvlib native bits.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential \
      cmake \
      git \
      libgomp1 \
      libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy packaging metadata first for layer-cache efficiency.
COPY pyproject.toml README.md ./
COPY src ./src

# Install into a self-contained prefix we copy forward.
RUN pip install --upgrade pip wheel setuptools \
 && pip install --prefix=/install ".[api]"

# ─── Stage 2 — Runtime ────────────────────────────────────────────────────────
FROM python:3.11-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    EDGEGRID_ENV=production \
    EDGEGRID_API_HOST=0.0.0.0 \
    EDGEGRID_API_PORT=8000

# Runtime-only libs (libgomp for LightGBM at import time).
RUN apt-get update && apt-get install -y --no-install-recommends \
      libgomp1 \
      curl \
    && rm -rf /var/lib/apt/lists/*

# Bring in the pre-installed site-packages + console scripts from the builder.
COPY --from=builder /install /usr/local

# Non-root runtime user.
RUN groupadd --system edgegrid && useradd --system --gid edgegrid --home /app edgegrid
WORKDIR /app

# Application code (read-only at runtime).
COPY --chown=edgegrid:edgegrid src      ./src
COPY --chown=edgegrid:edgegrid models   ./models
COPY --chown=edgegrid:edgegrid data     ./data
COPY --chown=edgegrid:edgegrid benchmarks ./benchmarks
COPY --chown=edgegrid:edgegrid docs     ./docs
COPY --chown=edgegrid:edgegrid pyproject.toml README.md ./

USER edgegrid

EXPOSE 8000

# Healthcheck — relies on /health (legacy) + /v5/healthz (new router).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${EDGEGRID_API_PORT}/v5/healthz" || exit 1

# Default entrypoint: serve the FastAPI app via uvicorn.
CMD ["uvicorn", "edgegrid_forecast.api.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
