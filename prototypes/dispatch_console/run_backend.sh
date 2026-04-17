#!/usr/bin/env bash
# Start the EdgeGrid Dispatch Console FastAPI backend.
#
# Requires the repo's Python env already set up (see pyproject.toml).
#
# From repo root:
#     ./prototypes/dispatch_console/run_backend.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

PORT="${EDGEGRID_PORT:-8000}"

# Prefer uv, fall back to python -m
if command -v uv >/dev/null 2>&1; then
    exec uv run uvicorn edgegrid_forecast.api.main:app --reload --host 0.0.0.0 --port "$PORT"
else
    exec python -m uvicorn edgegrid_forecast.api.main:app --reload --host 0.0.0.0 --port "$PORT"
fi
