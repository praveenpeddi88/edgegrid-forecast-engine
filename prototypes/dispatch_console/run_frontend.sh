#!/usr/bin/env bash
# Start the EdgeGrid Dispatch Console Vite frontend.
#
# First run: installs npm dependencies.
# Subsequent runs: starts the dev server on localhost:5173.
#
# From repo root:
#     ./prototypes/dispatch_console/run_frontend.sh

set -euo pipefail
cd "$(dirname "$0")/../.."/frontend

if [ ! -d node_modules ]; then
    echo "→ installing frontend dependencies (first run)…"
    npm install
fi

exec npm run dev -- --port 5173 --host
