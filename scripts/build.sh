#!/usr/bin/env bash
# Build the Svelte frontend into web/dist (served by FastAPI).
set -euo pipefail
cd "$(dirname "$0")/.."
cd web
npm run build