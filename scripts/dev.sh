#!/usr/bin/env bash
# Backend dev server (laptop). Serve SPA at :8000.
set -euo pipefail
cd "$(dirname "$0")/.."
cd backend
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload