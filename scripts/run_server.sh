#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

HOST="${STT_HOST:-0.0.0.0}"
PORT="${STT_PORT:-23000}"

if command -v uvicorn >/dev/null 2>&1; then
  PYTHONPATH=src uvicorn stt_gateway.server:app --host "$HOST" --port "$PORT"
else
  # uv-managed environment fallback
  PYTHONPATH=src uv run uvicorn stt_gateway.server:app --host "$HOST" --port "$PORT"
fi
