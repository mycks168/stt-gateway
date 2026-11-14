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
WORKERS="${STT_WORKERS:-1}"
MAX_REQUESTS="${STT_MAX_REQUESTS:-0}"
MAX_REQUESTS_JITTER="${STT_MAX_REQUESTS_JITTER:-0}"

# 0 以上の整数でない値のまま起動すると、意図しない並列数や入れ替え件数で動いてしまうため止める。
require_non_negative_integer() {
  local name="$1" value="$2"
  if ! [[ "$value" =~ ^[0-9]+$ ]]; then
    echo "invalid ${name}: '${value}' (must be a non-negative integer)" >&2
    exit 1
  fi
}
require_non_negative_integer STT_WORKERS "$WORKERS"
require_non_negative_integer STT_MAX_REQUESTS "$MAX_REQUESTS"
require_non_negative_integer STT_MAX_REQUESTS_JITTER "$MAX_REQUESTS_JITTER"
if [ "$WORKERS" -lt 1 ]; then
  echo "invalid STT_WORKERS: '${WORKERS}' (must be 1 or more)" >&2
  exit 1
fi

# 各ワーカーはモデルを 1 つずつ読み込む。ワーカーを増やすと GPU メモリもその数だけ必要になる。
UVICORN_ARGS=(stt_gateway.server:app --host "$HOST" --port "$PORT" --workers "$WORKERS")
# 指定件数を処理したワーカーを終了させ、肥大化したメモリを解放する。
# ワーカーが 2 つ以上なら uvicorn が、1 つならプロセスごと終了するので systemd が起動し直す。
if [ "$MAX_REQUESTS" -gt 0 ]; then
  UVICORN_ARGS+=(--limit-max-requests "$MAX_REQUESTS" --limit-max-requests-jitter "$MAX_REQUESTS_JITTER")
fi

# CTranslate2 は実行時に libcublas.so.12 (CUDA 12) を dlopen する。
# CUDA 13 のみのホストではシステムに .so.12 が無いため、モデルの読み込みは
# 成功するが最初の推論で CUBLAS_STATUS_NOT_SUPPORTED になる。
# nvidia-cublas-cu12 が提供するライブラリを LD_LIBRARY_PATH に追加して回避する。
# システム側に CUDA 12 が入っている環境では、見つからなくても実害はない。
setup_cublas_path() {
  local dir
  dir="$("$@" -c 'import nvidia.cublas.lib as m; print(list(m.__path__)[0])' 2>/dev/null || true)"
  if [ -n "$dir" ] && [ -d "$dir" ]; then
    export LD_LIBRARY_PATH="${dir}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}

# Haas-ja の依存（NeMo, torch）は extras に分けている。uv 管理の環境では起動時に入れる。
UV_RUN=(uv run)
if [ "${STT_BACKEND:-whisper}" = "haas" ]; then
  UV_RUN=(uv run --extra haas)
fi

if command -v uvicorn >/dev/null 2>&1; then
  setup_cublas_path python
  PYTHONPATH=src uvicorn "${UVICORN_ARGS[@]}"
else
  # uv-managed environment fallback
  setup_cublas_path "${UV_RUN[@]}" python
  PYTHONPATH=src "${UV_RUN[@]}" uvicorn "${UVICORN_ARGS[@]}"
fi
