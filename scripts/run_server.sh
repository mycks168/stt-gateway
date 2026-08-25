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

if command -v uvicorn >/dev/null 2>&1; then
  setup_cublas_path python
  PYTHONPATH=src uvicorn stt_gateway.server:app --host "$HOST" --port "$PORT"
else
  # uv-managed environment fallback
  setup_cublas_path uv run python
  PYTHONPATH=src uv run uvicorn stt_gateway.server:app --host "$HOST" --port "$PORT"
fi
