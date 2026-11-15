#!/usr/bin/env bash
set -euo pipefail

# 未設定のまま起動すると認証なしになるため、空のキーでも起動を拒否する。
TOKEN_FILE="${CREDENTIALS_DIRECTORY:?systemd LoadCredential is required}/api_key"
KEV_API_KEY="$(cat "$TOKEN_FILE")"
if [ -z "$KEV_API_KEY" ]; then
  echo "KEV API key must not be empty" >&2
  exit 1
fi
export KEV_API_KEY

# CUDA Graph を使うと GPU を約 4GB 使い、同じ GPU の Haas が推論の準備でメモリ不足になる。
# 使わなければ約 1.9GB で、判定は約 30ms（使うと約 20ms）なので、待つ上限の 200ms に収まる。
export KEV_CUDA_GRAPHS=0

# 実験環境と同じ依存を使い、常駐プロセスの起動時には依存の更新を行わない。
cd /opt/stt-gateway/.kev-runtime
exec .venv/bin/python -m kev.serve \
  --run jaredpalmer/kev-0.8b --device cuda --host 127.0.0.1 --port 8009
