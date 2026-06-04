# stt-gateway

faster-whisper ベースの STT API サーバ。

## ポート
- デフォルト: **23000**

## セットアップ
```bash
cd stt-gateway
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
cp .env.example .env
```

## 設定（.env）
```env
STT_HOST=0.0.0.0
STT_PORT=23000
STT_MODEL=large-v3-turbo
STT_DEVICE=auto
STT_COMPUTE_TYPE=int8_float16
STT_LANGUAGE=ja
STT_API_KEY=         # Bearer認証トークン（空にすると認証スキップ）
```

## 起動
```bash
bash scripts/run_server.sh
```
`uv sync` を使う環境でも動くように、`uvicorn` が見つからない場合は `uv run uvicorn` を自動利用します。

## ヘルスチェック
```bash
curl http://127.0.0.1:23000/health
```

## 推論API
`POST /transcribe` (multipart)

`STT_API_KEY` を設定した場合は `Authorization: Bearer <token>` ヘッダが必須。

例:
```bash
curl -X POST http://127.0.0.1:23000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/tmp/sample.wav" \
  -F "language=ja"
```

レスポンス例:
```json
{
  "ok": true,
  "text": "こんにちは",
  "language": "ja",
  "duration": 1.23,
  "latency_ms": 240,
  "model": "large-v3-turbo"
}
```
