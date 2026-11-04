# stt-gateway

音声ファイルを受け取り、文字起こし結果を JSON で返す STT API サーバ。
文字起こしエンジンは faster-whisper（多言語）と Haas-ja（日本語専用）から選べる。

## 動作要件
- Python 3.10 以上（Haas-ja を使う場合は 3.11 以上）
- [uv](https://docs.astral.sh/uv/)
- GPU を使う場合: NVIDIA ドライバ
- Haas-ja を使う場合: Linux と NVIDIA GPU（CPU では動かない）。ディスクは追加で約 8GB 必要

## セットアップ
```bash
uv sync
cp .env.example .env
```

Haas-ja を使う場合は、追加の依存を入れて `.env` の `STT_BACKEND` を `haas` にする。
```bash
uv sync --extra haas
```

## 設定（.env）

| 変数 | デフォルト | 説明 |
|---|---|---|
| `STT_HOST` | `0.0.0.0` | バインドするアドレス |
| `STT_PORT` | `23000` | ポート |
| `STT_BACKEND` | `whisper` | 文字起こしエンジン。`whisper` / `haas` |
| `STT_MODEL` | `large-v3-turbo` | faster-whisper のモデル名（whisper のみ） |
| `STT_DEVICE` | `auto` | `auto` / `cuda` / `cpu`（whisper のみ） |
| `STT_COMPUTE_TYPE` | `int8_float16` | `int8` / `int8_float16` / `float16` / `float32`（whisper のみ） |
| `STT_HAAS_MODEL` | `third-intelligence/Haas-ja-v1.0` | Haas-ja のモデル（haas のみ） |
| `STT_LANGUAGE` | `ja` | `language` 未指定時に使う言語 |
| `STT_API_KEY` | (空) | Bearer トークン。空の場合はトークンの値を検証しない |

- `STT_BACKEND` に上記以外の値を書くと、サーバは起動しない。
- モデルは最初の文字起こしのときに Hugging Face Hub からダウンロードされ、
  `~/.cache/huggingface` にキャッシュされる。`large-v3-turbo` は約 1.6GB。
  そのため最初のリクエストだけ時間がかかる。
- `haas` では日本語（`ja`）しか文字起こしできない。

## 起動
```bash
bash scripts/run_server.sh
```

systemd で常駐させる場合:
```bash
sudo cp stt-gateway.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now stt-gateway
sudo journalctl -u stt-gateway -f
```
`stt-gateway.service` は配置先を `/opt/stt-gateway` として定義している。別の場所に置く
場合は `WorkingDirectory` と `EnvironmentFile` を書き換える。

## API

### GET /health
認証不要。

```bash
curl http://127.0.0.1:23000/health
```
```json
{
  "ok": true,
  "backend": "whisper",
  "model": "large-v3-turbo",
  "device": "auto",
  "compute_type": "int8_float16"
}
```

### POST /transcribe
`multipart/form-data`。`Authorization: Bearer <token>` ヘッダが必須で、ヘッダが無い場合は
`401` を返す。`STT_API_KEY` が空のときはトークンの値を検証しない。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `file` | ○ | 音声ファイル |
| `language` | | 言語コード。省略時は `STT_LANGUAGE` |

#### 対応フォーマット
ファイルの拡張子で判定する。ファイル名または拡張子が無い場合は `.wav` として扱う。

| 拡張子 | 内容 |
|---|---|
| `.wav` | WAV（無圧縮） |
| `.ogg` / `.opus` | Opus（Ogg コンテナ） |
| `.webm` | Opus（WebM コンテナ） |

#### リクエスト例
```bash
curl -X POST http://127.0.0.1:23000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/tmp/sample.opus" \
  -F "language=ja"
```

#### レスポンス
```json
{
  "ok": true,
  "text": "こんにちは",
  "language": "ja",
  "duration": 1.23,
  "latency_ms": 240,
  "backend": "whisper",
  "model": "large-v3-turbo"
}
```

| ステータス | 条件 | `detail` |
|---|---|---|
| `400` | 対応外の拡張子 | `unsupported audio format: '.mp4' (allowed: .ogg, .opus, .wav, .webm)` |
| `400` | `haas` で `ja` 以外の言語を指定 | `unsupported language for backend 'haas': 'en' (supported: ja)` |
| `401` | ヘッダが無い | `Not authenticated` |
| `401` | トークンが一致しない | `invalid_token` |
| `500` | モデルの読み込みまたは文字起こしに失敗 | `transcribe_failed: <理由>` |

## テスト
```bash
uv sync --group dev
uv run pytest --cov=stt_gateway
```
GPU とモデル本体は不要。カバレッジが 80% を下回るとテストが失敗する。
