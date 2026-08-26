# stt-gateway

faster-whisper ベースの STT API サーバ。音声ファイルを受け取り、文字起こし結果を JSON で返す。

## 動作要件
- Python 3.10 以上
- [uv](https://docs.astral.sh/uv/)
- GPU を使う場合: NVIDIA ドライバ

CTranslate2 は実行時に CUDA 12 の `libcublas.so.12` を必要とする。これは依存パッケージ
`nvidia-cublas-cu12` に含まれており、`scripts/run_server.sh` が自動的に
`LD_LIBRARY_PATH` へ追加する。

## セットアップ
```bash
uv sync
cp .env.example .env
```

## 設定（.env）

| 変数 | デフォルト | 説明 |
|---|---|---|
| `STT_HOST` | `0.0.0.0` | バインドするアドレス |
| `STT_PORT` | `23000` | ポート |
| `STT_MODEL` | `large-v3-turbo` | faster-whisper のモデル名 |
| `STT_DEVICE` | `auto` | `auto` / `cuda` / `cpu` |
| `STT_COMPUTE_TYPE` | `int8_float16` | `int8` / `int8_float16` / `float16` / `float32` |
| `STT_LANGUAGE` | `ja` | `language` 未指定時に使う言語 |
| `STT_API_KEY` | (空) | Bearer トークン。空の場合はトークンの値を検証しない |

モデルは初回起動時に Hugging Face Hub からダウンロードされ、`~/.cache/huggingface` に
キャッシュされる。`large-v3-turbo` は約 1.6GB。

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

許可リストは `src/stt_gateway/formats.py` の `ALLOWED_SUFFIXES` で定義する。
デコードは faster-whisper が内部の PyAV(ffmpeg) で行う。

#### リクエスト例
```bash
curl -X POST http://127.0.0.1:23000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/tmp/sample.wav" \
  -F "language=ja"
```

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
  "model": "large-v3-turbo"
}
```

| ステータス | 条件 | `detail` |
|---|---|---|
| `400` | 対応外の拡張子 | `unsupported audio format: '.mp4' (allowed: .ogg, .opus, .wav, .webm)` |
| `401` | ヘッダが無い | `Not authenticated` |
| `401` | トークンが一致しない | `invalid_token` |
| `500` | 文字起こしに失敗 | `transcribe_failed: <理由>` |

## テスト
```bash
uv sync --group dev
uv run pytest --cov=stt_gateway
```
モデルはダミーに差し替えて実行するため、GPU とモデル本体は不要。フォーマット判定と
API の振る舞いを検証する。

カバレッジは 80% を下回るとテストが失敗する（`pyproject.toml` の
`[tool.coverage.report] fail_under`）。
