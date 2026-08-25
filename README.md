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

## 必要環境（GPU利用時）
`faster-whisper` が内部で使う CTranslate2 は、実行時に **CUDA 12 の `libcublas.so.12`**
を dlopen する。`nvidia-smi` が `CUDA Version: 13.x` と表示していても、CUDA 12 の cuBLAS
が入っていなければ利用できない点に注意。

不足している場合、**モデルの読み込みは成功するが最初の推論で失敗**し、
`CUBLAS_STATUS_NOT_SUPPORTED` となって `/transcribe` が 500 を返す。

確認:
```bash
ldconfig -p | grep -i cublas   # libcublas.so.12 が出るか
```

対策は依存関係に含めた `nvidia-cublas-cu12` で完結する。`scripts/run_server.sh` が
このパッケージのライブラリパスを自動的に `LD_LIBRARY_PATH` へ追加するため、
通常は追加作業は不要（システムに CUDA 12 が入っている環境でも実害はない）。

サーバを手動で起動する場合は、同じパスを明示的に通すこと:
```bash
export LD_LIBRARY_PATH="$(python -c 'import nvidia.cublas.lib as m; print(list(m.__path__)[0])'):$LD_LIBRARY_PATH"
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

### 対応フォーマット
アップロードするファイルの拡張子で判定する。それ以外の拡張子は `400` エラーになる。

| 拡張子 | 用途 |
|---|---|
| `.wav` | 既存クライアント向け（無圧縮） |
| `.ogg` / `.opus` | Opus（Oggコンテナ）。LTEなど帯域が限られる環境向け |
| `.webm` | Opus（WebMコンテナ、ブラウザ録音等） |

デコード自体は `faster-whisper` が内部のffmpeg(PyAV)で行うため、上記以外の
コンテナ/コーデックが必要になった場合は `src/stt_gateway/formats.py` の
許可リストに追加するだけでよい。

例（WAV、従来通り）:
```bash
curl -X POST http://127.0.0.1:23000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/tmp/sample.wav" \
  -F "language=ja"
```

例（Opus圧縮、LTE向け）:
```bash
curl -X POST http://127.0.0.1:23000/transcribe \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@/tmp/sample.opus" \
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

## テスト
```bash
uv sync --group dev
uv run pytest --cov=stt_gateway
```
モデル本体は読み込まず（ダミーモデルに差し替え済み）、フォーマット判定とAPIの
振る舞いのみを検証する。
