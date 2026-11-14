# stt-gateway

音声ファイルを受け取り、文字起こし結果を JSON で返す STT API サーバ。
文字起こしエンジンは faster-whisper（多言語）と Haas-ja（日本語専用）から選べる。

WebSocket（`/stream`）で音声をストリーミングで送ると、話している最中の相槌（「うん」「なるほど」）、
発話の文字起こし結果、「〇〇ね、確認するね。」のような一時的な返事を順に返す。

## 動作要件
- Python 3.10 以上（Haas-ja を使う場合は 3.11 以上）
- [uv](https://docs.astral.sh/uv/)
- GPU を使う場合: NVIDIA ドライバ
- Haas-ja を使う場合: Linux と NVIDIA GPU（CPU では動かない）。ディスクは追加で約 8GB 必要
- `/stream` を使う場合: [Ollama](https://ollama.com/)（一時的な返事を作る）。相槌を出すには、
  同時に話している 1 人あたり CPU 約 1.4 コア

## セットアップ
```bash
uv sync
cp .env.example .env
```

Haas-ja を使う場合は、追加の依存を入れて `.env` の `STT_BACKEND` を `haas` にする。
```bash
uv sync --extra haas
```

`/stream` で相槌を出す場合は、追加の依存を入れ、一時的な返事のモデルを Ollama に用意する。
```bash
uv sync --extra stream
ollama pull gemma4:e2b
```
- extras を複数使う場合は、まとめて指定する（例: `uv sync --extra haas --extra stream`）。
  1 つだけ指定して `uv sync` すると、指定しなかった extras は削除される。
- extras `stream` が無くても `/stream` は使える。その場合は相槌が出ない。

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
| `STT_WORKERS` | `1` | 同時に文字起こしできる件数（ワーカープロセス数） |
| `STT_MAX_REQUESTS` | `0` | 1 ワーカーがこの件数を処理したら、ワーカーを起動し直す。`0` で無効 |
| `STT_MAX_REQUESTS_JITTER` | `0` | 起動し直すまでの件数を、ワーカーごとに 0〜この値だけずらす |
| `STT_STREAM_MAX_UTTERANCE_SECONDS` | `60` | `/stream` の 1 発話の音声の上限（秒） |
| `STT_BACKCHANNEL_MAX_SESSIONS` | `4` | 1 ワーカーで同時に相槌を出せる発話の数。超えた発話は相槌なしになる |
| `STT_BACKCHANNEL_THRESHOLD` | `0.5` | 相槌を出す確率のしきい値（0 より大きく 1 以下）。小さいほど相槌が増える |
| `STT_BACKCHANNEL_MIN_INTERVAL_MS` | `1500` | 相槌と相槌の最短の間隔（ミリ秒） |
| `STT_BACKCHANNEL_PHRASES` | `うん,うんうん,はい,なるほど` | 相槌の文言（カンマ区切り）。順に使う |
| `STT_ACK_OLLAMA_URL` | `http://127.0.0.1:11434` | 一時的な返事を作る Ollama の URL |
| `STT_ACK_MODEL` | `gemma4:e2b` | 一時的な返事を作る Ollama のモデル |
| `STT_ACK_TIMEOUT_MS` | `1500` | 一時的な返事を待つ上限（ミリ秒） |
| `STT_ACK_MAX_CHARS` | `20` | 一時的な返事の文字数の上限 |
| `STT_ACK_FALLBACK_TEXT` | `ちょっと確認するね。` | 一時的な返事を作れなかったときに返す文言 |
| `STT_ACK_KEEP_ALIVE` | `30m` | Ollama がモデルを GPU に置いておく時間（Ollama の `keep_alive` の書式） |

- `STT_BACKEND` に上記以外の値を書くと、サーバは起動しない。
- モデルはサーバの起動時に読み込む。初回は Hugging Face Hub からダウンロードされ、
  `~/.cache/huggingface` にキャッシュされる。`large-v3-turbo` は約 1.6GB。
  読み込みが終わるまで `/transcribe` は応答を待たせる。
- ワーカーごとにモデルを読み込むため、GPU メモリも `STT_WORKERS` の数だけ必要になる
  （haas は 1 ワーカーあたり最大約 3.6GB）。
- 同時に来たリクエストが `STT_WORKERS` を超えると、空いたワーカーから順に処理する。
  待っている間もエラーにはならない。
- `STT_MAX_REQUESTS` の件数には `/health` も含まれる。起動し直しているワーカーは、
  モデルを読み込み終えるまでリクエストを処理しない。
- `STT_WORKERS=1` で `STT_MAX_REQUESTS` を使う場合は、systemd で常駐させる。
  `scripts/run_server.sh` を直接実行していると、指定件数でサーバが終了したままになる。
- `haas` では日本語（`ja`）しか文字起こしできない。
- `/stream` 用の設定が範囲外の値だと、サーバは起動しない。
- 一時的な返事のモデルは GPU メモリを使う（`gemma4:e2b` で約 1.7GB）。Ollama が動いていなくても
  `/stream` は使え、一時的な返事は `STT_ACK_FALLBACK_TEXT` になる。
- 相槌のモデルは、最初に相槌を出すときにダウンロードされる（`~/.cache/huggingface` と `~/.cache/cpc`）。

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

`latency_ms` は、ほかのリクエストの処理を待った時間を含む。

| ステータス | 条件 | `detail` |
|---|---|---|
| `400` | 対応外の拡張子 | `unsupported audio format: '.mp4' (allowed: .ogg, .opus, .wav, .webm)` |
| `400` | `haas` で `ja` 以外の言語を指定 | `unsupported language for backend 'haas': 'en' (supported: ja)` |
| `401` | ヘッダが無い | `Not authenticated` |
| `401` | トークンが一致しない | `invalid_token` |
| `500` | モデルの読み込みまたは文字起こしに失敗 | `transcribe_failed: <理由>` |

### WebSocket /stream
`Authorization: Bearer <token>` ヘッダが必須（`/transcribe` と同じ）。ヘッダが無い、またはトークンが
一致しない場合は、接続できない（HTTP `403`）。ブラウザの WebSocket はヘッダを付けられないため使えない。

1 つの接続で、発話を 1 つずつ順に送れる。

#### クライアントから送るもの
| 種類 | 形式 | 内容 |
|---|---|---|
| 発話の開始 | テキスト（JSON） | `{"type": "start", "encoding": "opus", "language": "ja"}`。`type` 以外は省略できる（省略時は `opus` と `STT_LANGUAGE`）。一時的な返事と相槌の指定（`ack`・`backchannel`）も付けられる（下記） |
| 音声 | バイナリ | `encoding` が `opus` なら Opus のパケット 1 つ（Ogg などのコンテナに入れない）。`pcm_s16le` なら 16kHz・モノラル・16bit リトルエンディアンの PCM |
| 発話の終了 | テキスト（JSON） | `{"type": "end"}` |

- Opus はモノラルで送る。フレーム長・ビットレート・エンコード時のサンプリングレートは自由。
  パケットは 1 つずつ、エンコードした順に送る。
- 言語は `ja` だけ。
- `end` を送ったら、結果を待たずに次の `start` を送ってよい。

#### 一時的な返事と相槌を発話ごとに指定する
`start` に `ack` と `backchannel` を付けると、その発話だけサーバの設定の代わりに使う。
省略した項目はサーバの設定のまま。

例：この後に別の本回答が続くクライアント向けに、相手の言葉を繰り返さない一言にする。
```json
{"type": "start", "encoding": "opus",
 "ack": {
   "system_prompt": "あなたは会話の聞き手です。相手の話を聞いて、相槌を一言だけ打ちます。あなたは答えを知らないので、質問や依頼には答えず「ちょっと見てみるね」「考えるね」のように受け止めるだけにします。相手の言葉は繰り返しません。聞き返しもしません。日本語で、15文字以内、くだけた口調、1文だけ。",
   "examples": [
     {"user": "明日の天気ってどう？", "reply": "ちょっと見てみるね。"},
     {"user": "なんか最近調子悪いんだよね", "reply": "そっか、ちょっと考えるね。"},
     {"user": "この関数どこで呼ばれてるんだっけ", "reply": "うん、探してみる。"},
     {"user": "お疲れさま、今日はもういいよ", "reply": "うん、お疲れさま！"}
   ],
   "fallback_text": "えっと、ちょっと待ってね。"
 },
 "backchannel": {"phrases": ["はい", "ええ"], "threshold": 0.4, "min_interval_ms": 1200}}
```

| フィールド | 範囲 | 省略時 |
|---|---|---|
| `ack.system_prompt` | 1〜2,000 文字 | 既定のプロンプト（相手の言葉を少し繰り返して受け止める） |
| `ack.examples` | `user`（1〜200 文字）と `reply`（1〜100 文字）の組、0〜8 個 | `system_prompt` も省略したときは既定の例、`system_prompt` を指定したときは例なし |
| `ack.fallback_text` | 1〜50 文字 | `STT_ACK_FALLBACK_TEXT` |
| `ack.max_chars` | 1〜100 | `STT_ACK_MAX_CHARS` |
| `backchannel.phrases` | 1〜20 文字の文言、1〜10 個 | `STT_BACKCHANNEL_PHRASES` |
| `backchannel.threshold` | 0 より大きく 1 以下 | `STT_BACKCHANNEL_THRESHOLD` |
| `backchannel.min_interval_ms` | 0〜60,000 | `STT_BACKCHANNEL_MIN_INTERVAL_MS` |

- `"ack": false` にすると、その発話では一時的な返事が届かない。
- 範囲外の値や知らないフィールドがあると `invalid_message` が届き、その発話は始まらない。
- 一時的な返事を作るモデルは、クライアントからは変えられない（`STT_ACK_MODEL`）。
- プロンプトは「会話の聞き手」として書き出し、「答えを知らない」「日本語で」を入れる。
  「アシスタント」や「つなぎ役」として書くと、質問に答えて事実でない内容を言い切ったり、
  日本語以外で返したりすることがある（`gemma4:e2b` で確認）。

#### サーバから届くもの（テキスト・JSON）
| `type` | 届くとき | 主なフィールド |
|---|---|---|
| `started` | `start` を受け付けたとき | `utterance_id`、`backchannel`（この発話で相槌が出るか）、`backchannel_unavailable_reason`（出ない理由。`busy`: 混雑、`unavailable`: 相槌のモデルが使えない） |
| `backchannel` | 話している最中、相槌を打つタイミング | `text`、`offset_ms`（発話の先頭からの位置）、`probability` |
| `transcript` | `end` の後 | `text`、`language`、`duration`、`latency_ms`、`backend`、`model`（`/transcribe` と同じ意味） |
| `ack` | `transcript` の後 | `text`、`source`（`llm`: LLM が作った、`fallback`: 固定の文言）、`latency_ms` |
| `error` | 下表 | `code`、`detail` |

- すべてに `utterance_id`（接続ごとに 1 から振る発話の番号）が付く。
- 1 つの発話では `transcript` が先、`ack` が後に届く。
- 相槌は `start` から `end` までの間だけ届く。
- 文字起こし結果が空のとき、または `"ack": false` のときは `ack` は届かない。

| `code` | 条件 |
|---|---|
| `invalid_message` | JSON として読めない、`type` や `encoding` が不明、`ack`・`backchannel` の指定が範囲外 |
| `invalid_audio` | Opus のパケットをデコードできない、PCM のバイト数が奇数 |
| `invalid_state` | 発話中に `start`、発話中でないときに音声や `end` |
| `unsupported_language` | `ja` 以外の言語 |
| `utterance_too_long` | 1 発話が `STT_STREAM_MAX_UTTERANCE_SECONDS` を超えた。その発話は捨てられ、`end` まで音声は無視される |
| `transcribe_failed` | 文字起こしに失敗した。その発話の `ack` は届かない |

`error` が届いても接続は切れない。ワーカーの入れ替え（`STT_MAX_REQUESTS`）のときは、接続が
クローズコード `1012` で切れるので、接続し直す。

#### 例（Python）
```python
import asyncio, json
import websockets

async def main(opus_packets: list[bytes]) -> None:
    headers = {"Authorization": "Bearer YOUR_TOKEN"}
    async with websockets.connect("ws://127.0.0.1:23000/stream", additional_headers=headers) as ws:
        await ws.send(json.dumps({"type": "start", "encoding": "opus"}))
        for packet in opus_packets:  # 20ms ごとにマイクからエンコードしたパケット
            await ws.send(packet)
        await ws.send(json.dumps({"type": "end"}))
        async for message in ws:
            event = json.loads(message)
            print(event)
            if event["type"] in ("ack", "error"):
                break
```

## テスト
```bash
uv sync --group dev
uv run pytest --cov=stt_gateway
```
GPU とモデル本体、Ollama は不要。カバレッジが 80% を下回るとテストが失敗する。
