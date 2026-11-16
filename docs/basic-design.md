# 基本設計

## 1. 全体構成

```
uvicorn（STT_WORKERS 個のワーカープロセス）
  └─ ワーカーごとに:
クライアント ──HTTP──▶ server.py（API・認証・lifespan で /stream の部品を用意）
          └──WebSocket──▶ stream_api.py（/stream・認証）
                               │    auth.py … Bearer トークンの検証（/transcribe と共有）
                               ▼
                          stream_session.py（1 接続の発話の進行）
                            ├─ backchannel_pool.py … 予測器の貸し出し（上限つき）
                            ├─ backchannel_stream.py … 予測器へ音声を渡し、合図を emitter へ
                            │    ├─ backchannel_triggers.py … 無音が続いたら相槌の合図
                            │    └─ backchannel_emitters.py … 区切りを判定して送る / 順に回して送る
                            │         └─ boundary_judge.py ◀─ causal_lm_boundary.py … Qwen3.5-0.8B-Base（GPU）
                            ├─ backchannel_policy.py ◀─ BackchannelPredictor（Protocol）
                            │                             └─ silero_vad_predictor.py … Silero VAD（CPU）
                            ├─ stream_options.py … 発話ごとの指定とサーバの設定の合成
                            ├─ ack_generator.py … Ollama の軽量 LLM
                            │    ├─ ack_prompt.py … プロンプトと会話例
                            │    └─ ack_text.py … LLM 出力の整形と検証
                            ├─ stream_audio_decoder.py … Opus / PCM から波形への変換
                            └─ pcm_audio.py … 波形と WAV の変換
                          ▼（発話の終了時）
              serial_transcriber.py（推論を 1 件ずつ実行。/transcribe と共有）
                          │ Transcriber（Protocol）
                          ▼
              backends.py（設定でバックエンドを選ぶ）
                 ├─ whisper_backend.py … faster-whisper
                 └─ haas_backend.py    … Haas-ja（haas-asr / NeMo）
                        └─ audio_decoding.py … PyAV でデコード
```

| モジュール | 責務 |
|---|---|
| `server.py` | HTTP API、Bearer 認証、一時ファイルの管理、例外から HTTP ステータスへの変換、起動時のモデル読み込みと `/stream` 用の部品（予測器の貸し出し・区切りの判定・一時的な返事の作成）の用意 |
| `auth.py` | Bearer トークンの取り出しと、API キーとの比較（`/transcribe` と `/stream` で共有） |
| `serial_transcriber.py` | ワーカー内でのモデルの保持と、推論を 1 件ずつ別スレッドで実行する順番待ち |
| `config.py` | 環境変数（`.env` を含む）の読み込みと検証 |
| `formats.py` | 拡張子による音声フォーマットの判定 |
| `transcription.py` | バックエンドの共通インタフェース（`Transcriber`）、結果の型（`Transcription`）、例外 |
| `backends.py` | 設定に応じたバックエンドの読み込みと、`/health` 用の情報の算出 |
| `whisper_backend.py` | faster-whisper による文字起こし |
| `haas_backend.py` | Haas-ja による文字起こし |
| `audio_decoding.py` | PyAV(ffmpeg) による音声ファイルから波形へのデコード（haas が使う） |
| `stream_api.py` | `/stream` の WebSocket 受け付け、認証、受信メッセージの解釈 |
| `stream_messages.py` | `/stream` で送受信するメッセージの型（Pydantic） |
| `stream_session.py` | 1 接続の中での発話の開始・音声の受け取り・終了の進行と、イベントの送信 |
| `backchannel_policy.py` | 相槌の判断に使う型（フレーム・合図・候補の結果）と文言の順番（純粋なロジック）。予測器とトリガーのインタフェース（`BackchannelPredictor`・`BackchannelTrigger`）もここに置く |
| `backchannel_pool.py` | ワーカー内の予測器を上限数まで用意して貸し出す。上限に達していれば貸さない |
| `backchannel_stream.py` | 1 つの発話の音声を予測器へ渡し、相槌を出すと決めたら emitter に渡す |
| `backchannel_emitters.py` | 相槌を出すと決めたフレームについて、送るかどうかを決めて送る（文言を順に回すだけか、途中の文字起こしで区切りを判定してから送るか） |
| `boundary_judge.py` | 途中までの書き起こしが言い終わりの区切りで止まっているかを、言語モデルで測る |
| `boundary_text.py` | 区切り（句読点・改行）を文字の種類（Unicode の分類）で見分ける（純粋なロジック） |
| `causal_lm_boundary.py` | Hugging Face の言語モデルを読み込み、文の続きに区切りが来る確率を計算する |
| `backchannel_triggers.py` | フレームの予測の列から、相槌の準備と相槌を出す合図を決める（無音が続いたら出す規則。純粋なロジック） |
| `silero_vad_predictor.py` | Silero VAD で、32ms ごとに話し手が話していない確率を予測する |
| `backchannel_interval.py` | 次の相槌を出してよいかを、送った相槌からの間隔（発話の経過時間と相槌の回数で伸ばす）と、新しく話した長さで決める（純粋なロジック） |
| `utterance_recording.py` | 発話の音声（WAV）と、毎フレームの確率・相槌の合図・候補の判定（JSON）をファイルに保存する |
| `sample_transcription.py` | 途中までの波形を一時ファイルにして `SerialTranscriber` で文字起こしする |
| `ack_generator.py` | Ollama の軽量 LLM で一時的な返事を作り、失敗時は固定の文言に差し替える |
| `ack_prompt.py` | 一時的な返事の既定のシステムプロンプトと会話例、chat API のメッセージ列の組み立て（純粋なロジック） |
| `stream_options.py` | `start` の `ack`・`backchannel` の指定とサーバの設定から、発話ごとの設定を決める（純粋なロジック） |
| `ack_text.py` | LLM 出力の整形と、返してよい文かどうかの検証（純粋なロジック） |
| `stream_audio_decoder.py` | 受信した Opus パケットまたは PCM バイト列を、16kHz モノラルの波形に変換する。Opus はデコーダの状態を発話ごとに持つ |
| `pcm_audio.py` | PCM バイト列と波形の相互変換、波形から WAV への書き出し（純粋なロジック） |

依存の向きは `server → serial_transcriber / backends → 各バックエンド → transcription` の一方向とする。
`server.py` は faster-whisper や NeMo を直接 import しない。

`/stream` の依存の向きは `server → stream_api → stream_session → backchannel_pool / backchannel_policy /
ack_generator / stream_audio_decoder / pcm_audio / serial_transcriber` とする。
`stream_messages.py`（メッセージの型）は `stream_api.py` と `stream_session.py` から使い、業務ロジックに依存しない。
`silero_vad_predictor.py` は `backchannel_policy.py` の `BackchannelPredictor` を実装する側で、
`server.py` の lifespan が予測器を作る関数を `backchannel_pool.py` に渡す。`stream_session.py` と
`backchannel_pool.py` は Silero VAD を直接 import しない。silero-vad は extras `stream` にあり、入っていない
環境でも `/transcribe` と、相槌なしの `/stream` が動くようにするため。
区切りの判定の言語モデル（`causal_lm_boundary.py`）も同じく、`server.py` が読み込んで `boundary_judge.py` に渡す。

## 2. 外部仕様

### 2.1 設定（環境変数）

| 変数 | デフォルト | 対象 | 説明 |
|---|---|---|---|
| `STT_HOST` | `0.0.0.0` | 共通 | バインドするアドレス（`scripts/run_server.sh` が使う） |
| `STT_PORT` | `23000` | 共通 | ポート（同上） |
| `STT_BACKEND` | `whisper` | 共通 | `whisper` / `haas`。それ以外の値では起動時に `ValueError` で停止する |
| `STT_MODEL` | `large-v3-turbo` | whisper | faster-whisper のモデル名 |
| `STT_DEVICE` | `auto` | whisper | `auto` / `cuda` / `cpu` |
| `STT_COMPUTE_TYPE` | `int8_float16` | whisper | `int8` / `int8_float16` / `float16` / `float32` |
| `STT_HAAS_MODEL` | `third-intelligence/Haas-ja-v1.0` | haas | Hugging Face のモデル ID または `.nemo` ファイルのパス |
| `STT_LANGUAGE` | `ja` | 共通 | リクエストで `language` を省略したときの言語 |
| `STT_API_KEY` | (空) | 共通 | Bearer トークン。空のときはトークンの値を検証しない |
| `STT_WORKERS` | `1` | 共通 | uvicorn のワーカープロセス数（`scripts/run_server.sh` が使う）。1 以上の整数 |
| `STT_MAX_REQUESTS` | `0` | 共通 | 1 ワーカーがこの件数のリクエストを処理したら終了させる。`0` で無効（同上） |
| `STT_MAX_REQUESTS_JITTER` | `0` | 共通 | 終了までの件数にワーカーごとに加える乱数の上限（同上） |

`STT_WORKERS` / `STT_MAX_REQUESTS` / `STT_MAX_REQUESTS_JITTER` が 0 以上の整数でない場合、
または `STT_WORKERS` が 0 の場合は、`scripts/run_server.sh` がエラーを出して起動しない。

`/stream` 用（いずれも 1 ワーカーあたりの値）：

| 変数 | デフォルト | 説明 |
|---|---|---|
| `STT_BACKCHANNEL_MAX_SESSIONS` | `4` | 同時に相槌を予測できる発話の数。超えた発話は相槌なしで受け付ける。1 以上の整数 |
| `STT_STREAM_MAX_UTTERANCE_SECONDS` | `60` | 1 発話の音声の上限（秒）。正の数 |
| `STT_STREAM_RECORDING_DIR` | (空) | 相槌を予測した発話の音声と判定の記録を保存するディレクトリ（下の「発話の録音」）。`~` はサーバを動かすユーザのホーム。空なら保存しない |
| `STT_BACKCHANNEL_SILENCE_MS` | `90` | 相槌を出すまでに待つ無音の長さ（ミリ秒）。32 以上の整数 |
| `STT_BACKCHANNEL_MIN_NEW_SPEECH_MS` | `1000` | 前の相槌（無ければ発話の先頭）から新しく話しているべき長さ（ミリ秒）。0 以上の整数 |
| `STT_BACKCHANNEL_MIN_INTERVAL_MS` | `1500` | 話し始めの、相槌を出してから次の相槌を出せるまでの最短の間隔（ミリ秒）。0 以上の整数 |
| `STT_BACKCHANNEL_INTERVAL_PER_MINUTE_MS` | `3000` | 1 分話すごとに最短の間隔へ足す長さ（ミリ秒）。0 以上の整数 |
| `STT_BACKCHANNEL_INTERVAL_FACTOR` | `1.1` | 相槌を 1 回出すごとに最短の間隔を何倍にするか。1.0 以上 3.0 以下 |
| `STT_BACKCHANNEL_MAX_INTERVAL_MS` | `8000` | 伸ばした最短の間隔の上限（ミリ秒）。0 以上の整数 |
| `STT_BACKCHANNEL_BOUNDARY_MODEL` | (空) | 言い終わりの区切りを判定する言語モデル（Hugging Face のモデル名か、モデルのディレクトリ）。空なら判定しない。読み込めなければ、ログに残して判定なしで起動する |
| `STT_BACKCHANNEL_BOUNDARY_THRESHOLD` | `0.35` | 相槌を出す区切りらしさの下限。0 以上 1 以下。0 なら判定しない |
| `STT_BACKCHANNEL_JUDGE_TIMEOUT_MS` | `200` | 相槌を出すと決めてから、途中の文字起こしと区切りの判定を待つ上限（ミリ秒）。正の整数 |
| `STT_BACKCHANNEL_PHRASES` | `うん,うんうん,はい,なるほど` | 相槌の文言の候補（カンマ区切り）。1 つ以上 |
| `STT_ACK_OLLAMA_URL` | `http://127.0.0.1:11434` | 一時的な返事を作る Ollama の URL |
| `STT_ACK_MODEL` | `gemma4:e2b` | 一時的な返事を作る Ollama のモデル |
| `STT_ACK_TIMEOUT_MS` | `1500` | Ollama の応答を待つ上限（ミリ秒）。正の整数 |
| `STT_ACK_MAX_CHARS` | `20` | 一時的な返事の文字数の上限。正の整数 |
| `STT_ACK_FALLBACK_TEXT` | `ちょっと確認するね。` | LLM の返事が使えないときに返す文言。空にはできない |
| `STT_ACK_KEEP_ALIVE` | `30m` | Ollama にモデルを GPU に置いたままにさせる時間（Ollama の `keep_alive` の書式） |
| `STT_ACK_LOAD_TIMEOUT_MS` | `60000` | Ollama にモデルを読み込ませるときに待つ上限（ミリ秒）。正の整数 |

範囲外の値、または数値として読めない値の場合は、起動時に `ValueError` で停止する。

Haas-ja は常に GPU（`cuda`）・bfloat16 で動かす。haas-asr が使う flash-linear-attention が
Triton と CUDA を必要とし、公表されている精度と速度も bfloat16 で測られているため。

### 2.2 GET /health

認証不要。モデルは読み込まず、設定上の値を返す。

| フィールド | 型 | 内容 |
|---|---|---|
| `ok` | bool | 常に `true` |
| `backend` | `"whisper"` \| `"haas"` | 使用中のバックエンド |
| `model` | string | whisper は `STT_MODEL`、haas は `STT_HAAS_MODEL` |
| `device` | string | whisper は `STT_DEVICE`、haas は `cuda` |
| `compute_type` | string | whisper は `STT_COMPUTE_TYPE`、haas は `bfloat16` |

### 2.3 POST /transcribe

`multipart/form-data`。`Authorization: Bearer <token>` ヘッダが必須。

| パラメータ | 必須 | 説明 |
|---|---|---|
| `file` | ○ | 音声ファイル |
| `language` | | 言語コード。省略時は `STT_LANGUAGE` |

対応フォーマットは拡張子で判定する（`formats.py` の `ALLOWED_SUFFIXES`）。ファイル名や拡張子が
無い場合は `.wav` として扱う。

| 拡張子 | 内容 |
|---|---|
| `.wav` | WAV（無圧縮） |
| `.ogg` / `.opus` | Opus（Ogg コンテナ） |
| `.webm` | Opus（WebM コンテナ） |

応答：

| フィールド | 型 | 内容 |
|---|---|---|
| `ok` | bool | 常に `true` |
| `text` | string | 文字起こし結果（前後の空白は除く） |
| `language` | string | whisper は Whisper が判定した言語、haas は指定された言語（`ja`） |
| `duration` | number \| null | 音声の長さ（秒） |
| `latency_ms` | int | 文字起こしにかかった時間（ミリ秒）。同じワーカーでの順番待ちの時間を含む |
| `backend` | `"whisper"` \| `"haas"` | 使用したバックエンド |
| `model` | string | 使用したモデル |

エラー：

| ステータス | 条件 | `detail` |
|---|---|---|
| `400` | 対応外の拡張子 | `unsupported audio format: '.mp4' (allowed: .ogg, .opus, .wav, .webm)` |
| `400` | バックエンドが対応しない言語（haas で `ja` 以外） | `unsupported language for backend 'haas': 'en' (supported: ja)` |
| `401` | ヘッダが無い | `Not authenticated` |
| `401` | トークンが一致しない | `invalid_token` |
| `500` | モデルの読み込みまたは文字起こしに失敗 | `transcribe_failed: <理由>` |

haas で extras が入っていない場合、理由は
`` STT_BACKEND=haas requires the 'haas' extra: run `uv sync --extra haas` `` になる。

### 2.4 WebSocket /stream

音声をストリーミングで受け取り、相槌（`backchannel`）・文字起こし結果（`transcript`）・
一時的な返事（`ack`）をイベントとして返す。1 つの接続で、複数の発話を順に扱える。

#### 接続と認証

- `Authorization: Bearer <token>` ヘッダが必須。検証は `/transcribe` と同じ（`STT_API_KEY` が
  空のときはヘッダの有無だけを見る）。
- ヘッダが無い、またはトークンが一致しない場合は、WebSocket を確立せず HTTP `403` で拒否する。
- 同時接続数に上限は設けない。相槌の予測ができないとき（下記）も接続は受け付け、
  その発話を相槌なしで扱う。`transcript` と `ack` はいつもどおり送る。
  - 同時に相槌を予測している発話が `STT_BACKCHANNEL_MAX_SESSIONS` に達している（`busy`）。
    相槌を出す発話はそれぞれ、無音のたびに途中の文字起こしと区切りの判定を GPU で順番に待つ。
    増やしすぎるとその待ちが重なり、全員の相槌が `STT_BACKCHANNEL_JUDGE_TIMEOUT_MS` に間に合わなくなるため。
  - extras `stream` が入っていない、または予測モデルを読み込めない（`unavailable`）。
- 相槌なしかどうかは、発話ごとに `started` の `backchannel` で知らせる。

#### クライアントから送るメッセージ

| 種類 | 形式 | 内容 |
|---|---|---|
| 発話の開始 | テキスト（JSON） | `{"type": "start", "encoding": "opus", "language": "ja", "ack": {...}, "backchannel": {...}}`。`type` 以外は省略可（省略時は `opus`・`STT_LANGUAGE`・サーバの設定）。`ack` と `backchannel` は下の「発話ごとの指定」 |
| 音声 | バイナリ | `encoding` に従う（下表） |
| 発話の終了 | テキスト（JSON） | `{"type": "end"}` |

音声の `encoding`：

| `encoding` | 1 メッセージの中身 | 補足 |
|---|---|---|
| `opus` | Opus のパケット 1 つ（コンテナなしの生パケット） | モノラル。フレーム長（2.5〜60ms）とビットレートは自由。エンコード時のサンプリングレートは問わない（デコーダは常に 48kHz で復元し、16kHz に変換する） |
| `pcm_s16le` | 16kHz・モノラル・符号付き 16bit リトルエンディアンの PCM | 長さは自由。バイト数は偶数 |

- 車載など回線の細いクライアントのため、`opus` を既定にする。音声 1 秒あたりのデータ量は、
  24kbps の Opus で約 3KB、`pcm_s16le` で 32KB。
- Opus のパケットは送った順にデコードする。パケットを抜かすと、その直後の音声が乱れる。

- 音声は `start` から `end` までの間だけ受け付ける。
- `end` を送った後、その発話の結果を待たずに次の `start` を送ってよい。
- 言語は `ja` のみ対応する。相槌の文言と区切りの判定を、日本語の話し言葉で調整しているため。

#### 発話ごとの指定（`start` の `ack` と `backchannel`）

一時的な返事と相槌は、どんな言い方が合うかがクライアントによって違う。たとえば、この後に
エージェントの本回答が続くクライアントでは、ack が内容を受け止めると本回答と同じ受け止めが
2 回続くため、中身のない「つなぎ」の一言が合う。そのため、発話ごとにクライアントが指定できる。
指定しない項目は、サーバの設定（2.1）と既定のプロンプト（下の「一時的な返事の作り方」）を使う。

`ack`：`false` を指定すると、その発話では一時的な返事を作らない（`transcript` で終わる）。
オブジェクトの場合のフィールドは次のとおり。

| フィールド | 型・範囲 | 省略時 |
|---|---|---|
| `system_prompt` | 文字列、1〜2,000 文字 | 既定のシステムプロンプト |
| `examples` | `{"user": 1〜200 文字, "reply": 1〜100 文字}` の配列、0〜8 個 | `system_prompt` も省略したときは既定の会話例、`system_prompt` を指定したときは会話例なし |
| `fallback_text` | 文字列、1〜50 文字 | `STT_ACK_FALLBACK_TEXT` |
| `max_chars` | 整数、1〜100 | `STT_ACK_MAX_CHARS` |

- `system_prompt` を指定して `examples` を省略したときに既定の会話例を使わないのは、既定の
  会話例が既定のシステムプロンプトの言い方（相手の言葉を繰り返す）に合わせてあり、別の
  プロンプトと混ぜると指示と例が食い違うため。
- 推論の設定（モデル、`think`、`temperature`、出力の上限）と、出力の整形・固定の文言への
  差し替えの規則は、指定によらず同じ。
- クライアントが書くプロンプトも、既定と同じく「答えを知らない聞き手」として書き出す必要がある。
  「音声アシスタントのつなぎ役」として書いた版では、`gemma4:e2b` が 6 件中 3 件を韓国語で返し、
  天気の質問に「明日の天気は晴れだよ。」と言い切った。サーバは出力の言語や内容の正しさを
  検証しないため、README にプロンプトの書き方を示す。

`backchannel`：

| フィールド | 型・範囲 | 省略時 |
|---|---|---|
| `phrases` | 文字列（1〜20 文字）の配列、1〜10 個 | `STT_BACKCHANNEL_PHRASES` |
| `silence_ms` | 整数、32〜2,000 | `STT_BACKCHANNEL_SILENCE_MS` |
| `min_new_speech_ms` | 整数、0〜10,000 | `STT_BACKCHANNEL_MIN_NEW_SPEECH_MS` |
| `min_interval_ms` | 整数、0〜60,000 | `STT_BACKCHANNEL_MIN_INTERVAL_MS` |
| `interval_per_minute_ms` | 整数、0〜60,000 | `STT_BACKCHANNEL_INTERVAL_PER_MINUTE_MS` |
| `interval_factor` | 数値、1.0〜3.0 | `STT_BACKCHANNEL_INTERVAL_FACTOR` |
| `max_interval_ms` | 整数、0〜60,000 | `STT_BACKCHANNEL_MAX_INTERVAL_MS` |
| `boundary_threshold` | 数値、0〜1。0 なら区切りを判定しない | `STT_BACKCHANNEL_BOUNDARY_THRESHOLD` |

- 範囲外の値、型の誤り、未知のフィールド（`system_promt` のような綴りの誤りを黙って無視しない
  ため）は `invalid_message` になり、その発話は始まらない。
- 文言は接続の中で順番を引き継ぐ。前の発話と `phrases` が違う場合は、新しい `phrases` の
  先頭から使う。送らなかった候補（言いかけの途中、発話の終了後に判定が終わったものなど）では
  順番を進めない。
- 予測器の上限（`STT_BACKCHANNEL_MAX_SESSIONS`）はクライアントから変えられない。

#### サーバが送るイベント（テキスト・JSON）

すべてのイベントに `type` と、発話の通し番号 `utterance_id`（接続ごとに 1 から）を付ける。
`error` のうち特定の発話に結び付かないものは `utterance_id` が `null` になる。

| `type` | 送るとき | フィールド |
|---|---|---|
| `started` | `start` を受け付けたとき | `utterance_id`、`backchannel`（この発話で相槌を出すか。bool）、`backchannel_unavailable_reason`（`backchannel` が `false` のとき `"busy"` \| `"unavailable"`、それ以外は `null`） |
| `backchannel` | 発話中、相槌を出すと決めたとき | `utterance_id`、`text`（相槌の文言）、`offset_ms`（発話の先頭からの音声上の位置。int）、`probability`（相槌を出したフレームの話していない確率） |
| `transcript` | `end` の後、文字起こしが終わったとき | `utterance_id`、`text`、`language`、`duration`、`latency_ms`、`backend`、`model`（各フィールドの意味は `/transcribe` の応答と同じ） |
| `ack` | `transcript` の後、一時的な返事ができたとき | `utterance_id`、`text`、`source`（`"llm"` \| `"fallback"`）、`latency_ms`（`transcript` を送ってから `ack` を送るまで） |
| `error` | 下表のとき | `utterance_id`、`code`、`detail` |

- 1 つの発話について、`transcript` は必ず `ack` より先に送る。一時的な返事は文字起こしの結果から
  作るため。
- 文字起こしの結果が空（無音など）の場合と、`start` で `"ack": false` を指定した場合は、`ack` は送らない。
- 音声を 1 つも送らずに `end` を送った発話は、文字起こしせずに `text` が空・`duration` が 0 の
  `transcript` を返す。
- `backchannel` は `start` から `end` までの音声に対してだけ送る。`end` の後に予測が
  届いても送らない。発話の区切りをクライアントが決めるため、クライアントが読み上げている
  間（発話中でない間）に相槌は出ない。

`error` の `code`：

| `code` | 条件 | 接続 |
|---|---|---|
| `invalid_message` | JSON として読めない、`type` や `encoding` が不明、`ack`・`backchannel` の指定が範囲外・未知のフィールド | 継続（`start` ならその発話は始まらない） |
| `invalid_audio` | Opus のパケットをデコードできない、`pcm_s16le` で奇数バイト | 継続（そのメッセージの音声は捨てる） |
| `invalid_state` | 発話中に `start`、発話中でないときに音声または `end` | 継続 |
| `unsupported_language` | `ja` 以外の言語 | 継続（その発話は始まらない） |
| `utterance_too_long` | 1 発話の音声が `STT_STREAM_MAX_UTTERANCE_SECONDS` を超えた | 継続（その発話は破棄し、`end` まで音声を捨てる） |
| `transcribe_failed` | 文字起こしに失敗（`detail` は `/transcribe` の `500` と同じ） | 継続（その発話の `ack` は送らない） |

一時的な返事を LLM で作れなかった場合は `error` にせず、`source: "fallback"` の `ack` を送る。

#### 相槌の決め方

Silero VAD が 32ms ごとに「話していない確率」を出し、0.5 以上を無音とみなす。
最後に話したフレームから無音が `backchannel.silence_ms`（省略時は `STT_BACKCHANNEL_SILENCE_MS`）
以上続いたフレームで相槌を出す。

- 1 つの無音では 1 回だけ出し、次は話し手がまた話してから出す。発話の先頭の無音
  （クライアントが話し始めの前から送る音声）では、まだ何も話していないため出さない。
- 前の相槌（無ければ発話の先頭）から新しく話した長さ（話しているフレームの合計）が
  `backchannel.min_new_speech_ms`（省略時は `STT_BACKCHANNEL_MIN_NEW_SPEECH_MS`）に足りなければ出さない。
  同じ言葉のまま間が空いただけの場所で、もう一度相槌を打たないため。
- 無音になった最初のフレームで、途中の文字起こしでの判定を先に始める（区切りを判定する発話のとき）。
  無音を待つ時間に判定の時間を重ねるため。同じ無音で相槌を出すときはその結果を使い、途中で
  話し始めたら使わない（次の無音で判定し直す）。
- タイミングを無音の長さで決めるのは、相槌の予測モデル（MaAI `vap_bc_jp`）より切れ目に合うため。
  ARGOS の音声（4 発話、200ms 以上の切れ目 27 個）では、MaAI の相槌 10 回がすべて話の途中
  （息継ぎ）で出たのに対し、無音（90ms）は 23 回のうち 20 回が切れ目の中に出た。切れ目の始まりからの
  遅れは中央値 211ms（Silero VAD は話し終わりから無音と判定するまでに 60〜90ms かかる）。
  RMS（音量）で無音を判定しないのは、車内のロードノイズなどで声と雑音の大きさが近いと
  誤判定が増えるため（人工の低域雑音で SNR 10dB にすると、誤反応が 5 回から 20 回に増えた。
  Silero VAD は SNR 0dB でも 3 回）。
- 送った相槌から最短の間隔が経っていなければ出さない。`offset_ms`（音声上の時刻）で測る。
  間隔と新しく話した長さは、送った相槌からだけ数え直す。送らなかった候補（言いかけの途中、
  時間切れなど）で数え直すと、言いかけで黙った直後の区切りで相槌を出せなくなるため。候補を送るかどうかが決まるまでは、次の候補を出さない。
  最短の間隔は、相槌を出そうとする位置（発話の先頭からの経過）と、それまでに出した相槌の回数で伸ばす。
  人は相手の話が長くなるほど相槌を減らすため。

  ```
  間隔 = min(上限, (min_interval_ms + 経過分 × interval_per_minute_ms) × interval_factor ^ n)
  ```

  - n は前の相槌より前に出した相槌の数（1 回目と 2 回目の間は n = 0）。
  - 上限は `max_interval_ms`。`min_interval_ms` より小さいときは `min_interval_ms`。
  - 各項目は `backchannel` で指定し、省略時は `STT_BACKCHANNEL_MIN_INTERVAL_MS`・
    `STT_BACKCHANNEL_INTERVAL_PER_MINUTE_MS`・`STT_BACKCHANNEL_INTERVAL_FACTOR`・
    `STT_BACKCHANNEL_MAX_INTERVAL_MS`。
  - 既定（1 分ごとに 3 秒、1.1 倍、上限 8 秒）で、ARGOS の 60 秒の発話で無音のタイミングが相槌を出した
    17 か所（`min_interval_ms` 2,500）に当てはめると、9 回に減り、間隔は 3.5 秒から 10 秒前後へ広がる。
- 文言（`backchannel.phrases`、省略時は `STT_BACKCHANNEL_PHRASES`）を順に使う。
  同じ文言が続かないようにするため。文言を話の内容から選ばないのは、ARGOS の録音で相槌が自然な
  35 か所すべてに「うん」が合い、内容から選ぶ仕組み（Kev-0.8B で話の種類を判定する）では
  6 か所で相槌が消え、4 か所で合わない文言を選んだため。
- 言い終わりの区切りを判定する発話（`STT_BACKCHANNEL_BOUNDARY_MODEL` を設定し、
  `boundary_threshold` が 0 より大きい）では、相槌を出すと決めたフレームまでの直近 15 秒の音声を
  文字起こしし、区切りらしさが `boundary_threshold` より小さければ相槌を出さない。
  - 直近 15 秒にするのは、発話の頭から文字起こしすると、長い話ほど判定が遅れて
    `STT_BACKCHANNEL_JUDGE_TIMEOUT_MS` に間に合わなくなるため（50 秒を過ぎると 150〜175ms かかった）。
    区切りの判定が見るのは文字起こしの末尾で、前の文脈は数文あれば足りる。
  - 区切りらしさは、文字起こし（末尾の記号と空白を除く）の続きに、言語モデルが区切りのトークン
    （空白・改行だけ、または記号で始まるもの）を予測する確率の合計。言いかけ（「なので」
    「〜の方で」）の後には言葉が続きやすく、言い終わり（「〜なんだよね」）の後には句読点が
    続きやすい。その違いをモデルの予測から読む。
  - 記号は Unicode の分類（P で始まる分類）で見分け、特定の記号や言葉は並べない。書き起こしや
    モデルの語彙に出る記号の種類に左右されないため。
  - 無音と文字だけで決める理由と限界：ARGOS の録音 25 本（文字起こしを最後まで読んで付けた
    正解で、相槌が自然 35 か所、黙るべき 85 か所）では、無音の長さは両者で分かれなかった
    （中央値 666ms と 474ms）。区切りらしさは言いかけを弾けるが、「〜じゃない」「〜ので」の
    ように続きも書ける区切りは低く出る。境目を決めたのとは別の 18 本では、今の決め方が
    自然 24 か所中 14・黙るべき 59 か所中 18 で打ったのに対し、区切りらしさ 0.35 以上・
    新しく話した長さ 1 秒以上では 11 と 3 だった。相槌は減るが、言いかけではほぼ打たない。
  - モデルは `Qwen3.5-0.8B-Base` で確かめた。GPU で 1 回約 20〜25ms、メモリは約 1.5GB。
    Ollama 経由では 1 回約 220ms かかる（モデルの計算は 10〜60ms で、残りは呼び出しごとの待ち）
    ため、gateway のプロセスに直接読み込む。モデルは同時に使えないため 1 件ずつ、別スレッドで動かす。
  - 区切りを判定できなかったとき（モデルの失敗）は、相槌が出なくなるのを避けるため、
    区切りに届いているとみなす。途中の文字起こしが失敗した・空だったときは出さない。
  - 途中の文字起こしと判定に `STT_BACKCHANNEL_JUDGE_TIMEOUT_MS` より長くかかったら、その相槌は
    出さない。遅れて出すと話の続きにかぶるため。
  - 時間切れでも途中の文字起こしは止めない。推論のスレッドは途中で止められず、待たずに止めると
    `SerialTranscriber` の順番待ちを抜けて GPU 上の推論が重なるため。結果は捨てる。
  - 起動スクリプトは `scripts/logging.json` で `stt_gateway` の INFO ログを有効にする。
    判定した相槌の候補ごとに、発話番号、音声位置、区切りらしさ（`boundary`）、理由、
    先に始めた判定を使ったか（`prepared`）、相槌を出すと決めてからの待ち時間を残す。
    理由は `sent`、`not_boundary`（言いかけの途中）、`timeout`、`judge_unavailable`（途中の
    文字起こしが得られない）、`closed`（発話の終了後）。発話本文と認証トークンは含めない。

#### 発話の録音

`STT_STREAM_RECORDING_DIR` を指定したときだけ、相槌を予測した発話（`end`・上限超過・切断で
終わったもの）ごとに 2 つのファイルを保存する。相槌を出した場所・黙った場所を、実際の音声と
突き合わせて調べ、相槌を打つ場所を音と文字から学習する材料にするため。

- ファイル名は `<発話の開始時刻 YYYYMMDD-HHMMSS-マイクロ秒>-u<utterance_id>` に `.wav` / `.json` を付ける。
  接続ごとに `utterance_id` は 1 から数え直すため、時刻で重ならないようにする。
- `.wav`：受け取った音声をデコードした 16kHz モノラル 16bit。予測器と文字起こしに渡したものと同じ。
- `.json`：`utterance_id`、`recorded_at`（発話の開始時刻）、`sample_rate_hz`、`duration_ms`、
  `silence_ms`・`min_new_speech_ms`、
  `min_interval_ms`・`interval_per_minute_ms`・`interval_factor`・`max_interval_ms`、
  `boundary_threshold`、`frames`（毎フレームの `offset_ms` と、話していない確率 `probability`）、
  `prepares`（判定を先に始めたフレーム）、`fires`（相槌を出すと決めたフレームと
  `has_prepared`）、`decisions`（途中の文字起こしで判定した候補ごとの `offset_ms`、`reason`、
  その時点の文字起こし `text`、`boundary_probability`。判定が間に合わなかった項目は null）。
  `decisions` に文字起こしと区切りらしさを残すのは、相槌を打つ場所を録音から学習するときの材料にするため。
- 予測と候補の判定が終わってから保存する。保存は別スレッドで行い、失敗しても発話の処理は続け、
  ログに保存先と一緒に残す。
- 録音には話した内容がそのまま入るため、ディレクトリは `0700`、ファイルは `0600` で作る。
  古いファイルは自動では消さない。

#### 一時的な返事の作り方

Ollama の `/api/chat` に、システムプロンプトと会話例を付けて文字起こし結果を渡す。
クライアントが `start` の `ack` で指定しなければ、次の既定のものを使う。
「アシスタント」として指示すると小型の LLM は質問に答えようとし、事実でない内容を言い切るため、
答えを知らない「聞き手」として相槌を打たせる。

既定のシステムプロンプト：

> あなたは会話の聞き手です。相手の話を聞いて、相槌を一言だけ打ちます。あなたは答えを知らないので、質問や依頼には答えず「調べるね」「確認するね」のように受け止めるだけにします。相手の言葉を少しだけ繰り返してもよいです。15文字以内、くだけた口調、1文だけ。

既定の会話例（ユーザー → 返事）：

| ユーザー | 返事 |
|---|---|
| 今日のニュースで何か大きいのあった？ | ニュースね、ちょっと見てみるね。 |
| 昨日ずっと雨でさ、出かけられなかったんだよね | そっかあ、雨だったんだ。 |
| この関数どこで呼ばれてるんだっけ | 呼び出し元ね、探してみる。 |
| お疲れさま、今日はもういいよ | うん、お疲れさま！ |

- 推論の設定：思考モードを使わない（`think: false`）、`temperature` 0.3、出力の上限 40 トークン。
- 出力は前後の空白と引用符（`「」"'`）を取り除き、最初の 1 行だけを使う。
- 次の場合は固定の文言（`ack.fallback_text`、省略時は `STT_ACK_FALLBACK_TEXT`）を
  `source: "fallback"` で返す。
  - `STT_ACK_TIMEOUT_MS` 以内に応答が無い、接続できない、応答が読めない
  - 整形後の文字数が 0、または文字数の上限（`ack.max_chars`、省略時は `STT_ACK_MAX_CHARS`）を超える

## 3. 処理の流れ

1. ワーカーの起動時（lifespan）に、設定されたバックエンドのモデルを読み込む。
   読み込みに失敗してもワーカーは起動を続け、失敗をログに出す。その後の `/transcribe` で
   読み込みを再試行し、失敗すれば `500` で理由を返す（extras 未導入などを利用者が知れるように）。
2. 拡張子を判定し、アップロードされた内容を同じ拡張子の一時ファイルに書き出す。
3. 同じワーカーで処理中の文字起こしがあれば、終わるまで待つ。
4. バックエンドで文字起こしする。一時ファイルは成否にかかわらず削除する。

### 3.0 並列処理

- 並列化はワーカープロセスを増やして行う（`STT_WORKERS`）。各ワーカーがモデルを 1 つずつ持つ。
  プロセス間で共有する状態が無いため、どちらのバックエンドでも同じ方法で並列にできる。
  GPU メモリはワーカー数に比例して必要になる。
- ワーカー内では推論を 1 件ずつ実行する（`SerialTranscriber`）。haas（NeMo）は `transcribe()` の
  たびに前処理の dither・`pad_to` やモジュールの freeze 状態を書き換えて元に戻し、デコーダの
  CUDA Graph も固定バッファを共有するため、同じモデルを複数スレッドから同時に呼べない。
- 推論とモデルの読み込みは別スレッドで実行し、イベントループを止めない。推論中も `/health` や
  ほかのリクエストのアップロード受信は処理される。順番待ちはイベントループ上のロック
  （`asyncio.Lock`）で行い、待っているリクエストがスレッドプールを占有しないようにする。
- 混雑時もリクエストを拒否せず、順番に処理する。ワーカー間の振り分けは OS の `accept` に任せる
  ため、空いているワーカーがあっても処理中のワーカーに割り当てられ、待つことがある。
- `STT_MAX_REQUESTS` が 1 以上のとき、uvicorn の `--limit-max-requests` で、指定件数を処理した
  ワーカーを終了させる。メモリが肥大化しても、プロセスごと解放して元に戻すため。
  - 終了するワーカーは処理中のリクエストを終えてから終了する。
  - ワーカーが 2 つ以上なら uvicorn が新しいワーカーを起動する。ワーカーが 1 つのときは
    プロセス全体が正常終了するため、systemd の `Restart=always` で起動し直す。
  - 件数は uvicorn がすべての HTTP リクエスト（`/health` を含む）で数える。
  - `STT_MAX_REQUESTS_JITTER` でワーカーごとに終了の件数をずらし、全ワーカーが同時に
    モデルを読み込み直して処理が止まることを避ける。

### 3.1 whisper バックエンド

faster-whisper にファイルパスを渡す。デコードは faster-whisper 内部の PyAV(ffmpeg) が行う。
VAD（発話区間検出）を有効にし、しきい値は 0.3 とする（既定の 0.5 より小さな声を拾う）。

### 3.2 haas バックエンド

1. 言語が `ja` 以外なら、デコードの前に `400` で拒否する。Haas-ja は日本語専用のため。
2. `audio_decoding.py` で、PyAV を使って 16kHz モノラル float32 の波形にする。
   - haas-asr 付属の `load_audio` は libsndfile を使い WebM を読めない。whisper バックエンドと
     対応フォーマットをそろえるため PyAV を使う。
   - faster-whisper の `decode_audio` は使わない。同じ処理だが最後に `gc.collect()` を呼び、
     NeMo と torch を読み込んだプロセスではそれだけで 1 回 200ms 以上かかるため。
   - ステレオは ffmpeg の標準の方法でモノラルにまとめる（faster-whisper と同じ結果になる）。
   - デコードに失敗した場合は `500`（`transcribe_failed: failed to decode audio: <パス>: <理由>`）。
3. `haas_asr.transcribe_longform` で文字起こしする。Haas-ja は 30 秒以下の発話で学習されており、
   長い音声は無音位置で分割して推論する必要があるため。30 秒以下の音声はそのまま 1 チャンクになる。

Haas-ja の出力は「、」はある程度付くが「。」はほとんど付かない。フィラー（「えー」など）の
扱いは実音声で確認していない。

### 3.3 /stream

1. 接続時に認証を確認する。
2. `start` を受けたら、発話の音声バッファを用意し、`backchannel_pool.py` から予測器を借りて
   状態を初期化し、`started` を返す。上限は `STT_BACKCHANNEL_MAX_SESSIONS`。
   - 予測器は発話の間だけ借り、`end` または切断で返す。接続したまま話していないクライアントが
     枠を握り続けないため。
   - 状態を初期化するのは、前の発話の文脈を次の発話の予測に持ち越さないため。
   - 作った予測器は、モデルを読み込み直さないよう捨てずに使い回す。初めて借りられたときに作る。
   - 借りられなければ、その発話は相槌なしで進める。
   - その発話で一時的な返事を作る場合は、Ollama に一時的な返事のモデルを読み込ませる要求を裏で送る
     （下の「一時的な返事のモデルの読み込み」）。
3. 音声を受けるたびに、`stream_audio_decoder.py` で 16kHz モノラルの波形にし、バッファへ足し、予測器へ渡す。
   - Opus は PyAV（ffmpeg の libopus）でパケットごとにデコードする。デコーダは前のパケットの
     状態を使うため、発話ごとに作り直す。1 パケット（20ms）のデコードは 0.1ms 未満で、
     遅延にはほぼ影響しない。
   - 予測器（Silero VAD）は、512 サンプル（32ms）ごとに 1 回推論し、RNN の状態と直前の 64 サンプルを
     発話の中で引き継ぐ。1 回の推論は 0.1ms 程度で、onnxruntime のスレッドは 1 つにする。
   - 予測器は別スレッドで推論し、結果をイベントループへ渡す。推論中も音声の受信を止めない。
   - 相槌を出すかどうかは `backchannel_triggers.py` と `backchannel_emitters.py` が決める
     （2.4 の「相槌の決め方」）。
   - 区切りを判定する発話では、判定を先に始めるときと相槌を出すと決めたときに、
     途中の文字起こしと区切りの判定を別のタスクで行う。予測の処理はその完了を待たずに続ける。
4. `end` を受けたら、その発話の予測器への入力を止め、バッファを WAV の一時ファイルに書き出して
   `SerialTranscriber` で文字起こしする。`/transcribe` と同じ順番待ちに入るため、GPU 上の推論が
   重ならない。一時ファイルは成否にかかわらず削除する。
5. `transcript` を送った後、文字起こし結果から一時的な返事を作り、`ack` を送る。
   文字起こしと一時的な返事の作成は、次の発話の受信と並行して進める。
6. 切断されたら、予測器を返し、送り終えていないイベントは捨てる。

- 一時的な返事のモデルの読み込み：ワーカーの起動時（lifespan）と、一時的な返事を作る発話の
  `start` を受けたときに、Ollama の `/api/generate` へ `prompt` を付けずに `model` と `keep_alive` だけを
  送る。Ollama は推論せずにモデルを GPU に読み込み、読み込み済みなら `keep_alive` を延ばすだけで
  すぐに応答する。
  - 一時的な返事のモデルは、`keep_alive` が切れたり他のモデルに追い出されたりすると GPU から外れる。
    外れた状態からの読み込みには数秒〜十数秒かかり（gemma4:e2b で約 8〜15 秒）、
    `STT_ACK_TIMEOUT_MS` に収まらない。
  - 一時的な返事の要求は `STT_ACK_TIMEOUT_MS` で切断し、そのとき Ollama は読み込みも中止する。
    そのため、読み込みは返事とは別の要求で `STT_ACK_LOAD_TIMEOUT_MS` まで待つ。返事が時間切れで
    固定の文言になっても読み込みは続き、次の発話の返事に間に合う。
  - 話している間に読み込みを済ませるため、`end` ではなく `start` で始める。読み込みが発話の長さ・
    文字起こしの時間・`STT_ACK_TIMEOUT_MS` の合計より長くかかると、その発話の返事は固定の文言になる。
  - 読み込みの要求はワーカーごとに同時に 1 本までとし、読み込み中に来た `start` では送らない。
    要求は接続とは別に持ち、切断されても取り消さない。ワーカーの終了時に取り消す。
  - 読み込みにかかった時間をログ（`ack model preloaded`）に残す。失敗してもログに出すだけで、
    発話の処理と起動は続ける。
- 予測器（Silero VAD）は CPU だけで動き、モデルは silero-vad のパッケージに同梱されている
  （ダウンロードしない）。区切りの判定の言語モデルは GPU を使い、ワーカーの起動時に読み込む。
- `/stream` の接続は `STT_MAX_REQUESTS` の件数に数えない（uvicorn は WebSocket の接続を数えない）。
  ワーカーの入れ替えでワーカーが終了するとき、接続中の `/stream` はクローズコード `1012` で
  切断される。クライアントは接続し直す。

## 4. 依存関係

| 区分 | 内容 |
|---|---|
| 必須 | fastapi、uvicorn、pydantic、python-multipart、python-dotenv、faster-whisper、av、numpy、websockets、httpx、nvidia-cublas-cu12（Linux） |
| extras `haas` | haas-asr（GitHub のコミット固定）、torch 2.14.0。Python 3.11 以上かつ Linux のみ |
| extras `stream` | silero-vad、onnxruntime。`/stream` で相槌を出す場合だけ必要 |
| extras `haas`（区切りの判定） | torch と transformers（haas-asr 経由）。`STT_BACKCHANNEL_BOUNDARY_MODEL` を使う場合だけ必要 |
| 外部サービス | Ollama（一時的な返事を作る LLM）。`/stream` を使う場合だけ必要 |

- **websockets**：uvicorn が WebSocket を扱うために必須の依存に入れる。extras `stream` が無くても
  `/stream` で文字起こしと一時的な返事を使えるようにするため。Opus のデコードは必須の av で行う。
- **httpx**：Ollama を非同期に呼ぶため。イベントループを止めずに、ほかの接続の音声を受け続ける。
- **silero-vad**：同梱の ONNX モデル（`silero_vad/data/silero_vad.onnx`、MIT）を onnxruntime で
  直接動かす。パッケージの推論関数は torch のテンソルを受け渡すが、1 チャンクごとに呼ぶため
  numpy のまま渡す。onnxruntime は直接使うため extras `stream` に明示する。silero-vad は torch に
  依存するため、extras `stream` は haas と同じくらい重い。
  faster-whisper にも Silero VAD のモデルが同梱されているが、音声全体をまとめて判定する形で、
  1 チャンクずつ状態を引き継いで呼ぶと確率が silero-vad と最大 0.83 ずれたため使わない。
- **`uv sync` は extras を指定し直す**：`uv sync` は指定しなかった extras のパッケージを削除する。
  haas と stream を両方使う環境では `uv sync --extra haas --extra stream` とする。
  extras `stream` から外したパッケージ（以前の依存）も、このとき削除される。
  `scripts/run_server.sh` の `uv run` は余分なパッケージを削除しないため、入れた extras は残る。

- **nvidia-cublas-cu12**：CTranslate2 は実行時に CUDA 12 の `libcublas.so.12` を読み込む。
  CUDA 13 だけのホストにはこれが無く、最初の推論で `CUBLAS_STATUS_NOT_SUPPORTED` になる。
  `scripts/run_server.sh` がこのパッケージのライブラリを `LD_LIBRARY_PATH` に追加する。
- **haas-asr のコミット固定**：PyPI に無く、NeMo の非公開 API に依存するため、動作確認した
  コミットに固定する。
- **torch の固定**：haas-asr 側に上限が無く、そのままでは新しい版が選ばれる。精度と速度を
  確認した 2.14.0 に固定する。
- **`[tool.uv] override-dependencies`**：haas-asr の uv.lock と同じ上書き（fsspec、hydra-core、
  packaging、transformers）。上書きはプロジェクト全体に効くため、whisper だけの環境でも
  fsspec と packaging はこの下限以上になる。
- **`[tool.uv] compile-bytecode = true`**：NeMo 2.7.3 の CUDA グラフ用デコーダは、実行時に
  コンパイルされると共有の `b" "` を書き換え、後からコンパイルされたモジュールの定数を壊す。
  インストール時に全モジュールをバイトコード化して、実行時のコンパイルを起こさない。

`scripts/run_server.sh` は、uv 管理の環境で `STT_BACKEND=haas` のとき `uv run --extra haas` で
起動する。

## 5. 認証

FastAPI の `HTTPBearer` で `Authorization` ヘッダを必須にし、`STT_API_KEY` と一致するかを
検証する。`STT_API_KEY` が空のときはヘッダの有無だけを検証する。

`/stream` も同じ `Authorization` ヘッダで同じ検証をする。`HTTPBearer` は WebSocket に使えないため、
ヘッダの解釈とトークンの比較は `/transcribe` と共通の関数にまとめ、両方から呼ぶ。
トークンは URL のクエリでは受け取らない。アクセスログやプロキシのログに残るため。

## 6. テスト

モデルはダミーに差し替え、GPU とモデル本体なしで実行する。`SerialTranscriber` は、
同時に投げた推論が重ならないこと、推論中もイベントループが進むこと、読み込みの失敗を
次のリクエストで再試行することを検証する。haas バックエンドは
`haas_asr` モジュールをダミーに差し替えて、読み込み・長尺処理の呼び出し・extras 未導入時の
エラーを検証する。カバレッジが 80% を下回るとテストが失敗する。

`/stream` は、予測器・区切りの判定のモデル・Ollama をダミーに差し替えて、GPU なしで実行する。
次を検証する。

- 認証の失敗
- 予測器の上限に達したとき・extras が無いときに、相槌なしで `transcript` と `ack` が届くこと
- メッセージの順序の誤り（発話中の `start` など）と、各 `error` の `code`
- 1 発話で `started` → `backchannel` → `transcript` → `ack` の順にイベントが届くこと
- 相槌の決め方（クライアントが指定した無音の長さ、最短の間隔、文言の順番、`end` 後の予測を送らないこと）
- 最短の間隔が経過時間と相槌の回数で伸び、上限で止まること（上限が最短の間隔より小さいときは最短の間隔）
- 間隔と新しく話した長さを送った相槌から数え、送らなかった候補では数え直さないこと。候補の結果が決まるまで次の候補を出さないこと
- 区切りらしさが境目より小さい候補では相槌を出さず、境目以上なら出すこと。境目が 0 の発話ではモデルを動かさないこと。
  モデルを読み込めない・判定に失敗したときは判定なしで相槌を出すこと。時間切れ・途中の文字起こしの
  失敗では出さないこと。送らなかった候補で文言の順番を進めないこと。区切りのトークンの見分け方と、
  確率の合計（ダミーの語彙とモデルで、実際の torch を使って確かめる）
- 判定に使う途中の音声が、直近の長さで切られること
- 黙り始めたフレームで始めた判定を、同じ無音の相槌でだけ使うこと
- 無音の決め方（無音が続いたら出す、1 つの無音で 1 回、先頭の無音で出さない、話し始めたら準備を捨てる）。
  Silero VAD の予測器は実際のモデルで、32ms ごとの位置と、無音を話していないと判定することを確かめる
- 録音を有効にした発話で、音声と毎フレームの確率・合図が保存されること、保存の失敗で止まらないこと
- `start` の `ack`・`backchannel` の指定が使われること、省略した項目がサーバの設定になること、
  範囲外の指定と未知のフィールド（`timing`・`phrases_by_kind` を含む）が `invalid_message` になること、
  `ack: false` で ack を送らないこと
- 一時的な返事の整形と、タイムアウト・接続失敗・文字数超過のときの固定文言への差し替え
- 一時的な返事のモデルの読み込み（読み込みだけの要求、同時に 1 本まで、返事が時間切れでも続くこと、
  失敗の記録、終了時の取り消し）と、`start` で読み込みを始めること（`ack: false` では始めない）
- Opus パケットと PCM のデコード（実際に PyAV でエンコードした Opus を使う）と、壊れたパケットの扱い

区切りの判定の言語モデルと Ollama を実際に使う確認は、自動テストとは別に手動で行う。
