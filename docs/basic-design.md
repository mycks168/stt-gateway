# 基本設計

## 1. 全体構成

```
クライアント ──HTTP──▶ server.py（API・認証）
                          │ Transcriber（Protocol）
                          ▼
              backends.py（設定でバックエンドを選ぶ）
                 ├─ whisper_backend.py … faster-whisper
                 └─ haas_backend.py    … Haas-ja（haas-asr / NeMo）
                        └─ audio_decoding.py … PyAV でデコード
```

| モジュール | 責務 |
|---|---|
| `server.py` | HTTP API、Bearer 認証、一時ファイルの管理、例外から HTTP ステータスへの変換 |
| `config.py` | 環境変数（`.env` を含む）の読み込みと検証 |
| `formats.py` | 拡張子による音声フォーマットの判定 |
| `transcription.py` | バックエンドの共通インタフェース（`Transcriber`）、結果の型（`Transcription`）、例外 |
| `backends.py` | 設定に応じたバックエンドの読み込みと、`/health` 用の情報の算出 |
| `whisper_backend.py` | faster-whisper による文字起こし |
| `haas_backend.py` | Haas-ja による文字起こし |
| `audio_decoding.py` | PyAV(ffmpeg) による音声ファイルから波形へのデコード（haas が使う） |

依存の向きは `server → backends → 各バックエンド → transcription` の一方向とする。
`server.py` は faster-whisper や NeMo を直接 import しない。

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
| `latency_ms` | int | 文字起こしにかかった時間（ミリ秒） |
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

## 3. 処理の流れ

1. 拡張子を判定し、アップロードされた内容を同じ拡張子の一時ファイルに書き出す。
2. 初回のリクエストで、設定されたバックエンドのモデルを読み込む（以降は使い回す）。
   起動直後に `/health` だけを呼んでもモデルは読み込まれない。
3. バックエンドで文字起こしする。一時ファイルは成否にかかわらず削除する。

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

## 4. 依存関係

| 区分 | 内容 |
|---|---|
| 必須 | fastapi、uvicorn、pydantic、python-multipart、python-dotenv、faster-whisper、av、numpy、nvidia-cublas-cu12（Linux） |
| extras `haas` | haas-asr（GitHub のコミット固定）、torch 2.14.0。Python 3.11 以上かつ Linux のみ |

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

## 6. テスト

モデルはダミーに差し替え、GPU とモデル本体なしで実行する。haas バックエンドは
`haas_asr` モジュールをダミーに差し替えて、読み込み・長尺処理の呼び出し・extras 未導入時の
エラーを検証する。カバレッジが 80% を下回るとテストが失敗する。
