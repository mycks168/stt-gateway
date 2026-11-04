# 変更履歴（決定事項と却下案）

## 2026-11-04 Haas-ja バックエンドの追加

### 背景

Third Intelligence 社が日本語音声認識モデル Haas-ja-v1.0 を公開した。RTX 5060 Ti で
large-v3-turbo と比較した（JSUT・FLEURS 各 200 件、1 発話ずつ推論）。

| モデル | データ | CER strict | CER kana | レイテンシ p50 |
|---|---|---|---|---|
| Haas-ja | JSUT | 4.38% | 0.49% | 28ms |
| large-v3-turbo | JSUT | 6.40% | 0.69% | 182ms |
| Haas-ja | FLEURS | 3.97% | 2.47% | 37ms |
| large-v3-turbo | FLEURS | 4.62% | 2.84% | 239ms |

VRAM のピークは約 3.6GB。読み上げ音声のみの評価で、フィラーや話し言葉は未評価。

### 決定事項

- `STT_BACKEND=whisper|haas` でバックエンドを切り替える。faster-whisper は残す。
- デフォルトは `whisper` とし、稼働中のサービスの動作を変えない。
- Haas-ja の依存（NeMo、torch）は extras `haas` に分ける。venv が約 8GB になるため。
- API の入出力の形は変えず、`/health` と `/transcribe` の応答に `backend` を追加する。
- 認証の環境変数は既存の `STT_API_KEY` のまま変えない。
- haas で `ja` 以外の言語が指定されたら `400` を返す。
- 音声のデコードは PyAV を直接呼ぶ。faster-whisper の `decode_audio` は内部で `gc.collect()` を
  呼び、NeMo を読み込んだプロセスでは 1 回 230ms ほどかかった（推論は約 30ms）。直接呼ぶと
  5〜8ms で、波形の差は最大 1.5e-5。

### 動作確認（テスト用サーバ、RTX 5060 Ti）

| バックエンド | 音声 | レイテンシ（2 回目以降） |
|---|---|---|
| haas | JSUT 4.9 秒（WAV / Opus / WebM） | 40〜47ms |
| haas | JSUT 10 発話を連結した 44 秒（WAV） | 106〜160ms |
| whisper | JSUT 4.9 秒（WAV / WebM） | 約 250ms |

初回のリクエストはモデルの読み込みを含み、haas で約 10 秒かかる。

### 却下した案

- **Whisper を Haas-ja で置き換える**：Haas-ja は日本語専用かつ NVIDIA GPU 専用で、
  多言語や CPU の環境で使えなくなるため。
- **haas-asr の `load_audio` を使う**：WebM を読めず、対応フォーマットが whisper と食い違うため。
  PyAV で直接デコードする。
