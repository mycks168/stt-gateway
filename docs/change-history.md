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
- Haas-ja の依存（NeMo、torch）は extras `haas` に分ける。venv が約 7.5GB になるため。
- API の入出力の形は変えず、`/health` と `/transcribe` の応答に `backend` を追加する。
- 認証の環境変数は既存の `STT_API_KEY` のまま変えない。
- haas で `ja` 以外の言語が指定されたら `400` を返す。

### 却下した案

- **Whisper を Haas-ja で置き換える**：Haas-ja は日本語専用かつ NVIDIA GPU 専用で、
  多言語や CPU の環境で使えなくなるため。
- **haas-asr の `load_audio` を使う**：WebM を読めず、対応フォーマットが whisper と食い違うため。
  faster-whisper のデコーダ（PyAV）を共用する。
