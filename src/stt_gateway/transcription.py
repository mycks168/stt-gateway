"""文字起こしバックエンドの共通インタフェースと結果の型。

API 層はこのモジュールだけに依存し、faster-whisper や NeMo には直接依存しない。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

BackendName = Literal["whisper", "haas"]


@dataclass(frozen=True)
class Transcription:
    """1 ファイル分の文字起こし結果。"""

    text: str
    language: str
    duration_seconds: float | None


class UnsupportedLanguageError(ValueError):
    """バックエンドが対応していない言語を指定されたときに送出する。"""

    def __init__(self, backend: BackendName, language: str, supported: tuple[str, ...]) -> None:
        """エラーメッセージに対応言語の一覧を含めて初期化する。"""
        super().__init__(
            f"unsupported language for backend '{backend}': '{language}' "
            f"(supported: {', '.join(supported)})"
        )


class Transcriber(Protocol):
    """音声ファイルをテキストに変換するバックエンド。"""

    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """音声ファイルを文字起こしする。対応外の言語では UnsupportedLanguageError を送出する。"""
        ...
