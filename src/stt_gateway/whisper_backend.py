"""faster-whisper による文字起こしバックエンド。"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from .transcription import Transcription

if TYPE_CHECKING:
    from faster_whisper import WhisperModel

# 発話区間検出（VAD）のしきい値。faster-whisper の既定値 0.5 より低くし、小さな声も発話として扱う。
VAD_THRESHOLD = 0.3


class WhisperTranscriber:
    """faster-whisper の WhisperModel を使うバックエンド。言語は Whisper が対応するものすべて。"""

    def __init__(self, model: WhisperModel) -> None:
        """読み込み済みのモデルを受け取る。"""
        self._model = model

    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """VAD で無音を除いて文字起こしする。デコードは faster-whisper 内部の PyAV が行う。"""
        segments, info = self._model.transcribe(
            str(audio_path),
            language=language,
            vad_filter=True,
            vad_parameters={"threshold": VAD_THRESHOLD},
        )
        text = "".join(segment.text for segment in segments).strip()
        return Transcription(text=text, language=info.language, duration_seconds=info.duration)


def load_whisper_transcriber(model_name: str, device: str, compute_type: str) -> WhisperTranscriber:
    """モデルを読み込んでバックエンドを作る。初回は Hugging Face Hub からダウンロードする。"""
    from faster_whisper import WhisperModel

    model = WhisperModel(model_name, device=device, compute_type=compute_type)
    return WhisperTranscriber(model)
