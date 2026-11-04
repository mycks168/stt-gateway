from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stt_gateway.transcription import Transcription
from stt_gateway.whisper_backend import VAD_THRESHOLD, WhisperTranscriber


@dataclass
class _FakeSegment:
    text: str


@dataclass
class _FakeInfo:
    language: str
    duration: float


@dataclass
class _FakeWhisperModel:
    """WhisperModel.transcribe の引数を記録する。"""

    calls: list[tuple[str, dict[str, Any]]] = field(default_factory=list)

    def transcribe(self, path: str, **kwargs: Any) -> tuple[list[_FakeSegment], _FakeInfo]:
        """セグメントを 2 つ返す。"""
        self.calls.append((path, kwargs))
        return [_FakeSegment(" こんにちは"), _FakeSegment("世界 ")], _FakeInfo("ja", 2.5)


def test_whisper_transcriber_joins_segments_and_uses_vad() -> None:
    model = _FakeWhisperModel()
    result = WhisperTranscriber(model).transcribe(Path("/tmp/a.wav"), "ja")

    assert result == Transcription(text="こんにちは世界", language="ja", duration_seconds=2.5)
    path, kwargs = model.calls[0]
    assert path == "/tmp/a.wav"
    assert kwargs["language"] == "ja"
    assert kwargs["vad_filter"] is True
    assert kwargs["vad_parameters"] == {"threshold": VAD_THRESHOLD}
