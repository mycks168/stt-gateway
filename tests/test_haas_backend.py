from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from stt_gateway import haas_backend
from stt_gateway.haas_backend import SAMPLE_RATE_HZ, AudioSamples, HaasTranscriber
from stt_gateway.transcription import UnsupportedLanguageError

_TWO_SECONDS = np.zeros(SAMPLE_RATE_HZ * 2, dtype=np.float32)


def _decode_two_seconds(audio_path: Path) -> AudioSamples:
    """2 秒分の無音を返すダミーのデコーダ。"""
    return _TWO_SECONDS


def test_haas_transcriber_returns_text_and_duration() -> None:
    received: list[AudioSamples] = []

    def recognize(samples: AudioSamples) -> str:
        """受け取った波形を記録する。"""
        received.append(samples)
        return " こんにちは "

    result = HaasTranscriber(recognize, _decode_two_seconds).transcribe(Path("a.opus"), "ja")

    assert result.text == "こんにちは"
    assert result.language == "ja"
    assert result.duration_seconds == pytest.approx(2.0)
    assert received[0] is _TWO_SECONDS


def test_haas_transcriber_rejects_non_japanese_before_decoding() -> None:
    decoded: list[Path] = []

    def decode(audio_path: Path) -> AudioSamples:
        """呼ばれたら記録する。"""
        decoded.append(audio_path)
        return _TWO_SECONDS

    transcriber = HaasTranscriber(lambda samples: "", decode)
    with pytest.raises(UnsupportedLanguageError) as exc_info:
        transcriber.transcribe(Path("a.wav"), "en")
    assert "'en'" in str(exc_info.value)
    assert not decoded


def test_decode_with_pyav_requests_model_sample_rate(monkeypatch: pytest.MonkeyPatch) -> None:
    import faster_whisper

    calls: list[tuple[str, int]] = []

    def fake_decode_audio(path: str, sampling_rate: int) -> AudioSamples:
        """引数を記録する。"""
        calls.append((path, sampling_rate))
        return _TWO_SECONDS

    monkeypatch.setattr(faster_whisper, "decode_audio", fake_decode_audio)
    assert haas_backend.decode_with_pyav(Path("/tmp/a.webm")) is _TWO_SECONDS
    assert calls == [("/tmp/a.webm", SAMPLE_RATE_HZ)]


def test_load_haas_transcriber_uses_longform_recognition(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    longform_calls: list[tuple[Any, AudioSamples, int]] = []
    fake_model = object()

    def transcribe_longform(model: Any, samples: AudioSamples, batch_size: int) -> str:
        """引数を記録する。"""
        longform_calls.append((model, samples, batch_size))
        return "長い音声"

    fake_haas_asr = types.SimpleNamespace(
        load_model=lambda name, device: fake_model,
        transcribe_longform=transcribe_longform,
    )
    monkeypatch.setitem(sys.modules, "haas_asr", fake_haas_asr)
    monkeypatch.setattr(haas_backend, "decode_with_pyav", _decode_two_seconds)

    transcriber = haas_backend.load_haas_transcriber("third-intelligence/Haas-ja-v1.0")

    assert transcriber.transcribe(Path("a.wav"), "ja").text == "長い音声"
    assert longform_calls == [(fake_model, _TWO_SECONDS, haas_backend.LONGFORM_BATCH_SIZE)]


def test_load_haas_transcriber_explains_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def import_without_haas(name: str, *args: Any, **kwargs: Any) -> Any:
        """haas_asr だけ未導入として扱う。"""
        if name == "haas_asr":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "haas_asr", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_haas)

    with pytest.raises(RuntimeError) as exc_info:
        haas_backend.load_haas_transcriber("third-intelligence/Haas-ja-v1.0")
    assert "uv sync --extra haas" in str(exc_info.value)
