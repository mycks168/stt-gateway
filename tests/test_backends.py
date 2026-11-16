from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from stt_gateway import backends
from stt_gateway.backend_selection import BackendSelection
from stt_gateway.config import Settings
from stt_gateway.whisper_backend import WhisperTranscriber

_BASE_SETTINGS = Settings(
    backends=BackendSelection("whisper", ("whisper",)),
    model="large-v3-turbo",
    device="auto",
    compute_type="int8_float16",
    haas_model="third-intelligence/Haas-ja-v1.0",
)


def test_describe_backend_whisper_uses_whisper_settings() -> None:
    description = backends.describe_backend(_BASE_SETTINGS, "whisper")
    assert description == backends.BackendDescription(
        "whisper", "large-v3-turbo", "auto", "int8_float16"
    )


def test_describe_backend_haas_ignores_whisper_settings() -> None:
    description = backends.describe_backend(_BASE_SETTINGS, "haas")
    assert description == backends.BackendDescription(
        "haas", "third-intelligence/Haas-ja-v1.0", "cuda", "bfloat16"
    )


@dataclass
class _FakeWhisperModel:
    """WhisperModel のコンストラクタ引数を記録する。"""

    name: str
    device: str
    compute_type: str


def test_load_transcriber_whisper_builds_whisper_model(monkeypatch: pytest.MonkeyPatch) -> None:
    import faster_whisper

    monkeypatch.setattr(faster_whisper, "WhisperModel", _FakeWhisperModel)
    transcriber = backends.load_transcriber(_BASE_SETTINGS, "whisper")
    assert isinstance(transcriber, WhisperTranscriber)
    assert transcriber._model == _FakeWhisperModel("large-v3-turbo", "auto", "int8_float16")


def test_load_transcriber_haas_loads_haas_model(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[tuple[str, str]] = []
    fake_haas_asr = types.SimpleNamespace(
        load_model=lambda name, device: loaded.append((name, device))
    )
    monkeypatch.setitem(sys.modules, "haas_asr", fake_haas_asr)

    backends.load_transcriber(_BASE_SETTINGS, "haas")
    assert loaded == [("third-intelligence/Haas-ja-v1.0", "cuda")]
