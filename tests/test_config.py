from __future__ import annotations

from pathlib import Path

import pytest

from stt_gateway.config import (
    BackchannelSettings,
    load_ack_settings,
    load_backchannel_settings,
    load_max_utterance_seconds,
    load_stream_recording_dir,
    parse_backend,
)


@pytest.mark.parametrize("value", ["whisper", "haas"])
def test_parse_backend_accepts_known_backends(value: str) -> None:
    assert parse_backend(value) == value


def test_parse_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_backend("Haas")
    message = str(exc_info.value)
    assert "'Haas'" in message
    assert "whisper, haas" in message


_STREAM_ENV_NAMES = (
    "STT_BACKCHANNEL_MAX_SESSIONS",
    "STT_BACKCHANNEL_SILENCE_MS",
    "STT_BACKCHANNEL_MIN_INTERVAL_MS",
    "STT_BACKCHANNEL_INTERVAL_PER_MINUTE_MS",
    "STT_BACKCHANNEL_INTERVAL_FACTOR",
    "STT_BACKCHANNEL_MIN_NEW_SPEECH_MS",
    "STT_BACKCHANNEL_BOUNDARY_MODEL",
    "STT_BACKCHANNEL_BOUNDARY_THRESHOLD",
    "STT_BACKCHANNEL_JUDGE_TIMEOUT_MS",
    "STT_BACKCHANNEL_MAX_INTERVAL_MS",
    "STT_BACKCHANNEL_PHRASES",
    "STT_ACK_OLLAMA_URL",
    "STT_ACK_MODEL",
    "STT_ACK_TIMEOUT_MS",
    "STT_ACK_MAX_CHARS",
    "STT_ACK_FALLBACK_TEXT",
    "STT_ACK_KEEP_ALIVE",
    "STT_STREAM_MAX_UTTERANCE_SECONDS",
)


@pytest.fixture
def clean_stream_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """ローカルの .env の値に依存しないよう、/stream の環境変数を消す。"""
    for name in _STREAM_ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.usefixtures("clean_stream_env")
def test_stream_settings_defaults() -> None:
    backchannel = load_backchannel_settings()
    ack = load_ack_settings()
    assert backchannel == BackchannelSettings(
        max_sessions=4,
        silence_ms=90,
        min_new_speech_ms=1000,
        min_interval_ms=1500,
        interval_per_minute_ms=3000,
        interval_factor=1.1,
        max_interval_ms=8000,
        boundary_model="",
        boundary_threshold=0.35,
        judge_timeout_ms=200,
        phrases=("うん", "うんうん", "はい", "なるほど"),
    )
    assert ack.ollama_url == "http://127.0.0.1:11434"
    assert ack.model == "gemma4:e2b"
    assert ack.timeout_ms == 1500
    assert ack.max_chars == 20
    assert ack.fallback_text == "ちょっと確認するね。"
    assert ack.keep_alive == "30m"
    assert load_max_utterance_seconds() == 60.0


def test_stream_settings_read_environment(clean_stream_env: pytest.MonkeyPatch) -> None:
    clean_stream_env.setenv("STT_BACKCHANNEL_PHRASES", " うん , ,へえ ")
    clean_stream_env.setenv("STT_ACK_OLLAMA_URL", "http://ollama:11434/")
    clean_stream_env.setenv("STT_BACKCHANNEL_SILENCE_MS", "200")
    clean_stream_env.setenv("STT_BACKCHANNEL_BOUNDARY_MODEL", " /models/qwen ")
    clean_stream_env.setenv("STT_BACKCHANNEL_JUDGE_TIMEOUT_MS", "150")
    backchannel = load_backchannel_settings()
    assert backchannel.phrases == ("うん", "へえ")
    assert backchannel.silence_ms == 200
    assert backchannel.boundary_model == "/models/qwen"
    assert backchannel.judge_timeout_ms == 150
    assert load_ack_settings().ollama_url == "http://ollama:11434"


@pytest.mark.parametrize(
    ("name", "value", "expected_message"),
    [
        ("STT_BACKCHANNEL_MAX_SESSIONS", "0", "must be 1 or more"),
        ("STT_BACKCHANNEL_MAX_SESSIONS", "four", "must be an integer"),
        ("STT_BACKCHANNEL_MIN_INTERVAL_MS", "-1", "must be 0 or more"),
        ("STT_BACKCHANNEL_JUDGE_TIMEOUT_MS", "0", "must be 1 or more"),
        ("STT_BACKCHANNEL_INTERVAL_FACTOR", "high", "must be a number"),
        ("STT_BACKCHANNEL_PHRASES", " , ", "at least one phrase"),
    ],
)
def test_backchannel_settings_reject_invalid_values(
    clean_stream_env: pytest.MonkeyPatch, name: str, value: str, expected_message: str
) -> None:
    clean_stream_env.setenv(name, value)
    with pytest.raises(ValueError, match=name) as exc_info:
        load_backchannel_settings()
    assert expected_message in str(exc_info.value)


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("STT_ACK_TIMEOUT_MS", "0"),
        ("STT_ACK_MAX_CHARS", "0"),
        ("STT_ACK_FALLBACK_TEXT", "  "),
        ("STT_ACK_MODEL", ""),
    ],
)
def test_ack_settings_reject_invalid_values(
    clean_stream_env: pytest.MonkeyPatch, name: str, value: str
) -> None:
    clean_stream_env.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_ack_settings()


def test_max_utterance_seconds_must_be_positive(clean_stream_env: pytest.MonkeyPatch) -> None:
    clean_stream_env.setenv("STT_STREAM_MAX_UTTERANCE_SECONDS", "0")
    with pytest.raises(ValueError, match="STT_STREAM_MAX_UTTERANCE_SECONDS"):
        load_max_utterance_seconds()


def test_stream_recording_dir(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("STT_STREAM_RECORDING_DIR", raising=False)
    assert load_stream_recording_dir() is None
    monkeypatch.setenv("STT_STREAM_RECORDING_DIR", "  ")
    assert load_stream_recording_dir() is None
    monkeypatch.setenv("STT_STREAM_RECORDING_DIR", "~/stream-recordings")
    assert load_stream_recording_dir() == Path.home() / "stream-recordings"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("STT_BACKCHANNEL_SILENCE_MS", "31"),
        ("STT_BACKCHANNEL_INTERVAL_FACTOR", "0.9"),
        ("STT_BACKCHANNEL_INTERVAL_FACTOR", "3.5"),
        ("STT_BACKCHANNEL_MIN_NEW_SPEECH_MS", "-1"),
        ("STT_BACKCHANNEL_BOUNDARY_THRESHOLD", "1.5"),
        ("STT_BACKCHANNEL_BOUNDARY_THRESHOLD", "high"),
    ],
)
def test_backchannel_settings_reject_out_of_range_values(
    clean_stream_env: pytest.MonkeyPatch, name: str, value: str
) -> None:
    clean_stream_env.setenv(name, value)
    with pytest.raises(ValueError, match=name):
        load_backchannel_settings()
