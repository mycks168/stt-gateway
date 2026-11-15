from __future__ import annotations

from pathlib import Path

import pytest

from stt_gateway.config import (
    BackchannelJudgeSettings,
    BackchannelSettings,
    load_ack_settings,
    load_backchannel_judge_settings,
    load_backchannel_settings,
    load_max_utterance_seconds,
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
    "STT_BACKCHANNEL_THRESHOLD",
    "STT_BACKCHANNEL_MIN_INTERVAL_MS",
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
        threshold=0.5,
        min_interval_ms=1500,
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
    clean_stream_env.setenv("STT_BACKCHANNEL_THRESHOLD", "1")
    assert load_backchannel_settings().phrases == ("うん", "へえ")
    assert load_backchannel_settings().threshold == 1.0
    assert load_ack_settings().ollama_url == "http://ollama:11434"


@pytest.mark.parametrize(
    ("name", "value", "expected_message"),
    [
        ("STT_BACKCHANNEL_MAX_SESSIONS", "0", "must be 1 or more"),
        ("STT_BACKCHANNEL_MAX_SESSIONS", "four", "must be an integer"),
        ("STT_BACKCHANNEL_MIN_INTERVAL_MS", "-1", "must be 0 or more"),
        ("STT_BACKCHANNEL_THRESHOLD", "0", "greater than 0.0"),
        ("STT_BACKCHANNEL_THRESHOLD", "1.5", "at most 1.0"),
        ("STT_BACKCHANNEL_THRESHOLD", "high", "must be a number"),
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


def test_backchannel_judge_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "STT_BACKCHANNEL_JUDGE_URL",
        "STT_BACKCHANNEL_JUDGE_TOKEN_FILE",
        "STT_BACKCHANNEL_JUDGE_TIMEOUT_MS",
    ):
        monkeypatch.delenv(name, raising=False)
    defaults = load_backchannel_judge_settings()
    assert defaults == BackchannelJudgeSettings("", Path("~/.config/kev/api_key"), 200)

    monkeypatch.setenv("STT_BACKCHANNEL_JUDGE_URL", " http://127.0.0.1:8009/ ")
    monkeypatch.setenv("STT_BACKCHANNEL_JUDGE_TIMEOUT_MS", "150")
    assert load_backchannel_judge_settings().url == "http://127.0.0.1:8009"
    assert load_backchannel_judge_settings().timeout_ms == 150

    monkeypatch.setenv("STT_BACKCHANNEL_JUDGE_TIMEOUT_MS", "0")
    with pytest.raises(ValueError, match="STT_BACKCHANNEL_JUDGE_TIMEOUT_MS"):
        load_backchannel_judge_settings()
