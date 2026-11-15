"""環境変数（.env を含む）から読み込むサーバ設定。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import get_args

from dotenv import load_dotenv

from .backchannel_interval import (
    MAX_BACKCHANNEL_INTERVAL_FACTOR,
    MIN_BACKCHANNEL_INTERVAL_FACTOR,
)
from .backchannel_policy import (
    BACKCHANNEL_TIMING_NAMES,
    MIN_BACKCHANNEL_SILENCE_MS,
    BackchannelTimingName,
)
from .transcription import BackendName

load_dotenv(override=False)

BACKEND_NAMES: tuple[BackendName, ...] = get_args(BackendName)

DEFAULT_BACKCHANNEL_PHRASES = "うん,うんうん,はい,なるほど"


def parse_backend(value: str) -> BackendName:
    """STT_BACKEND の値を検証する。誤記のまま別のバックエンドで起動しないよう ValueError にする。"""
    for backend in BACKEND_NAMES:
        if value == backend:
            return backend
    raise ValueError(f"invalid STT_BACKEND: '{value}' (allowed: {', '.join(BACKEND_NAMES)})")


def _read_int(name: str, default: str, minimum: int) -> int:
    """整数の環境変数を読み、minimum 未満なら ValueError にする。"""
    raw = os.getenv(name, default)
    try:
        value = int(raw)
    except ValueError as error:
        raise ValueError(f"invalid {name}: '{raw}' (must be an integer)") from error
    if value < minimum:
        raise ValueError(f"invalid {name}: '{raw}' (must be {minimum} or more)")
    return value


def _read_float(
    name: str, default: str, *, lower_exclusive: float, upper_inclusive: float
) -> float:
    """実数の環境変数を読み、(lower_exclusive, upper_inclusive] の範囲外なら ValueError にする。"""
    raw = os.getenv(name, default)
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(f"invalid {name}: '{raw}' (must be a number)") from error
    if not lower_exclusive < value <= upper_inclusive:
        raise ValueError(
            f"invalid {name}: '{raw}' (must be greater than {lower_exclusive} "
            f"and at most {upper_inclusive})"
        )
    return value


def _read_interval_factor() -> float:
    """相槌のたびに間隔を伸ばす倍率を読む。1 未満は間隔が縮むため受け付けない。"""
    name = "STT_BACKCHANNEL_INTERVAL_FACTOR"
    value = _read_float(
        name, "1.1", lower_exclusive=0.0, upper_inclusive=MAX_BACKCHANNEL_INTERVAL_FACTOR
    )
    if value < MIN_BACKCHANNEL_INTERVAL_FACTOR:
        raise ValueError(
            f"invalid {name}: '{value}' (must be {MIN_BACKCHANNEL_INTERVAL_FACTOR} or more)"
        )
    return value


def _read_non_empty(name: str, default: str) -> str:
    """空にできない文字列の環境変数を読む。"""
    value = os.getenv(name, default).strip()
    if not value:
        raise ValueError(f"invalid {name}: must not be empty")
    return value


def parse_backchannel_timing(value: str) -> BackchannelTimingName:
    """相槌のタイミングの決め方の名前を読む。"""
    name = value.strip()
    for timing in BACKCHANNEL_TIMING_NAMES:
        if name == timing:
            return timing
    raise ValueError(
        f"invalid STT_BACKCHANNEL_TIMING: '{value}' "
        f"(must be one of {', '.join(BACKCHANNEL_TIMING_NAMES)})"
    )


def parse_phrases(name: str, raw: str) -> tuple[str, ...]:
    """カンマ区切りの文言の一覧を読む。空の要素は捨て、1 つも無ければ ValueError にする。"""
    phrases = tuple(phrase.strip() for phrase in raw.split(",") if phrase.strip())
    if not phrases:
        raise ValueError(f"invalid {name}: '{raw}' (must contain at least one phrase)")
    return phrases


@dataclass(frozen=True)
class BackchannelSettings:
    """/stream の相槌の設定。"""

    max_sessions: int
    timing: BackchannelTimingName
    threshold: float
    silence_ms: int
    min_interval_ms: int
    interval_per_minute_ms: int
    interval_factor: float
    max_interval_ms: int
    phrases: tuple[str, ...]


@dataclass(frozen=True)
class AckSettings:
    """/stream の一時的な返事（Ollama の軽量 LLM）の設定。"""

    ollama_url: str
    model: str
    timeout_ms: int
    max_chars: int
    fallback_text: str
    keep_alive: str


@dataclass(frozen=True)
class BackchannelJudgeSettings:
    """/stream で相槌の文言を話の種類に合わせて選ぶ、Kev の設定。url が空なら使わない。"""

    url: str
    token_file: Path
    timeout_ms: int


def load_backchannel_settings() -> BackchannelSettings:
    """相槌の設定を環境変数から読む。"""
    return BackchannelSettings(
        max_sessions=_read_int("STT_BACKCHANNEL_MAX_SESSIONS", "4", minimum=1),
        timing=parse_backchannel_timing(os.getenv("STT_BACKCHANNEL_TIMING", "maai")),
        threshold=_read_float(
            "STT_BACKCHANNEL_THRESHOLD", "0.5", lower_exclusive=0.0, upper_inclusive=1.0
        ),
        silence_ms=_read_int(
            "STT_BACKCHANNEL_SILENCE_MS", "90", minimum=MIN_BACKCHANNEL_SILENCE_MS
        ),
        min_interval_ms=_read_int("STT_BACKCHANNEL_MIN_INTERVAL_MS", "1500", minimum=0),
        interval_per_minute_ms=_read_int(
            "STT_BACKCHANNEL_INTERVAL_PER_MINUTE_MS", "3000", minimum=0
        ),
        interval_factor=_read_interval_factor(),
        max_interval_ms=_read_int("STT_BACKCHANNEL_MAX_INTERVAL_MS", "8000", minimum=0),
        phrases=parse_phrases(
            "STT_BACKCHANNEL_PHRASES",
            os.getenv("STT_BACKCHANNEL_PHRASES", DEFAULT_BACKCHANNEL_PHRASES),
        ),
    )


def load_backchannel_judge_settings() -> BackchannelJudgeSettings:
    """話の種類の判定（Kev）の設定を環境変数から読む。"""
    return BackchannelJudgeSettings(
        url=os.getenv("STT_BACKCHANNEL_JUDGE_URL", "").strip().rstrip("/"),
        token_file=Path(os.getenv("STT_BACKCHANNEL_JUDGE_TOKEN_FILE", "~/.config/kev/api_key")),
        timeout_ms=_read_int("STT_BACKCHANNEL_JUDGE_TIMEOUT_MS", "200", minimum=1),
    )


def load_ack_settings() -> AckSettings:
    """一時的な返事の設定を環境変数から読む。"""
    return AckSettings(
        ollama_url=_read_non_empty("STT_ACK_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/"),
        model=_read_non_empty("STT_ACK_MODEL", "gemma4:e2b"),
        timeout_ms=_read_int("STT_ACK_TIMEOUT_MS", "1500", minimum=1),
        max_chars=_read_int("STT_ACK_MAX_CHARS", "20", minimum=1),
        fallback_text=_read_non_empty("STT_ACK_FALLBACK_TEXT", "ちょっと確認するね。"),
        keep_alive=_read_non_empty("STT_ACK_KEEP_ALIVE", "30m"),
    )


def load_max_utterance_seconds() -> float:
    """1 発話の音声の上限（秒）を環境変数から読む。"""
    return _read_float(
        "STT_STREAM_MAX_UTTERANCE_SECONDS",
        "60",
        lower_exclusive=0.0,
        upper_inclusive=float("inf"),
    )


def load_stream_recording_dir() -> Path | None:
    """/stream の録音の保存先を環境変数から読む。空なら録音しない。"""
    raw = os.getenv("STT_STREAM_RECORDING_DIR", "").strip()
    return Path(raw).expanduser() if raw else None


@dataclass(frozen=True)
class Settings:
    """サーバ全体の設定。STT_MODEL / STT_DEVICE / STT_COMPUTE_TYPE は whisper バックエンド用。"""

    host: str = os.getenv("STT_HOST", "0.0.0.0")
    port: int = int(os.getenv("STT_PORT", "23000"))
    backend: BackendName = parse_backend(os.getenv("STT_BACKEND", "whisper"))
    model: str = os.getenv("STT_MODEL", "large-v3-turbo")
    device: str = os.getenv("STT_DEVICE", "auto")
    compute_type: str = os.getenv("STT_COMPUTE_TYPE", "int8_float16")
    haas_model: str = os.getenv("STT_HAAS_MODEL", "third-intelligence/Haas-ja-v1.0")
    language: str = os.getenv("STT_LANGUAGE", "ja")
    api_key: str = os.getenv("STT_API_KEY", "")
    stream_max_utterance_seconds: float = field(default_factory=load_max_utterance_seconds)
    backchannel: BackchannelSettings = field(default_factory=load_backchannel_settings)
    backchannel_judge: BackchannelJudgeSettings = field(
        default_factory=load_backchannel_judge_settings
    )
    ack: AckSettings = field(default_factory=load_ack_settings)
    stream_recording_dir: Path | None = field(default_factory=load_stream_recording_dir)


settings = Settings()
