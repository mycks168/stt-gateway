"""環境変数（.env を含む）から読み込むサーバ設定。"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import get_args

from dotenv import load_dotenv

from .transcription import BackendName

load_dotenv(override=False)

BACKEND_NAMES: tuple[BackendName, ...] = get_args(BackendName)


def parse_backend(value: str) -> BackendName:
    """STT_BACKEND の値を検証する。誤記のまま別のバックエンドで起動しないよう ValueError にする。"""
    for backend in BACKEND_NAMES:
        if value == backend:
            return backend
    raise ValueError(f"invalid STT_BACKEND: '{value}' (allowed: {', '.join(BACKEND_NAMES)})")


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


settings = Settings()
