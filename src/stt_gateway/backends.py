"""設定に応じて文字起こしバックエンドを選ぶ。"""

from __future__ import annotations

from dataclasses import dataclass

from .config import Settings
from .transcription import BackendName, Transcriber

# Haas-ja は GPU 専用で、haas-asr が bfloat16 にキャストして読み込む。
HAAS_DEVICE = "cuda"
HAAS_COMPUTE_TYPE = "bfloat16"


@dataclass(frozen=True)
class BackendDescription:
    """/health で返す、使用中のバックエンドとモデルの情報。"""

    backend: BackendName
    model: str
    device: str
    compute_type: str


def describe_backend(settings: Settings) -> BackendDescription:
    """モデルを読み込まずに、設定から使用中のバックエンドの情報を求める。"""
    if settings.backend == "haas":
        return BackendDescription("haas", settings.haas_model, HAAS_DEVICE, HAAS_COMPUTE_TYPE)
    return BackendDescription("whisper", settings.model, settings.device, settings.compute_type)


def load_transcriber(settings: Settings) -> Transcriber:
    """設定されたバックエンドのモデルを読み込む。重い依存は選ばれた側だけ import する。"""
    if settings.backend == "haas":
        from .haas_backend import load_haas_transcriber

        return load_haas_transcriber(settings.haas_model)

    from .whisper_backend import load_whisper_transcriber

    return load_whisper_transcriber(settings.model, settings.device, settings.compute_type)
