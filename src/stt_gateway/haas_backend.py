"""Haas-ja（Third Intelligence 社の日本語音声認識モデル）による文字起こしバックエンド。

haas-asr と NeMo は extras `haas` でだけ入るため、import は読み込み関数の中で行う。
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from .transcription import Transcription, UnsupportedLanguageError

AudioSamples = npt.NDArray[np.float32]
Recognizer = Callable[[AudioSamples], str]
AudioDecoder = Callable[[Path], AudioSamples]

# Haas-ja の入力サンプリングレート。
SAMPLE_RATE_HZ = 16_000
# Haas-ja は日本語専用のモデル。
SUPPORTED_LANGUAGES = ("ja",)
# 長い音声を分割したチャンクをまとめて推論する件数。haas-asr の既定値に合わせる。
LONGFORM_BATCH_SIZE = 64


class HaasTranscriber:
    """音声をデコードして Haas-ja に渡すバックエンド。"""

    def __init__(self, recognize: Recognizer, decode_audio: AudioDecoder) -> None:
        """推論関数と、ファイルを 16kHz モノラルの波形にする関数を受け取る。"""
        self._recognize = recognize
        self._decode_audio = decode_audio

    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """日本語以外を拒否してから文字起こしする。"""
        if language not in SUPPORTED_LANGUAGES:
            raise UnsupportedLanguageError("haas", language, SUPPORTED_LANGUAGES)
        samples = self._decode_audio(audio_path)
        text = self._recognize(samples).strip()
        return Transcription(
            text=text,
            language=language,
            duration_seconds=len(samples) / SAMPLE_RATE_HZ,
        )


def decode_with_pyav(audio_path: Path) -> AudioSamples:
    """PyAV(ffmpeg) で 16kHz モノラルにデコードする。

    haas-asr 付属の load_audio は libsndfile を使うため WebM を読めない。
    Whisper バックエンドと同じ対応フォーマットにそろえるため、faster-whisper のデコーダを使う。
    """
    from faster_whisper import decode_audio

    samples: AudioSamples = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE_HZ)
    return samples


def load_haas_transcriber(model_name: str) -> HaasTranscriber:
    """Haas-ja を GPU に読み込む。初回は Hugging Face Hub からダウンロードする。"""
    try:
        haas_asr = importlib.import_module("haas_asr")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "STT_BACKEND=haas requires the 'haas' extra: run `uv sync --extra haas`"
        ) from error

    model = haas_asr.load_model(model_name, device="cuda")

    def recognize(samples: AudioSamples) -> str:
        """30 秒を超える音声は無音位置で分割して推論する（haas-asr の長尺処理）。"""
        text: str = haas_asr.transcribe_longform(model, samples, batch_size=LONGFORM_BATCH_SIZE)
        return text

    return HaasTranscriber(recognize, decode_with_pyav)
