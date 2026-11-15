"""/stream で受け取った波形を、/transcribe と同じ順番待ちで文字起こしする。"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from .audio_decoding import AudioSamples
from .pcm_audio import write_wav
from .serial_transcriber import SerialTranscriber
from .stream_audio_decoder import STREAM_SAMPLE_RATE_HZ
from .transcription import Transcription


async def transcribe_samples(
    serial_transcriber: SerialTranscriber, samples: AudioSamples, language: str
) -> Transcription:
    """波形を WAV の一時ファイルにして文字起こしする。一時ファイルは成否にかかわらず消す。"""
    if len(samples) == 0:
        # 音声の無い発話はバックエンドに渡すと失敗しうるため、推論せずに空の結果にする。
        return Transcription(text="", language=language, duration_seconds=0.0)
    # 60 秒の音声で約 2MB を書き出すため、ファイル操作もイベントループの外で行う。
    audio_path = await asyncio.to_thread(_write_temporary_wav, samples)
    try:
        return await serial_transcriber.transcribe(audio_path, language)
    finally:
        await asyncio.to_thread(audio_path.unlink, missing_ok=True)


def _write_temporary_wav(samples: AudioSamples) -> Path:
    """波形を WAV の一時ファイルに書き出してパスを返す。削除は呼び出し側が行う。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = Path(temp_file.name)
    write_wav(audio_path, samples, STREAM_SAMPLE_RATE_HZ)
    return audio_path
