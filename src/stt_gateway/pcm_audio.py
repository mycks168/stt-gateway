"""PCM バイト列と波形の相互変換、波形から WAV への書き出し。"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from .audio_decoding import AudioSamples

PCM_S16_BYTES_PER_SAMPLE = 2
# int16 の最大振幅。波形（-1.0〜1.0）との変換に使う。
PCM_S16_FULL_SCALE = 32768.0
PCM_S16_MAX = 32767


def pcm_s16le_to_samples(payload: bytes) -> AudioSamples:
    """符号付き 16bit リトルエンディアンの PCM を、-1.0〜1.0 の float32 波形にする。"""
    if len(payload) % PCM_S16_BYTES_PER_SAMPLE:
        raise ValueError(f"pcm_s16le payload must have an even number of bytes: {len(payload)}")
    samples: AudioSamples = (
        np.frombuffer(payload, dtype="<i2").astype(np.float32) / PCM_S16_FULL_SCALE
    )
    return samples


def samples_to_pcm_s16le(samples: AudioSamples) -> bytes:
    """float32 波形を 16bit PCM にする。範囲外の値は飽和させる（折り返すと大きな雑音になるため）。"""
    scaled = np.clip(np.round(samples * PCM_S16_FULL_SCALE), -PCM_S16_FULL_SCALE, PCM_S16_MAX)
    return scaled.astype("<i2").tobytes()


def write_wav(path: Path, samples: AudioSamples, sample_rate_hz: int) -> None:
    """モノラル 16bit の WAV ファイルとして書き出す。"""
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(PCM_S16_BYTES_PER_SAMPLE)
        wav_file.setframerate(sample_rate_hz)
        wav_file.writeframes(samples_to_pcm_s16le(samples))
