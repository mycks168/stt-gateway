from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from stt_gateway.audio_decoding import decode_to_mono

_SOURCE_RATE_HZ = 8_000
_TARGET_RATE_HZ = 16_000
_SOURCE_SECONDS = 1
_AMPLITUDE = 0.5
_TONE_HZ = 440
# resampler の遅延で端のサンプル数がずれるため、長さの比較に許す誤差
_LENGTH_TOLERANCE_SAMPLES = 64
_INT16_FULL_SCALE = 32767


def _write_tone_wav(path: Path, frame_count: int, channel_count: int) -> None:
    """全チャンネル同じ正弦波の 16bit WAV を書き出す。"""
    times = np.arange(frame_count) / _SOURCE_RATE_HZ
    mono = (_AMPLITUDE * np.sin(2 * np.pi * _TONE_HZ * times) * _INT16_FULL_SCALE).astype(np.int16)
    frames = np.repeat(mono[:, None], channel_count, axis=1)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(channel_count)
        wav_file.setsampwidth(2)
        wav_file.setframerate(_SOURCE_RATE_HZ)
        wav_file.writeframes(frames.tobytes())


def test_decode_to_mono_resamples_keeping_amplitude(tmp_path: Path) -> None:
    audio_path = tmp_path / "tone.wav"
    _write_tone_wav(audio_path, _SOURCE_RATE_HZ * _SOURCE_SECONDS, channel_count=1)

    samples = decode_to_mono(audio_path, _TARGET_RATE_HZ)

    assert samples.dtype == np.float32
    assert abs(len(samples) - _TARGET_RATE_HZ * _SOURCE_SECONDS) <= _LENGTH_TOLERANCE_SAMPLES
    assert float(np.abs(samples).max()) == pytest.approx(_AMPLITUDE, abs=0.05)


def test_decode_to_mono_downmixes_stereo(tmp_path: Path) -> None:
    audio_path = tmp_path / "stereo.wav"
    _write_tone_wav(audio_path, _SOURCE_RATE_HZ * _SOURCE_SECONDS, channel_count=2)

    samples = decode_to_mono(audio_path, _TARGET_RATE_HZ)

    assert samples.ndim == 1
    assert abs(len(samples) - _TARGET_RATE_HZ * _SOURCE_SECONDS) <= _LENGTH_TOLERANCE_SAMPLES


def test_decode_to_mono_returns_empty_for_silent_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "empty.wav"
    _write_tone_wav(audio_path, 0, channel_count=1)

    samples = decode_to_mono(audio_path, _TARGET_RATE_HZ)

    assert samples.dtype == np.float32
    assert len(samples) == 0


def test_decode_to_mono_reports_path_for_broken_file(tmp_path: Path) -> None:
    audio_path = tmp_path / "broken.webm"
    audio_path.write_bytes(b"not an audio file")

    with pytest.raises(ValueError) as exc_info:
        decode_to_mono(audio_path, _TARGET_RATE_HZ)
    assert str(audio_path) in str(exc_info.value)
