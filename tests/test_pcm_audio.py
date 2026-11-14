from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from stt_gateway.pcm_audio import pcm_s16le_to_samples, samples_to_pcm_s16le, write_wav


def test_pcm_round_trip_keeps_values() -> None:
    samples = np.array([0.0, 0.5, -0.5, -1.0], dtype=np.float32)
    restored = pcm_s16le_to_samples(samples_to_pcm_s16le(samples))
    assert restored.dtype == np.float32
    np.testing.assert_allclose(restored, samples, atol=1 / 32768)


def test_samples_to_pcm_saturates_out_of_range_values() -> None:
    pcm = samples_to_pcm_s16le(np.array([2.0, -2.0], dtype=np.float32))
    assert np.frombuffer(pcm, dtype="<i2").tolist() == [32767, -32768]


def test_pcm_rejects_odd_number_of_bytes() -> None:
    with pytest.raises(ValueError, match="even number of bytes: 3"):
        pcm_s16le_to_samples(b"\x00\x01\x02")


def test_write_wav_writes_mono_16bit(tmp_path: Path) -> None:
    path = tmp_path / "a.wav"
    write_wav(path, np.zeros(1600, dtype=np.float32), 16000)
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.getnframes() == 1600
