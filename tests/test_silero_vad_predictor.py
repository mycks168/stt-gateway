from __future__ import annotations

import sys
from importlib import resources

import numpy as np
import pytest

from stt_gateway import silero_vad_predictor
from stt_gateway.silero_vad_predictor import create_silero_predictor

SAMPLE_RATE_HZ = 16000


def _speech_like(seconds: float) -> np.ndarray:
    """声に近い音（基本周波数と倍音を持ち、音量が揺れる）を作る。"""
    times = np.arange(int(SAMPLE_RATE_HZ * seconds)) / SAMPLE_RATE_HZ
    voice = sum(np.sin(2 * np.pi * 150 * harmonic * times) / harmonic for harmonic in range(1, 8))
    envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 4 * times)
    samples: np.ndarray = (0.1 * voice * envelope).astype(np.float32)
    return samples


def test_predictor_reports_silence_every_32ms_across_chunk_boundaries() -> None:
    predictor = create_silero_predictor()
    # 80ms ずつ受け取っても、512 サンプル（32ms）ごとに判定する。
    silence = np.zeros(1280, dtype=np.float32)
    frames = [frame for _ in range(4) for frame in predictor.predict(silence)]
    assert [frame.offset_ms for frame in frames] == [32 * index for index in range(1, 11)]
    assert all(frame.probability > 0.9 for frame in frames), "無音は話していない確率が高い"


def test_predictor_reset_restarts_offsets() -> None:
    predictor = create_silero_predictor()
    predictor.predict(np.zeros(1000, dtype=np.float32))
    predictor.reset()
    assert [frame.offset_ms for frame in predictor.predict(np.zeros(512, dtype=np.float32))] == [32]


def test_predictor_distinguishes_speech_from_silence() -> None:
    predictor = create_silero_predictor()
    frames = predictor.predict(
        np.concatenate([np.zeros(8000, dtype=np.float32), _speech_like(1.0)])
    )
    leading = [frame.probability for frame in frames if frame.offset_ms <= 480]
    assert min(leading) > 0.5
    # 合成音は声として判定されにくいため、無音より話していない確率が下がることだけを確かめる。
    voiced = [frame.probability for frame in frames if frame.offset_ms > 600]
    assert min(voiced) < min(leading)


def test_create_reports_missing_silero_package(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(_package: str) -> None:
        raise ModuleNotFoundError("No module named 'silero_vad'")

    monkeypatch.setattr(resources, "files", missing)
    with pytest.raises(RuntimeError, match="silero-vad"):
        create_silero_predictor()


def test_create_reports_missing_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    with pytest.raises(RuntimeError, match="onnxruntime"):
        silero_vad_predictor.create_silero_predictor()
