from __future__ import annotations

import builtins
import queue
import sys
import types
from typing import Any

import numpy as np
import pytest

from stt_gateway import maai_backchannel
from stt_gateway.audio_decoding import AudioSamples
from stt_gateway.maai_backchannel import (
    SAMPLES_PER_FRAME,
    MaaiBackchannelPredictor,
    create_maai_predictor,
)


class _FakeMaai:
    """1 フレーム渡されるたびに、決まった確率を結果のキューに入れるダミー。"""

    def __init__(self, probabilities: list[float]) -> None:
        self.result_dict_queue: queue.Queue[dict[str, object]] = queue.Queue()
        self._probabilities = probabilities
        self.received: list[tuple[AudioSamples, AudioSamples]] = []
        self.reset_count = 0

    def reset_runtime_state(self) -> None:
        """呼ばれた回数を記録する。"""
        self.reset_count += 1

    def process(self, x1: AudioSamples, x2: AudioSamples) -> None:
        """受け取った音声を記録し、確率を 1 つ結果に入れる。"""
        self.received.append((x1, x2))
        self.result_dict_queue.put({"p_bc": np.array(self._probabilities[len(self.received) - 1])})


def test_predict_splits_audio_into_frames_and_feeds_silence_as_system_channel() -> None:
    model = _FakeMaai([0.1, 0.7, 0.2])
    predictor = MaaiBackchannelPredictor(model)

    frames = predictor.predict(np.ones(SAMPLES_PER_FRAME + 100, dtype=np.float32))
    frames += predictor.predict(np.ones(SAMPLES_PER_FRAME * 2 - 100, dtype=np.float32))

    assert [f.offset_ms for f in frames] == [100, 200, 300]
    assert [f.probability for f in frames] == pytest.approx([0.1, 0.7, 0.2])
    assert all(len(user) == SAMPLES_PER_FRAME for user, _ in model.received)
    assert all(not system.any() for _, system in model.received), "システム側は無音"


def test_reset_clears_context_and_offsets() -> None:
    model = _FakeMaai([0.1, 0.9])
    predictor = MaaiBackchannelPredictor(model)
    predictor.predict(np.ones(SAMPLES_PER_FRAME + 100, dtype=np.float32))
    model.result_dict_queue.put({"p_bc": 0.0})  # 前の発話の取り残し

    predictor.reset()
    frames = predictor.predict(np.ones(SAMPLES_PER_FRAME, dtype=np.float32))

    assert model.reset_count == 1
    assert [f.offset_ms for f in frames] == [100], "位置は発話の先頭から数え直す"
    assert len(model.received[-1][0]) == SAMPLES_PER_FRAME, "前の発話の残りの音声を混ぜない"


def test_create_maai_predictor_uses_bc_model_on_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    created: dict[str, object] = {}

    def fake_maai(**kwargs: object) -> _FakeMaai:
        """引数を記録する。"""
        created.update(kwargs)
        return _FakeMaai([])

    fake_module = types.SimpleNamespace(
        Maai=fake_maai, MaaiInput=types.SimpleNamespace(Chunk=object)
    )
    monkeypatch.setitem(sys.modules, "maai", fake_module)

    predictor = create_maai_predictor()

    assert isinstance(predictor, MaaiBackchannelPredictor)
    assert created["mode"] == "bc"
    assert created["lang"] == "jp"
    assert created["device"] == "cpu"
    assert created["frame_rate"] == maai_backchannel.FRAME_RATE_HZ


def test_create_maai_predictor_explains_missing_extra(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def import_without_maai(name: str, *args: Any, **kwargs: Any) -> Any:
        """maai だけ未導入として扱う。"""
        if name == "maai":
            raise ModuleNotFoundError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "maai", raising=False)
    monkeypatch.setattr(builtins, "__import__", import_without_maai)
    with pytest.raises(RuntimeError, match="uv sync --extra stream"):
        create_maai_predictor()
