from __future__ import annotations

import pytest

from stt_gateway.backchannel_policy import (
    BackchannelFrame,
    BackchannelTrigger,
    PhraseRotation,
)


def _observe_all(trigger: BackchannelTrigger, probabilities: list[float]) -> list[BackchannelFrame]:
    """100ms 間隔のフレームとして順に渡し、相槌を出すと決めたフレームを返す。"""
    fired = []
    for index, probability in enumerate(probabilities, start=1):
        frame = trigger.observe(BackchannelFrame(index * 100, probability))
        if frame is not None:
            fired.append(frame)
    return fired


def test_trigger_fires_once_per_rising_edge() -> None:
    trigger = BackchannelTrigger(0.5, 0)
    fired = _observe_all(trigger, [0.1, 0.6, 0.7, 0.8, 0.2, 0.55])
    assert [(f.offset_ms, f.probability) for f in fired] == [(200, 0.6), (600, 0.55)]


def test_trigger_treats_threshold_as_above() -> None:
    assert _observe_all(BackchannelTrigger(0.5, 0), [0.5])[0].offset_ms == 100


def test_trigger_skips_rising_edges_within_min_interval() -> None:
    trigger = BackchannelTrigger(0.5, 1500)
    # 200ms で発火、900ms は 700ms 後なので出さない、1700ms は 1500ms 後なので出す
    probabilities = [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6] + [0.0] * 7 + [0.6]
    assert [f.offset_ms for f in _observe_all(trigger, probabilities)] == [200, 1700]


def test_phrase_rotation_cycles() -> None:
    rotation = PhraseRotation(("うん", "はい"))
    assert [rotation.next_phrase() for _ in range(3)] == ["うん", "はい", "うん"]


def test_phrase_rotation_requires_phrases() -> None:
    with pytest.raises(ValueError, match="at least one phrase"):
        PhraseRotation(())
