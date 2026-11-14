from __future__ import annotations

import pytest

from stt_gateway.backchannel_policy import (
    BackchannelDecision,
    BackchannelFrame,
    BackchannelTrigger,
    PhraseRotation,
)


def _observe_all(
    trigger: BackchannelTrigger, probabilities: list[float]
) -> list[BackchannelDecision]:
    """100ms 間隔のフレームとして順に渡し、出た相槌を返す。"""
    decisions = []
    for index, probability in enumerate(probabilities, start=1):
        decision = trigger.observe(BackchannelFrame(index * 100, probability))
        if decision is not None:
            decisions.append(decision)
    return decisions


def test_trigger_fires_once_per_rising_edge() -> None:
    trigger = BackchannelTrigger(0.5, 0, PhraseRotation(("うん",)))
    decisions = _observe_all(trigger, [0.1, 0.6, 0.7, 0.8, 0.2, 0.55])
    assert [(d.offset_ms, d.probability) for d in decisions] == [(200, 0.6), (600, 0.55)]


def test_trigger_treats_threshold_as_above() -> None:
    trigger = BackchannelTrigger(0.5, 0, PhraseRotation(("うん",)))
    assert _observe_all(trigger, [0.5])[0].offset_ms == 100


def test_trigger_skips_rising_edges_within_min_interval() -> None:
    trigger = BackchannelTrigger(0.5, 1500, PhraseRotation(("うん",)))
    # 200ms で発火、900ms は 700ms 後なので出さない、1700ms は 1500ms 後なので出す
    probabilities = [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6] + [0.0] * 7 + [0.6]
    decisions = _observe_all(trigger, probabilities)
    assert [d.offset_ms for d in decisions] == [200, 1700]


def test_phrase_rotation_cycles_and_is_shared_between_triggers() -> None:
    rotation = PhraseRotation(("うん", "はい"))
    first = BackchannelTrigger(0.5, 0, rotation)
    second = BackchannelTrigger(0.5, 0, rotation)
    texts = [d.text for d in _observe_all(first, [0.9, 0.0, 0.9])]
    texts += [d.text for d in _observe_all(second, [0.9])]
    assert texts == ["うん", "はい", "うん"]


def test_phrase_rotation_requires_phrases() -> None:
    with pytest.raises(ValueError, match="at least one phrase"):
        PhraseRotation(())
