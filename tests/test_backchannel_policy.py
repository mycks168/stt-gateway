from __future__ import annotations

import pytest

from stt_gateway.backchannel_policy import PhraseRotation


def test_phrase_rotation_cycles() -> None:
    rotation = PhraseRotation(("うん", "はい"))
    assert [rotation.next_phrase() for _ in range(3)] == ["うん", "はい", "うん"]


def test_phrase_rotation_requires_phrases() -> None:
    with pytest.raises(ValueError, match="at least one phrase"):
        PhraseRotation(())
