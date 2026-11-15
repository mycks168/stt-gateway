from __future__ import annotations

import pytest

from stt_gateway.boundary_text import is_break_token, strip_trailing_breaks


@pytest.mark.parametrize("token", ["、", "。", "？", "?", "」", "\n", " ", "", "、で"])
def test_break_tokens(token: str) -> None:
    assert is_break_token(token)


@pytest.mark.parametrize("token", ["で", "なので", " the", "1", "ね、"])
def test_tokens_that_continue_the_phrase(token: str) -> None:
    assert not is_break_token(token)


def test_strip_trailing_breaks_keeps_inner_punctuation() -> None:
    assert (
        strip_trailing_breaks("あのさ、してもらったんだよね。 \n") == "あのさ、してもらったんだよね"
    )
    assert strip_trailing_breaks("なので") == "なので"
    assert strip_trailing_breaks("。、 ") == ""
