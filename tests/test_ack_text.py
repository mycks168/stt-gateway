from __future__ import annotations

import pytest

from stt_gateway.ack_prompt import DEFAULT_ACK_PROMPT, AckPrompt, build_ack_messages
from stt_gateway.ack_text import normalize_ack_text


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("  明日の天気ね、確認するね。 ", "明日の天気ね、確認するね。"),
        ("「調べるね。」", "調べるね。"),
        ('"調べるね"', "調べるね"),
        ("調べるね。\n明日は晴れです。", "調べるね。"),
        ("", None),
        ("「」", None),
        ("あ" * 21, None),
    ],
)
def test_normalize_ack_text(raw: str, expected: str | None) -> None:
    assert normalize_ack_text(raw, max_chars=20) == expected


def test_build_ack_messages_places_transcript_last() -> None:
    messages = build_ack_messages("明日の天気は？", DEFAULT_ACK_PROMPT)
    assert messages[0] == {"role": "system", "content": DEFAULT_ACK_PROMPT.system_prompt}
    assert len(messages) == 1 + 2 * len(DEFAULT_ACK_PROMPT.examples) + 1
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[-1] == {"role": "user", "content": "明日の天気は？"}


def test_build_ack_messages_uses_given_prompt_and_examples() -> None:
    prompt = AckPrompt("つなぎ役です。", (("天気は？", "ちょっと見てみるね。"),))
    assert build_ack_messages("ニュースある？", prompt) == [
        {"role": "system", "content": "つなぎ役です。"},
        {"role": "user", "content": "天気は？"},
        {"role": "assistant", "content": "ちょっと見てみるね。"},
        {"role": "user", "content": "ニュースある？"},
    ]


def test_build_ack_messages_without_examples() -> None:
    messages = build_ack_messages("ニュースある？", AckPrompt("つなぎ役です。", ()))
    assert [message["role"] for message in messages] == ["system", "user"]
