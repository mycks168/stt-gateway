from __future__ import annotations

import pytest

from stt_gateway.ack_prompt import FEW_SHOT_EXAMPLES, SYSTEM_PROMPT, build_ack_messages
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
    messages = build_ack_messages("明日の天気は？")
    assert messages[0] == {"role": "system", "content": SYSTEM_PROMPT}
    assert len(messages) == 1 + 2 * len(FEW_SHOT_EXAMPLES) + 1
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[-1] == {"role": "user", "content": "明日の天気は？"}
