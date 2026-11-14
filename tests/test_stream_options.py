from __future__ import annotations

import pytest
from pydantic import ValidationError

from stt_gateway.ack_generator import AckRequest
from stt_gateway.ack_prompt import DEFAULT_ACK_PROMPT, AckPrompt
from stt_gateway.config import AckSettings, BackchannelSettings
from stt_gateway.stream_messages import (
    CLIENT_MESSAGE_ADAPTER,
    AckExample,
    AckOptions,
    BackchannelOptions,
    StartMessage,
)
from stt_gateway.stream_options import (
    BackchannelRule,
    resolve_ack_request,
    resolve_backchannel_rule,
)

_ACK_DEFAULTS = AckSettings(
    "http://ollama.test", "gemma4:e2b", 1500, 20, "ちょっと確認するね。", "30m"
)
_BACKCHANNEL_DEFAULTS = BackchannelSettings(4, 0.5, 1500, ("うん", "はい"))
# 会話例の上限（8 個）を 1 つ超える数
_NINE_EXAMPLES = ",".join(['{"user": "a", "reply": "b"}'] * 9)


def test_backchannel_rule_defaults_to_server_settings() -> None:
    assert resolve_backchannel_rule(None, _BACKCHANNEL_DEFAULTS) == BackchannelRule(
        0.5, 1500, ("うん", "はい")
    )
    assert resolve_backchannel_rule(BackchannelOptions(), _BACKCHANNEL_DEFAULTS) == BackchannelRule(
        0.5, 1500, ("うん", "はい")
    )


def test_backchannel_rule_uses_given_fields_only() -> None:
    options = BackchannelOptions(threshold=0.3, phrases=["はい", "ええ"])
    assert resolve_backchannel_rule(options, _BACKCHANNEL_DEFAULTS) == BackchannelRule(
        0.3, 1500, ("はい", "ええ")
    )
    assert resolve_backchannel_rule(
        BackchannelOptions(min_interval_ms=0), _BACKCHANNEL_DEFAULTS
    ) == BackchannelRule(0.5, 0, ("うん", "はい"))


def test_ack_request_defaults_and_disable() -> None:
    assert resolve_ack_request(None, _ACK_DEFAULTS) == AckRequest(
        DEFAULT_ACK_PROMPT, "ちょっと確認するね。", 20
    )
    assert resolve_ack_request(AckOptions(), _ACK_DEFAULTS) == AckRequest(
        DEFAULT_ACK_PROMPT, "ちょっと確認するね。", 20
    )
    assert resolve_ack_request(False, _ACK_DEFAULTS) is None


def test_ack_request_with_custom_prompt_drops_default_examples() -> None:
    request = resolve_ack_request(AckOptions(system_prompt="つなぎ役です。"), _ACK_DEFAULTS)
    assert request == AckRequest(AckPrompt("つなぎ役です。", ()), "ちょっと確認するね。", 20)


def test_ack_request_uses_given_examples_and_limits() -> None:
    options = AckOptions(
        examples=[AckExample(user="天気は？", reply="見てみるね。")],
        fallback_text="待ってね。",
        max_chars=10,
    )
    request = resolve_ack_request(options, _ACK_DEFAULTS)
    assert request == AckRequest(
        AckPrompt(DEFAULT_ACK_PROMPT.system_prompt, (("天気は？", "見てみるね。"),)),
        "待ってね。",
        10,
    )


def test_ack_request_allows_explicitly_empty_examples() -> None:
    request = resolve_ack_request(AckOptions(examples=[]), _ACK_DEFAULTS)
    assert request is not None
    assert request.prompt == AckPrompt(DEFAULT_ACK_PROMPT.system_prompt, ())


def test_start_message_parses_options() -> None:
    message = CLIENT_MESSAGE_ADAPTER.validate_json(
        '{"type": "start", "ack": {"system_prompt": "  つなぎ役です。 "},'
        ' "backchannel": {"phrases": ["はい"], "threshold": 1}}'
    )
    assert isinstance(message, StartMessage)
    assert message.ack == AckOptions(system_prompt="つなぎ役です。")
    assert message.backchannel == BackchannelOptions(phrases=["はい"], threshold=1.0)
    disabled = CLIENT_MESSAGE_ADAPTER.validate_json('{"type": "start", "ack": false}')
    assert isinstance(disabled, StartMessage)
    assert disabled.ack is False


@pytest.mark.parametrize(
    "start",
    [
        '{"type": "start", "ack": true}',
        '{"type": "start", "ack": {"system_promt": "typo"}}',
        '{"type": "start", "ack": {"system_prompt": "   "}}',
        '{"type": "start", "ack": {"system_prompt": "' + "あ" * 2001 + '"}}',
        '{"type": "start", "ack": {"examples": [' + _NINE_EXAMPLES + "]}}",
        '{"type": "start", "ack": {"examples": [{"user": "a"}]}}',
        '{"type": "start", "ack": {"max_chars": 0}}',
        '{"type": "start", "ack": {"fallback_text": "' + "あ" * 51 + '"}}',
        '{"type": "start", "backchannel": {"threshold": 0}}',
        '{"type": "start", "backchannel": {"threshold": 1.5}}',
        '{"type": "start", "backchannel": {"phrases": []}}',
        '{"type": "start", "backchannel": {"phrases": ["' + "あ" * 21 + '"]}}',
        '{"type": "start", "backchannel": {"min_interval_ms": 60001}}',
        '{"type": "start", "backchannel": {"volume": 0.5}}',
    ],
)
def test_start_message_rejects_invalid_options(start: str) -> None:
    with pytest.raises(ValidationError):
        CLIENT_MESSAGE_ADAPTER.validate_json(start)
