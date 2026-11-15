from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from stt_gateway.backchannel_judge import (
    KIND_CRITERIA,
    BackchannelKindJudge,
    read_judge_token,
)
from stt_gateway.backchannel_policy import BACKCHANNEL_KINDS, BackchannelKind

Handler = Callable[[httpx.Request], httpx.Response]


def _judge(handler: Handler, text: str = "猫が死んじゃってさ。") -> BackchannelKind | None:
    """ダミーの Kev につないだ判定器で 1 回判定する。"""

    async def scenario() -> BackchannelKind | None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await BackchannelKindJudge("http://kev.test/", "secret", client).judge(text)

    return asyncio.run(scenario())


def test_judge_sends_english_kind_question_with_bearer_token() -> None:
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """リクエストを記録して種類を返す。"""
        requests.append(request)
        return httpx.Response(
            200, json={"answers": {"kind": {"choice": "sad_or_trouble", "confidence": 0.4}}}
        )

    assert _judge(handler) == "sad_or_trouble"
    request = requests[0]
    assert str(request.url) == "http://kev.test/v1/systemone"
    assert request.headers["Authorization"] == "Bearer secret"
    body = json.loads(request.content)
    assert body["state"] == "猫が死んじゃってさ。"
    question = body["questions"]["kind"]
    assert question["type"] == "choice"
    assert "transcript of Japanese speech" in question["instructions"]
    assert list(question["criteria"]) == list(BACKCHANNEL_KINDS)


def test_kind_criteria_cover_every_kind() -> None:
    assert set(KIND_CRITERIA) == set(BACKCHANNEL_KINDS)


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "試さないとわからないかな。でも相槌が来てない気がするんだけど。",
            "でも相槌が来てない気がするんだけど。",
        ),
        ("どうすればいい？設定を変えたら直ったんだ。", "設定を変えたら直ったんだ。"),
        ("それで。そのあと電源を入れ直して", "そのあと電源を入れ直して"),
        ("えっと、それで、その、", "えっと、それで、その、"),
        ("  昨日の話。今日は「疲れた！」  ", "今日は「疲れた！」"),
        ("昨日の話！\nでも、今日は違うんだよね?", "でも、今日は違うんだよね?"),
        ("。！？", "。！？"),
        ("", ""),
    ],
)
def test_judge_classifies_current_sentence_without_earlier_speech(text: str, expected: str) -> None:
    """過去の質問・不確かさを残さず、現在の文とその末尾をモデルへ渡す。"""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        """判定対象を記録する。"""
        requests.append(request)
        return httpx.Response(200, json={"answers": {"kind": {"choice": "continuing"}}})

    assert _judge(handler, text) == "continuing"
    assert json.loads(requests[0].content)["state"] == expected


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(401, json={"detail": "unauthorized"}),
        httpx.Response(200, json={"answers": {"kind": {"choice": "angry"}}}),
        httpx.Response(200, json={"unexpected": True}),
    ],
)
def test_judge_returns_none_on_bad_response(
    response: httpx.Response, caplog: pytest.LogCaptureFixture
) -> None:
    assert _judge(lambda _request: response) is None
    assert "http://kev.test/v1/systemone" in caplog.text, "ログに接続先を含めること"


def test_judge_returns_none_when_unreachable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        """接続できない状態を再現する。"""
        raise httpx.ConnectError("connection refused", request=request)

    assert _judge(handler) is None


def test_read_judge_token(tmp_path: Path) -> None:
    token_file = tmp_path / "api_key"
    token_file.write_text("  secret\n")
    assert read_judge_token(token_file) == "secret"


def test_read_judge_token_rejects_missing_or_empty_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="failed to read backchannel judge token"):
        read_judge_token(tmp_path / "missing")
    empty = tmp_path / "empty"
    empty.write_text("\n")
    with pytest.raises(ValueError, match="is empty"):
        read_judge_token(empty)


def test_kind_criteria_include_added_kinds() -> None:
    added = {
        "seeking_agreement",
        "opinion",
        "happy",
        "complaint",
        "step_by_step",
        "thinking_aloud",
        "wrapping_up",
    }
    assert added <= set(KIND_CRITERIA)
    assert len(KIND_CRITERIA) == 14
