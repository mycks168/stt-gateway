from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Coroutine

import httpx
import pytest

from stt_gateway.ack_generator import Ack, AckGenerator
from stt_gateway.config import AckSettings

_SETTINGS = AckSettings(
    ollama_url="http://ollama.test:11434",
    model="gemma4:e2b",
    timeout_ms=200,
    max_chars=20,
    fallback_text="ちょっと確認するね。",
    keep_alive="30m",
)

Handler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


def _reply(content: str) -> httpx.Response:
    """Ollama chat API（stream: false）と同じ形の応答を作る。"""
    return httpx.Response(
        200, json={"message": {"role": "assistant", "content": content}, "done": True}
    )


def _generate(handler: Handler, transcript: str = "明日の天気は？") -> Ack:
    """ダミーの Ollama につないだ AckGenerator で返事を 1 つ作る。"""

    async def scenario() -> Ack:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await AckGenerator(_SETTINGS, client).generate(transcript)

    return asyncio.run(scenario())


def test_generate_sends_listener_prompt_and_returns_llm_text() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """リクエストを記録して返事を返す。"""
        requests.append(request)
        return _reply("「明日の天気ね、確認するね。」")

    assert _generate(handler) == Ack("明日の天気ね、確認するね。", "llm")

    body = json.loads(requests[0].content)
    assert str(requests[0].url) == "http://ollama.test:11434/api/chat"
    assert body["model"] == "gemma4:e2b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["keep_alive"] == "30m"
    assert "聞き手" in body["messages"][0]["content"]
    assert body["messages"][-1] == {"role": "user", "content": "明日の天気は？"}


@pytest.mark.parametrize("content", ["明日は晴れで最高気温は二十五度になる見込みです。", "  "])
def test_generate_falls_back_when_text_is_unusable(content: str) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        """長すぎる、または空の返事を返す。"""
        return _reply(content)

    assert _generate(handler) == Ack("ちょっと確認するね。", "fallback")


def test_generate_falls_back_on_timeout() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        """STT_ACK_TIMEOUT_MS より長く待たせる。"""
        await asyncio.sleep(1)
        return _reply("調べるね。")

    assert _generate(handler).source == "fallback"


@pytest.mark.parametrize(
    "response",
    [httpx.Response(500, text="model not found"), httpx.Response(200, json={"unexpected": True})],
)
def test_generate_falls_back_on_bad_response(response: httpx.Response) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        """エラーや想定外の形の応答を返す。"""
        return response

    assert _generate(handler).source == "fallback"


def test_generate_falls_back_when_ollama_is_unreachable(caplog: pytest.LogCaptureFixture) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        """接続できない状態を再現する。"""
        raise httpx.ConnectError("connection refused", request=request)

    assert _generate(handler).source == "fallback"
    assert "http://ollama.test:11434/api/chat" in caplog.text, "ログに接続先を含めること"


def test_warm_up_requests_once_and_ignores_failure(caplog: pytest.LogCaptureFixture) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """記録してから失敗させる。"""
        requests.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await AckGenerator(_SETTINGS, client).warm_up()

    asyncio.run(scenario())
    assert len(requests) == 1
    assert "ack warm-up failed" in caplog.text
