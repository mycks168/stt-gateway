from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Coroutine

import httpx
import pytest

from stt_gateway.ack_generator import Ack, AckGenerator, AckRequest
from stt_gateway.ack_prompt import DEFAULT_ACK_PROMPT, AckPrompt
from stt_gateway.config import AckSettings

_SETTINGS = AckSettings(
    ollama_url="http://ollama.test:11434",
    model="gemma4:e2b",
    timeout_ms=200,
    max_chars=20,
    fallback_text="ちょっと確認するね。",
    keep_alive="30m",
    load_timeout_ms=1000,
)

_DEFAULT_REQUEST = AckRequest(DEFAULT_ACK_PROMPT, "ちょっと確認するね。", 20)

Handler = Callable[[httpx.Request], Coroutine[None, None, httpx.Response]]


def _reply(content: str) -> httpx.Response:
    """Ollama chat API（stream: false）と同じ形の応答を作る。"""
    return httpx.Response(
        200, json={"message": {"role": "assistant", "content": content}, "done": True}
    )


def _generate(
    handler: Handler, transcript: str = "明日の天気は？", request: AckRequest = _DEFAULT_REQUEST
) -> Ack:
    """ダミーの Ollama につないだ AckGenerator で返事を 1 つ作る。"""

    async def scenario() -> Ack:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await AckGenerator(_SETTINGS, client).generate(transcript, request)

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


def _load_reply() -> httpx.Response:
    """Ollama の読み込みだけの要求への応答を作る。"""
    return httpx.Response(200, json={"model": "gemma4:e2b", "done": True, "done_reason": "load"})


def test_preload_sends_load_only_request_and_logs_elapsed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """リクエストを記録して、読み込み完了を返す。"""
        requests.append(request)
        return _load_reply()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = AckGenerator(_SETTINGS, client)
            generator.request_preload()
            generator.request_preload()  # 読み込み中は 2 本目を送らない
            await asyncio.sleep(0.05)
            await generator.close()

    with caplog.at_level(logging.INFO):
        asyncio.run(scenario())
    assert len(requests) == 1
    assert str(requests[0].url) == "http://ollama.test:11434/api/generate"
    assert json.loads(requests[0].content) == {"model": "gemma4:e2b", "keep_alive": "30m"}
    assert "ack model preloaded (model=gemma4:e2b, elapsed_ms=" in caplog.text


def test_preload_continues_after_reply_times_out() -> None:
    finished_paths: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """読み込みも返事も、STT_ACK_TIMEOUT_MS より長くかかる状態を再現する。"""
        await asyncio.sleep(0.4)
        finished_paths.append(request.url.path)
        return _load_reply() if request.url.path == "/api/generate" else _reply("調べるね。")

    async def scenario() -> Ack:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = AckGenerator(_SETTINGS, client)
            generator.request_preload()
            ack = await generator.generate("明日の天気は？", _DEFAULT_REQUEST)
            await asyncio.sleep(0.4)
            return ack

    assert asyncio.run(scenario()).source == "fallback"
    assert finished_paths == ["/api/generate"], "返事が時間切れでも、読み込みは最後まで待つこと"


def test_preload_failure_is_logged_and_can_be_retried(caplog: pytest.LogCaptureFixture) -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        """記録してから失敗させる。"""
        requests.append(request)
        raise httpx.ConnectError("connection refused", request=request)

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = AckGenerator(_SETTINGS, client)
            for _ in range(2):
                generator.request_preload()
                await asyncio.sleep(0.05)

    asyncio.run(scenario())
    assert len(requests) == 2, "前の読み込みが終わっていれば、次の要求でまた読み込む"
    assert "ack model preload failed (url=http://ollama.test:11434/api/generate" in caplog.text


def test_close_cancels_preload_in_progress() -> None:
    is_cancelled = asyncio.Event()

    async def handler(_request: httpx.Request) -> httpx.Response:
        """取り消されるまで読み込みを終えない。"""
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            is_cancelled.set()
            raise
        return _load_reply()

    async def scenario() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            generator = AckGenerator(_SETTINGS, client)
            generator.request_preload()
            await asyncio.sleep(0.05)
            await generator.close()
            await generator.close()  # 終わった後に呼んでも何もしない

    asyncio.run(scenario())
    assert is_cancelled.is_set()


def test_generate_uses_request_prompt_limit_and_fallback() -> None:
    requests: list[httpx.Request] = []
    request = AckRequest(AckPrompt("つなぎ役です。", ()), "えっと、ちょっと待ってね。", 5)

    async def handler(http_request: httpx.Request) -> httpx.Response:
        """リクエストを記録して、上限の 5 文字を超える返事を返す。"""
        requests.append(http_request)
        return _reply("ちょっと見てみるね。")

    assert _generate(handler, request=request) == Ack("えっと、ちょっと待ってね。", "fallback")
    messages = json.loads(requests[0].content)["messages"]
    assert messages == [
        {"role": "system", "content": "つなぎ役です。"},
        {"role": "user", "content": "明日の天気は？"},
    ]
