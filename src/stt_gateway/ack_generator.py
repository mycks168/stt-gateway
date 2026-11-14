"""Ollama の軽量 LLM で一時的な返事を作り、使えないときは固定の文言に差し替える。"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Literal

import httpx
from pydantic import BaseModel, ValidationError

from .ack_prompt import DEFAULT_ACK_PROMPT, AckPrompt, build_ack_messages
from .ack_text import normalize_ack_text
from .config import AckSettings

logger = logging.getLogger(__name__)

AckSource = Literal["llm", "fallback"]

OLLAMA_CHAT_PATH = "/api/chat"
# 思考モードは返事が遅くなるだけで、一言の相槌には不要。
OLLAMA_THINK = False
OLLAMA_TEMPERATURE = 0.3
OLLAMA_MAX_OUTPUT_TOKENS = 40
WARMUP_TEXT = "こんにちは"
MILLISECONDS_PER_SECOND = 1000


@dataclass(frozen=True)
class Ack:
    """一時的な返事と、その出どころ。"""

    text: str
    source: AckSource


@dataclass(frozen=True)
class AckRequest:
    """1 つの発話の一時的な返事の作り方。クライアントの指定とサーバの設定から決まる。"""

    prompt: AckPrompt
    fallback_text: str
    max_chars: int


class _OllamaMessage(BaseModel):
    """Ollama chat API の応答のうち、使う部分。"""

    content: str


class _OllamaChatResponse(BaseModel):
    """Ollama chat API（stream: false）の応答。"""

    message: _OllamaMessage


class AckGenerator:
    """文字起こし結果から一時的な返事を作る。"""

    def __init__(self, settings: AckSettings, client: httpx.AsyncClient) -> None:
        """設定と、Ollama への接続に使う HTTP クライアントを受け取る。"""
        self._settings = settings
        self._client = client

    async def generate(self, transcript_text: str, request: AckRequest) -> Ack:
        """一時的な返事を作る。時間切れ・失敗・長すぎる場合は固定の文言を返す。"""
        timeout_seconds = self._settings.timeout_ms / MILLISECONDS_PER_SECOND
        try:
            raw_text = await asyncio.wait_for(
                self._request_reply(transcript_text, request.prompt), timeout_seconds
            )
        except (asyncio.TimeoutError, httpx.HTTPError, ValidationError) as error:
            logger.warning(
                "ack generation failed (url=%s, model=%s): %r; using fallback text",
                self._chat_url,
                self._settings.model,
                error,
            )
            return self._fallback(request)
        text = normalize_ack_text(raw_text, request.max_chars)
        if text is None:
            logger.info(
                "ack text rejected (model=%s): %r; using fallback text",
                self._settings.model,
                raw_text,
            )
            return self._fallback(request)
        return Ack(text, "llm")

    async def warm_up(self) -> None:
        """モデルを GPU に読み込ませておく。最初の返事がモデルの読み込みを待たないようにするため。"""
        try:
            await self._request_reply(WARMUP_TEXT, DEFAULT_ACK_PROMPT)
        except (httpx.HTTPError, ValidationError) as error:
            logger.warning(
                "ack warm-up failed (url=%s, model=%s): %r",
                self._chat_url,
                self._settings.model,
                error,
            )

    @property
    def _chat_url(self) -> str:
        """Ollama chat API の URL。"""
        return f"{self._settings.ollama_url}{OLLAMA_CHAT_PATH}"

    async def _request_reply(self, transcript_text: str, prompt: AckPrompt) -> str:
        """Ollama に返事を 1 つ作らせ、本文を返す。"""
        response = await self._client.post(
            self._chat_url,
            json={
                "model": self._settings.model,
                "stream": False,
                "think": OLLAMA_THINK,
                "keep_alive": self._settings.keep_alive,
                "options": {
                    "temperature": OLLAMA_TEMPERATURE,
                    "num_predict": OLLAMA_MAX_OUTPUT_TOKENS,
                },
                "messages": build_ack_messages(transcript_text, prompt),
            },
        )
        response.raise_for_status()
        return _OllamaChatResponse.model_validate_json(response.content).message.content

    @staticmethod
    def _fallback(request: AckRequest) -> Ack:
        """固定の文言の返事を作る。"""
        return Ack(request.fallback_text, "fallback")
