"""/stream の WebSocket を受け付け、認証して、受信したメッセージを StreamSession に渡す。"""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, FastAPI, WebSocket, WebSocketDisconnect, status

from .auth import extract_bearer_token, is_token_accepted
from .config import settings
from .stream_messages import StreamEvent
from .stream_session import StreamDependencies, StreamSession

router = APIRouter()


class WebSocketEventSender:
    """イベントを JSON のテキストメッセージとして送る。

    文字起こしと相槌のタスクが同時に送ることがあるため、送信をロックで 1 つずつにする。
    切断後に届いたイベントは捨てる。
    """

    def __init__(self, websocket: WebSocket) -> None:
        """送り先の WebSocket を受け取る。"""
        self._websocket = websocket
        self._lock = asyncio.Lock()
        self._is_closed = False

    async def send(self, event: StreamEvent) -> None:
        """イベントを送る。切断済みなら何もしない。"""
        async with self._lock:
            if self._is_closed:
                return
            try:
                await self._websocket.send_text(event.model_dump_json())
            except (WebSocketDisconnect, RuntimeError):
                # RuntimeError は Starlette がクローズ後の送信で送出する。
                self._is_closed = True

    def mark_closed(self) -> None:
        """切断されたことを記録し、以降の送信を捨てる。"""
        self._is_closed = True


def get_stream_dependencies(app: FastAPI) -> StreamDependencies:
    """lifespan で用意した、このワーカーの /stream 用の部品を返す。"""
    dependencies = getattr(app.state, "stream_dependencies", None)
    if not isinstance(dependencies, StreamDependencies):
        raise RuntimeError("stream_dependencies is not initialized: app started without lifespan")
    return dependencies


def is_authorized(websocket: WebSocket) -> bool:
    """Authorization ヘッダの Bearer トークンを /transcribe と同じ規則で検証する。"""
    token = extract_bearer_token(websocket.headers.get("authorization"))
    return token is not None and is_token_accepted(token, settings.api_key)


@router.websocket("/stream")
async def stream(websocket: WebSocket) -> None:
    """音声を受け取り、相槌・文字起こし結果・一時的な返事をイベントとして返す。"""
    if not is_authorized(websocket):
        # accept 前に閉じると、WebSocket を確立せず HTTP 403 で拒否される。
        await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
        return
    await websocket.accept()
    sender = WebSocketEventSender(websocket)
    session = StreamSession(get_stream_dependencies(websocket.app), sender)
    try:
        await _receive_until_disconnected(websocket, session)
    finally:
        sender.mark_closed()
        await session.close()


async def _receive_until_disconnected(websocket: WebSocket, session: StreamSession) -> None:
    """切断されるまでメッセージを受け取り、テキストとバイナリで振り分ける。"""
    while True:
        message = await websocket.receive()
        if message["type"] == "websocket.disconnect":
            return
        text = message.get("text")
        if text is not None:
            await session.handle_text(text)
            continue
        payload = message.get("bytes")
        if payload is not None:
            await session.handle_audio(payload)
