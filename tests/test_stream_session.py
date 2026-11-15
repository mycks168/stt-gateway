from __future__ import annotations

import asyncio
import threading
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx
import numpy as np
import pytest

from stt_gateway.ack_generator import AckGenerator
from stt_gateway.audio_decoding import AudioSamples
from stt_gateway.backchannel_policy import BackchannelFrame, BackchannelPredictor
from stt_gateway.backchannel_pool import BackchannelPool
from stt_gateway.backends import BackendDescription
from stt_gateway.config import AckSettings, BackchannelSettings
from stt_gateway.pcm_audio import samples_to_pcm_s16le
from stt_gateway.serial_transcriber import SerialTranscriber
from stt_gateway.stream_messages import AckEvent, BackchannelEvent, StreamEvent
from stt_gateway.stream_session import StreamDependencies, StreamSession
from stt_gateway.transcription import Transcription

_ACK_SETTINGS = AckSettings(
    "http://ollama.test", "gemma4:e2b", 1000, 20, "ちょっと確認するね。", "30m"
)
_BACKCHANNEL_SETTINGS = BackchannelSettings(1, 32, 0, 0, 0, 1.0, 0, "", 0.0, 200, ("うん",))
_ONE_FRAME_PCM = samples_to_pcm_s16le(np.zeros(1600, dtype=np.float32))


class _RecordingSender:
    """送られたイベントを記録する。"""

    def __init__(self) -> None:
        self.events: list[StreamEvent] = []

    async def send(self, event: StreamEvent) -> None:
        """記録する。"""
        self.events.append(event)


class _ReleaseWatchingPool(BackchannelPool):
    """予測器が返されたことを待てるプール。"""

    def __init__(self, predictor: BackchannelPredictor) -> None:
        super().__init__(1, lambda: predictor)
        self.released = asyncio.Event()

    def release(self, predictor: BackchannelPredictor) -> None:
        """返されたことを記録する。"""
        super().release(predictor)
        self.released.set()


class _GatedPredictor:
    """gate が開くまで予測を止め、その後は話してから黙ったフレームを返す予測器。"""

    def __init__(self) -> None:
        self.gate = threading.Event()
        self.is_predicting = threading.Event()

    def reset(self) -> None:
        """何もしない。"""

    def predict(self, samples: AudioSamples) -> list[BackchannelFrame]:
        """gate が開くのを待ってから、相槌を出せる無音までのフレームを返す。"""
        self.is_predicting.set()
        self.gate.wait(timeout=5)
        return [BackchannelFrame(32, 0.1), BackchannelFrame(64, 0.9)]


class _FailingPredictor:
    """予測で必ず失敗する予測器。"""

    def __init__(self) -> None:
        self.has_failed = threading.Event()

    def reset(self) -> None:
        """何もしない。"""

    def predict(self, samples: AudioSamples) -> list[BackchannelFrame]:
        """失敗させる。"""
        self.has_failed.set()
        raise RuntimeError("onnxruntime crashed")


class _FixedTranscriber:
    """固定の結果を返すダミー。"""

    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """固定の結果を返す。"""
        return Transcription("こんにちは", language, 0.1)


def _session(
    pool: BackchannelPool, sender: _RecordingSender, client: httpx.AsyncClient
) -> StreamSession:
    """ダミーの部品で StreamSession を作る。"""
    dependencies = StreamDependencies(
        serial_transcriber=SerialTranscriber(_FixedTranscriber),
        backchannel_pool=pool,
        ack_generator=AckGenerator(_ACK_SETTINGS, client),
        backend=BackendDescription("whisper", "large-v3-turbo", "auto", "int8_float16"),
        default_language="ja",
        max_utterance_seconds=60,
        backchannel=_BACKCHANNEL_SETTINGS,
        ack=_ACK_SETTINGS,
        speech_judging=None,
    )
    return StreamSession(dependencies, sender)


def _unreachable_ollama() -> httpx.AsyncClient:
    """接続できない Ollama につながる HTTP クライアント。"""

    def handler(request: httpx.Request) -> httpx.Response:
        """接続できない状態を再現する。"""
        raise httpx.ConnectError("connection refused", request=request)

    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


SessionActions = Callable[[StreamSession], Awaitable[None]]


def _run_utterance(predictor: BackchannelPredictor, actions: SessionActions) -> list[StreamEvent]:
    """PCM の発話を始めて actions を実行し、予測器が返されてタスクが終わるまで待つ。"""

    async def scenario() -> list[StreamEvent]:
        pool = _ReleaseWatchingPool(predictor)
        sender = _RecordingSender()
        async with _unreachable_ollama() as client:
            session = _session(pool, sender, client)
            await session.handle_text('{"type": "start", "encoding": "pcm_s16le"}')
            await actions(session)
            await asyncio.wait_for(pool.released.wait(), 5)
            await session.wait_for_background_tasks()
        return sender.events

    return asyncio.run(scenario())


def test_backchannel_predicted_after_end_is_not_sent() -> None:
    predictor = _GatedPredictor()

    async def end_while_predicting(session: StreamSession) -> None:
        """予測の途中で発話を終え、その後に予測を終わらせる。"""
        await session.handle_audio(_ONE_FRAME_PCM)
        await asyncio.to_thread(predictor.is_predicting.wait, 5)
        await session.handle_text('{"type": "end"}')
        predictor.gate.set()

    events = _run_utterance(predictor, end_while_predicting)
    assert not [event for event in events if isinstance(event, BackchannelEvent)]
    assert [event.type for event in events] == ["started", "transcript", "ack"]
    ack = events[-1]
    assert isinstance(ack, AckEvent)
    assert ack.source == "fallback"


def test_close_stops_backchannel_and_returns_predictor() -> None:
    predictor = _GatedPredictor()
    predictor.gate.set()

    async def disconnect(session: StreamSession) -> None:
        """発話の途中で切断する。"""
        await session.close()

    events = _run_utterance(predictor, disconnect)
    assert [event.type for event in events] == ["started"], "切断した発話は文字起こししない"


def test_prediction_failure_disables_backchannel_but_keeps_utterance(
    caplog: pytest.LogCaptureFixture,
) -> None:
    predictor = _FailingPredictor()

    async def speak_after_failure(session: StreamSession) -> None:
        """予測が失敗した後も音声を送り、発話を終える。"""
        await session.handle_audio(_ONE_FRAME_PCM)
        await asyncio.to_thread(predictor.has_failed.wait, 5)
        # 失敗した後に届いた音声は、終了の印が来るまで捨てられる
        await session.handle_audio(_ONE_FRAME_PCM)
        await session.handle_text('{"type": "end"}')

    events = _run_utterance(predictor, speak_after_failure)
    assert [event.type for event in events] == ["started", "transcript", "ack"]
    assert "backchannel prediction failed" in caplog.text
