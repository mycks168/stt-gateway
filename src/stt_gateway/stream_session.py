"""/stream の 1 接続の中で、発話の開始・音声の受け取り・終了を進め、イベントを送る。"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
from pydantic import ValidationError

from .ack_generator import AckGenerator
from .audio_decoding import AudioSamples
from .backchannel_policy import BackchannelPredictor, BackchannelTrigger, PhraseRotation
from .backchannel_pool import BackchannelPool, LeasedPredictor
from .backends import BackendDescription
from .config import BackchannelSettings
from .pcm_audio import write_wav
from .serial_transcriber import SerialTranscriber
from .stream_audio_decoder import (
    STREAM_SAMPLE_RATE_HZ,
    AudioDecodeError,
    StreamAudioDecoder,
    create_stream_audio_decoder,
)
from .stream_messages import (
    CLIENT_MESSAGE_ADAPTER,
    AckEvent,
    BackchannelEvent,
    EndMessage,
    ErrorEvent,
    StartedEvent,
    StartMessage,
    StreamErrorCode,
    StreamEvent,
    TranscriptEvent,
    UtteranceId,
)
from .transcription import Transcription

logger = logging.getLogger(__name__)

# 相槌の予測モデルが日本語用のため、/stream は日本語だけを受け付ける。
SUPPORTED_LANGUAGES: tuple[str, ...] = ("ja",)
MILLISECONDS_PER_SECOND = 1000


class EventSender(Protocol):
    """クライアントへイベントを送る。切断後に呼ばれても例外にしない。"""

    async def send(self, event: StreamEvent) -> None:
        """イベントを 1 つ送る。"""
        ...


@dataclass(frozen=True)
class StreamDependencies:
    """/stream の処理に必要な、ワーカー内で共有する部品と設定。"""

    serial_transcriber: SerialTranscriber
    backchannel_pool: BackchannelPool
    ack_generator: AckGenerator
    backend: BackendDescription
    default_language: str
    max_utterance_seconds: float
    backchannel: BackchannelSettings


class _BackchannelStream:
    """1 つの発話の音声を予測器へ渡し、相槌のイベントを送る。

    予測は CPU を使う重い処理のため別スレッドで行い、音声の受信を止めない。
    予測器は発話が終わってスレッド上の処理も終わってから返す。予測中に返すと、
    次の発話が同じ予測器を同時に使ってしまうため。
    """

    def __init__(
        self,
        utterance_id: UtteranceId,
        predictor: BackchannelPredictor,
        trigger: BackchannelTrigger,
        pool: BackchannelPool,
        sender: EventSender,
    ) -> None:
        """予測に使う部品を受け取り、予測のタスクを始める。"""
        self._utterance_id = utterance_id
        self._predictor = predictor
        self._trigger = trigger
        self._pool = pool
        self._sender = sender
        self._inputs: asyncio.Queue[AudioSamples | None] = asyncio.Queue()
        self._is_open = True
        self.task = asyncio.create_task(self._run())

    def feed(self, samples: AudioSamples) -> None:
        """音声を予測の待ち行列に入れる。"""
        self._inputs.put_nowait(samples)

    def close(self) -> None:
        """発話の終了。これ以降に確定した予測は送らない。"""
        self._is_open = False
        self._inputs.put_nowait(None)

    async def _run(self) -> None:
        """待ち行列の音声を予測し、相槌を出すと決めたらイベントを送る。最後に予測器を返す。"""
        try:
            await asyncio.to_thread(self._predictor.reset)
            while (samples := await self._next_batch()) is not None:
                if self._is_open:
                    await self._predict_and_send(samples)
        except Exception:
            logger.exception("backchannel prediction failed (utterance_id=%d)", self._utterance_id)
            await self._discard_until_closed()
        finally:
            self._pool.release(self._predictor)

    async def _next_batch(self) -> AudioSamples | None:
        """次の音声を取り出す。予測が遅れて溜まっていれば、まとめて 1 回で予測する。"""
        first = await self._inputs.get()
        if first is None:
            return None
        batch = [first]
        while not self._inputs.empty():
            queued = self._inputs.get_nowait()
            if queued is None:
                # 終了の印は戻し、溜まっていた分を予測した後のループで受け取る。
                self._inputs.put_nowait(None)
                break
            batch.append(queued)
        return np.concatenate(batch)

    async def _predict_and_send(self, samples: AudioSamples) -> None:
        """予測して、相槌を出すフレームがあればイベントを送る。"""
        frames = await asyncio.to_thread(self._predictor.predict, samples)
        for frame in frames:
            decision = self._trigger.observe(frame)
            if decision is None or not self._is_open:
                continue
            await self._sender.send(
                BackchannelEvent(
                    utterance_id=self._utterance_id,
                    text=decision.text,
                    offset_ms=decision.offset_ms,
                    probability=decision.probability,
                )
            )

    async def _discard_until_closed(self) -> None:
        """予測に失敗した後は、終了の印が来るまで音声を捨てる。"""
        while await self._inputs.get() is not None:
            pass


@dataclass
class _ReceivingUtterance:
    """音声を受け取っている発話。"""

    utterance_id: UtteranceId
    language: str
    decoder: StreamAudioDecoder
    backchannel: _BackchannelStream | None
    chunks: list[AudioSamples] = field(default_factory=list)
    sample_count: int = 0


@dataclass(frozen=True)
class _DiscardingUtterance:
    """上限を超えて破棄し、end まで音声を捨てている発話。"""

    utterance_id: UtteranceId


@dataclass(frozen=True)
class _FinishedUtterance:
    """end を受け取り、文字起こしと一時的な返事を作る発話。"""

    utterance_id: UtteranceId
    language: str
    samples: AudioSamples


class StreamSession:
    """1 つの WebSocket 接続の中での発話の進行。発話は 1 つずつ順に受け付ける。"""

    def __init__(self, dependencies: StreamDependencies, sender: EventSender) -> None:
        """共有の部品と、イベントの送り先を受け取る。"""
        self._dependencies = dependencies
        self._sender = sender
        self._rotation = PhraseRotation(dependencies.backchannel.phrases)
        self._current: _ReceivingUtterance | _DiscardingUtterance | None = None
        self._last_utterance_id = 0
        self._background_tasks: set[asyncio.Task[None]] = set()

    async def handle_text(self, text: str) -> None:
        """テキストのメッセージ（start / end）を処理する。"""
        try:
            message = CLIENT_MESSAGE_ADAPTER.validate_json(text)
        except ValidationError as error:
            await self._send_error(
                "invalid_message", f"invalid message: {error.errors()[0]['msg']}"
            )
            return
        if isinstance(message, StartMessage):
            await self._start_utterance(message)
        else:
            await self._end_utterance(message)

    async def handle_audio(self, payload: bytes) -> None:
        """バイナリのメッセージ（音声）を処理する。"""
        utterance = self._current
        if utterance is None:
            await self._send_error(
                "invalid_state", "audio received while no utterance is in progress"
            )
            return
        if isinstance(utterance, _DiscardingUtterance):
            return
        try:
            samples = utterance.decoder.decode(payload)
        except AudioDecodeError as error:
            await self._send_error("invalid_audio", str(error))
            return
        utterance.chunks.append(samples)
        utterance.sample_count += len(samples)
        if utterance.sample_count > self._max_utterance_samples:
            await self._discard_too_long(utterance)
            return
        if utterance.backchannel is not None:
            utterance.backchannel.feed(samples)

    async def close(self) -> None:
        """切断時の後始末。受け取り途中の発話は文字起こしせずに捨てる。

        文字起こし中のタスクは止めない。推論のスレッドは途中で止められず、待たずに止めると
        SerialTranscriber の順番待ちを抜けて、GPU 上の推論が重なってしまうため。
        """
        if isinstance(self._current, _ReceivingUtterance) and self._current.backchannel is not None:
            self._current.backchannel.close()
        self._current = None

    async def wait_for_background_tasks(self) -> None:
        """文字起こしと一時的な返事のタスクがすべて終わるまで待つ。"""
        while self._background_tasks:
            await asyncio.gather(*self._background_tasks)

    @property
    def _max_utterance_samples(self) -> int:
        """1 発話の音声のサンプル数の上限。"""
        return int(self._dependencies.max_utterance_seconds * STREAM_SAMPLE_RATE_HZ)

    async def _start_utterance(self, message: StartMessage) -> None:
        """発話を始め、予測器を借りられれば相槌の予測も始める。"""
        if self._current is not None:
            await self._send_error(
                "invalid_state", "start received while an utterance is in progress"
            )
            return
        language = message.language or self._dependencies.default_language
        if language not in SUPPORTED_LANGUAGES:
            await self._send_error(
                "unsupported_language",
                f"unsupported language for stream: '{language}' (supported: {', '.join(SUPPORTED_LANGUAGES)})",
            )
            return
        self._last_utterance_id += 1
        utterance_id = UtteranceId(self._last_utterance_id)
        lease = await self._dependencies.backchannel_pool.acquire()
        backchannel = (
            self._start_backchannel(utterance_id, lease)
            if isinstance(lease, LeasedPredictor)
            else None
        )
        self._current = _ReceivingUtterance(
            utterance_id, language, create_stream_audio_decoder(message.encoding), backchannel
        )
        await self._sender.send(
            StartedEvent(
                utterance_id=utterance_id,
                backchannel=backchannel is not None,
                backchannel_unavailable_reason=None
                if isinstance(lease, LeasedPredictor)
                else lease.reason,
            )
        )

    def _start_backchannel(
        self, utterance_id: UtteranceId, lease: LeasedPredictor
    ) -> _BackchannelStream:
        """借りた予測器で、この発話の相槌の予測を始める。"""
        settings = self._dependencies.backchannel
        trigger = BackchannelTrigger(settings.threshold, settings.min_interval_ms, self._rotation)
        return _BackchannelStream(
            utterance_id,
            lease.predictor,
            trigger,
            self._dependencies.backchannel_pool,
            self._sender,
        )

    async def _end_utterance(self, message: EndMessage) -> None:
        """発話を終え、文字起こしと一時的な返事をバックグラウンドで作る。"""
        utterance = self._current
        if utterance is None:
            await self._send_error(
                "invalid_state", f"'{message.type}' received while no utterance is in progress"
            )
            return
        self._current = None
        if isinstance(utterance, _DiscardingUtterance):
            return
        if utterance.backchannel is not None:
            utterance.backchannel.close()
        samples = (
            np.concatenate(utterance.chunks) if utterance.chunks else np.zeros(0, dtype=np.float32)
        )
        self._spawn(
            self._finish(_FinishedUtterance(utterance.utterance_id, utterance.language, samples))
        )

    async def _discard_too_long(self, utterance: _ReceivingUtterance) -> None:
        """上限を超えた発話を破棄し、end まで音声を捨てる。"""
        if utterance.backchannel is not None:
            utterance.backchannel.close()
        self._current = _DiscardingUtterance(utterance.utterance_id)
        await self._send_error(
            "utterance_too_long",
            f"utterance exceeds {self._dependencies.max_utterance_seconds:g} seconds",
        )

    async def _finish(self, utterance: _FinishedUtterance) -> None:
        """文字起こし結果を送り、文字起こしできたら一時的な返事を送る。"""
        transcript = await self._send_transcript(utterance)
        if transcript is None or not transcript.text:
            return
        sent_at = time.perf_counter()
        ack = await self._dependencies.ack_generator.generate(transcript.text)
        await self._sender.send(
            AckEvent(
                utterance_id=utterance.utterance_id,
                text=ack.text,
                source=ack.source,
                latency_ms=_elapsed_ms(sent_at),
            )
        )

    async def _send_transcript(self, utterance: _FinishedUtterance) -> Transcription | None:
        """文字起こしして transcript を送る。失敗したら error を送って None を返す。"""
        started_at = time.perf_counter()
        try:
            transcription = await self._transcribe(utterance)
        except Exception as error:
            logger.exception(
                "stream transcription failed (utterance_id=%d)", utterance.utterance_id
            )
            await self._send_error(
                "transcribe_failed", f"transcribe_failed: {error}", utterance.utterance_id
            )
            return None
        backend = self._dependencies.backend
        await self._sender.send(
            TranscriptEvent(
                utterance_id=utterance.utterance_id,
                text=transcription.text,
                language=transcription.language,
                duration=transcription.duration_seconds,
                latency_ms=_elapsed_ms(started_at),
                backend=backend.backend,
                model=backend.model,
            )
        )
        return transcription

    async def _transcribe(self, utterance: _FinishedUtterance) -> Transcription:
        """音声を WAV の一時ファイルにして、/transcribe と同じ順番待ちで文字起こしする。"""
        if len(utterance.samples) == 0:
            # 音声の無い発話はバックエンドに渡すと失敗しうるため、推論せずに空の結果にする。
            return Transcription(text="", language=utterance.language, duration_seconds=0.0)
        # 60 秒の音声で約 2MB を書き出すため、ファイル操作もイベントループの外で行う。
        audio_path = await asyncio.to_thread(_write_temporary_wav, utterance.samples)
        try:
            return await self._dependencies.serial_transcriber.transcribe(
                audio_path, utterance.language
            )
        finally:
            await asyncio.to_thread(audio_path.unlink, missing_ok=True)

    async def _send_error(
        self, code: StreamErrorCode, detail: str, utterance_id: UtteranceId | None = None
    ) -> None:
        """error を送る。発話を指定しなければ、受け取り中の発話に結び付ける。"""
        if utterance_id is None and self._current is not None:
            utterance_id = self._current.utterance_id
        await self._sender.send(ErrorEvent(utterance_id=utterance_id, code=code, detail=detail))

    def _spawn(self, coroutine: Coroutine[object, object, None]) -> None:
        """バックグラウンドのタスクを始め、終わるまで参照を持っておく（途中で回収されないため）。"""
        task = asyncio.create_task(coroutine)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


def _write_temporary_wav(samples: AudioSamples) -> Path:
    """波形を WAV の一時ファイルに書き出してパスを返す。削除は呼び出し側が行う。"""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
        audio_path = Path(temp_file.name)
    write_wav(audio_path, samples, STREAM_SAMPLE_RATE_HZ)
    return audio_path


def _elapsed_ms(started_at: float) -> int:
    """started_at（perf_counter の値）からの経過時間をミリ秒で返す。"""
    return int((time.perf_counter() - started_at) * MILLISECONDS_PER_SECOND)
