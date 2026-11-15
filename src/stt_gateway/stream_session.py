"""/stream の 1 接続の中で、発話の開始・音声の受け取り・終了を進め、イベントを送る。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine
from dataclasses import dataclass, field

import numpy as np
from pydantic import ValidationError

from .ack_generator import AckGenerator, AckRequest
from .audio_decoding import AudioSamples
from .backchannel_policy import BackchannelKind, BackchannelTrigger, PhraseRotation
from .backchannel_pool import BackchannelPool, LeasedPredictor, PredictorRefusal
from .backchannel_stream import (
    BackchannelEmitter,
    BackchannelStream,
    KindEmitter,
    KindJudging,
    KindTarget,
    RotationEmitter,
    samples_until_offset,
)
from .backends import BackendDescription
from .config import AckSettings, BackchannelSettings
from .sample_transcription import transcribe_samples
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
    EndMessage,
    ErrorEvent,
    EventSender,
    StartedEvent,
    StartMessage,
    StreamErrorCode,
    TranscriptEvent,
    UtteranceId,
)
from .stream_options import (
    BackchannelRule,
    KindPhrasing,
    RotationPhrasing,
    resolve_ack_request,
    resolve_backchannel_rule,
)
from .transcription import Transcription

logger = logging.getLogger(__name__)

# 相槌の予測モデルが日本語用のため、/stream は日本語だけを受け付ける。
SUPPORTED_LANGUAGES: tuple[str, ...] = ("ja",)
MILLISECONDS_PER_SECOND = 1000


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
    ack: AckSettings
    kind_judging: KindJudging | None


@dataclass
class _ReceivingUtterance:
    """音声を受け取っている発話。"""

    utterance_id: UtteranceId
    language: str
    decoder: StreamAudioDecoder
    ack_request: AckRequest | None
    backchannel: BackchannelStream | None = None
    chunks: list[AudioSamples] = field(default_factory=list)
    sample_count: int = 0

    def samples_until(self, offset_ms: int) -> AudioSamples:
        """受け取った音声の先頭から offset_ms までを返す。"""
        return samples_until_offset(self.chunks, offset_ms)


@dataclass(frozen=True)
class _KindPlan:
    """話の種類で文言を選ぶときの、種類ごとの文言と判定の部品。"""

    phrasing: KindPhrasing
    judging: KindJudging


@dataclass(frozen=True)
class _DiscardingUtterance:
    """上限を超えて破棄し、end まで音声を捨てている発話。"""

    utterance_id: UtteranceId


@dataclass(frozen=True)
class _FinishedUtterance:
    """end を受け取り、文字起こしと一時的な返事を作る発話。ack_request が None なら返事は作らない。"""

    utterance_id: UtteranceId
    language: str
    samples: AudioSamples
    ack_request: AckRequest | None


class StreamSession:
    """1 つの WebSocket 接続の中での発話の進行。発話は 1 つずつ順に受け付ける。"""

    def __init__(self, dependencies: StreamDependencies, sender: EventSender) -> None:
        """共有の部品と、イベントの送り先を受け取る。"""
        self._dependencies = dependencies
        self._sender = sender
        self._rotation_phrases = dependencies.backchannel.phrases
        self._rotation = PhraseRotation(self._rotation_phrases)
        self._kind_rotations: dict[BackchannelKind, tuple[tuple[str, ...], PhraseRotation]] = {}
        self._current: _ReceivingUtterance | _DiscardingUtterance | None = None
        self._last_utterance_id = 0
        self._background_tasks: set[asyncio.Task[object]] = set()

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
        rule = resolve_backchannel_rule(message.backchannel, self._dependencies.backchannel)
        utterance = _ReceivingUtterance(
            utterance_id,
            language,
            create_stream_audio_decoder(message.encoding),
            resolve_ack_request(message.ack, self._dependencies.ack),
        )
        plan = self._plan_phrasing(rule.phrasing)
        lease = (
            PredictorRefusal("unavailable")
            if plan is None
            else await self._dependencies.backchannel_pool.acquire()
        )
        if isinstance(lease, LeasedPredictor) and plan is not None:
            utterance.backchannel = self._start_backchannel(utterance, lease, rule, plan)
        self._current = utterance
        await self._sender.send(
            StartedEvent(
                utterance_id=utterance_id,
                backchannel=utterance.backchannel is not None,
                backchannel_unavailable_reason=None
                if isinstance(lease, LeasedPredictor)
                else lease.reason,
            )
        )

    def _plan_phrasing(
        self, phrasing: RotationPhrasing | KindPhrasing
    ) -> RotationPhrasing | _KindPlan | None:
        """文言の決め方を決める。話の種類で選ぶ指定なのに判定を使えないときは None を返す。"""
        if isinstance(phrasing, RotationPhrasing):
            return phrasing
        if self._dependencies.kind_judging is None:
            return None
        return _KindPlan(phrasing, self._dependencies.kind_judging)

    def _start_backchannel(
        self,
        utterance: _ReceivingUtterance,
        lease: LeasedPredictor,
        rule: BackchannelRule,
        plan: RotationPhrasing | _KindPlan,
    ) -> BackchannelStream:
        """借りた予測器で、この発話の相槌の予測を始める。"""
        return BackchannelStream(
            utterance.utterance_id,
            lease.predictor,
            BackchannelTrigger(rule.threshold, rule.min_interval_ms),
            self._dependencies.backchannel_pool,
            self._emitter_for(utterance, plan),
        )

    def _emitter_for(
        self, utterance: _ReceivingUtterance, plan: RotationPhrasing | _KindPlan
    ) -> BackchannelEmitter:
        """文言を順に回すか、話の種類で選ぶかに応じた emitter を作る。"""
        if isinstance(plan, RotationPhrasing):
            return RotationEmitter(
                utterance.utterance_id, self._rotation_for(plan.phrases), self._sender
            )
        return KindEmitter(
            KindTarget(utterance.utterance_id, utterance.language, utterance.samples_until),
            self._kind_rotations_for(plan.phrasing),
            plan.judging,
            self._sender,
            self._track,
        )

    def _rotation_for(self, phrases: tuple[str, ...]) -> PhraseRotation:
        """文言の順番を返す。前の発話と文言が違うときは、新しい文言の先頭から数え直す。"""
        if phrases != self._rotation_phrases:
            self._rotation_phrases = phrases
            self._rotation = PhraseRotation(phrases)
        return self._rotation

    def _kind_rotations_for(self, phrasing: KindPhrasing) -> dict[BackchannelKind, PhraseRotation]:
        """種類ごとの文言の順番を返す。前の発話と文言が同じ種類は、順番を引き継ぐ。"""
        rotations: dict[BackchannelKind, PhraseRotation] = {}
        for kind, phrases in phrasing.phrases_by_kind.items():
            previous = self._kind_rotations.get(kind)
            if previous is None or previous[0] != phrases:
                previous = (phrases, PhraseRotation(phrases))
                self._kind_rotations[kind] = previous
            rotations[kind] = previous[1]
        return rotations

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
            self._finish(
                _FinishedUtterance(
                    utterance.utterance_id, utterance.language, samples, utterance.ack_request
                )
            )
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
        if transcript is None or not transcript.text or utterance.ack_request is None:
            return
        sent_at = time.perf_counter()
        ack = await self._dependencies.ack_generator.generate(
            transcript.text, utterance.ack_request
        )
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
            transcription = await transcribe_samples(
                self._dependencies.serial_transcriber, utterance.samples, utterance.language
            )
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

    async def _send_error(
        self, code: StreamErrorCode, detail: str, utterance_id: UtteranceId | None = None
    ) -> None:
        """error を送る。発話を指定しなければ、受け取り中の発話に結び付ける。"""
        if utterance_id is None and self._current is not None:
            utterance_id = self._current.utterance_id
        await self._sender.send(ErrorEvent(utterance_id=utterance_id, code=code, detail=detail))

    def _spawn(self, coroutine: Coroutine[object, object, None]) -> None:
        """バックグラウンドのタスクを始めて登録する。"""
        self._track(asyncio.create_task(coroutine))

    def _track(self, task: asyncio.Task[object]) -> None:
        """タスクが終わるまで参照を持っておく（途中で回収されないため）。"""
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)


def _elapsed_ms(started_at: float) -> int:
    """started_at（perf_counter の値）からの経過時間をミリ秒で返す。"""
    return int((time.perf_counter() - started_at) * MILLISECONDS_PER_SECOND)
