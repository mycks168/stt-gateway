"""/stream の 1 接続の中で、発話の開始・音声の受け取り・終了を進め、イベントを送る。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
from pydantic import ValidationError

from .ack_generator import AckGenerator, AckRequest
from .audio_decoding import AudioSamples
from .backchannel_emitters import (
    BackchannelDelivery,
    BackchannelEmitter,
    JudgedUtterance,
    JudgingEmitter,
    JudgingRule,
    RotationEmitter,
    SpeechJudging,
)
from .backchannel_interval import BackchannelPacing
from .backchannel_policy import PhraseRotation
from .backchannel_pool import BackchannelPool, LeasedPredictor
from .backchannel_stream import BackchannelStream, recent_samples
from .backchannel_triggers import create_pacing, create_trigger
from .backend_selection import BackendSelection, UnsupportedBackendError, select_backend
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
    resolve_ack_request,
    resolve_backchannel_rule,
)
from .transcription import BackendName, Transcription
from .utterance_recording import UtteranceRecorder, UtteranceRecording

logger = logging.getLogger(__name__)

# 相槌の文言と区切りの判定を日本語の話し言葉で調整しているため、/stream は日本語だけを受け付ける。
SUPPORTED_LANGUAGES: tuple[str, ...] = ("ja",)
MILLISECONDS_PER_SECOND = 1000
# 相槌の判定では、途中の文字起こしを直近のこの長さだけにする。発話の頭から文字起こしすると
# 長い話ほど判定が遅れて時間切れになるため。区切りの判定が見るのは文字起こしの末尾で、
# 前の文脈は数文あれば足りる。
PARTIAL_TRANSCRIPTION_WINDOW_MS = 15_000


@dataclass(frozen=True)
class StreamDependencies:
    """/stream の処理に必要な、ワーカー内で共有する部品と設定。

    backend_descriptions は、有効なエンジンごとの transcript に載せる情報。
    """

    serial_transcriber: SerialTranscriber
    backchannel_pool: BackchannelPool
    ack_generator: AckGenerator
    backends: BackendSelection
    backend_descriptions: Mapping[BackendName, BackendDescription]
    default_language: str
    max_utterance_seconds: float
    backchannel: BackchannelSettings
    ack: AckSettings
    speech_judging: SpeechJudging | None
    recorder: UtteranceRecorder | None = None


@dataclass(frozen=True)
class _BackchannelRun:
    """発話の相槌の予測と、その決め方。"""

    stream: BackchannelStream
    rule: BackchannelRule


@dataclass(frozen=True)
class _UtteranceTarget:
    """発話の言語と、文字起こし結果に使うエンジン。"""

    language: str
    backend: BackendName


@dataclass
class _ReceivingUtterance:
    """音声を受け取っている発話。"""

    utterance_id: UtteranceId
    target: _UtteranceTarget
    decoder: StreamAudioDecoder
    ack_request: AckRequest | None
    started_at: datetime
    backchannel: _BackchannelRun | None = None
    chunks: list[AudioSamples] = field(default_factory=list)
    sample_count: int = 0

    def samples_until(self, offset_ms: int) -> AudioSamples:
        """相槌の判定に使う、offset_ms までの直近 PARTIAL_TRANSCRIPTION_WINDOW_MS の音声を返す。"""
        return recent_samples(self.chunks, offset_ms, PARTIAL_TRANSCRIPTION_WINDOW_MS)

    def all_samples(self) -> AudioSamples:
        """受け取った音声すべてを 1 つの波形にする。"""
        if not self.chunks:
            return np.zeros(0, dtype=np.float32)
        samples: AudioSamples = np.concatenate(self.chunks)
        return samples


@dataclass(frozen=True)
class _DiscardingUtterance:
    """上限を超えて破棄し、end まで音声を捨てている発話。"""

    utterance_id: UtteranceId


@dataclass(frozen=True)
class _FinishedUtterance:
    """end を受け取り、文字起こしと一時的な返事を作る発話。ack_request が None なら返事は作らない。"""

    utterance_id: UtteranceId
    target: _UtteranceTarget
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
            utterance.backchannel.stream.feed(samples)

    async def close(self) -> None:
        """切断時の後始末。受け取り途中の発話は文字起こしせずに捨てる。

        文字起こし中のタスクは止めない。推論のスレッドは途中で止められず、待たずに止めると
        SerialTranscriber の順番待ちを抜けて、GPU 上の推論が重なってしまうため。
        """
        if isinstance(self._current, _ReceivingUtterance):
            self._close_backchannel(self._current)
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
        target = await self._resolve_target(message)
        if target is None:
            return
        self._last_utterance_id += 1
        utterance_id = UtteranceId(self._last_utterance_id)
        rule = resolve_backchannel_rule(message.backchannel, self._dependencies.backchannel)
        utterance = _ReceivingUtterance(
            utterance_id,
            target,
            create_stream_audio_decoder(message.encoding),
            resolve_ack_request(message.ack, self._dependencies.ack),
            datetime.now().astimezone(),
        )
        if utterance.ack_request is not None:
            # 話している間に読み込みを済ませ、発話の終わりの返事がモデルの読み込みを待たないようにする。
            self._dependencies.ack_generator.request_preload()
        pacing = create_pacing(rule.timing, rule.interval)
        pool = self._dependencies.backchannel_pool
        lease = await pool.acquire()
        if isinstance(lease, LeasedPredictor):
            stream = BackchannelStream(
                utterance_id,
                lease.predictor,
                create_trigger(rule.timing, pacing),
                pool,
                self._emitter_for(utterance, rule, pacing),
            )
            utterance.backchannel = _BackchannelRun(stream, rule)
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

    async def _resolve_target(self, message: StartMessage) -> _UtteranceTarget | None:
        """発話の言語とエンジンを決める。使えなければ error を送って None を返す。"""
        language = message.language or self._dependencies.default_language
        if language not in SUPPORTED_LANGUAGES:
            await self._send_error(
                "unsupported_language",
                f"unsupported language for stream: '{language}' "
                f"(supported: {', '.join(SUPPORTED_LANGUAGES)})",
            )
            return None
        try:
            backend = select_backend(message.backend, self._dependencies.backends)
        except UnsupportedBackendError as error:
            await self._send_error("unsupported_backend", str(error))
            return None
        return _UtteranceTarget(language, backend)

    def _emitter_for(
        self, utterance: _ReceivingUtterance, rule: BackchannelRule, pacing: BackchannelPacing
    ) -> BackchannelEmitter:
        """言い終わりの区切りを判定して送るか、文言を順に回すだけかに応じた emitter を作る。

        区切りの判定を使えない、または区切りを判定しない指定（boundary_threshold が 0）なら回すだけにする。
        """
        judging = self._dependencies.speech_judging
        rotation = self._rotation_for(rule.phrases)
        delivery = BackchannelDelivery(self._sender, pacing)
        if judging is None or rule.boundary_threshold <= 0:
            return RotationEmitter(utterance.utterance_id, rotation, delivery)
        return JudgingEmitter(
            JudgedUtterance(
                utterance.utterance_id, utterance.target.language, utterance.samples_until
            ),
            JudgingRule(rotation, rule.boundary_threshold),
            judging,
            delivery,
            self._track,
        )

    def _rotation_for(self, phrases: tuple[str, ...]) -> PhraseRotation:
        """文言の順番を返す。前の発話と文言が違うときは、新しい文言の先頭から数え直す。"""
        if phrases != self._rotation_phrases:
            self._rotation_phrases = phrases
            self._rotation = PhraseRotation(phrases)
        return self._rotation

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
        self._close_backchannel(utterance)
        samples = utterance.all_samples()
        self._spawn(
            self._finish(
                _FinishedUtterance(
                    utterance.utterance_id, utterance.target, samples, utterance.ack_request
                )
            )
        )

    async def _discard_too_long(self, utterance: _ReceivingUtterance) -> None:
        """上限を超えた発話を破棄し、end まで音声を捨てる。"""
        self._close_backchannel(utterance)
        self._current = _DiscardingUtterance(utterance.utterance_id)
        await self._send_error(
            "utterance_too_long",
            f"utterance exceeds {self._dependencies.max_utterance_seconds:g} seconds",
        )

    def _close_backchannel(self, utterance: _ReceivingUtterance) -> None:
        """相槌の予測を終える。録音する設定なら、予測が終わってから音声と確率を保存する。"""
        run = utterance.backchannel
        if run is None:
            return
        run.stream.close()
        recorder = self._dependencies.recorder
        if recorder is not None:
            self._spawn(self._record(recorder, utterance, run))

    async def _record(
        self, recorder: UtteranceRecorder, utterance: _ReceivingUtterance, run: _BackchannelRun
    ) -> None:
        """予測と候補の判定が終わるのを待ち、発話の音声と確率を保存する。"""
        await run.stream.wait_for_decisions()
        samples = utterance.all_samples()
        await recorder.record(
            UtteranceRecording(
                utterance.utterance_id,
                utterance.started_at,
                samples,
                run.rule.timing,
                run.rule.interval,
                run.rule.boundary_threshold,
                run.stream.trace(),
            )
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
                self._dependencies.serial_transcriber,
                utterance.samples,
                utterance.target.language,
                utterance.target.backend,
            )
        except Exception as error:
            logger.exception(
                "stream transcription failed (utterance_id=%d)", utterance.utterance_id
            )
            await self._send_error(
                "transcribe_failed", f"transcribe_failed: {error}", utterance.utterance_id
            )
            return None
        backend = self._dependencies.backend_descriptions[utterance.target.backend]
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
