"""1 つの発話の音声から相槌のタイミングを予測し、文言を決めてイベントを送る。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .audio_decoding import AudioSamples
from .backchannel_judge import BackchannelKindJudge
from .backchannel_policy import (
    BackchannelFrame,
    BackchannelKind,
    BackchannelPredictor,
    BackchannelTrigger,
    PhraseRotation,
)
from .backchannel_pool import BackchannelPool
from .sample_transcription import transcribe_samples
from .serial_transcriber import SerialTranscriber
from .stream_audio_decoder import STREAM_SAMPLE_RATE_HZ
from .stream_messages import BackchannelEvent, EventSender, UtteranceId

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
IsOpen = Callable[[], bool]
TrackTask = Callable[["asyncio.Task[object]"], None]
SamplesUntil = Callable[[int], AudioSamples]


class BackchannelEmitter(Protocol):
    """相槌を出すと決めたフレームの文言を決めて送る。予測を止めないよう、すぐに戻る。"""

    async def emit(self, frame: BackchannelFrame, is_open: IsOpen) -> None:
        """フレームの相槌を送る。is_open が False になった後は送らない。"""
        ...


class RotationEmitter:
    """文言を順に回して送る。"""

    def __init__(
        self, utterance_id: UtteranceId, rotation: PhraseRotation, sender: EventSender
    ) -> None:
        """発話・文言の順番・送り先を受け取る。"""
        self._utterance_id = utterance_id
        self._rotation = rotation
        self._sender = sender

    async def emit(self, frame: BackchannelFrame, is_open: IsOpen) -> None:
        """次の文言で相槌を送る。"""
        if not is_open():
            return
        await self._sender.send(
            BackchannelEvent(
                utterance_id=self._utterance_id,
                text=self._rotation.next_phrase(),
                offset_ms=frame.offset_ms,
                probability=frame.probability,
            )
        )


@dataclass(frozen=True)
class KindJudging:
    """話の種類で文言を選ぶのに使う、ワーカー内で共有する部品。"""

    judge: BackchannelKindJudge
    serial_transcriber: SerialTranscriber
    timeout_ms: int


@dataclass(frozen=True)
class KindTarget:
    """話の種類を判定する発話。samples_until は発話の先頭から指定の位置までの音声を返す。"""

    utterance_id: UtteranceId
    language: str
    samples_until: SamplesUntil


class KindEmitter:
    """フレームまでの音声を文字起こしして話の種類を判定し、その種類の文言を送る。

    判定は別のタスクで行い、予測の処理を待たせない。timeout_ms に間に合わなければ出さない。
    遅れた相槌は話の続きにかぶるため。
    """

    def __init__(
        self,
        target: KindTarget,
        rotations: Mapping[BackchannelKind, PhraseRotation],
        judging: KindJudging,
        sender: EventSender,
        track: TrackTask,
    ) -> None:
        """判定する発話・種類ごとの文言の順番・判定の部品・送り先・タスクの登録先を受け取る。"""
        self._target = target
        self._rotations = rotations
        self._judging = judging
        self._sender = sender
        self._track = track

    async def emit(self, frame: BackchannelFrame, is_open: IsOpen) -> None:
        """判定のタスクを始めてすぐに戻る。"""
        task = asyncio.create_task(self._judge_and_send(frame, is_open))
        self._track(task)

    async def _judge_and_send(self, frame: BackchannelFrame, is_open: IsOpen) -> None:
        """種類を判定し、時間内に決まって対応する文言があれば送る。"""
        started = time.perf_counter()
        judging = asyncio.create_task(self._judge(self._target.samples_until(frame.offset_ms)))
        try:
            # 時間切れでも判定（途中の文字起こし）は止めない。推論のスレッドは途中で止められず、
            # 止めると SerialTranscriber の順番待ちを抜けて GPU 上の推論が重なるため。結果だけ捨てる。
            kind = await asyncio.wait_for(
                asyncio.shield(judging), self._judging.timeout_ms / MILLISECONDS_PER_SECOND
            )
        except asyncio.TimeoutError:
            self._log_decision(frame, "timeout", started)
            self._track(judging)
            return
        rotation = None if kind is None else self._rotations.get(kind)
        if rotation is None:
            reason = "judge_unavailable" if kind is None else "unmapped_kind"
            self._log_decision(frame, reason, started, kind)
            return
        if not is_open():
            # 送らない相槌で文言の順番を進めないよう、送る直前まで次の文言を取らない。
            self._log_decision(frame, "closed", started, kind)
            return
        await self._sender.send(
            BackchannelEvent(
                utterance_id=self._target.utterance_id,
                text=rotation.next_phrase(),
                offset_ms=frame.offset_ms,
                probability=frame.probability,
                kind=kind,
            )
        )
        self._log_decision(frame, "sent", started, kind)

    def _log_decision(
        self,
        frame: BackchannelFrame,
        reason: str,
        started: float,
        kind: BackchannelKind | None = None,
    ) -> None:
        """送信と見送りを区別して残す。発話本文やトークンはログに含めない。"""
        logger.info(
            "backchannel decision (utterance_id=%d, offset_ms=%d, kind=%s, reason=%s, elapsed_ms=%.1f)",
            self._target.utterance_id,
            frame.offset_ms,
            kind,
            reason,
            (time.perf_counter() - started) * MILLISECONDS_PER_SECOND,
        )

    async def _judge(self, samples: AudioSamples) -> BackchannelKind | None:
        """途中の音声を文字起こしし、話の種類を判定する。失敗したら None を返す。"""
        try:
            transcription = await transcribe_samples(
                self._judging.serial_transcriber, samples, self._target.language
            )
        except Exception:
            logger.exception(
                "partial transcription for backchannel failed (utterance_id=%d)",
                self._target.utterance_id,
            )
            return None
        if not transcription.text:
            return None
        return await self._judging.judge.judge(transcription.text)


class BackchannelStream:
    """1 つの発話の音声を予測器へ渡し、相槌を出すと決めたら emitter に渡す。

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
        emitter: BackchannelEmitter,
    ) -> None:
        """予測に使う部品を受け取り、予測のタスクを始める。"""
        self._utterance_id = utterance_id
        self._predictor = predictor
        self._trigger = trigger
        self._pool = pool
        self._emitter = emitter
        self._inputs: asyncio.Queue[AudioSamples | None] = asyncio.Queue()
        self._is_open = True
        self.task = asyncio.create_task(self._run())

    def is_open(self) -> bool:
        """発話の途中か（end や切断の前か）を返す。"""
        return self._is_open

    def feed(self, samples: AudioSamples) -> None:
        """音声を予測の待ち行列に入れる。"""
        self._inputs.put_nowait(samples)

    def close(self) -> None:
        """発話の終了。これ以降に確定した予測は送らない。"""
        self._is_open = False
        self._inputs.put_nowait(None)

    async def _run(self) -> None:
        """待ち行列の音声を予測し、相槌を出すと決めたら emitter に渡す。最後に予測器を返す。"""
        try:
            await asyncio.to_thread(self._predictor.reset)
            while (samples := await self._next_batch()) is not None:
                if self._is_open:
                    await self._predict_and_emit(samples)
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

    async def _predict_and_emit(self, samples: AudioSamples) -> None:
        """予測して、相槌を出すフレームがあれば emitter に渡す。"""
        frames = await asyncio.to_thread(self._predictor.predict, samples)
        for frame in frames:
            fired = self._trigger.observe(frame)
            if fired is not None and self._is_open:
                await self._emitter.emit(fired, self.is_open)

    async def _discard_until_closed(self) -> None:
        """予測に失敗した後は、終了の印が来るまで音声を捨てる。"""
        while await self._inputs.get() is not None:
            pass


def samples_until_offset(chunks: list[AudioSamples], offset_ms: int) -> AudioSamples:
    """受け取った音声の先頭から offset_ms までを 1 つの波形にする。"""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    end = offset_ms * STREAM_SAMPLE_RATE_HZ // MILLISECONDS_PER_SECOND
    samples: AudioSamples = np.concatenate(chunks)[:end]
    return samples
