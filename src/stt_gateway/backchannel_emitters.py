"""相槌を出すと決めたフレームについて、送るかどうかと文言を決めてイベントを送る。"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Protocol

from .audio_decoding import AudioSamples
from .backchannel_interval import BackchannelPacing
from .backchannel_judge import BackchannelKindJudge
from .backchannel_policy import (
    BackchannelDecision,
    BackchannelDecisionReason,
    BackchannelFire,
    BackchannelFrame,
    BackchannelKind,
    BackchannelPrepare,
    PhraseRotation,
)
from .boundary_judge import BoundaryJudge
from .sample_transcription import transcribe_samples
from .serial_transcriber import SerialTranscriber
from .stream_messages import BackchannelEvent, EventSender, UtteranceId

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
IsOpen = Callable[[], bool]
TrackTask = Callable[["asyncio.Task[object]"], None]
SamplesUntil = Callable[[int], AudioSamples]
# 文言を順に回すか、話の種類ごとの文言から選ぶか。
PhraseSource = PhraseRotation | Mapping[BackchannelKind, PhraseRotation]


class BackchannelDelivery:
    """相槌を送り、候補を送ったかどうかを pacing に伝える。"""

    def __init__(self, sender: EventSender, pacing: BackchannelPacing) -> None:
        """送り先と、次の相槌を出してよいかの判定を受け取る。"""
        self._sender = sender
        self._pacing = pacing

    async def deliver(self, event: BackchannelEvent) -> None:
        """相槌を送る。送れなかったときも、次の候補を出せるよう pacing に伝える。"""
        try:
            await self._sender.send(event)
        except BaseException:
            self._pacing.settle(None)
            raise
        self._pacing.settle(event.offset_ms)

    def skip(self) -> None:
        """候補を送らないと決めた。"""
        self._pacing.settle(None)


class BackchannelEmitter(Protocol):
    """相槌を出すと決めたフレームの文言を決めて送る。予測を止めないよう、すぐに戻る。

    候補ごとに、送ったか送らなかったかを BackchannelDelivery で伝える。
    """

    def prepare(self, signal: BackchannelPrepare) -> None:
        """相槌を出しそうなフレームで、文言を決める準備を先に始める。"""
        ...

    async def emit(self, signal: BackchannelFire, is_open: IsOpen) -> None:
        """フレームの相槌を送る。is_open が False になった後は送らない。"""
        ...

    async def wait(self) -> None:
        """出した候補の結果がすべて決まるまで待つ。"""
        ...

    def decisions(self) -> tuple[BackchannelDecision, ...]:
        """途中の文字起こしで判定した候補の結果を返す。判定しない emitter は空を返す。"""
        ...


class RotationEmitter:
    """文言を順に回して送る。"""

    def __init__(
        self, utterance_id: UtteranceId, rotation: PhraseRotation, delivery: BackchannelDelivery
    ) -> None:
        """発話・文言の順番・送り先を受け取る。"""
        self._utterance_id = utterance_id
        self._rotation = rotation
        self._delivery = delivery

    def prepare(self, signal: BackchannelPrepare) -> None:
        """文言は順に回すだけで、先に決めておくものが無いため何もしない。"""

    async def emit(self, signal: BackchannelFire, is_open: IsOpen) -> None:
        """次の文言で相槌を送る。"""
        if not is_open():
            self._delivery.skip()
            return
        await self._delivery.deliver(
            BackchannelEvent(
                utterance_id=self._utterance_id,
                text=self._rotation.next_phrase(),
                offset_ms=signal.frame.offset_ms,
                probability=signal.frame.probability,
            )
        )

    async def wait(self) -> None:
        """emit の中で送り終えているため、待つものは無い。"""

    def decisions(self) -> tuple[BackchannelDecision, ...]:
        """途中の文字起こしで判定しないため、空を返す。"""
        return ()


@dataclass(frozen=True)
class SpeechJudging:
    """途中の文字起こしから相槌を判定するのに使う、ワーカー内で共有する部品。

    kind_judge は話の種類の判定（Kev）、boundary_judge は言い終わりの区切りの判定。
    使えないものは None。
    """

    serial_transcriber: SerialTranscriber
    timeout_ms: int
    kind_judge: BackchannelKindJudge | None
    boundary_judge: BoundaryJudge | None


@dataclass(frozen=True)
class JudgedUtterance:
    """途中の文字起こしで判定する発話。samples_until は指定の位置までの直近の音声を返す。"""

    utterance_id: UtteranceId
    language: str
    samples_until: SamplesUntil


@dataclass(frozen=True)
class JudgingRule:
    """判定の結果から、相槌を送るかどうかと文言を決める規則。

    boundary_threshold は、相槌を送る区切りらしさの下限。0 なら区切りを判定しない。
    """

    phrases: PhraseSource
    boundary_threshold: float


@dataclass(frozen=True)
class _SpeechJudgment:
    """途中の文字起こしと、そこから判定した結果。判定しなかった・失敗した項目は None。"""

    text: str
    boundary_probability: float | None
    kind: BackchannelKind | None


_Judging = asyncio.Task[_SpeechJudgment | None]


class JudgingEmitter:
    """フレームまでの音声を文字起こしして判定し、送るかどうかと文言を決める。

    言い終わりの区切りらしさが boundary_threshold に届かなければ送らない（言いかけの途中のため）。
    文言を話の種類ごとに選ぶ指定なら、種類も判定してその種類の文言を送る。
    判定は別のタスクで行い、予測の処理を待たせない。timeout_ms に間に合わなければ出さない。
    遅れた相槌は話の続きにかぶるため。準備の合図を受けたら判定を先に始め、同じ山で相槌を出すときは
    その結果を使う。判定（途中の文字起こしとモデル）の時間だけ相槌が遅れるのを避けるため。
    """

    def __init__(
        self,
        target: JudgedUtterance,
        rule: JudgingRule,
        judging: SpeechJudging,
        delivery: BackchannelDelivery,
        track: TrackTask,
    ) -> None:
        """判定する発話・規則・判定の部品・送り先・タスクの登録先を受け取る。"""
        self._target = target
        self._rule = rule
        self._judging = judging
        self._delivery = delivery
        self._track = track
        self._prepared: _Judging | None = None
        self._settling: set[asyncio.Task[None]] = set()
        self._decisions: list[BackchannelDecision] = []

    def prepare(self, signal: BackchannelPrepare) -> None:
        """準備のフレームまでの音声で、判定を始める。"""
        self._prepared = self._start_judging(signal.frame)

    async def emit(self, signal: BackchannelFire, is_open: IsOpen) -> None:
        """判定のタスクを始めてすぐに戻る。同じ山で始めた判定があれば、それを使う。"""
        judging = self._prepared if signal.has_prepared else None
        self._prepared = None
        if judging is None:
            judging = self._start_judging(signal.frame)
        settling = asyncio.create_task(self._settle(signal, judging, is_open))
        self._settling.add(settling)
        settling.add_done_callback(self._settling.discard)
        self._track(settling)

    async def wait(self) -> None:
        """出した候補の結果がすべて決まるまで待つ。"""
        while self._settling:
            await asyncio.gather(*self._settling)

    def decisions(self) -> tuple[BackchannelDecision, ...]:
        """ここまでに判定した候補の結果を返す。"""
        return tuple(self._decisions)

    def _start_judging(self, frame: BackchannelFrame) -> _Judging:
        """フレームまでの音声で判定のタスクを始める。

        結果を使わなくなっても判定（途中の文字起こし）は止めない。推論のスレッドは途中で止められず、
        止めると SerialTranscriber の順番待ちを抜けて GPU 上の推論が重なるため。結果だけ捨てる。
        """
        judging = asyncio.create_task(self._judge(self._target.samples_until(frame.offset_ms)))
        self._track(judging)
        return judging

    async def _settle(self, signal: BackchannelFire, judging: _Judging, is_open: IsOpen) -> None:
        """判定を待って相槌を送るかどうかを決め、送るなら送って、結果を記録する。"""
        started = time.perf_counter()
        frame = signal.frame
        decision, rotation = await self._decide(frame, judging, is_open)
        if rotation is None:
            self._delivery.skip()
        else:
            await self._delivery.deliver(
                BackchannelEvent(
                    utterance_id=self._target.utterance_id,
                    text=rotation.next_phrase(),
                    offset_ms=frame.offset_ms,
                    probability=frame.probability,
                    kind=decision.kind,
                )
            )
        self._decisions.append(decision)
        self._log_decision(decision, started, has_prepared=signal.has_prepared)

    async def _decide(
        self, frame: BackchannelFrame, judging: _Judging, is_open: IsOpen
    ) -> tuple[BackchannelDecision, PhraseRotation | None]:
        """判定を待ち、候補の結果と、送るならその文言の順番を返す。

        送らない相槌で文言の順番を進めないよう、次の文言は送る側が取る。
        """
        try:
            judgment = await asyncio.wait_for(
                asyncio.shield(judging), self._judging.timeout_ms / MILLISECONDS_PER_SECOND
            )
        except asyncio.TimeoutError:
            return _decision(frame, "timeout", None), None
        if judgment is None:
            return _decision(frame, "judge_unavailable", None), None
        if self._is_before_boundary(judgment):
            return _decision(frame, "not_boundary", judgment), None
        rotation = self._rotation_for(judgment.kind)
        if rotation is None:
            reason: BackchannelDecisionReason = (
                "judge_unavailable" if judgment.kind is None else "unmapped_kind"
            )
            return _decision(frame, reason, judgment), None
        if not is_open():
            return _decision(frame, "closed", judgment), None
        return _decision(frame, "sent", judgment), rotation

    def _is_before_boundary(self, judgment: _SpeechJudgment) -> bool:
        """言い終わりの区切りに届いていない（言いかけの途中）かを返す。

        区切りを判定できなかったときは、相槌が出なくなるのを避けるため届いているとみなす。
        """
        probability = judgment.boundary_probability
        return probability is not None and probability < self._rule.boundary_threshold

    def _rotation_for(self, kind: BackchannelKind | None) -> PhraseRotation | None:
        """送る文言の順番を返す。種類ごとに選ぶ指定で、種類が無い・対応が無いときは None。"""
        phrases = self._rule.phrases
        if isinstance(phrases, PhraseRotation):
            return phrases
        return None if kind is None else phrases.get(kind)

    def _log_decision(
        self, decision: BackchannelDecision, started: float, *, has_prepared: bool
    ) -> None:
        """送信と見送りを区別して残す。elapsed_ms は相槌を出すと決めてからの待ち時間。

        発話本文やトークンはログに含めない。
        """
        boundary = decision.boundary_probability
        logger.info(
            "backchannel decision (utterance_id=%d, offset_ms=%d, kind=%s, boundary=%s, "
            "reason=%s, prepared=%s, elapsed_ms=%.1f)",
            self._target.utterance_id,
            decision.offset_ms,
            decision.kind,
            None if boundary is None else f"{boundary:.2f}",
            decision.reason,
            has_prepared,
            (time.perf_counter() - started) * MILLISECONDS_PER_SECOND,
        )

    async def _judge(self, samples: AudioSamples) -> _SpeechJudgment | None:
        """途中の音声を文字起こしし、区切りらしさと話の種類を判定する。失敗したら None を返す。"""
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
        text = transcription.text
        if not text:
            return None
        boundary_probability, kind = await asyncio.gather(
            self._judge_boundary(text), self._judge_kind(text)
        )
        return _SpeechJudgment(text, boundary_probability, kind)

    async def _judge_boundary(self, text: str) -> float | None:
        """区切りらしさを返す。区切りを判定しない指定・判定を使えないときは None。"""
        judge = self._judging.boundary_judge
        if judge is None or self._rule.boundary_threshold <= 0:
            return None
        return await judge.judge(text)

    async def _judge_kind(self, text: str) -> BackchannelKind | None:
        """話の種類を返す。文言を順に回す指定・判定を使えないときは None。"""
        judge = self._judging.kind_judge
        if judge is None or isinstance(self._rule.phrases, PhraseRotation):
            return None
        return await judge.judge(text)


def _decision(
    frame: BackchannelFrame, reason: BackchannelDecisionReason, judgment: _SpeechJudgment | None
) -> BackchannelDecision:
    """フレームの候補の結果を作る。"""
    if judgment is None:
        return BackchannelDecision(frame.offset_ms, reason, None, None, None)
    return BackchannelDecision(
        frame.offset_ms, reason, judgment.text, judgment.boundary_probability, judgment.kind
    )
