from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from stt_gateway.backchannel_emitters import (
    BackchannelDelivery,
    JudgedUtterance,
    JudgingEmitter,
    JudgingRule,
    RotationEmitter,
    SpeechJudging,
)
from stt_gateway.backchannel_interval import BackchannelInterval, BackchannelPacing
from stt_gateway.backchannel_policy import (
    BackchannelDecision,
    BackchannelFire,
    BackchannelFrame,
    BackchannelPrepare,
    PhraseRotation,
)
from stt_gateway.boundary_judge import BoundaryJudge
from stt_gateway.serial_transcriber import SerialTranscriber
from stt_gateway.stream_messages import BackchannelEvent, StreamEvent, UtteranceId
from stt_gateway.transcription import Transcription

_EMITTER_LOGGER = "stt_gateway.backchannel_emitters"
_FIRE = BackchannelFire(BackchannelFrame(300, 0.7), False)
# 前の相槌から 1 秒は次を出さない。送ったかどうかを pacing の状態で確かめるため。
_INTERVAL = BackchannelInterval(1000, 0, 1.0, 1000)
_BOUNDARY_THRESHOLD = 0.35


class _RecordingSender:
    """送られたイベントを記録する。"""

    def __init__(self) -> None:
        self.events: list[StreamEvent] = []

    async def send(self, event: StreamEvent) -> None:
        """記録する。"""
        self.events.append(event)


class _FixedTranscriber:
    """決まった文字起こし結果を返す。error があれば送出する。"""

    def __init__(self, text: str, error: Exception | None = None) -> None:
        self._text = text
        self._error = error

    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """決まった結果を返す。"""
        if self._error is not None:
            raise self._error
        return Transcription(self._text, language, 0.1)


@dataclass
class _Setup:
    """JudgingEmitter を動かす条件。boundary_probability が None ならモデルが失敗する。"""

    text: str = "先週引っ越したんだ。"
    transcription_error: Exception | None = None
    phrases: tuple[str, ...] = ("うん", "はい")
    boundary_probability: float | None = 0.9
    timeout_ms: int = 1000


@dataclass
class _Outcome:
    """JudgingEmitter を動かした結果。"""

    events: list[StreamEvent]
    decisions: tuple[BackchannelDecision, ...]
    boundary_texts: list[str]
    is_closed_after: bool


EmitterScenario = Callable[[JudgingEmitter], Awaitable[None]]


async def _emit_fire(emitter: JudgingEmitter) -> None:
    """準備なしで 1 フレーム分の相槌を渡す。"""
    await emitter.emit(_FIRE, lambda: True)


def _run_emitter(setup: _Setup, scenario: EmitterScenario = _emit_fire) -> _Outcome:
    """JudgingEmitter を scenario で動かし、結果を返す。"""
    boundary_texts: list[str] = []

    def boundary_model(text: str) -> float:
        """受け取った文を記録し、決まった区切りらしさを返す。"""
        boundary_texts.append(text)
        if setup.boundary_probability is None:
            raise RuntimeError("CUDA out of memory")
        return setup.boundary_probability

    async def run() -> _Outcome:
        sender = _RecordingSender()
        pacing = BackchannelPacing(_INTERVAL, 0)
        tasks: list[asyncio.Task[object]] = []
        judging = SpeechJudging(
            SerialTranscriber(
                {"haas": lambda: _FixedTranscriber(setup.text, setup.transcription_error)}
            ),
            setup.timeout_ms,
            BoundaryJudge(boundary_model),
            "haas",
        )
        emitter = JudgingEmitter(
            JudgedUtterance(
                UtteranceId(1),
                "ja",
                lambda offset_ms: np.ones(offset_ms * 16, dtype=np.float32),
            ),
            JudgingRule(PhraseRotation(setup.phrases), _BOUNDARY_THRESHOLD),
            judging,
            BackchannelDelivery(sender, pacing),
            tasks.append,
        )
        pacing.hold()
        await scenario(emitter)
        await emitter.wait()
        # 準備で始めて使わなかった判定も、区切りの判定が終わるまで待つ。
        await asyncio.gather(*tasks)
        return _Outcome(
            sender.events,
            emitter.decisions(),
            boundary_texts,
            pacing.is_closed(_FIRE.frame.offset_ms + 1),
        )

    return asyncio.run(run())


def _sent(text: str) -> list[BackchannelEvent]:
    """_FIRE の位置で送られるはずの相槌。"""
    return [BackchannelEvent(utterance_id=UtteranceId(1), text=text, offset_ms=300, probability=0.7)]


def test_judging_emitter_sends_next_phrase_at_boundary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=_EMITTER_LOGGER)
    outcome = _run_emitter(_Setup())
    assert outcome.events == _sent("うん")
    assert outcome.boundary_texts == ["先週引っ越したんだ"], "末尾の句点は除いて判定する"
    assert outcome.decisions == (BackchannelDecision(300, "sent", "先週引っ越したんだ。", 0.9),)
    assert outcome.is_closed_after, "送った相槌から間隔を数える"
    assert "reason=sent" in caplog.text
    assert "boundary=0.90" in caplog.text
    assert "offset_ms=300" in caplog.text
    assert "先週引っ越したんだ" not in caplog.text


def test_judging_emitter_sends_at_threshold() -> None:
    outcome = _run_emitter(_Setup(boundary_probability=_BOUNDARY_THRESHOLD))
    assert outcome.events == _sent("うん"), "境目ちょうどは区切りとみなす"


def test_judging_emitter_stays_silent_before_boundary(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=_EMITTER_LOGGER)
    outcome = _run_emitter(_Setup(text="あのさ、なので", boundary_probability=0.1))
    assert outcome.events == []
    assert outcome.decisions == (BackchannelDecision(300, "not_boundary", "あのさ、なので", 0.1),)
    assert not outcome.is_closed_after, "送らなかった候補では間隔を数え直さない"
    assert "reason=not_boundary" in caplog.text


def test_judging_emitter_sends_when_boundary_model_fails(caplog: pytest.LogCaptureFixture) -> None:
    outcome = _run_emitter(_Setup(boundary_probability=None))
    assert outcome.events == _sent("うん"), "区切りを判定できなければ、相槌が消えないよう送る"
    assert outcome.decisions[0].boundary_probability is None
    assert "backchannel boundary judge failed" in caplog.text


def test_judging_emitter_drops_backchannel_after_end(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=_EMITTER_LOGGER)

    async def scenario(emitter: JudgingEmitter) -> None:
        await emitter.emit(_FIRE, lambda: False)

    outcome = _run_emitter(_Setup(), scenario)
    assert outcome.events == []
    assert "reason=closed" in caplog.text


def test_judging_emitter_skips_empty_transcript_without_judging(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=_EMITTER_LOGGER)
    outcome = _run_emitter(_Setup(text=""))
    assert (outcome.events, outcome.boundary_texts) == ([], [])
    assert outcome.decisions == (BackchannelDecision(300, "judge_unavailable", None, None),)
    assert "reason=judge_unavailable" in caplog.text


def test_judging_emitter_skips_when_partial_transcription_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    outcome = _run_emitter(_Setup(transcription_error=RuntimeError("CUDA out of memory")))
    assert outcome.events == []
    assert "partial transcription for backchannel failed" in caplog.text


def test_judging_emitter_does_not_advance_phrase_for_skipped_candidate() -> None:
    async def scenario(emitter: JudgingEmitter) -> None:
        await emitter.emit(_FIRE, lambda: False)
        await emitter.wait()
        await emitter.emit(_FIRE, lambda: True)

    outcome = _run_emitter(_Setup(), scenario)
    assert outcome.events == _sent("うん"), "送らなかった候補では文言の順番を進めない"


def test_judging_emitter_times_out(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger=_EMITTER_LOGGER)

    async def scenario(emitter: JudgingEmitter) -> None:
        # イベントループを 1 度も譲らずに時間切れにするため、判定のタスクを進めない。
        await emitter.emit(_FIRE, lambda: True)

    outcome = _run_emitter(_Setup(timeout_ms=0), scenario)
    assert outcome.events == []
    assert outcome.decisions == (BackchannelDecision(300, "timeout", None, None),)
    assert "reason=timeout" in caplog.text


def _prepare_then_emit(*, has_prepared: bool) -> _Outcome:
    """準備の合図の後に相槌を出す。"""

    async def scenario(emitter: JudgingEmitter) -> None:
        emitter.prepare(BackchannelPrepare(BackchannelFrame(200, 0.6)))
        await emitter.emit(BackchannelFire(_FIRE.frame, has_prepared), lambda: True)

    return _run_emitter(_Setup(), scenario)


def test_judging_emitter_uses_judgment_started_by_prepare(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger=_EMITTER_LOGGER)
    outcome = _prepare_then_emit(has_prepared=True)
    assert len(outcome.boundary_texts) == 1, "準備で始めた判定を使い、判定し直さない"
    assert outcome.events == _sent("うん")
    assert "prepared=True" in caplog.text


def test_judging_emitter_judges_again_when_preparation_is_not_for_this_silence() -> None:
    outcome = _prepare_then_emit(has_prepared=False)
    assert len(outcome.boundary_texts) == 2, "別の無音の準備は使わず、相槌を出す位置で判定し直す"
    assert len(outcome.events) == 1


def _run_rotation_emitter(*, is_open: bool) -> tuple[list[StreamEvent], bool]:
    """RotationEmitter に相槌を 1 つ渡し、送られたイベントと、その後に pacing が閉じたかを返す。"""

    async def run() -> tuple[list[StreamEvent], bool]:
        sender = _RecordingSender()
        pacing = BackchannelPacing(_INTERVAL, 0)
        emitter = RotationEmitter(
            UtteranceId(1), PhraseRotation(("うん",)), BackchannelDelivery(sender, pacing)
        )
        pacing.hold()
        emitter.prepare(BackchannelPrepare(BackchannelFrame(200, 0.45)))
        await emitter.emit(_FIRE, lambda: is_open)
        await emitter.wait()
        assert emitter.decisions() == ()
        return sender.events, pacing.is_closed(_FIRE.frame.offset_ms + 1)

    return asyncio.run(run())


def test_rotation_emitter_sends_next_phrase() -> None:
    events, is_closed_after = _run_rotation_emitter(is_open=True)
    assert events == _sent("うん")
    assert is_closed_after, "送った相槌から間隔を数える"


def test_rotation_emitter_drops_backchannel_after_end() -> None:
    events, is_closed_after = _run_rotation_emitter(is_open=False)
    assert events == []
    assert not is_closed_after


def test_delivery_reopens_pacing_when_sending_fails() -> None:
    class _BrokenSender:
        """いつも送信に失敗する送り先。"""

        async def send(self, event: StreamEvent) -> None:
            """切断された接続のふりをする。"""
            raise ConnectionError(f"connection closed before sending {event.type}")

    async def run() -> bool:
        pacing = BackchannelPacing(_INTERVAL, 0)
        pacing.hold()
        with pytest.raises(ConnectionError):
            await BackchannelDelivery(_BrokenSender(), pacing).deliver(_sent("うん")[0])
        return pacing.is_closed(_FIRE.frame.offset_ms + 1)

    assert not asyncio.run(run()), "送れなかった候補では間隔を数えず、次の候補を出せる"
