from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import httpx
import numpy as np
import pytest

from stt_gateway.backchannel_judge import BackchannelKindJudge
from stt_gateway.backchannel_policy import BackchannelFrame, PhraseRotation
from stt_gateway.backchannel_stream import (
    KindEmitter,
    KindJudging,
    KindTarget,
    samples_until_offset,
)
from stt_gateway.serial_transcriber import SerialTranscriber
from stt_gateway.stream_messages import BackchannelEvent, StreamEvent, UtteranceId
from stt_gateway.transcription import Transcription


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


def _emit_once(transcriber: _FixedTranscriber, *, is_open: bool = True) -> list[StreamEvent]:
    """1 フレーム分の相槌を KindEmitter に渡し、送られたイベントを返す。"""

    def kev(_request: httpx.Request) -> httpx.Response:
        """いつも news を返すダミーの Kev。"""
        return httpx.Response(200, json={"answers": {"kind": {"choice": "news"}}})

    async def scenario() -> list[StreamEvent]:
        sender = _RecordingSender()
        tasks: list[asyncio.Task[object]] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(kev)) as client:
            judging = KindJudging(
                BackchannelKindJudge("http://kev.test", "secret", client),
                SerialTranscriber(lambda: transcriber),
                1000,
            )
            emitter = KindEmitter(
                KindTarget(
                    UtteranceId(1),
                    "ja",
                    lambda offset_ms: np.ones(offset_ms * 16, dtype=np.float32),
                ),
                {"news": PhraseRotation(("そうなんだ",))},
                judging,
                sender,
                tasks.append,
            )
            await emitter.emit(BackchannelFrame(300, 0.7), lambda: is_open)
            await asyncio.gather(*tasks)
        return sender.events

    return asyncio.run(scenario())


def test_kind_emitter_sends_phrase_for_judged_kind(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="stt_gateway.backchannel_stream")
    events = _emit_once(_FixedTranscriber("先週引っ越したんだ。"))
    assert events == [
        BackchannelEvent(
            utterance_id=UtteranceId(1),
            text="そうなんだ",
            offset_ms=300,
            probability=0.7,
            kind="news",
        )
    ]
    assert "reason=sent" in caplog.text
    assert "kind=news" in caplog.text
    assert "offset_ms=300" in caplog.text
    assert "先週引っ越したんだ" not in caplog.text


def test_kind_emitter_drops_backchannel_after_end(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level(logging.INFO, logger="stt_gateway.backchannel_stream")
    assert _emit_once(_FixedTranscriber("先週引っ越したんだ。"), is_open=False) == []
    assert "reason=closed" in caplog.text


def test_kind_emitter_skips_empty_transcript_without_asking_kev(
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level(logging.INFO, logger="stt_gateway.backchannel_stream")
    assert _emit_once(_FixedTranscriber("")) == []
    assert "reason=judge_unavailable" in caplog.text


def test_kind_emitter_skips_when_partial_transcription_fails(
    caplog: pytest.LogCaptureFixture,
) -> None:
    assert _emit_once(_FixedTranscriber("", RuntimeError("CUDA out of memory"))) == []
    assert "partial transcription for backchannel failed" in caplog.text


def test_samples_until_offset_cuts_at_frame() -> None:
    chunks = [np.ones(1000, dtype=np.float32), np.zeros(3000, dtype=np.float32)]
    assert len(samples_until_offset(chunks, 200)) == 3200
    assert len(samples_until_offset(chunks, 1000)) == 4000, "受け取った分より先は無い"
    assert len(samples_until_offset([], 100)) == 0
