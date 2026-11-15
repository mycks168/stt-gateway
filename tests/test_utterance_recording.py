from __future__ import annotations

import asyncio
import dataclasses
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from stt_gateway.backchannel_policy import BackchannelFire, BackchannelFrame
from stt_gateway.backchannel_triggers import ProbabilityTiming, SilenceTiming
from stt_gateway.stream_messages import UtteranceId
from stt_gateway.utterance_recording import (
    BackchannelTrace,
    UtteranceRecorder,
    UtteranceRecording,
    recording_document,
)


def _recording() -> UtteranceRecording:
    """準備と発火を 1 回ずつ含む、0.3 秒の発話。"""
    frames = (
        BackchannelFrame(100, 0.1),
        BackchannelFrame(200, 0.4),
        BackchannelFrame(300, 0.7),
    )
    return UtteranceRecording(
        UtteranceId(2),
        datetime(2026, 11, 15, 17, 30, 0, 123456, tzinfo=timezone.utc),
        np.zeros(4800, dtype=np.float32),
        ProbabilityTiming(0.6),
        1500,
        BackchannelTrace(frames, (frames[1],), (BackchannelFire(frames[2], True),)),
    )


def test_recording_document_lists_frames_and_signals() -> None:
    document = recording_document(_recording())
    assert document["recorded_at"] == "2026-11-15T17:30:00.123456+00:00"
    assert document["timing"] == "maai"
    assert document["duration_ms"] == 300
    assert document["prepare_threshold"] == pytest.approx(0.4)
    assert document["prepares"] == [{"offset_ms": 200, "probability": 0.4}]
    assert document["fires"] == [{"offset_ms": 300, "probability": 0.7, "has_prepared": True}]


def test_recording_document_for_silence_timing() -> None:
    recording = dataclasses.replace(_recording(), timing=SilenceTiming(120))
    document = recording_document(recording)
    assert (document["timing"], document["silence_ms"]) == ("silence", 120)
    assert "threshold" not in document


def test_recorder_names_files_by_time_and_utterance(tmp_path: Path) -> None:
    asyncio.run(UtteranceRecorder(tmp_path).record(_recording()))
    assert sorted(path.name for path in tmp_path.iterdir()) == [
        "20261115-173000-123456-u2.json",
        "20261115-173000-123456-u2.wav",
    ]


def test_recorder_logs_failure_without_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    not_a_directory = tmp_path / "file"
    not_a_directory.write_text("", encoding="utf-8")
    asyncio.run(UtteranceRecorder(not_a_directory).record(_recording()))
    assert "failed to save stream recording" in caplog.text
    assert str(not_a_directory) in caplog.text
