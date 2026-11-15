from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from stt_gateway.backchannel_interval import BackchannelInterval
from stt_gateway.backchannel_policy import (
    BackchannelDecision,
    BackchannelFire,
    BackchannelFrame,
)
from stt_gateway.backchannel_triggers import SilenceTiming
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
        BackchannelFrame(200, 0.8),
        BackchannelFrame(300, 0.9),
    )
    return UtteranceRecording(
        UtteranceId(2),
        datetime(2026, 11, 15, 17, 30, 0, 123456, tzinfo=timezone.utc),
        np.zeros(4800, dtype=np.float32),
        SilenceTiming(90, 1000),
        BackchannelInterval(1500, 3000, 1.1, 8000),
        0.35,
        BackchannelTrace(
            frames,
            (frames[1],),
            (BackchannelFire(frames[2], True),),
            (BackchannelDecision(300, "not_boundary", "あのさ、なので", 0.12),),
        ),
    )


def test_recording_document_lists_frames_and_signals() -> None:
    document = recording_document(_recording())
    assert document["recorded_at"] == "2026-11-15T17:30:00.123456+00:00"
    assert (document["silence_ms"], document["min_new_speech_ms"]) == (90, 1000)
    assert (document["min_interval_ms"], document["interval_factor"]) == (1500, 1.1)
    assert document["duration_ms"] == 300
    assert document["prepares"] == [{"offset_ms": 200, "probability": 0.8}]
    assert document["fires"] == [{"offset_ms": 300, "probability": 0.9, "has_prepared": True}]
    assert document["boundary_threshold"] == 0.35
    assert document["decisions"] == [
        {
            "offset_ms": 300,
            "reason": "not_boundary",
            "text": "あのさ、なので",
            "boundary_probability": 0.12,
        }
    ]


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
