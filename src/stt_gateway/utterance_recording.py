"""/stream の発話の音声と相槌の確率を、相槌の調整のためにファイルへ残す。"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .audio_decoding import AudioSamples
from .backchannel_interval import BackchannelInterval
from .backchannel_policy import BackchannelDecision, BackchannelFire, BackchannelFrame
from .backchannel_triggers import SilenceTiming
from .pcm_audio import write_wav
from .stream_audio_decoder import STREAM_SAMPLE_RATE_HZ
from .stream_messages import UtteranceId

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000
# 録音には話した内容がそのまま入るため、サーバを動かすユーザ以外は読めないようにする。
RECORDING_DIRECTORY_MODE = 0o700
RECORDING_FILE_MODE = 0o600


@dataclass(frozen=True)
class BackchannelTrace:
    """1 つの発話で予測した話していない確率と、相槌の準備・発火の合図、判定した候補の結果。"""

    frames: tuple[BackchannelFrame, ...]
    prepares: tuple[BackchannelFrame, ...]
    fires: tuple[BackchannelFire, ...]
    decisions: tuple[BackchannelDecision, ...]


@dataclass(frozen=True)
class UtteranceRecording:
    """ファイルに残す 1 つの発話。"""

    utterance_id: UtteranceId
    recorded_at: datetime
    samples: AudioSamples
    timing: SilenceTiming
    interval: BackchannelInterval
    boundary_threshold: float
    trace: BackchannelTrace


def recording_document(recording: UtteranceRecording) -> dict[str, object]:
    """WAV と並べて保存する JSON の中身を作る。"""
    trace = recording.trace
    return {
        "utterance_id": recording.utterance_id,
        "recorded_at": recording.recorded_at.isoformat(),
        "sample_rate_hz": STREAM_SAMPLE_RATE_HZ,
        "duration_ms": len(recording.samples) * MILLISECONDS_PER_SECOND // STREAM_SAMPLE_RATE_HZ,
        "silence_ms": recording.timing.silence_ms,
        "min_new_speech_ms": recording.timing.min_new_speech_ms,
        "min_interval_ms": recording.interval.min_interval_ms,
        "interval_per_minute_ms": recording.interval.per_minute_ms,
        "interval_factor": recording.interval.factor,
        "max_interval_ms": recording.interval.max_interval_ms,
        "boundary_threshold": recording.boundary_threshold,
        "frames": [_frame_document(frame) for frame in trace.frames],
        "prepares": [_frame_document(frame) for frame in trace.prepares],
        "fires": [
            {**_frame_document(fire.frame), "has_prepared": fire.has_prepared}
            for fire in trace.fires
        ],
        "decisions": [dataclasses.asdict(decision) for decision in trace.decisions],
    }


def _frame_document(frame: BackchannelFrame) -> dict[str, object]:
    """フレームを JSON の値にする。"""
    return {"offset_ms": frame.offset_ms, "probability": frame.probability}


class UtteranceRecorder:
    """発話ごとに、音声（WAV）と確率（JSON）を同じ名前で directory に保存する。"""

    def __init__(self, directory: Path) -> None:
        """保存先のディレクトリを受け取る。無ければ保存のときに作る。"""
        self._directory = directory

    async def record(self, recording: UtteranceRecording) -> None:
        """イベントループを止めないよう別スレッドで保存する。失敗してもログに残すだけにする。

        録音は調整のための補助で、失敗しても発話の処理（文字起こしなど）は続けるため。
        """
        try:
            wav_path = await asyncio.to_thread(self._save, recording)
        except OSError:
            logger.exception(
                "failed to save stream recording (utterance_id=%d, directory=%s)",
                recording.utterance_id,
                self._directory,
            )
            return
        logger.info(
            "stream recording saved (utterance_id=%d, path=%s)", recording.utterance_id, wav_path
        )

    def _save(self, recording: UtteranceRecording) -> Path:
        """WAV と JSON を書き出し、WAV のパスを返す。"""
        self._directory.mkdir(parents=True, exist_ok=True, mode=RECORDING_DIRECTORY_MODE)
        # 接続ごとに発話の番号は 1 から数え直すため、時刻を名前に入れて重ならないようにする。
        stem = f"{recording.recorded_at:%Y%m%d-%H%M%S-%f}-u{recording.utterance_id}"
        wav_path = self._directory / f"{stem}.wav"
        json_path = self._directory / f"{stem}.json"
        write_wav(wav_path, recording.samples, STREAM_SAMPLE_RATE_HZ)
        wav_path.chmod(RECORDING_FILE_MODE)
        json_path.write_text(
            json.dumps(recording_document(recording), ensure_ascii=False), encoding="utf-8"
        )
        json_path.chmod(RECORDING_FILE_MODE)
        return wav_path
