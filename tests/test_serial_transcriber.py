from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path

import pytest

from stt_gateway.serial_transcriber import SerialTranscriber
from stt_gateway.transcription import Transcription

# 推論が重なれば必ず検出できる程度に、1 件の推論を引き延ばす時間。
_INFERENCE_SECONDS = 0.05
_CONCURRENT_REQUESTS = 4
# 起動時の読み込みに 1 回失敗し、次のリクエストで読み込み直したときの試行回数。
_LOAD_ATTEMPTS_AFTER_ONE_FAILURE = 2


class _OverlapDetectingTranscriber:
    """同時に実行されている推論の最大数を記録するダミー。"""

    def __init__(self) -> None:
        self.max_running = 0
        self._running = 0
        self._counter_lock = threading.Lock()

    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """実行中の件数を数えながら、少し時間をかけて結果を返す。"""
        with self._counter_lock:
            self._running += 1
            self.max_running = max(self.max_running, self._running)
        time.sleep(_INFERENCE_SECONDS)
        with self._counter_lock:
            self._running -= 1
        return Transcription(text=audio_path.name, language=language, duration_seconds=None)


def test_transcriptions_run_one_at_a_time() -> None:
    backend = _OverlapDetectingTranscriber()

    async def _send_concurrently() -> list[Transcription]:
        """同時に複数件の文字起こしを投げる。"""
        serial = SerialTranscriber(lambda: backend)
        requests = (
            serial.transcribe(Path(f"{index}.wav"), "ja") for index in range(_CONCURRENT_REQUESTS)
        )
        return await asyncio.gather(*requests)

    results = asyncio.run(_send_concurrently())
    assert [result.text for result in results] == [f"{i}.wav" for i in range(_CONCURRENT_REQUESTS)]
    assert backend.max_running == 1


def test_event_loop_keeps_running_during_inference() -> None:
    backend = _OverlapDetectingTranscriber()

    async def _tick_during_inference() -> int:
        """推論中にイベントループ上の別の処理が進んだ回数を返す。"""
        serial = SerialTranscriber(lambda: backend)
        ticks = 0
        task = asyncio.create_task(serial.transcribe(Path("a.wav"), "ja"))
        while not task.done():
            ticks += 1
            await asyncio.sleep(0)
        await task
        return ticks

    assert asyncio.run(_tick_during_inference()) > 1


def test_model_is_loaded_once_across_preload_and_requests() -> None:
    loads: list[int] = []

    def _load() -> _OverlapDetectingTranscriber:
        """読み込み回数を数える。"""
        loads.append(1)
        return _OverlapDetectingTranscriber()

    async def _preload_then_transcribe() -> None:
        """起動時の読み込みの後に、同時に文字起こしする。"""
        serial = SerialTranscriber(_load)
        await serial.preload()
        await asyncio.gather(
            serial.transcribe(Path("a.wav"), "ja"), serial.transcribe(Path("b.wav"), "ja")
        )

    asyncio.run(_preload_then_transcribe())
    assert len(loads) == 1


def test_failed_load_is_retried_on_next_request() -> None:
    attempts: list[int] = []

    def _fail_first_time() -> _OverlapDetectingTranscriber:
        """1 回目だけ読み込みに失敗する。"""
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("download failed")
        return _OverlapDetectingTranscriber()

    async def _preload_then_transcribe() -> Transcription:
        """起動時の読み込み失敗の後に文字起こしする。"""
        serial = SerialTranscriber(_fail_first_time)
        with pytest.raises(RuntimeError, match="download failed"):
            await serial.preload()
        return await serial.transcribe(Path("a.wav"), "ja")

    assert asyncio.run(_preload_then_transcribe()).text == "a.wav"
    assert len(attempts) == _LOAD_ATTEMPTS_AFTER_ONE_FAILURE
