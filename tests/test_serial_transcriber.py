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
        serial = SerialTranscriber({"whisper": lambda: backend})
        requests = (
            serial.transcribe(Path(f"{index}.wav"), "ja", "whisper")
            for index in range(_CONCURRENT_REQUESTS)
        )
        return await asyncio.gather(*requests)

    results = asyncio.run(_send_concurrently())
    assert [result.text for result in results] == [f"{i}.wav" for i in range(_CONCURRENT_REQUESTS)]
    assert backend.max_running == 1


def test_event_loop_keeps_running_during_inference() -> None:
    backend = _OverlapDetectingTranscriber()

    async def _tick_during_inference() -> int:
        """推論中にイベントループ上の別の処理が進んだ回数を返す。"""
        serial = SerialTranscriber({"whisper": lambda: backend})
        ticks = 0
        task = asyncio.create_task(serial.transcribe(Path("a.wav"), "ja", "whisper"))
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
        serial = SerialTranscriber({"whisper": _load})
        await serial.preload("whisper")
        await asyncio.gather(
            serial.transcribe(Path("a.wav"), "ja", "whisper"),
            serial.transcribe(Path("b.wav"), "ja", "whisper"),
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
        serial = SerialTranscriber({"whisper": _fail_first_time})
        with pytest.raises(RuntimeError, match="download failed"):
            await serial.preload("whisper")
        return await serial.transcribe(Path("a.wav"), "ja", "whisper")

    assert asyncio.run(_preload_then_transcribe()).text == "a.wav"
    assert len(attempts) == _LOAD_ATTEMPTS_AFTER_ONE_FAILURE


def test_transcriptions_do_not_overlap_across_backends() -> None:
    whisper = _OverlapDetectingTranscriber()
    haas = _OverlapDetectingTranscriber()
    shared_counter = _OverlapDetectingTranscriber()

    class _Counting:
        """エンジンごとのダミーに渡しつつ、エンジンをまたいだ同時実行数も数える。"""

        def __init__(self, inner: _OverlapDetectingTranscriber) -> None:
            self._inner = inner

        def transcribe(self, audio_path: Path, language: str) -> Transcription:
            """全体の同時実行数を数えながら、エンジンのダミーで文字起こしする。"""
            shared_counter.transcribe(audio_path, language)
            return self._inner.transcribe(audio_path, language)

    async def _send_to_both() -> None:
        """2 つのエンジンへ同時に文字起こしを投げる。"""
        serial = SerialTranscriber(
            {"whisper": lambda: _Counting(whisper), "haas": lambda: _Counting(haas)}
        )
        await asyncio.gather(
            serial.transcribe(Path("a.wav"), "ja", "whisper"),
            serial.transcribe(Path("b.wav"), "ja", "haas"),
        )

    asyncio.run(_send_to_both())
    assert shared_counter.max_running == 1


def test_loading_one_backend_does_not_block_another() -> None:
    is_loading = threading.Event()
    release_loading = threading.Event()

    def _slow_load() -> _OverlapDetectingTranscriber:
        """解放されるまで読み込みを終えない。"""
        is_loading.set()
        release_loading.wait(timeout=5)
        return _OverlapDetectingTranscriber()

    async def _transcribe_while_loading() -> Transcription:
        """whisper の読み込み中に haas で文字起こしする。"""
        serial = SerialTranscriber({"haas": _OverlapDetectingTranscriber, "whisper": _slow_load})
        loading = asyncio.create_task(serial.transcribe(Path("w.wav"), "ja", "whisper"))
        while not is_loading.is_set():
            await asyncio.sleep(0.01)
        result = await asyncio.wait_for(serial.transcribe(Path("h.wav"), "ja", "haas"), 2)
        release_loading.set()
        await loading
        return result

    assert asyncio.run(_transcribe_while_loading()).text == "h.wav"


def test_backend_without_loader_is_rejected() -> None:
    serial = SerialTranscriber({"whisper": _OverlapDetectingTranscriber})
    with pytest.raises(ValueError, match="backend 'haas' is not enabled in this worker"):
        asyncio.run(serial.transcribe(Path("a.wav"), "ja", "haas"))
