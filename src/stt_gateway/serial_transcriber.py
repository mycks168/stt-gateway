"""1 プロセス内でモデルの読み込みと文字起こしを 1 件ずつ実行する。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from .transcription import Transcriber, Transcription

TranscriberLoader = Callable[[], Transcriber]


class SerialTranscriber:
    """モデルを 1 つだけ読み込み、推論を 1 件ずつ別スレッドで実行する。

    Haas（NeMo）は推論のたびにモデルの状態（dither 等）を書き換え、デコーダの CUDA Graph の
    固定バッファも共有するため、同じモデルを複数スレッドから同時に呼べない。
    待ちはイベントループ上のロックで行い、待機中のリクエストでスレッドプールを埋めない。
    """

    def __init__(self, load: TranscriberLoader) -> None:
        """モデルを読み込む関数を受け取る。読み込みは preload か最初の文字起こしで行う。"""
        self._load = load
        self._transcriber: Transcriber | None = None
        self._lock = asyncio.Lock()

    async def preload(self) -> None:
        """ワーカー起動時にモデルを読み込み、最初のリクエストが読み込みを待たないようにする。"""
        async with self._lock:
            await self._ensure_loaded()

    async def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """前のリクエストの推論が終わるのを待ってから文字起こしする。"""
        async with self._lock:
            transcriber = await self._ensure_loaded()
            return await run_in_threadpool(transcriber.transcribe, audio_path, language)

    async def _ensure_loaded(self) -> Transcriber:
        """ロックを持った状態で呼ぶ。未読み込みならイベントループを止めないよう別スレッドで読み込む。"""
        if self._transcriber is None:
            self._transcriber = await run_in_threadpool(self._load)
        return self._transcriber
