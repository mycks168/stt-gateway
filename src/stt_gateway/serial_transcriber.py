"""1 プロセス内で、文字起こしエンジンのモデルの読み込みと文字起こしを 1 件ずつ実行する。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from pathlib import Path

from fastapi.concurrency import run_in_threadpool

from .transcription import BackendName, Transcriber, Transcription

TranscriberLoader = Callable[[], Transcriber]


class SerialTranscriber:
    """エンジンごとにモデルを 1 つずつ読み込み、推論はエンジンをまたいで 1 件ずつ実行する。

    Haas（NeMo）は推論のたびにモデルの状態（dither 等）を書き換え、デコーダの CUDA Graph の
    固定バッファも共有するため、同じモデルを複数スレッドから同時に呼べない。エンジンが違っても
    推論を重ねないのは、GPU を取り合って相槌の判定（短い時間で終える必要がある）が遅れるため。
    モデルの読み込みは推論の順番待ちとは別に行い、読み込みに数秒かかる間もほかのエンジンの
    推論を止めない。待ちはイベントループ上のロックで行い、待機中のリクエストでスレッドプールを埋めない。
    """

    def __init__(self, loaders: Mapping[BackendName, TranscriberLoader]) -> None:
        """エンジンごとのモデルを読み込む関数を受け取る。

        読み込みは preload か、そのエンジンでの最初の文字起こしで行う。
        """
        self._loaders = dict(loaders)
        self._transcribers: dict[BackendName, Transcriber] = {}
        self._load_locks = {backend: asyncio.Lock() for backend in self._loaders}
        self._inference_lock = asyncio.Lock()

    async def preload(self, backend: BackendName) -> None:
        """ワーカー起動時にモデルを読み込み、最初のリクエストが読み込みを待たないようにする。"""
        await self._ensure_loaded(backend)

    async def transcribe(
        self, audio_path: Path, language: str, backend: BackendName
    ) -> Transcription:
        """指定したエンジンで、前のリクエストの推論が終わるのを待ってから文字起こしする。"""
        transcriber = await self._ensure_loaded(backend)
        async with self._inference_lock:
            return await run_in_threadpool(transcriber.transcribe, audio_path, language)

    async def _ensure_loaded(self, backend: BackendName) -> Transcriber:
        """未読み込みなら、イベントループを止めないよう別スレッドで読み込む。

        失敗したら記録せず、次の呼び出しで読み込み直す。
        """
        loader = self._loaders.get(backend)
        if loader is None:
            raise ValueError(
                f"backend '{backend}' is not enabled in this worker "
                f"(enabled: {', '.join(self._loaders)})"
            )
        async with self._load_locks[backend]:
            transcriber = self._transcribers.get(backend)
            if transcriber is None:
                transcriber = await run_in_threadpool(loader)
                self._transcribers[backend] = transcriber
            return transcriber
