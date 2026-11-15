"""1 つの発話の音声から相槌のタイミングを予測し、相槌を出すと決めたら emitter に渡す。"""

from __future__ import annotations

import asyncio
import logging

import numpy as np

from .audio_decoding import AudioSamples
from .backchannel_emitters import BackchannelEmitter
from .backchannel_peaks import summarize_probabilities
from .backchannel_policy import (
    BackchannelFire,
    BackchannelFrame,
    BackchannelPredictor,
    BackchannelPrepare,
    BackchannelTrigger,
)
from .backchannel_pool import BackchannelPool
from .stream_audio_decoder import STREAM_SAMPLE_RATE_HZ
from .stream_messages import UtteranceId
from .utterance_recording import BackchannelTrace

logger = logging.getLogger(__name__)

MILLISECONDS_PER_SECOND = 1000


class BackchannelStream:
    """1 つの発話の音声を予測器へ渡し、相槌を出すと決めたら emitter に渡す。

    予測は CPU を使う重い処理のため別スレッドで行い、音声の受信を止めない。
    予測器は発話が終わってスレッド上の処理も終わってから返す。予測中に返すと、
    次の発話が同じ予測器を同時に使ってしまうため。
    """

    def __init__(
        self,
        utterance_id: UtteranceId,
        predictor: BackchannelPredictor,
        trigger: BackchannelTrigger,
        pool: BackchannelPool,
        emitter: BackchannelEmitter,
    ) -> None:
        """予測に使う部品を受け取り、予測のタスクを始める。"""
        self._utterance_id = utterance_id
        self._predictor = predictor
        self._trigger = trigger
        self._pool = pool
        self._emitter = emitter
        self._inputs: asyncio.Queue[AudioSamples | None] = asyncio.Queue()
        self._frames: list[BackchannelFrame] = []
        self._prepares: list[BackchannelFrame] = []
        self._fires: list[BackchannelFire] = []
        self._is_open = True
        self.task = asyncio.create_task(self._run())

    def is_open(self) -> bool:
        """発話の途中か（end や切断の前か）を返す。"""
        return self._is_open

    def trace(self) -> BackchannelTrace:
        """ここまでに予測した確率・出した合図・候補の結果を返す。

        発話全体の分は wait_for_decisions の後に揃う。
        """
        return BackchannelTrace(
            tuple(self._frames),
            tuple(self._prepares),
            tuple(self._fires),
            self._emitter.decisions(),
        )

    async def wait_for_decisions(self) -> None:
        """予測と、出した候補の結果がすべて決まるまで待つ。close の後に呼ぶ。"""
        await self.task
        await self._emitter.wait()

    def feed(self, samples: AudioSamples) -> None:
        """音声を予測の待ち行列に入れる。"""
        self._inputs.put_nowait(samples)

    def close(self) -> None:
        """発話の終了。これ以降に確定した予測は送らない。"""
        self._is_open = False
        self._inputs.put_nowait(None)

    async def _run(self) -> None:
        """待ち行列の音声を予測し、相槌を出すと決めたら emitter に渡す。最後に予測器を返す。"""
        try:
            await asyncio.to_thread(self._predictor.reset)
            while (samples := await self._next_batch()) is not None:
                if self._is_open:
                    await self._predict_and_emit(samples)
        except Exception:
            logger.exception("backchannel prediction failed (utterance_id=%d)", self._utterance_id)
            await self._discard_until_closed()
        finally:
            self._log_probabilities()
            self._pool.release(self._predictor)

    async def _next_batch(self) -> AudioSamples | None:
        """次の音声を取り出す。予測が遅れて溜まっていれば、まとめて 1 回で予測する。"""
        first = await self._inputs.get()
        if first is None:
            return None
        batch = [first]
        while not self._inputs.empty():
            queued = self._inputs.get_nowait()
            if queued is None:
                # 終了の印は戻し、溜まっていた分を予測した後のループで受け取る。
                self._inputs.put_nowait(None)
                break
            batch.append(queued)
        return np.concatenate(batch)

    async def _predict_and_emit(self, samples: AudioSamples) -> None:
        """予測して、相槌を出すフレームがあれば emitter に渡す。"""
        frames = await asyncio.to_thread(self._predictor.predict, samples)
        self._frames.extend(frames)
        for frame in frames:
            signal = self._trigger.observe(frame)
            if signal is None or not self._is_open:
                continue
            if isinstance(signal, BackchannelPrepare):
                self._prepares.append(signal.frame)
                self._emitter.prepare(signal)
            else:
                self._fires.append(signal)
                await self._emitter.emit(signal, self.is_open)

    def _log_probabilities(self) -> None:
        """確率の山を残す。しきい値をクライアントの音声に合わせて調整するときの手がかりにする。"""
        summary = summarize_probabilities(self._frames)
        logger.info(
            "backchannel probabilities (utterance_id=%d, frames=%d, max=%.2f, peaks=%s)",
            self._utterance_id,
            summary.frame_count,
            summary.max_probability,
            summary.format_peaks(),
        )

    async def _discard_until_closed(self) -> None:
        """予測に失敗した後は、終了の印が来るまで音声を捨てる。"""
        while await self._inputs.get() is not None:
            pass


def recent_samples(chunks: list[AudioSamples], end_ms: int, window_ms: int) -> AudioSamples:
    """受け取った音声のうち、end_ms までの直近 window_ms を 1 つの波形にする。"""
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    end = end_ms * STREAM_SAMPLE_RATE_HZ // MILLISECONDS_PER_SECOND
    start = max(0, (end_ms - window_ms) * STREAM_SAMPLE_RATE_HZ // MILLISECONDS_PER_SECOND)
    samples: AudioSamples = np.concatenate(chunks)[start:end]
    return samples
