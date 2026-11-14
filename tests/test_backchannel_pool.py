from __future__ import annotations

import asyncio

import pytest

from stt_gateway.audio_decoding import AudioSamples
from stt_gateway.backchannel_policy import BackchannelFrame
from stt_gateway.backchannel_pool import BackchannelPool, LeasedPredictor, PredictorRefusal


class _NullPredictor:
    """何も予測しないダミー。"""

    def reset(self) -> None:
        """何もしない。"""

    def predict(self, samples: AudioSamples) -> list[BackchannelFrame]:
        """常に空を返す。"""
        return []


def test_pool_creates_up_to_limit_then_reports_busy() -> None:
    created: list[_NullPredictor] = []

    def create() -> _NullPredictor:
        """作った予測器を記録する。"""
        created.append(_NullPredictor())
        return created[-1]

    async def scenario() -> None:
        pool = BackchannelPool(2, create)
        first = await pool.acquire()
        second = await pool.acquire()
        third = await pool.acquire()
        assert isinstance(first, LeasedPredictor)
        assert isinstance(second, LeasedPredictor)
        assert third == PredictorRefusal("busy")

        pool.release(first.predictor)
        reused = await pool.acquire()
        assert isinstance(reused, LeasedPredictor)
        assert reused.predictor is first.predictor, "返された予測器を作り直さずに使うこと"
        assert len(created) == 2

    asyncio.run(scenario())


def test_pool_reports_unavailable_and_retries_after_failure(
    caplog: pytest.LogCaptureFixture,
) -> None:
    attempts: list[int] = []

    def create() -> _NullPredictor:
        """1 回目だけ失敗する。"""
        attempts.append(1)
        if len(attempts) == 1:
            raise RuntimeError("maai is not installed")
        return _NullPredictor()

    async def scenario() -> None:
        pool = BackchannelPool(1, create)
        assert await pool.acquire() == PredictorRefusal("unavailable")
        # 失敗した分の枠は戻り、次の発話でもう一度作れること
        assert isinstance(await pool.acquire(), LeasedPredictor)

    asyncio.run(scenario())
    assert "failed to create backchannel predictor" in caplog.text
