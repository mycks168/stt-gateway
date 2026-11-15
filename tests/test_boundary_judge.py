from __future__ import annotations

import asyncio
import threading

import pytest

from stt_gateway.boundary_judge import BoundaryJudge


def test_judge_passes_text_without_trailing_punctuation() -> None:
    received: list[str] = []

    def model(text: str) -> float:
        """受け取った文を記録し、決まった確率を返す。"""
        received.append(text)
        return 0.8

    assert asyncio.run(BoundaryJudge(model).judge("してもらったんだよね。")) == 0.8
    assert received == ["してもらったんだよね"], "書き起こしの句読点の有無で予測を変えない"


def test_judge_skips_text_without_words() -> None:
    def model(text: str) -> float:
        """呼ばれないはずのモデル。"""
        raise AssertionError(f"model must not be called: {text!r}")

    assert asyncio.run(BoundaryJudge(model).judge("。 ")) is None


def test_judge_returns_none_when_model_fails(caplog: pytest.LogCaptureFixture) -> None:
    def model(text: str) -> float:
        """いつも失敗するモデル。"""
        raise RuntimeError(f"CUDA out of memory ({len(text)} chars)")

    assert asyncio.run(BoundaryJudge(model).judge("引っ越したんだ")) is None
    assert "backchannel boundary judge failed" in caplog.text
    assert "引っ越したんだ" not in caplog.text, "発話本文はログに出さない"


def test_judge_runs_model_one_at_a_time() -> None:
    received: list[str] = []
    running = 0
    max_running = 0
    lock = threading.Lock()
    release = threading.Event()

    def model(text: str) -> float:
        """同時に動いている数の最大を数える。"""
        nonlocal running, max_running
        received.append(text)
        with lock:
            running += 1
            max_running = max(max_running, running)
        release.wait(0.05)
        with lock:
            running -= 1
        return 0.5

    async def run() -> list[float | None]:
        judge = BoundaryJudge(model)
        return list(await asyncio.gather(judge.judge("なので"), judge.judge("それで")))

    assert asyncio.run(run()) == [0.5, 0.5]
    assert sorted(received) == ["それで", "なので"]
    assert max_running == 1
