"""ワーカー内の相槌の予測器を、上限数まで用意して発話ごとに貸し出す。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass

from .backchannel_policy import BackchannelPredictor, BackchannelUnavailableReason

logger = logging.getLogger(__name__)

PredictorFactory = Callable[[], BackchannelPredictor]


@dataclass(frozen=True)
class LeasedPredictor:
    """貸し出し中の予測器。"""

    predictor: BackchannelPredictor


@dataclass(frozen=True)
class PredictorRefusal:
    """予測器を貸せなかった理由。"""

    reason: BackchannelUnavailableReason


PredictorLease = LeasedPredictor | PredictorRefusal


class BackchannelPool:
    """予測器を最大 max_sessions 個まで作り、使い回す。

    相槌を出す発話はそれぞれ、無音のたびに途中の文字起こしと区切りの判定を GPU で順番に待つ。
    同時に相槌を出す発話を増やしすぎると、その待ちが重なって全員の相槌が間に合わなくなるため上限を設ける。
    返された予測器は、モデルを読み込み直さないよう捨てずに次の発話で使う。
    """

    def __init__(self, max_sessions: int, create_predictor: PredictorFactory) -> None:
        """上限数と、予測器を作る関数を受け取る。作るのは初めて必要になったとき。"""
        self._max_sessions = max_sessions
        self._create_predictor = create_predictor
        self._idle: list[BackchannelPredictor] = []
        self._created_count = 0

    async def acquire(self) -> PredictorLease:
        """予測器を借りる。上限に達していれば busy、作れなければ unavailable を返す。"""
        if self._idle:
            return LeasedPredictor(self._idle.pop())
        if self._created_count >= self._max_sessions:
            return PredictorRefusal("busy")
        # 作っている間に別の発話が上限を超えて作らないよう、先に枠を確保する。
        self._created_count += 1
        try:
            predictor = await asyncio.to_thread(self._create_predictor)
        except Exception:
            self._created_count -= 1
            logger.exception(
                "failed to create backchannel predictor; continuing without backchannel"
            )
            return PredictorRefusal("unavailable")
        return LeasedPredictor(predictor)

    def release(self, predictor: BackchannelPredictor) -> None:
        """予測器を返す。返した後に predict() が動いていないことは呼び出し側が保証する。"""
        self._idle.append(predictor)
