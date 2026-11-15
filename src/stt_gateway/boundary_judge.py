"""途中までの書き起こしが、言い終わりの区切りで止まっているかを言語モデルで測る。"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

from .boundary_text import strip_trailing_breaks

logger = logging.getLogger(__name__)

# 文を受け取り、その続きに区切り（句読点・改行）が来る確率を返す。GPU を使う重い処理。
BreakProbabilityModel = Callable[[str], float]


class BoundaryJudge:
    """書き起こしの続きに区切りが来る確率を、言い終わりの区切りらしさとして返す。

    言いかけ（「なので」「〜の方で」）の後には言葉が続きやすく、言い終わり（「〜なんだよね」）の
    後には句読点が続きやすい。その違いを言語モデルの予測から読む。
    """

    def __init__(self, model: BreakProbabilityModel) -> None:
        """続きに区切りが来る確率を返すモデルを受け取る。"""
        self._model = model
        self._lock = asyncio.Lock()

    async def judge(self, transcript_text: str) -> float | None:
        """区切りらしさ（0〜1）を返す。文字が無い・モデルが失敗した場合は None を返す。

        モデルは同時に使えないため 1 件ずつ、イベントループを止めないよう別スレッドで動かす。
        """
        text = strip_trailing_breaks(transcript_text)
        if not text:
            return None
        try:
            async with self._lock:
                return await asyncio.to_thread(self._model, text)
        except Exception:
            logger.exception("backchannel boundary judge failed (chars=%d)", len(text))
            return None
