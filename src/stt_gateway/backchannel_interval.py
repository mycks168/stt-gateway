"""相槌どうしの最短の間隔を、発話の経過時間と相槌の回数に合わせて伸ばす。

人は相手の話が長くなるほど相槌を減らすため、話し始めは min_interval_ms で打ち、
話が続くほど・相槌を重ねるほど間隔を空ける。
"""

from __future__ import annotations

from dataclasses import dataclass

MILLISECONDS_PER_MINUTE = 60_000
# 1 未満だと相槌のたびに間隔が縮み、大きすぎると 2〜3 回で上限に張り付くため、この範囲だけ受け付ける。
MIN_BACKCHANNEL_INTERVAL_FACTOR = 1.0
MAX_BACKCHANNEL_INTERVAL_FACTOR = 3.0


@dataclass(frozen=True)
class BackchannelInterval:
    """最短の間隔の決め方。

    間隔 = min(max_interval_ms, (min_interval_ms + 経過分 × per_minute_ms) × factor ^ n)。
    n は前の相槌より前に出した相槌の数（1 回目と 2 回目の間は n = 0）。
    max_interval_ms が min_interval_ms より小さいときは min_interval_ms を上限にする。
    """

    min_interval_ms: int
    per_minute_ms: int
    factor: float
    max_interval_ms: int

    def required_ms(self, elapsed_ms: int, earlier_count: int) -> float:
        """発話の先頭から elapsed_ms の位置で、前の相槌から空ける間隔を返す。

        earlier_count は前の相槌より前に出した相槌の数。
        """
        grown = self.min_interval_ms + self.per_minute_ms * elapsed_ms / MILLISECONDS_PER_MINUTE
        return min(
            max(self.max_interval_ms, self.min_interval_ms), grown * self.factor**earlier_count
        )


class IntervalGate:
    """1 つの発話の中で、前の相槌から必要な間隔が経ったかを判定する。"""

    def __init__(self, interval: BackchannelInterval) -> None:
        """間隔の決め方を受け取る。"""
        self._interval = interval
        self._last_fired_offset_ms: int | None = None
        self._fired_count = 0

    def is_closed(self, offset_ms: int) -> bool:
        """offset_ms では、前の相槌からの間隔が足りず出せないかを返す。"""
        if self._last_fired_offset_ms is None:
            return False
        required_ms = self._interval.required_ms(offset_ms, self._fired_count - 1)
        return offset_ms - self._last_fired_offset_ms < required_ms

    def record_fire(self, offset_ms: int) -> None:
        """相槌を出したことを記録する。"""
        self._last_fired_offset_ms = offset_ms
        self._fired_count += 1
