"""次の相槌を出してよいかを、送った相槌からの間隔と、新しく話した長さで決める。

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


class BackchannelPacing:
    """1 つの発話の中で、次の相槌の候補を出してよいかを判定する。

    次のどれかに当たる間は出さない。
    - 送った相槌からの間隔が足りない。
    - 送った相槌（無ければ発話の先頭）から新しく話した長さが min_new_speech_ms に足りない。
      同じ言葉のまま間が空いただけの場所で、もう一度相槌を打たないため。
    - 前の候補を送るかどうかが、まだ決まっていない。
    間隔と話した長さは、送った相槌からだけ数え直す。送らなかった候補で数え直すと、
    言いかけで黙った直後の区切りで相槌を出せなくなるため。
    """

    def __init__(self, interval: BackchannelInterval, min_new_speech_ms: int) -> None:
        """間隔の決め方と、相槌の前に新しく話しているべき長さを受け取る。"""
        self._gate = IntervalGate(interval)
        self._min_new_speech_ms = min_new_speech_ms
        self._new_speech_ms = 0
        self._is_held = False

    def add_speech(self, duration_ms: int) -> None:
        """話していた長さを足す。"""
        self._new_speech_ms += duration_ms

    def is_closed(self, offset_ms: int) -> bool:
        """offset_ms では候補を出せないかを返す。"""
        return (
            self._is_held
            or self._new_speech_ms < self._min_new_speech_ms
            or self._gate.is_closed(offset_ms)
        )

    def hold(self) -> None:
        """候補を出した。送るかどうかが決まるまで、次の候補を出さない。"""
        self._is_held = True

    def settle(self, sent_offset_ms: int | None) -> None:
        """候補の結果が決まった。送ったなら、その位置から間隔と話した長さを数え直す。"""
        self._is_held = False
        if sent_offset_ms is None:
            return
        self._gate.record_fire(sent_offset_ms)
        self._new_speech_ms = 0
