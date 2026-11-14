"""相槌の確率の列から、相槌を出すかどうかと文言を決める。予測器のインタフェースもここに置く。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

from .audio_decoding import AudioSamples

# 相槌なしで発話を進める理由。busy は予測器の上限、unavailable は予測器を作れないとき。
BackchannelUnavailableReason = Literal["busy", "unavailable"]


@dataclass(frozen=True)
class BackchannelFrame:
    """予測器が 1 フレームごとに出す結果。offset_ms は発話の先頭からの音声上の位置。"""

    offset_ms: int
    probability: float


class BackchannelPredictor(Protocol):
    """ユーザの音声から、相槌を打つべき確率をフレームごとに予測する。

    1 つの予測器を複数の発話で使い回すため、発話の開始時に reset() を呼ぶ。
    predict() は CPU を使う重い処理で、別スレッドから呼ばれる。
    """

    def reset(self) -> None:
        """前の発話の文脈を捨てる。"""
        ...

    def predict(self, samples: AudioSamples) -> list[BackchannelFrame]:
        """16kHz モノラルの音声を追加し、新しく確定したフレームの予測を返す。"""
        ...


@dataclass(frozen=True)
class BackchannelDecision:
    """相槌を出すと決めたときの内容。"""

    text: str
    offset_ms: int
    probability: float


class PhraseRotation:
    """相槌の文言を順に返す。同じ文言が続くと機械的に聞こえるため、接続の間は順番を引き継ぐ。"""

    def __init__(self, phrases: tuple[str, ...]) -> None:
        """1 つ以上の文言を受け取る。"""
        if not phrases:
            raise ValueError("phrases must contain at least one phrase")
        self._phrases = phrases
        self._next_index = 0

    def next_phrase(self) -> str:
        """次の文言を返す。"""
        phrase = self._phrases[self._next_index]
        self._next_index = (self._next_index + 1) % len(self._phrases)
        return phrase


class BackchannelTrigger:
    """1 つの発話の中で、確率がしきい値を下から上へ超えたフレームでだけ相槌を出す。

    しきい値を超えている間ずっと出すと、1 つの間に何度も相槌を打ってしまうため。
    前の相槌から min_interval_ms 経つまでは出さない。予測は近い位置で続けて山を作ることがあるため。
    """

    def __init__(self, threshold: float, min_interval_ms: int, rotation: PhraseRotation) -> None:
        """しきい値・最短の間隔・文言の順番を受け取る。"""
        self._threshold = threshold
        self._min_interval_ms = min_interval_ms
        self._rotation = rotation
        self._was_above_threshold = False
        self._last_fired_offset_ms: int | None = None

    def observe(self, frame: BackchannelFrame) -> BackchannelDecision | None:
        """フレームの予測を受け取り、相槌を出すなら内容を返す。"""
        is_above_threshold = frame.probability >= self._threshold
        is_rising_edge = is_above_threshold and not self._was_above_threshold
        self._was_above_threshold = is_above_threshold
        if not is_rising_edge or self._is_within_min_interval(frame.offset_ms):
            return None
        self._last_fired_offset_ms = frame.offset_ms
        return BackchannelDecision(self._rotation.next_phrase(), frame.offset_ms, frame.probability)

    def _is_within_min_interval(self, offset_ms: int) -> bool:
        """前の相槌から最短の間隔が経っていないかを返す。"""
        if self._last_fired_offset_ms is None:
            return False
        return offset_ms - self._last_fired_offset_ms < self._min_interval_ms
