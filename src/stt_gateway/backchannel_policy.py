"""相槌の判断に使う型（話の種類・フレーム・合図）と、予測器・トリガー・文言の順番のインタフェース。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, get_args

from .audio_decoding import AudioSamples

# 相槌なしで発話を進める理由。busy は予測器の上限、unavailable は予測器や話の種類の判定を使えないとき。
BackchannelUnavailableReason = Literal["busy", "unavailable"]

# 相槌を打つ時点で、話し手が今何をしているか。クライアントは種類ごとに相槌の文言を指定する。
BackchannelKind = Literal[
    "continuing",
    "hesitating",
    "question_or_request",
    "sad_or_trouble",
    "explanation",
    "surprising",
    "news",
    "seeking_agreement",
    "opinion",
    "happy",
    "complaint",
    "step_by_step",
    "thinking_aloud",
    "wrapping_up",
]
BACKCHANNEL_KINDS: tuple[BackchannelKind, ...] = get_args(BackchannelKind)

# 相槌のタイミングの決め方。maai は MaAI の確率、silence は Silero VAD で無音が続いたことで決める。
BackchannelTimingName = Literal["maai", "silence"]
BACKCHANNEL_TIMING_NAMES: tuple[BackchannelTimingName, ...] = get_args(BackchannelTimingName)
# Silero VAD は 32ms ごとに判定するため、それより短い無音は見分けられない。
MIN_BACKCHANNEL_SILENCE_MS = 32


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


@dataclass(frozen=True)
class BackchannelPrepare:
    """相槌を出しそうなフレーム。文言を決める準備（話の種類の判定）を先に始めてよい。"""

    frame: BackchannelFrame


@dataclass(frozen=True)
class BackchannelFire:
    """相槌を出すと決めたフレーム。

    has_prepared は、同じ山（maai）または同じ無音（silence）で直前に BackchannelPrepare を出したか。
    """

    frame: BackchannelFrame
    has_prepared: bool


BackchannelSignal = BackchannelPrepare | BackchannelFire

# 相槌の候補をどうしたか。sent 以外は送らなかった理由。
BackchannelDecisionReason = Literal[
    "sent", "timeout", "not_boundary", "judge_unavailable", "unmapped_kind", "closed"
]


@dataclass(frozen=True)
class BackchannelDecision:
    """途中の文字起こしで判定した相槌の候補 1 つの結果。

    text はその時点までの文字起こし、boundary_probability は言い終わりの区切りらしさ。
    判定が間に合わなかった・使えなかった項目は None。
    """

    offset_ms: int
    reason: BackchannelDecisionReason
    text: str | None
    boundary_probability: float | None
    kind: BackchannelKind | None


class BackchannelTrigger(Protocol):
    """フレームの予測を順に受け取り、準備や相槌の合図を出す。1 つの発話の中で状態を持つ。"""

    def observe(self, frame: BackchannelFrame) -> BackchannelSignal | None:
        """フレームを受け取り、準備を始めるか相槌を出すなら、その合図を返す。"""
        ...
