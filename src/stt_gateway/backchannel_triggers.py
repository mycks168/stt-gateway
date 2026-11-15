"""フレームの予測の列から、相槌の準備と相槌を出す合図を決める。"""

from __future__ import annotations

from dataclasses import dataclass

from .backchannel_interval import BackchannelInterval, BackchannelPacing
from .backchannel_policy import (
    BackchannelFire,
    BackchannelFrame,
    BackchannelPrepare,
    BackchannelSignal,
    BackchannelTrigger,
)

# 予測器の確率がこれ以上のフレームを、話していないとみなす（Silero VAD の推奨の境目）。
SILENT_PROBABILITY = 0.5


class SilenceTrigger:
    """話していない状態が silence_ms 続いたフレームで相槌を出す。

    フレームの確率は「話していない確率」（Silero VAD）。話していない状態になった最初のフレームで
    準備の合図を出し、送るかどうかの判定を待つ時間を無音を待つ時間に重ねる。
    1 つの切れ目では 1 回だけ出し、次は話し手がまた話してから出す。発話の先頭の無音
    （クライアントが話し始めの前から送る音声）では、まだ何も話していないため出さない。
    話していたフレームの長さは pacing に足す。
    """

    def __init__(self, silence_ms: int, pacing: BackchannelPacing) -> None:
        """相槌を出すまで待つ無音の長さと、次の相槌を出してよいかの判定を受け取る。"""
        self._silence_ms = silence_ms
        self._pacing = pacing
        self._previous_offset_ms = 0
        self._last_speech_offset_ms: int | None = None
        self._has_prepared = False

    def observe(self, frame: BackchannelFrame) -> BackchannelSignal | None:
        """フレームを受け取り、準備を始めるか相槌を出すなら、その合図を返す。"""
        frame_ms = frame.offset_ms - self._previous_offset_ms
        self._previous_offset_ms = frame.offset_ms
        if frame.probability < SILENT_PROBABILITY:
            self._pacing.add_speech(frame_ms)
            self._last_speech_offset_ms = frame.offset_ms
            # 無音の途中で話し始めたら、その無音で始めた準備は使わない。
            self._has_prepared = False
            return None
        if self._last_speech_offset_ms is None or self._pacing.is_closed(frame.offset_ms):
            return None
        if frame.offset_ms - self._last_speech_offset_ms >= self._silence_ms:
            return self._fire(frame)
        if self._has_prepared:
            return None
        self._has_prepared = True
        return BackchannelPrepare(frame)

    def _fire(self, frame: BackchannelFrame) -> BackchannelFire:
        """相槌を出す合図を作り、話し手がまた話すまで次を出さない。"""
        signal = BackchannelFire(frame, self._has_prepared)
        self._pacing.hold()
        self._last_speech_offset_ms = None
        self._has_prepared = False
        return signal


@dataclass(frozen=True)
class SilenceTiming:
    """話していない状態が silence_ms 続いたところで相槌を出す。

    前の相槌（無ければ発話の先頭）から新しく min_new_speech_ms 話していなければ出さない。
    """

    silence_ms: int
    min_new_speech_ms: int


def create_pacing(timing: SilenceTiming, interval: BackchannelInterval) -> BackchannelPacing:
    """次の相槌を出してよいかの判定を作る。"""
    return BackchannelPacing(interval, timing.min_new_speech_ms)


def create_trigger(timing: SilenceTiming, pacing: BackchannelPacing) -> BackchannelTrigger:
    """無音が続いたところで相槌を出すトリガーを作る。"""
    return SilenceTrigger(timing.silence_ms, pacing)
