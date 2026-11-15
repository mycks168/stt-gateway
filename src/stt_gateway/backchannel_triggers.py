"""フレームの予測の列から、相槌の準備と相槌を出す合図を決める（timing ごとの規則）。"""

from __future__ import annotations

from dataclasses import dataclass

from .backchannel_interval import BackchannelInterval, IntervalGate
from .backchannel_policy import (
    BackchannelFire,
    BackchannelFrame,
    BackchannelPrepare,
    BackchannelSignal,
    BackchannelTrigger,
)

# しきい値に対して、話の種類の判定を先に始める確率の割合。確率はしきい値に届く 1〜2 フレーム前から
# 上がり始めるため、そこで判定を始めておけば、しきい値を超えたときに待たずに送れる。
PREPARE_THRESHOLD_RATIO = 2 / 3
# 予測器の確率がこれ以上のフレームを、話していないとみなす（Silero VAD の推奨の境目）。
SILENT_PROBABILITY = 0.5


class ProbabilityTrigger:
    """MaAI の確率がしきい値を下から上へ超えたフレームでだけ相槌を出す（timing が maai のとき）。

    しきい値を超えている間ずっと出すと、1 つの間に何度も相槌を打ってしまうため。
    前の相槌から interval の間隔が経つまでは出さない。予測は近い位置で続けて山を作ることがあるため。
    しきい値の PREPARE_THRESHOLD_RATIO 倍を超えたフレームでは、準備の合図を出す。
    文言は決めない（順に回すか、話の種類で選ぶかは呼び出し側が決める）。
    """

    def __init__(self, threshold: float, interval: BackchannelInterval) -> None:
        """しきい値と、相槌どうしの間隔の決め方を受け取る。"""
        self._threshold = threshold
        self._prepare_threshold = threshold * PREPARE_THRESHOLD_RATIO
        self._gate = IntervalGate(interval)
        self._was_above_threshold = False
        self._was_above_prepare_threshold = False
        self._has_prepared = False

    def observe(self, frame: BackchannelFrame) -> BackchannelSignal | None:
        """フレームの予測を受け取り、準備を始めるか相槌を出すなら、その合図を返す。"""
        is_above_threshold = frame.probability >= self._threshold
        is_above_prepare_threshold = frame.probability >= self._prepare_threshold
        is_rising_edge = is_above_threshold and not self._was_above_threshold
        is_prepare_rising_edge = (
            is_above_prepare_threshold and not self._was_above_prepare_threshold
        )
        self._was_above_threshold = is_above_threshold
        self._was_above_prepare_threshold = is_above_prepare_threshold
        if not is_above_prepare_threshold:
            # 山が終わったら、その山で始めた準備は次の山では使わない。
            self._has_prepared = False
        if self._gate.is_closed(frame.offset_ms):
            return None
        if is_rising_edge:
            return self._fire(frame)
        if is_prepare_rising_edge:
            self._has_prepared = True
            return BackchannelPrepare(frame)
        return None

    def _fire(self, frame: BackchannelFrame) -> BackchannelFire:
        """相槌を出す合図を作る。準備は 1 回の相槌でだけ使う。"""
        self._gate.record_fire(frame.offset_ms)
        signal = BackchannelFire(frame, self._has_prepared)
        self._has_prepared = False
        return signal


class SilenceTrigger:
    """話していない状態が silence_ms 続いたフレームで相槌を出す（timing が silence のとき）。

    フレームの確率は「話していない確率」（Silero VAD）。話していない状態になった最初のフレームで
    準備の合図を出し、話の種類の判定を待つ時間を無音を待つ時間に重ねる。
    1 つの切れ目では 1 回だけ出し、次は話し手がまた話してから出す。発話の先頭の無音
    （クライアントが話し始めの前から送る音声）では、まだ何も話していないため出さない。
    """

    def __init__(self, silence_ms: int, interval: BackchannelInterval) -> None:
        """相槌を出すまで待つ無音の長さと、相槌どうしの間隔の決め方を受け取る。"""
        self._silence_ms = silence_ms
        self._gate = IntervalGate(interval)
        self._last_speech_offset_ms: int | None = None
        self._has_prepared = False

    def observe(self, frame: BackchannelFrame) -> BackchannelSignal | None:
        """フレームを受け取り、準備を始めるか相槌を出すなら、その合図を返す。"""
        if frame.probability < SILENT_PROBABILITY:
            self._last_speech_offset_ms = frame.offset_ms
            # 無音の途中で話し始めたら、その無音で始めた準備は使わない。
            self._has_prepared = False
            return None
        if self._last_speech_offset_ms is None or self._gate.is_closed(frame.offset_ms):
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
        self._gate.record_fire(frame.offset_ms)
        self._last_speech_offset_ms = None
        self._has_prepared = False
        return signal


@dataclass(frozen=True)
class ProbabilityTiming:
    """MaAI の確率がしきい値を超えたところで相槌を出す。"""

    threshold: float


@dataclass(frozen=True)
class SilenceTiming:
    """話していない状態が silence_ms 続いたところで相槌を出す。"""

    silence_ms: int


BackchannelTiming = ProbabilityTiming | SilenceTiming


def create_trigger(timing: BackchannelTiming, interval: BackchannelInterval) -> BackchannelTrigger:
    """タイミングの決め方に合うトリガーを作る。"""
    if isinstance(timing, ProbabilityTiming):
        return ProbabilityTrigger(timing.threshold, interval)
    return SilenceTrigger(timing.silence_ms, interval)
