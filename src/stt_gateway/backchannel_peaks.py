"""1 つの発話の相槌の確率の列を、ログに残す山の一覧にまとめる。"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .backchannel_policy import BackchannelFrame

# これより低い確率は山として残さない。しきい値の調整に使うのは、しきい値の近くまで上がった山のため。
PEAK_FLOOR_PROBABILITY = 0.15


@dataclass(frozen=True)
class ProbabilitySummary:
    """発話全体の確率の要約。peaks は PEAK_FLOOR_PROBABILITY 以上が続いた区間ごとの最大のフレーム。"""

    frame_count: int
    max_probability: float
    peaks: tuple[BackchannelFrame, ...]

    def format_peaks(self) -> str:
        """山を「位置ms:確率」の並びにする。"""
        return ",".join(f"{peak.offset_ms}:{peak.probability:.2f}" for peak in self.peaks)


def summarize_probabilities(frames: Sequence[BackchannelFrame]) -> ProbabilitySummary:
    """フレームの列から、確率の最大値と山の一覧を作る。"""
    peaks: list[BackchannelFrame] = []
    current_peak: BackchannelFrame | None = None
    for frame in frames:
        if frame.probability < PEAK_FLOOR_PROBABILITY:
            if current_peak is not None:
                peaks.append(current_peak)
            current_peak = None
        elif current_peak is None or frame.probability > current_peak.probability:
            current_peak = frame
    if current_peak is not None:
        peaks.append(current_peak)
    max_probability = max((frame.probability for frame in frames), default=0.0)
    return ProbabilitySummary(len(frames), max_probability, tuple(peaks))
