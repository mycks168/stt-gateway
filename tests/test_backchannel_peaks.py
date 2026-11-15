from __future__ import annotations

from stt_gateway.backchannel_peaks import summarize_probabilities
from stt_gateway.backchannel_policy import BackchannelFrame


def _frames(probabilities: list[float]) -> list[BackchannelFrame]:
    """100ms 間隔のフレームを作る。"""
    return [
        BackchannelFrame(index * 100, probability)
        for index, probability in enumerate(probabilities, start=1)
    ]


def test_summary_keeps_highest_frame_of_each_peak() -> None:
    summary = summarize_probabilities(_frames([0.0, 0.2, 0.5, 0.3, 0.1, 0.16, 0.05, 0.4]))
    assert summary.frame_count == 8
    assert summary.max_probability == 0.5
    assert summary.peaks == (
        BackchannelFrame(300, 0.5),
        BackchannelFrame(600, 0.16),
        BackchannelFrame(800, 0.4),
    )
    assert summary.format_peaks() == "300:0.50,600:0.16,800:0.40"


def test_summary_ignores_probabilities_below_floor() -> None:
    summary = summarize_probabilities(_frames([0.1, 0.14]))
    assert summary.peaks == ()
    assert summary.format_peaks() == ""


def test_summary_of_no_frames() -> None:
    summary = summarize_probabilities([])
    assert (summary.frame_count, summary.max_probability, summary.peaks) == (0, 0.0, ())
