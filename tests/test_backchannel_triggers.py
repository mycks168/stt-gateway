from __future__ import annotations

from stt_gateway.backchannel_interval import BackchannelInterval
from stt_gateway.backchannel_policy import (
    BackchannelFire,
    BackchannelFrame,
    BackchannelPrepare,
    BackchannelSignal,
    BackchannelTrigger,
)
from stt_gateway.backchannel_triggers import (
    ProbabilityTiming,
    ProbabilityTrigger,
    SilenceTiming,
    SilenceTrigger,
    create_trigger,
)


def _fixed(interval_ms: int) -> BackchannelInterval:
    """伸びない、決まった間隔。"""
    return BackchannelInterval(interval_ms, 0, 1.0, interval_ms)


def _signals(trigger: BackchannelTrigger, probabilities: list[float]) -> list[BackchannelSignal]:
    """100ms 間隔のフレームとして順に渡し、出た合図を返す。"""
    signals = []
    for index, probability in enumerate(probabilities, start=1):
        signal = trigger.observe(BackchannelFrame(index * 100, probability))
        if signal is not None:
            signals.append(signal)
    return signals


def _observe_all(trigger: BackchannelTrigger, probabilities: list[float]) -> list[BackchannelFrame]:
    """100ms 間隔のフレームとして順に渡し、相槌を出すと決めたフレームを返す。"""
    return [
        signal.frame
        for signal in _signals(trigger, probabilities)
        if isinstance(signal, BackchannelFire)
    ]


def test_trigger_fires_once_per_rising_edge() -> None:
    trigger = ProbabilityTrigger(0.5, _fixed(0))
    fired = _observe_all(trigger, [0.1, 0.6, 0.7, 0.8, 0.2, 0.55])
    assert [(f.offset_ms, f.probability) for f in fired] == [(200, 0.6), (600, 0.55)]


def test_trigger_treats_threshold_as_above() -> None:
    assert _observe_all(ProbabilityTrigger(0.5, _fixed(0)), [0.5])[0].offset_ms == 100


def test_trigger_skips_rising_edges_within_min_interval() -> None:
    trigger = ProbabilityTrigger(0.5, _fixed(1500))
    # 200ms で発火、900ms は 700ms 後なので出さない、1700ms は 1500ms 後なので出す
    probabilities = [0.0, 0.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.6] + [0.0] * 7 + [0.6]
    assert [f.offset_ms for f in _observe_all(trigger, probabilities)] == [200, 1700]


def test_trigger_prepares_before_firing_in_same_peak() -> None:
    # しきい値 0.6 の 2/3 は 0.4。0.45 で準備し、0.7 で準備済みとして発火する。
    signals = _signals(ProbabilityTrigger(0.6, _fixed(0)), [0.1, 0.45, 0.7, 0.8])
    assert signals == [
        BackchannelPrepare(BackchannelFrame(200, 0.45)),
        BackchannelFire(BackchannelFrame(300, 0.7), has_prepared=True),
    ]


def test_trigger_fires_unprepared_when_jumping_over_both_levels() -> None:
    signals = _signals(ProbabilityTrigger(0.6, _fixed(0)), [0.1, 0.7])
    assert signals == [BackchannelFire(BackchannelFrame(200, 0.7), has_prepared=False)]


def test_trigger_does_not_reuse_preparation_from_ended_peak() -> None:
    # 準備した山は発火せずに終わり、次の山ではいきなりしきい値を超える。
    signals = _signals(ProbabilityTrigger(0.6, _fixed(0)), [0.45, 0.1, 0.7])
    assert signals[-1] == BackchannelFire(BackchannelFrame(300, 0.7), has_prepared=False)


def test_trigger_uses_preparation_only_for_one_backchannel() -> None:
    # 発火の後、準備の水準より上のまましきい値を下回り、また超えたときは準備し直していない。
    signals = _signals(ProbabilityTrigger(0.6, _fixed(0)), [0.45, 0.7, 0.5, 0.7])
    assert signals[1:] == [
        BackchannelFire(BackchannelFrame(200, 0.7), has_prepared=True),
        BackchannelFire(BackchannelFrame(400, 0.7), has_prepared=False),
    ]


def test_trigger_does_not_prepare_within_min_interval() -> None:
    signals = _signals(ProbabilityTrigger(0.6, _fixed(1000)), [0.7, 0.1, 0.45, 0.7])
    assert signals == [BackchannelFire(BackchannelFrame(100, 0.7), has_prepared=False)]


# 32ms ごとの「話していない確率」。0.5 以上を無音とみなす。
SPEECH, SILENT = 0.1, 0.9


def _silence_signals(
    trigger: BackchannelTrigger, probabilities: list[float]
) -> list[BackchannelSignal]:
    """32ms 間隔のフレームとして順に渡し、出た合図を返す。"""
    signals = []
    for index, probability in enumerate(probabilities, start=1):
        signal = trigger.observe(BackchannelFrame(index * 32, probability))
        if signal is not None:
            signals.append(signal)
    return signals


def test_silence_trigger_prepares_at_silence_start_and_fires_after_silence_ms() -> None:
    signals = _silence_signals(
        SilenceTrigger(96, _fixed(0)), [SPEECH, SPEECH, SILENT, SILENT, SILENT, SILENT]
    )
    assert signals == [
        BackchannelPrepare(BackchannelFrame(96, SILENT)),
        BackchannelFire(BackchannelFrame(160, SILENT), has_prepared=True),
    ]


def test_silence_trigger_fires_once_per_silence() -> None:
    signals = _silence_signals(
        SilenceTrigger(64, _fixed(0)), [SPEECH] + [SILENT] * 10 + [SPEECH] + [SILENT] * 3
    )
    fires = [signal.frame.offset_ms for signal in signals if isinstance(signal, BackchannelFire)]
    assert fires == [96, 448], "同じ無音では 1 回だけ。話してからの次の無音でまた出す"


def test_silence_trigger_drops_preparation_when_speech_resumes() -> None:
    signals = _silence_signals(
        SilenceTrigger(96, _fixed(0)), [SPEECH, SILENT, SPEECH, SILENT, SILENT, SILENT, SILENT]
    )
    assert [type(signal) for signal in signals] == [
        BackchannelPrepare,
        BackchannelPrepare,
        BackchannelFire,
    ]
    assert signals[-1] == BackchannelFire(BackchannelFrame(192, SILENT), has_prepared=True)


def test_silence_trigger_ignores_leading_silence() -> None:
    assert _silence_signals(SilenceTrigger(64, _fixed(0)), [SILENT] * 10) == [], (
        "まだ何も話していない"
    )


def test_silence_trigger_waits_min_interval() -> None:
    probabilities = [SPEECH, SILENT, SILENT, SILENT, SPEECH, SILENT, SILENT, SILENT]
    signals = _silence_signals(SilenceTrigger(64, _fixed(1000)), probabilities)
    assert [
        signal.frame.offset_ms for signal in signals if isinstance(signal, BackchannelFire)
    ] == [96]


def test_create_trigger_matches_timing() -> None:
    assert isinstance(create_trigger(ProbabilityTiming(0.5), _fixed(0)), ProbabilityTrigger)
    assert isinstance(create_trigger(SilenceTiming(120), _fixed(0)), SilenceTrigger)
