from __future__ import annotations

from stt_gateway.backchannel_interval import BackchannelInterval, BackchannelPacing
from stt_gateway.backchannel_policy import (
    BackchannelFire,
    BackchannelFrame,
    BackchannelPrepare,
    BackchannelSignal,
    BackchannelTrigger,
)
from stt_gateway.backchannel_triggers import (
    SilenceTiming,
    SilenceTrigger,
    create_pacing,
    create_trigger,
)


def _fixed(interval_ms: int, min_new_speech_ms: int = 0) -> BackchannelPacing:
    """伸びない、決まった間隔で相槌を出せる判定。"""
    return BackchannelPacing(
        BackchannelInterval(interval_ms, 0, 1.0, interval_ms), min_new_speech_ms
    )


def _observe_frames(
    trigger: BackchannelTrigger,
    pacing: BackchannelPacing,
    probabilities: list[float],
    frame_ms: int,
    *,
    is_sent: bool,
) -> list[BackchannelSignal]:
    """frame_ms 間隔のフレームとして順に渡し、出た合図を返す。

    相槌の合図が出たら、その場で送った（is_sent が False なら送らなかった）ことにする。
    """
    signals = []
    for index, probability in enumerate(probabilities, start=1):
        signal = trigger.observe(BackchannelFrame(index * frame_ms, probability))
        if signal is None:
            continue
        signals.append(signal)
        if isinstance(signal, BackchannelFire):
            pacing.settle(signal.frame.offset_ms if is_sent else None)
    return signals


# 32ms ごとの「話していない確率」。0.5 以上を無音とみなす。
SPEECH, SILENT = 0.1, 0.9


def _silence_signals(
    pacing: BackchannelPacing,
    silence_ms: int,
    probabilities: list[float],
    *,
    is_sent: bool = True,
) -> list[BackchannelSignal]:
    """話していない確率を 32ms 間隔で渡し、出た合図を返す。"""
    return _observe_frames(
        SilenceTrigger(silence_ms, pacing), pacing, probabilities, 32, is_sent=is_sent
    )


def _fire_offsets(signals: list[BackchannelSignal]) -> list[int]:
    """相槌を出すと決めたフレームの位置を返す。"""
    return [signal.frame.offset_ms for signal in signals if isinstance(signal, BackchannelFire)]


def test_silence_trigger_prepares_at_silence_start_and_fires_after_silence_ms() -> None:
    signals = _silence_signals(_fixed(0), 96, [SPEECH, SPEECH, SILENT, SILENT, SILENT, SILENT])
    assert signals == [
        BackchannelPrepare(BackchannelFrame(96, SILENT)),
        BackchannelFire(BackchannelFrame(160, SILENT), has_prepared=True),
    ]


def test_silence_trigger_fires_once_per_silence() -> None:
    signals = _silence_signals(_fixed(0), 64, [SPEECH] + [SILENT] * 10 + [SPEECH] + [SILENT] * 3)
    assert _fire_offsets(signals) == [96, 448], (
        "同じ無音では 1 回だけ。話してからの次の無音でまた出す"
    )


def test_silence_trigger_drops_preparation_when_speech_resumes() -> None:
    signals = _silence_signals(
        _fixed(0), 96, [SPEECH, SILENT, SPEECH, SILENT, SILENT, SILENT, SILENT]
    )
    assert [type(signal) for signal in signals] == [
        BackchannelPrepare,
        BackchannelPrepare,
        BackchannelFire,
    ]
    assert signals[-1] == BackchannelFire(BackchannelFrame(192, SILENT), has_prepared=True)


def test_silence_trigger_ignores_leading_silence() -> None:
    assert _silence_signals(_fixed(0), 64, [SILENT] * 10) == [], "まだ何も話していない"


def test_silence_trigger_waits_min_interval() -> None:
    probabilities = [SPEECH, SILENT, SILENT, SILENT, SPEECH, SILENT, SILENT, SILENT]
    assert _fire_offsets(_silence_signals(_fixed(1000), 64, probabilities)) == [96]


def test_silence_trigger_does_not_wait_interval_after_unsent_candidate() -> None:
    probabilities = [SPEECH, SILENT, SILENT, SILENT, SPEECH, SILENT, SILENT, SILENT]
    signals = _silence_signals(_fixed(1000), 64, probabilities, is_sent=False)
    assert _fire_offsets(signals) == [96, 224], "送らなかった候補では間隔を数え直さない"


def test_silence_trigger_waits_for_new_speech() -> None:
    # 話したのは 1 フレーム（32ms）ずつ。64ms 話すまでは候補を出さない。
    probabilities = [SPEECH, SILENT, SILENT, SILENT, SPEECH, SILENT, SILENT, SILENT]
    assert _fire_offsets(_silence_signals(_fixed(0, 64), 64, probabilities)) == [224]
    # 送った後は数え直すため、次の 1 フレームだけでは出さない。
    longer = [*probabilities, SPEECH, SILENT, SILENT, SILENT]
    assert _fire_offsets(_silence_signals(_fixed(0, 64), 64, longer)) == [224]


def test_trigger_waits_until_previous_candidate_is_settled() -> None:
    pacing = _fixed(0)
    trigger = SilenceTrigger(64, pacing)
    probabilities = [SPEECH, SILENT, SILENT, SILENT, SPEECH, SILENT, SILENT, SILENT]
    signals = [trigger.observe(BackchannelFrame(i * 32, p)) for i, p in enumerate(probabilities, 1)]
    fires = [signal for signal in signals if isinstance(signal, BackchannelFire)]
    assert [fire.frame.offset_ms for fire in fires] == [96], (
        "前の候補を送るかどうかが決まるまで、次の候補を出さない"
    )


def test_create_trigger_uses_silence_ms() -> None:
    pacing = _fixed(0)
    trigger = create_trigger(SilenceTiming(64, 0), pacing)
    signals = _observe_frames(trigger, pacing, [SPEECH, SILENT, SILENT, SILENT], 32, is_sent=True)
    assert _fire_offsets(signals) == [96]


def test_create_pacing_requires_new_speech() -> None:
    interval = BackchannelInterval(0, 0, 1.0, 0)
    assert create_pacing(SilenceTiming(120, 1000), interval).is_closed(500), "まだ話していない"
    assert not create_pacing(SilenceTiming(120, 0), interval).is_closed(500)
