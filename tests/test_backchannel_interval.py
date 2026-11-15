from __future__ import annotations

from stt_gateway.backchannel_interval import BackchannelInterval, IntervalGate


def test_interval_grows_with_elapsed_time() -> None:
    interval = BackchannelInterval(2000, 3000, 1.0, 10000)
    assert interval.required_ms(0, 0) == 2000
    assert interval.required_ms(60_000, 0) == 5000, "1 分話すと 3 秒伸びる"


def test_interval_grows_with_backchannel_count() -> None:
    interval = BackchannelInterval(2000, 0, 1.5, 10000)
    assert [interval.required_ms(0, count) for count in range(3)] == [2000, 3000, 4500]


def test_interval_is_capped_but_never_below_min_interval() -> None:
    assert BackchannelInterval(2000, 3000, 2.0, 6000).required_ms(60_000, 3) == 6000
    assert BackchannelInterval(2000, 0, 1.0, 1000).required_ms(0, 0) == 2000, (
        "上限が最短の間隔より小さければ最短の間隔"
    )


def test_gate_waits_longer_after_each_backchannel() -> None:
    gate = IntervalGate(BackchannelInterval(1000, 0, 2.0, 10000))
    assert not gate.is_closed(0), "まだ相槌を出していない"
    gate.record_fire(0)
    assert gate.is_closed(999)
    assert not gate.is_closed(1000), "1 回目と 2 回目の間は min_interval_ms"
    gate.record_fire(1000)
    assert gate.is_closed(2999)
    assert not gate.is_closed(3000), "2 回目と 3 回目の間は 2 倍"


def test_gate_measures_interval_at_candidate_position() -> None:
    gate = IntervalGate(BackchannelInterval(1000, 6000, 1.0, 10000))
    gate.record_fire(30_000)
    # 候補の位置 t で必要な間隔は 1000 + 6000 × t / 60000（34.4 秒なら 4440ms、34.5 秒なら 4450ms）。
    assert gate.is_closed(34_400)
    assert not gate.is_closed(34_500)
