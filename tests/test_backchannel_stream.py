from __future__ import annotations

import numpy as np

from stt_gateway.backchannel_stream import recent_samples


def test_recent_samples_cuts_window_before_frame() -> None:
    chunks = [np.ones(1000, dtype=np.float32), np.zeros(3000, dtype=np.float32)]
    assert len(recent_samples(chunks, 200, 1000)) == 3200
    assert len(recent_samples(chunks, 1000, 1000)) == 4000, "受け取った分より先は無い"
    window = recent_samples(chunks, 200, 100)
    assert len(window) == 1600, "直近 100ms だけ"
    assert window.sum() == 0, "先頭の 1000 サンプルは窓の外"
    assert len(recent_samples([], 100, 1000)) == 0
