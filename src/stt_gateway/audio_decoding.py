"""音声ファイルを PyAV(ffmpeg) でモノラル float32 の波形にデコードする。"""

from __future__ import annotations

from pathlib import Path

import av
import numpy as np
import numpy.typing as npt

AudioSamples = npt.NDArray[np.float32]


def decode_to_mono(audio_path: Path, sample_rate_hz: int) -> AudioSamples:
    """指定したサンプリングレートのモノラル float32 にデコードする。

    faster-whisper の decode_audio と同じ処理だが、最後の gc.collect() を呼ばない。
    NeMo と torch を読み込んだプロセスでは、gc.collect() だけで 1 回あたり 200ms 以上かかるため。
    PyAV の resampler が作る循環参照は、Python の通常の世代別 GC で回収される。
    """
    resampler = av.audio.resampler.AudioResampler(
        format="flt", layout="mono", rate=sample_rate_hz
    )
    chunks: list[AudioSamples] = []
    try:
        with av.open(str(audio_path), metadata_errors="ignore") as container:
            for frame in container.decode(audio=0):
                chunks.extend(_to_samples(resampled) for resampled in resampler.resample(frame))
            # resampler 内部に残ったサンプルを取り出す
            chunks.extend(_to_samples(resampled) for resampled in resampler.resample(None))
    except av.error.FFmpegError as error:
        raise ValueError(f"failed to decode audio: {audio_path}: {error}") from error
    if not chunks:
        return np.zeros(0, dtype=np.float32)
    return np.concatenate(chunks)


def _to_samples(frame: av.AudioFrame) -> AudioSamples:
    """モノラル flt のフレームを 1 次元の float32 配列にする。"""
    samples: AudioSamples = frame.to_ndarray().reshape(-1).astype(np.float32, copy=False)
    return samples
