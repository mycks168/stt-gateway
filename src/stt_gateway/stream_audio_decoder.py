"""/stream で受け取った音声メッセージを、16kHz モノラルの波形に変換する。"""

from __future__ import annotations

from typing import Literal, Protocol

import av
import numpy as np

from .audio_decoding import AudioSamples
from .pcm_audio import pcm_s16le_to_samples

AudioEncoding = Literal["opus", "pcm_s16le"]

STREAM_SAMPLE_RATE_HZ = 16000
# Opus はエンコード時のレートに関係なく 48kHz で復元できるため、デコーダは常に 48kHz で開く。
OPUS_DECODE_SAMPLE_RATE_HZ = 48000


class AudioDecodeError(ValueError):
    """受け取った音声メッセージを波形にできないときに送出する。"""


class StreamAudioDecoder(Protocol):
    """音声メッセージ 1 つを波形に変換する。発話ごとに作り直す。"""

    def decode(self, payload: bytes) -> AudioSamples:
        """メッセージ 1 つ分の音声を 16kHz モノラルの波形にする。"""
        ...


class PcmS16leDecoder:
    """16kHz モノラルの 16bit PCM を波形にする。"""

    def decode(self, payload: bytes) -> AudioSamples:
        """PCM を波形にする。奇数バイトは AudioDecodeError にする。"""
        try:
            return pcm_s16le_to_samples(payload)
        except ValueError as error:
            raise AudioDecodeError(str(error)) from error


class OpusPacketDecoder:
    """コンテナなしの Opus パケットを 1 つずつデコードする。

    Opus は前のパケットの状態を使って復元するため、1 つの発話の間は同じデコーダを使い続ける。
    """

    def __init__(self) -> None:
        """libopus のデコーダと、16kHz モノラルへのリサンプラを用意する。"""
        self._decoder = av.CodecContext.create("libopus", "r")
        self._decoder.sample_rate = OPUS_DECODE_SAMPLE_RATE_HZ
        self._decoder.layout = "mono"
        self._resampler = av.AudioResampler(format="flt", layout="mono", rate=STREAM_SAMPLE_RATE_HZ)

    def decode(self, payload: bytes) -> AudioSamples:
        """パケット 1 つをデコードする。壊れたパケットは AudioDecodeError にする。"""
        if not payload:
            raise AudioDecodeError("opus packet must not be empty")
        chunks: list[AudioSamples] = []
        try:
            for frame in self._decoder.decode(av.Packet(payload)):
                chunks.extend(
                    resampled.to_ndarray().reshape(-1).astype(np.float32, copy=False)
                    for resampled in self._resampler.resample(frame)
                )
        except av.error.FFmpegError as error:
            raise AudioDecodeError(
                f"failed to decode opus packet ({len(payload)} bytes): {error}"
            ) from error
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(chunks)


def create_stream_audio_decoder(encoding: AudioEncoding) -> StreamAudioDecoder:
    """発話の開始時に指定された encoding のデコーダを作る。"""
    if encoding == "opus":
        return OpusPacketDecoder()
    return PcmS16leDecoder()
