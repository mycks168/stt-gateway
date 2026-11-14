"""/stream のテストで共有する音声の生成処理。"""

from __future__ import annotations

import av
import numpy as np

from stt_gateway.audio_decoding import AudioSamples

OPUS_ENCODE_SAMPLE_RATE_HZ = 48000
OPUS_BITRATE_BPS = 24000


def sine_wave(duration_seconds: float, sample_rate_hz: int = 16000) -> AudioSamples:
    """振幅 0.3 の 440Hz の正弦波を作る。無音だと Opus のデコード結果を確かめにくいため。"""
    times = np.arange(int(duration_seconds * sample_rate_hz)) / sample_rate_hz
    wave: AudioSamples = (0.3 * np.sin(2 * np.pi * 440 * times)).astype(np.float32)
    return wave


def encode_opus_packets(duration_seconds: float) -> list[bytes]:
    """正弦波を 20ms フレームの Opus にエンコードし、コンテナなしのパケットの列を返す。"""
    encoder = av.CodecContext.create("libopus", "w")
    encoder.sample_rate = OPUS_ENCODE_SAMPLE_RATE_HZ
    encoder.layout = "mono"
    encoder.format = "s16"
    encoder.bit_rate = OPUS_BITRATE_BPS
    encoder.open()
    pcm = (sine_wave(duration_seconds, OPUS_ENCODE_SAMPLE_RATE_HZ) * 32767).astype("<i2")
    packets: list[bytes] = []
    for offset in range(0, len(pcm) - encoder.frame_size + 1, encoder.frame_size):
        frame = av.AudioFrame.from_ndarray(
            pcm[offset : offset + encoder.frame_size].reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = OPUS_ENCODE_SAMPLE_RATE_HZ
        packets.extend(bytes(packet) for packet in encoder.encode(frame))
    packets.extend(bytes(packet) for packet in encoder.encode(None))
    return packets
