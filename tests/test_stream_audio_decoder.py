from __future__ import annotations

import numpy as np
import pytest
from stream_helpers import encode_opus_packets

from stt_gateway.pcm_audio import samples_to_pcm_s16le
from stt_gateway.stream_audio_decoder import (
    AudioDecodeError,
    OpusPacketDecoder,
    PcmS16leDecoder,
    create_stream_audio_decoder,
)


def test_opus_packets_decode_to_16khz_mono() -> None:
    packets = encode_opus_packets(1.0)
    decoder = OpusPacketDecoder()
    samples = np.concatenate([decoder.decode(packet) for packet in packets])
    assert samples.dtype == np.float32
    assert samples.ndim == 1
    # 1 秒ぶん（エンコーダの先読みとリサンプラの遅れで多少前後する）
    assert 15000 <= len(samples) <= 17000
    assert np.abs(samples).max() > 0.1, "正弦波が復元されていること"


def test_opus_decoder_rejects_empty_packet() -> None:
    with pytest.raises(AudioDecodeError, match="must not be empty"):
        OpusPacketDecoder().decode(b"")


def test_opus_decoder_rejects_broken_packet() -> None:
    # TOC バイトだけの不正なパケット（コード 3 で、フレーム数のバイトが無い）
    with pytest.raises(AudioDecodeError, match="failed to decode opus packet"):
        OpusPacketDecoder().decode(b"\xff")


def test_pcm_decoder_converts_and_rejects_odd_bytes() -> None:
    decoder = PcmS16leDecoder()
    samples = decoder.decode(samples_to_pcm_s16le(np.array([0.5], dtype=np.float32)))
    assert samples.tolist() == pytest.approx([0.5], abs=1e-4)
    with pytest.raises(AudioDecodeError, match="even number"):
        decoder.decode(b"\x00")


def test_create_stream_audio_decoder_selects_by_encoding() -> None:
    assert isinstance(create_stream_audio_decoder("opus"), OpusPacketDecoder)
    assert isinstance(create_stream_audio_decoder("pcm_s16le"), PcmS16leDecoder)
