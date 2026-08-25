from __future__ import annotations

import pytest

from stt_gateway.formats import UnsupportedAudioFormat, resolve_suffix


@pytest.mark.parametrize(
    "filename",
    ["sample.wav", "SAMPLE.WAV", "voice.ogg", "voice.opus", "voice.webm"],
)
def test_resolve_suffix_accepts_allowed_formats(filename: str) -> None:
    suffix = resolve_suffix(filename)
    assert suffix == suffix.lower()


def test_resolve_suffix_defaults_to_wav_when_no_filename() -> None:
    assert resolve_suffix(None) == ".wav"
    assert resolve_suffix("") == ".wav"


def test_resolve_suffix_defaults_to_wav_when_no_extension() -> None:
    assert resolve_suffix("recording") == ".wav"


def test_resolve_suffix_rejects_unsupported_format() -> None:
    with pytest.raises(UnsupportedAudioFormat) as exc_info:
        resolve_suffix("video.mp4")
    assert ".mp4" in str(exc_info.value)
    assert ".wav" in str(exc_info.value)
