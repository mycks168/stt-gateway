from __future__ import annotations

import pytest

from stt_gateway.config import parse_backend


@pytest.mark.parametrize("value", ["whisper", "haas"])
def test_parse_backend_accepts_known_backends(value: str) -> None:
    assert parse_backend(value) == value


def test_parse_backend_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError) as exc_info:
        parse_backend("Haas")
    message = str(exc_info.value)
    assert "'Haas'" in message
    assert "whisper, haas" in message
