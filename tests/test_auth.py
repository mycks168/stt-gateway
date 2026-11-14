from __future__ import annotations

import pytest

from stt_gateway.auth import extract_bearer_token, is_token_accepted


def test_is_token_accepted_skips_check_without_api_key() -> None:
    assert is_token_accepted("anything", "")


def test_is_token_accepted_compares_with_api_key() -> None:
    assert is_token_accepted("secret", "secret")
    assert not is_token_accepted("wrong", "secret")


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Bearer abc", "abc"),
        ("bearer  abc ", "abc"),
        (None, None),
        ("", None),
        ("Basic abc", None),
        ("Bearer ", None),
        ("Bearer", None),
    ],
)
def test_extract_bearer_token(header: str | None, expected: str | None) -> None:
    assert extract_bearer_token(header) == expected
