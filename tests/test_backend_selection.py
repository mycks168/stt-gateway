from __future__ import annotations

import pytest

from stt_gateway.backend_selection import (
    BackendSelection,
    UnsupportedBackendError,
    partial_transcription_backend,
    select_backend,
    startup_backends,
)

_HAAS_ONLY = BackendSelection("haas", ("haas",))
_WHISPER_ONLY = BackendSelection("whisper", ("whisper",))
_WHISPER_DEFAULT_WITH_HAAS = BackendSelection("whisper", ("whisper", "haas"))
_HAAS_DEFAULT_WITH_WHISPER = BackendSelection("haas", ("haas", "whisper"))


def test_selection_rejects_default_that_is_not_enabled() -> None:
    with pytest.raises(ValueError, match="default backend 'haas' is not enabled"):
        BackendSelection("haas", ("whisper",))


def test_select_backend_uses_default_when_omitted() -> None:
    assert select_backend(None, _HAAS_DEFAULT_WITH_WHISPER) == "haas"


def test_select_backend_returns_enabled_backend() -> None:
    assert select_backend("whisper", _HAAS_DEFAULT_WITH_WHISPER) == "whisper"


@pytest.mark.parametrize("requested", ["whisper", "", "HAAS"])
def test_select_backend_rejects_unknown_or_disabled_backend(requested: str) -> None:
    with pytest.raises(UnsupportedBackendError) as exc_info:
        select_backend(requested, _HAAS_ONLY)
    assert str(exc_info.value) == f"unsupported backend: '{requested}' (enabled: haas)"


@pytest.mark.parametrize(
    ("selection", "expected"),
    [
        (_HAAS_ONLY, "haas"),
        (_WHISPER_ONLY, "whisper"),
        (_WHISPER_DEFAULT_WITH_HAAS, "haas"),
        (_HAAS_DEFAULT_WITH_WHISPER, "haas"),
    ],
)
def test_partial_transcription_prefers_haas(selection: BackendSelection, expected: str) -> None:
    assert partial_transcription_backend(selection) == expected


def test_startup_loads_only_default_without_judging() -> None:
    assert startup_backends(_WHISPER_DEFAULT_WITH_HAAS, is_judging_enabled=False) == ("whisper",)


def test_startup_also_loads_judging_backend() -> None:
    assert startup_backends(_WHISPER_DEFAULT_WITH_HAAS, is_judging_enabled=True) == (
        "whisper",
        "haas",
    )


def test_startup_does_not_load_unused_backend_when_judging_uses_default() -> None:
    assert startup_backends(_HAAS_DEFAULT_WITH_WHISPER, is_judging_enabled=True) == ("haas",)
