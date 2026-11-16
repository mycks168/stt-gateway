"""リクエストで指定された文字起こしエンジンを、サーバで使えるものの中から決める。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import get_args

from .transcription import BackendName

BACKEND_NAMES: tuple[BackendName, ...] = get_args(BackendName)

# 相槌の判定に使う途中の文字起こしで、優先するエンジンの順。判定は
# STT_BACKCHANNEL_JUDGE_TIMEOUT_MS（既定 200ms）以内に終える必要があり、直近 15 秒の音声で
# haas は約 40ms、whisper は約 320ms かかるため、haas を先にする。
PARTIAL_TRANSCRIPTION_PREFERENCE: tuple[BackendName, ...] = ("haas", "whisper")


class UnsupportedBackendError(ValueError):
    """知らないエンジン、またはサーバで有効にしていないエンジンを指定されたときに送出する。"""

    def __init__(self, requested: str, enabled: tuple[BackendName, ...]) -> None:
        """エラーメッセージに、有効なエンジンの一覧を含めて初期化する。"""
        super().__init__(f"unsupported backend: '{requested}' (enabled: {', '.join(enabled)})")


@dataclass(frozen=True)
class BackendSelection:
    """サーバで使える文字起こしエンジンと、指定が無いときに使うエンジン。"""

    default: BackendName
    enabled: tuple[BackendName, ...]

    def __post_init__(self) -> None:
        """既定のエンジンが有効なエンジンに含まれない設定を拒否する。"""
        if self.default not in self.enabled:
            raise ValueError(
                f"default backend '{self.default}' is not enabled "
                f"(enabled: {', '.join(self.enabled)})"
            )


def parse_backend_name(value: str) -> BackendName | None:
    """エンジンの名前を BackendName にする。知らない名前なら None を返す。"""
    for backend in BACKEND_NAMES:
        if value == backend:
            return backend
    return None


def select_backend(requested: str | None, selection: BackendSelection) -> BackendName:
    """指定されたエンジンを返す。省略時は既定のエンジン。使えなければ UnsupportedBackendError。"""
    if requested is None:
        return selection.default
    backend = parse_backend_name(requested)
    if backend is None or backend not in selection.enabled:
        raise UnsupportedBackendError(requested, selection.enabled)
    return backend


def partial_transcription_backend(selection: BackendSelection) -> BackendName:
    """相槌の判定に使う途中の文字起こしのエンジンを、有効なエンジンの中から速い順に選ぶ。"""
    return min(selection.enabled, key=PARTIAL_TRANSCRIPTION_PREFERENCE.index)


def startup_backends(
    selection: BackendSelection, is_judging_enabled: bool
) -> tuple[BackendName, ...]:
    """起動時に読み込むエンジン。ほかのエンジンは初めて指定されたときに読み込む。

    相槌の判定に使うエンジンも起動時に読み込む。判定は短い時間で終える必要があり、
    最初の判定がモデルの読み込みを待つと時間切れになるため。
    """
    if not is_judging_enabled:
        return (selection.default,)
    judging_backend = partial_transcription_backend(selection)
    if judging_backend == selection.default:
        return (selection.default,)
    return (selection.default, judging_backend)
