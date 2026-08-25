"""アップロード音声の拡張子を検証するモジュール。

WAV（既存クライアント向け）と、LTEなど帯域が限られる環境向けの
圧縮フォーマット（Opus系）のみを許可し、それ以外は明示的に拒否する。
実際のデコードは faster-whisper (内部でPyAV/ffmpegを使用) に任せるため、
ここでは受け口のIFとして拡張子の許可判定のみを行う。
"""

from __future__ import annotations

from pathlib import Path

# 拡張子 -> フォーマット種別。値は現状ログ・将来拡張用の識別子。
ALLOWED_SUFFIXES: dict[str, str] = {
    ".wav": "wav",
    ".ogg": "opus",  # Ogg Opus コンテナ
    ".opus": "opus",  # Raw Opus (Oggコンテナ)
    ".webm": "opus",  # WebM Opus コンテナ（ブラウザ録音等）
}

DEFAULT_SUFFIX = ".wav"


class UnsupportedAudioFormat(ValueError):
    """許可リストにない拡張子が指定された場合の例外。"""


def resolve_suffix(filename: str | None) -> str:
    """アップロードされたファイル名から許可済みの拡張子を判定する。

    ファイル名が無い/拡張子が無い場合は、既存クライアント(WAV)との
    互換性のためデフォルトでWAVとして扱う。
    """
    suffix = Path(filename).suffix.lower() if filename else ""
    suffix = suffix or DEFAULT_SUFFIX

    if suffix not in ALLOWED_SUFFIXES:
        allowed = ", ".join(sorted(ALLOWED_SUFFIXES))
        raise UnsupportedAudioFormat(
            f"unsupported audio format: '{suffix}' (allowed: {allowed})"
        )
    return suffix
