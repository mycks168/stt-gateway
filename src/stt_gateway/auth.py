"""Bearer トークンの検証。/transcribe（HTTP）と /stream（WebSocket）で同じ規則を使う。"""

from __future__ import annotations

import secrets

BEARER_SCHEME = "bearer"


def is_token_accepted(token: str, api_key: str) -> bool:
    """トークンが API キーと一致するかを返す。API キーが空なら値を検証しない。

    比較にかかる時間からキーを推測されないよう、定数時間で比較する。
    """
    if not api_key:
        return True
    return secrets.compare_digest(token.encode(), api_key.encode())


def extract_bearer_token(authorization: str | None) -> str | None:
    """Authorization ヘッダの値から Bearer トークンを取り出す。形式が違えば None を返す。"""
    if not authorization:
        return None
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != BEARER_SCHEME or not token.strip():
        return None
    return token.strip()
