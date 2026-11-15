"""言い終わりの区切り（句読点・改行）を、文字の種類で見分ける。

特定の記号を並べず Unicode の分類で決める。書き起こしやモデルの語彙に出る記号の種類に
左右されないため。
"""

from __future__ import annotations

import unicodedata

PUNCTUATION_CATEGORY_PREFIX = "P"


def _is_punctuation(character: str) -> bool:
    """句読点・括弧などの記号かを返す。"""
    return unicodedata.category(character).startswith(PUNCTUATION_CATEGORY_PREFIX)


def is_break_token(token_text: str) -> bool:
    """モデルの 1 トークンが、区切り（空白・改行だけ、または記号で始まる）かを返す。"""
    stripped = token_text.strip()
    return not stripped or _is_punctuation(stripped[0])


def strip_trailing_breaks(text: str) -> str:
    """末尾の記号と空白を取り除く。

    書き起こしが末尾に句読点を付けるかどうかで、続きの予測が変わらないようにするため。
    """
    end = len(text)
    while end > 0 and (_is_punctuation(text[end - 1]) or text[end - 1].isspace()):
        end -= 1
    return text[:end]
