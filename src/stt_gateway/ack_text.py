"""LLM が出した一時的な返事を整形し、返してよい文かどうかを判定する。"""

from __future__ import annotations

# LLM は返事を引用符で囲むことがあり、そのまま読み上げると不自然になるため取り除く。
QUOTE_CHARACTERS = "「」『』\"'“”‘’"


def normalize_ack_text(raw_text: str, max_chars: int) -> str | None:
    """最初の 1 行の前後の空白と引用符を取り除く。空、または max_chars を超えるなら None を返す。

    長すぎる返事は「相槌」ではなく回答を始めている可能性が高いため、使わずに固定の文言にする。
    """
    lines = raw_text.strip().splitlines()
    if not lines:
        return None
    text = lines[0].strip().strip(QUOTE_CHARACTERS).strip()
    if not text or len(text) > max_chars:
        return None
    return text
