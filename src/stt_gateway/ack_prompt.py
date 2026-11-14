"""一時的な返事を作るための、Ollama chat API に渡すメッセージを組み立てる。"""

from __future__ import annotations

from typing import Literal, TypedDict

# 「アシスタント」として指示すると小型の LLM は質問に答えようとし、事実でない内容を言い切る。
# 答えを知らない「聞き手」として相槌を打たせると、受け止めるだけの一言で止まる。
SYSTEM_PROMPT = (
    "あなたは会話の聞き手です。相手の話を聞いて、相槌を一言だけ打ちます。"
    "あなたは答えを知らないので、質問や依頼には答えず「調べるね」「確認するね」のように"
    "受け止めるだけにします。相手の言葉を少しだけ繰り返してもよいです。"
    "15文字以内、くだけた口調、1文だけ。"
)

# 小型の LLM は指示文より会話例に従うため、期待する返事の形を例で示す。
# 質問・依頼だけでなく、質問でない話やあいさつへの返し方も含める。
FEW_SHOT_EXAMPLES: tuple[tuple[str, str], ...] = (
    ("今日のニュースで何か大きいのあった？", "ニュースね、ちょっと見てみるね。"),
    ("昨日ずっと雨でさ、出かけられなかったんだよね", "そっかあ、雨だったんだ。"),
    ("この関数どこで呼ばれてるんだっけ", "呼び出し元ね、探してみる。"),
    ("お疲れさま、今日はもういいよ", "うん、お疲れさま！"),
)


class ChatMessage(TypedDict):
    """Ollama chat API の 1 メッセージ。"""

    role: Literal["system", "user", "assistant"]
    content: str


def build_ack_messages(transcript_text: str) -> list[ChatMessage]:
    """システムプロンプト・会話例・文字起こし結果を、chat API のメッセージ列にする。"""
    messages: list[ChatMessage] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for user_text, reply_text in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": user_text})
        messages.append({"role": "assistant", "content": reply_text})
    messages.append({"role": "user", "content": transcript_text})
    return messages
