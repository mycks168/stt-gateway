"""途中までの書き起こしから、Kev（判断だけを返すモデル）で話の種類を判定する。"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import httpx
from pydantic import BaseModel, ValidationError

from .backchannel_policy import BackchannelKind

logger = logging.getLogger(__name__)

KEV_SYSTEMONE_PATH = "/v1/systemone"
KEV_MODEL = "kev-latest"
# 1 回の判定は約 20ms。待つ上限は呼び出し側が別に決めるため、ここは接続が固まったときの保険。
KEV_REQUEST_TIMEOUT_SECONDS = 2.0
KIND_QUESTION_ID = "kind"

# Kev は英語の分類で学習されているため、指示と選択肢の説明は英語で書く（書き起こしは日本語のまま）。
# 日本語で書くと、26 件の比較で合っていた件数が 23 件から 18 件に下がった。
# 種類は 14 にしている。47 件の比較で、7 種類のときより合っていた件数が 37 件から 40 件に増えた。
KIND_INSTRUCTION = (
    "This is a transcript of Japanese speech, possibly cut off where the speaker paused. "
    "What best describes what the speaker is doing at this point?"
)
KIND_CRITERIA: dict[BackchannelKind, str] = {
    "continuing": (
        "The speaker paused mid-story and will clearly continue (e.g. ends with けど、 / で、 / さ、)"
    ),
    "hesitating": "The speaker is hesitating or searching for words (fillers like えっと、その、あのー)",
    "question_or_request": "The speaker is asking a question or making a request",
    "sad_or_trouble": "The speaker reports something sad, painful, or troublesome",
    "explanation": "The speaker explains a reason, a cause, or how a problem got solved",
    "surprising": "The speaker reports something surprising, impressive, or exciting",
    "news": "The speaker finished reporting an ordinary event or piece of news",
    "seeking_agreement": (
        "The speaker states something and seeks agreement (e.g. ends with だよね / でしょ)"
    ),
    "opinion": "The speaker states their own opinion or impression (e.g. 〜と思う)",
    "happy": "The speaker reports something happy, fun, or a relief",
    "complaint": "The speaker complains or is annoyed or frustrated about something",
    "step_by_step": (
        "The speaker is explaining steps in order and will continue with the next step"
    ),
    "thinking_aloud": "The speaker is thinking aloud or undecided (e.g. どうしようかな)",
    "wrapping_up": (
        "The speaker is wrapping up and about to finish talking"
        " (e.g. って感じ, 以上です, そんなところかな)"
    ),
}


def latest_sentence(transcript_text: str) -> str:
    """相槌を打つ現在の文を取り出す。過去の質問や言いよどみで現在の判定を変えないため。"""
    sentences = re.findall(r'[^。！？!?]+[。！？!?]*[」』”’"）)]*', transcript_text.strip())
    return sentences[-1].strip() if sentences else transcript_text.strip()


class _KindAnswer(BaseModel):
    """Kev の choice の答えのうち、使う部分。"""

    choice: BackchannelKind


class _KevAnswers(BaseModel):
    """Kev の answers のうち、使う部分。"""

    kind: _KindAnswer


class _KevResponse(BaseModel):
    """Kev /v1/systemone の応答のうち、使う部分。"""

    answers: _KevAnswers


def read_judge_token(token_file: Path) -> str:
    """Kev の Bearer トークンをファイルから読む。読めない・空なら ValueError にする。"""
    try:
        token = token_file.expanduser().read_text(encoding="utf-8").strip()
    except OSError as error:
        raise ValueError(
            f"failed to read backchannel judge token: {token_file}: {error}"
        ) from error
    if not token:
        raise ValueError(f"backchannel judge token file is empty: {token_file}")
    return token


class BackchannelKindJudge:
    """Kev に、話し手が今何をしているかを 1 つ選ばせる。"""

    def __init__(self, base_url: str, token: str, client: httpx.AsyncClient) -> None:
        """Kev の URL・トークン・HTTP クライアントを受け取る。"""
        self._url = f"{base_url.rstrip('/')}{KEV_SYSTEMONE_PATH}"
        self._headers = {"Authorization": f"Bearer {token}"}
        self._client = client

    async def judge(self, transcript_text: str) -> BackchannelKind | None:
        """話の種類を返す。接続できない・失敗した・応答が読めない場合は None を返す。"""
        try:
            response = await self._client.post(
                self._url,
                headers=self._headers,
                timeout=KEV_REQUEST_TIMEOUT_SECONDS,
                json={
                    "state": latest_sentence(transcript_text),
                    "model": KEV_MODEL,
                    "questions": {
                        KIND_QUESTION_ID: {
                            "type": "choice",
                            "instructions": KIND_INSTRUCTION,
                            "criteria": KIND_CRITERIA,
                        }
                    },
                },
            )
            response.raise_for_status()
            return _KevResponse.model_validate_json(response.content).answers.kind.choice
        except (httpx.HTTPError, ValidationError) as error:
            logger.warning("backchannel kind judge failed (url=%s): %r", self._url, error)
            return None
