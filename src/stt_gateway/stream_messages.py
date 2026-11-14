"""/stream で送受信するメッセージの型。"""

from __future__ import annotations

from typing import Annotated, Literal, NewType

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from .ack_generator import AckSource
from .backchannel_policy import BackchannelUnavailableReason
from .stream_audio_decoder import AudioEncoding
from .transcription import BackendName

# 接続ごとに 1 から振る発話の通し番号。ほかの整数（時刻やミリ秒）と取り違えないよう専用の型にする。
UtteranceId = NewType("UtteranceId", int)

StreamErrorCode = Literal[
    "invalid_message",
    "invalid_audio",
    "invalid_state",
    "unsupported_language",
    "utterance_too_long",
    "transcribe_failed",
]


class StartMessage(BaseModel):
    """クライアントからの発話の開始。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["start"]
    encoding: AudioEncoding = "opus"
    language: str | None = None


class EndMessage(BaseModel):
    """クライアントからの発話の終了。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["end"]


ClientMessage = Annotated[StartMessage | EndMessage, Field(discriminator="type")]
CLIENT_MESSAGE_ADAPTER: TypeAdapter[StartMessage | EndMessage] = TypeAdapter(ClientMessage)


class StartedEvent(BaseModel):
    """発話の開始を受け付けたことを知らせる。"""

    type: Literal["started"] = "started"
    utterance_id: UtteranceId
    backchannel: bool
    backchannel_unavailable_reason: BackchannelUnavailableReason | None


class BackchannelEvent(BaseModel):
    """相槌。"""

    type: Literal["backchannel"] = "backchannel"
    utterance_id: UtteranceId
    text: str
    offset_ms: int
    probability: float


class TranscriptEvent(BaseModel):
    """発話の文字起こし結果。各フィールドの意味は /transcribe の応答と同じ。"""

    type: Literal["transcript"] = "transcript"
    utterance_id: UtteranceId
    text: str
    language: str
    duration: float | None
    latency_ms: int
    backend: BackendName
    model: str


class AckEvent(BaseModel):
    """一時的な返事。latency_ms は transcript を送ってから ack を送るまでの時間。"""

    type: Literal["ack"] = "ack"
    utterance_id: UtteranceId
    text: str
    source: AckSource
    latency_ms: int


class ErrorEvent(BaseModel):
    """エラー。特定の発話に結び付かないものは utterance_id が None。"""

    type: Literal["error"] = "error"
    utterance_id: UtteranceId | None
    code: StreamErrorCode
    detail: str


StreamEvent = StartedEvent | BackchannelEvent | TranscriptEvent | AckEvent | ErrorEvent
