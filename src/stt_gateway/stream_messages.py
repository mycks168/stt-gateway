"""/stream で送受信するメッセージの型。"""

from __future__ import annotations

from typing import Annotated, Literal, NewType, Protocol

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    model_validator,
)

from .ack_generator import AckSource
from .backchannel_interval import (
    MAX_BACKCHANNEL_INTERVAL_FACTOR,
    MIN_BACKCHANNEL_INTERVAL_FACTOR,
)
from .backchannel_policy import (
    BACKCHANNEL_KINDS,
    MIN_BACKCHANNEL_SILENCE_MS,
    BackchannelKind,
    BackchannelTimingName,
    BackchannelUnavailableReason,
)
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


# クライアントが指定できる長さと個数の上限。極端に長いプロンプトで LLM の応答が遅くなるのを防ぐ。
MAX_SYSTEM_PROMPT_CHARS = 2000
MAX_ACK_EXAMPLES = 8
MAX_EXAMPLE_USER_CHARS = 200
MAX_EXAMPLE_REPLY_CHARS = 100
MAX_FALLBACK_TEXT_CHARS = 50
MAX_ACK_CHARS_LIMIT = 100
MAX_BACKCHANNEL_PHRASES = 10
MAX_BACKCHANNEL_PHRASE_CHARS = 20
MAX_BACKCHANNEL_MIN_INTERVAL_MS = 60000
# 無音を長く待つほど、相槌が話の続きにかぶる。
MAX_BACKCHANNEL_SILENCE_MS = 2000
# 相槌の前に新しく話しているべき長さの上限。これ以上は、相槌がほぼ出なくなる。
MAX_BACKCHANNEL_MIN_NEW_SPEECH_MS = 10000
MAX_PHRASES_PER_KIND = 5


def _text(max_length: int) -> StringConstraints:
    """前後の空白を除いて 1〜max_length 文字の文字列にする制約。"""
    return StringConstraints(strip_whitespace=True, min_length=1, max_length=max_length)


class _StrictOptions(BaseModel):
    """未知のフィールドを拒否する。綴りの誤り（system_promt など）を黙って無視しないため。"""

    model_config = ConfigDict(extra="forbid")


class AckExample(_StrictOptions):
    """一時的な返事の会話例 1 つ。"""

    user: Annotated[str, _text(MAX_EXAMPLE_USER_CHARS)]
    reply: Annotated[str, _text(MAX_EXAMPLE_REPLY_CHARS)]


class AckOptions(_StrictOptions):
    """発話ごとの一時的な返事の指定。省略した項目はサーバの設定を使う。"""

    system_prompt: Annotated[str, _text(MAX_SYSTEM_PROMPT_CHARS)] | None = None
    examples: Annotated[list[AckExample], Field(max_length=MAX_ACK_EXAMPLES)] | None = None
    fallback_text: Annotated[str, _text(MAX_FALLBACK_TEXT_CHARS)] | None = None
    max_chars: Annotated[int, Field(ge=1, le=MAX_ACK_CHARS_LIMIT)] | None = None


_BackchannelPhrase = Annotated[str, _text(MAX_BACKCHANNEL_PHRASE_CHARS)]
# 1 つの種類に複数の文言を指定すると、同じ文言が続かないよう順に使う。
_KindPhraseList = Annotated[
    list[_BackchannelPhrase], Field(min_length=1, max_length=MAX_PHRASES_PER_KIND)
]


class BackchannelOptions(_StrictOptions):
    """発話ごとの相槌の指定。省略した項目はサーバの設定を使う。"""

    phrases: (
        Annotated[
            list[Annotated[str, _text(MAX_BACKCHANNEL_PHRASE_CHARS)]],
            Field(min_length=1, max_length=MAX_BACKCHANNEL_PHRASES),
        ]
        | None
    ) = None
    timing: BackchannelTimingName | None = None
    threshold: Annotated[float, Field(gt=0.0, le=1.0)] | None = None
    silence_ms: (
        Annotated[int, Field(ge=MIN_BACKCHANNEL_SILENCE_MS, le=MAX_BACKCHANNEL_SILENCE_MS)] | None
    ) = None
    min_new_speech_ms: Annotated[int, Field(ge=0, le=MAX_BACKCHANNEL_MIN_NEW_SPEECH_MS)] | None = (
        None
    )
    min_interval_ms: Annotated[int, Field(ge=0, le=MAX_BACKCHANNEL_MIN_INTERVAL_MS)] | None = None
    interval_per_minute_ms: (
        Annotated[int, Field(ge=0, le=MAX_BACKCHANNEL_MIN_INTERVAL_MS)] | None
    ) = None
    interval_factor: (
        Annotated[
            float,
            Field(ge=MIN_BACKCHANNEL_INTERVAL_FACTOR, le=MAX_BACKCHANNEL_INTERVAL_FACTOR),
        ]
        | None
    ) = None
    max_interval_ms: Annotated[int, Field(ge=0, le=MAX_BACKCHANNEL_MIN_INTERVAL_MS)] | None = None
    boundary_threshold: Annotated[float, Field(ge=0.0, le=1.0)] | None = None
    phrases_by_kind: (
        Annotated[
            dict[BackchannelKind, _BackchannelPhrase | _KindPhraseList],
            Field(min_length=1, max_length=len(BACKCHANNEL_KINDS)),
        ]
        | None
    ) = None

    @model_validator(mode="after")
    def _reject_both_phrasings(self) -> BackchannelOptions:
        """順に回す文言と種類ごとの文言は、どちらを使うか曖昧になるため同時に指定させない。"""
        if self.phrases is not None and self.phrases_by_kind is not None:
            raise ValueError("specify either phrases or phrases_by_kind, not both")
        return self


class StartMessage(BaseModel):
    """クライアントからの発話の開始。ack が False ならその発話では一時的な返事を作らない。"""

    model_config = ConfigDict(extra="ignore")

    type: Literal["start"]
    encoding: AudioEncoding = "opus"
    language: str | None = None
    ack: AckOptions | Literal[False] | None = None
    backchannel: BackchannelOptions | None = None


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
    """相槌。kind は話の種類で選んだときの種類（順に回したときは None）。

    probability は、timing が maai なら相槌を打つべき確率、silence なら話していない確率。
    """

    type: Literal["backchannel"] = "backchannel"
    utterance_id: UtteranceId
    text: str
    offset_ms: int
    probability: float
    kind: BackchannelKind | None = None


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


class EventSender(Protocol):
    """クライアントへイベントを送る。切断後に呼ばれても例外にしない。"""

    async def send(self, event: StreamEvent) -> None:
        """イベントを 1 つ送る。"""
        ...
