"""start の ack・backchannel の指定とサーバの設定から、発話ごとの設定を決める。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .ack_generator import AckRequest
from .ack_prompt import DEFAULT_ACK_PROMPT, AckPrompt
from .backchannel_policy import BackchannelKind
from .config import AckSettings, BackchannelSettings
from .stream_messages import AckOptions, BackchannelOptions


@dataclass(frozen=True)
class RotationPhrasing:
    """文言を順に回す。"""

    phrases: tuple[str, ...]


@dataclass(frozen=True)
class KindPhrasing:
    """話の種類を判定し、その種類の文言を順に使う。対応の無い種類では相槌を出さない。"""

    phrases_by_kind: Mapping[BackchannelKind, tuple[str, ...]]


@dataclass(frozen=True)
class BackchannelRule:
    """1 つの発話で相槌を出す規則。"""

    threshold: float
    min_interval_ms: int
    phrasing: RotationPhrasing | KindPhrasing


def resolve_backchannel_rule(
    options: BackchannelOptions | None, defaults: BackchannelSettings
) -> BackchannelRule:
    """指定された項目はそのまま、省略された項目はサーバの設定を使う。"""
    if options is None:
        return BackchannelRule(
            defaults.threshold, defaults.min_interval_ms, RotationPhrasing(defaults.phrases)
        )
    return BackchannelRule(
        threshold=defaults.threshold if options.threshold is None else options.threshold,
        min_interval_ms=(
            defaults.min_interval_ms if options.min_interval_ms is None else options.min_interval_ms
        ),
        phrasing=_resolve_phrasing(options, defaults),
    )


def _resolve_phrasing(
    options: BackchannelOptions, defaults: BackchannelSettings
) -> RotationPhrasing | KindPhrasing:
    """種類ごとの文言が指定されていればそれを、無ければ順に回す文言を使う。"""
    if options.phrases_by_kind is not None:
        return KindPhrasing(
            {
                kind: (phrases,) if isinstance(phrases, str) else tuple(phrases)
                for kind, phrases in options.phrases_by_kind.items()
            }
        )
    return RotationPhrasing(defaults.phrases if options.phrases is None else tuple(options.phrases))


def resolve_ack_request(
    options: AckOptions | Literal[False] | None, defaults: AckSettings
) -> AckRequest | None:
    """一時的な返事の作り方を決める。ack: false なら作らないことを None で表す。"""
    if options is False:
        return None
    if options is None:
        return AckRequest(DEFAULT_ACK_PROMPT, defaults.fallback_text, defaults.max_chars)
    return AckRequest(
        prompt=_resolve_prompt(options),
        fallback_text=defaults.fallback_text
        if options.fallback_text is None
        else options.fallback_text,
        max_chars=defaults.max_chars if options.max_chars is None else options.max_chars,
    )


def _resolve_prompt(options: AckOptions) -> AckPrompt:
    """プロンプトと会話例を決める。

    system_prompt だけを指定したときは既定の会話例を使わない。既定の会話例は既定のプロンプトの
    言い方（相手の言葉を繰り返す）に合わせてあり、別の指示と混ぜると食い違うため。
    """
    if options.system_prompt is None:
        system_prompt = DEFAULT_ACK_PROMPT.system_prompt
        default_examples = DEFAULT_ACK_PROMPT.examples
    else:
        system_prompt = options.system_prompt
        default_examples = ()
    if options.examples is None:
        return AckPrompt(system_prompt, default_examples)
    return AckPrompt(
        system_prompt, tuple((example.user, example.reply) for example in options.examples)
    )
