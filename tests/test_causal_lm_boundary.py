from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

from stt_gateway.causal_lm_boundary import load_break_probability_model

# ダミーの語彙。番号 1 と 3 が区切り（句読点・改行）。
_VOCABULARY = ["で", "、", "ね", "\n"]
# モデルの出力は語彙より 1 つ長い（使われない番号の分）。
_LOGIT_COUNT = len(_VOCABULARY) + 1


class _FakeTokenizer:
    """1 文字を 1 トークンにするダミーのトークナイザ。"""

    def __len__(self) -> int:
        return len(_VOCABULARY)

    def decode(self, token_ids: list[int]) -> str:
        """番号を文字に戻す。"""
        return "".join(_VOCABULARY[token_id] for token_id in token_ids)

    def __call__(self, text: str, return_tensors: str) -> SimpleNamespace:
        """文字を番号の列にする。"""
        assert return_tensors == "pt"
        return SimpleNamespace(input_ids=torch.tensor([[_VOCABULARY.index(c) for c in text]]))


class _FakeModel:
    """最後のトークンの番号に応じて、決まった次のトークンの分布を返すダミーのモデル。"""

    device = torch.device("cpu")

    def to(self, device: str) -> _FakeModel:
        """GPU が無い環境として読み込まれるため、CPU を指定される。"""
        assert device == "cpu"
        return self

    def eval(self) -> _FakeModel:
        """推論用にする（ダミーでは何もしない）。"""
        return self

    def __call__(self, token_ids: torch.Tensor) -> SimpleNamespace:
        """「ね」の後は区切りが 80%、それ以外の後は 10% になる分布を返す。"""
        probabilities = (
            [0.1, 0.5, 0.1, 0.3, 0.0]
            if int(token_ids[0, -1]) == _VOCABULARY.index("ね")
            else [0.6, 0.05, 0.3, 0.05, 0.0]
        )
        logits = torch.log(torch.tensor(probabilities) + 1e-9)
        return SimpleNamespace(logits=logits.expand(1, token_ids.shape[1], _LOGIT_COUNT))


def test_break_probability_sums_break_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[str] = []
    dtypes: list[torch.dtype] = []
    transformers = importlib.import_module("transformers")

    def load_tokenizer(model_name: str) -> _FakeTokenizer:
        """読み込むモデル名を記録する。"""
        loaded.append(model_name)
        return _FakeTokenizer()

    def load_model(model_name: str, dtype: torch.dtype) -> _FakeModel:
        """読み込むモデル名を記録する。"""
        loaded.append(model_name)
        dtypes.append(dtype)
        return _FakeModel()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", load_model)

    break_probability = load_break_probability_model("test/model")

    assert loaded == ["test/model", "test/model"]
    assert dtypes == [torch.float32], "GPU が無ければ float32 で読む"
    assert break_probability("でね") == pytest.approx(0.8), "区切りのトークンの確率の合計"
    assert break_probability("ねで") == pytest.approx(0.1)


def test_load_reports_missing_libraries(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(name: str) -> None:
        """torch が入っていない環境のふりをする。"""
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(RuntimeError, match="uv sync --extra haas"):
        load_break_probability_model("test/model")
