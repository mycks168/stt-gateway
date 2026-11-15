"""Hugging Face の言語モデルで、文の続きに区切りが来る確率を計算する。

torch と transformers は extras `haas` でだけ入るため、import は読み込み関数の中で行う。
"""

from __future__ import annotations

import importlib

from .boundary_judge import BreakProbabilityModel
from .boundary_text import is_break_token


def load_break_probability_model(model_name: str) -> BreakProbabilityModel:
    """言語モデルを読み込み、続きに区切りが来る確率を返す関数を作る。

    model_name は Hugging Face のモデル名か、モデルを置いたディレクトリ。GPU があれば GPU に載せる。
    """
    try:
        torch = importlib.import_module("torch")
        transformers = importlib.import_module("transformers")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "STT_BACKCHANNEL_BOUNDARY_MODEL requires torch and transformers: "
            "run `uv sync --extra haas`"
        ) from error

    has_gpu = torch.cuda.is_available()
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    # GPU では半分の精度で読み、メモリを抑える。CPU は bfloat16 の計算が遅いため float32 のまま。
    model = transformers.AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.bfloat16 if has_gpu else torch.float32
    )
    model = model.to("cuda" if has_gpu else "cpu").eval()
    # 区切りのトークンは語彙から 1 度だけ調べておく。判定のたびに調べると数百 ms かかるため。
    break_mask = torch.tensor(
        [is_break_token(tokenizer.decode([token_id])) for token_id in range(len(tokenizer))],
        device=model.device,
    )

    def break_probability(text: str) -> float:
        """text の次のトークンが区切りである確率を返す。"""
        with torch.inference_mode():
            token_ids = tokenizer(text, return_tensors="pt").input_ids.to(model.device)
            probabilities = model(token_ids).logits[0, -1].float().softmax(-1)
            # モデルの出力は語彙より長いことがある（使われない番号の分）。語彙の範囲だけを見る。
            return float(probabilities[: break_mask.shape[0]][break_mask].sum())

    return break_probability
