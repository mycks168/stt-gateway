"""MaAI（maai-kyoto/vap_bc_jp）で相槌を打つべき確率を予測する。

maai は extras `stream` にあるため、このモジュールの import では読み込まず、
create_maai_predictor() の中で読み込む。
"""

from __future__ import annotations

import queue
from typing import Protocol

import numpy as np

from .audio_decoding import AudioSamples
from .backchannel_policy import BackchannelFrame

MAAI_MODE = "bc"
MAAI_LANGUAGE = "jp"
FRAME_RATE_HZ = 10
CONTEXT_SECONDS = 20
MAAI_SAMPLE_RATE_HZ = 16000
SAMPLES_PER_FRAME = MAAI_SAMPLE_RATE_HZ // FRAME_RATE_HZ
MILLISECONDS_PER_FRAME = 1000 // FRAME_RATE_HZ
MAAI_DEVICE = "cpu"


class MaaiModel(Protocol):
    """このモジュールが使う maai.Maai の機能。"""

    result_dict_queue: queue.Queue[dict[str, object]]

    def reset_runtime_state(self) -> None:
        """音声バッファと推論のキャッシュを捨てる。"""
        ...

    def process(self, x1: AudioSamples, x2: AudioSamples) -> None:
        """2 チャネル分の音声を 1 フレーム分受け取り、確定した結果を result_dict_queue に入れる。"""
        ...


class MaaiBackchannelPredictor:
    """maai.Maai をフレーム単位で直接呼び出す予測器。

    Maai.start() は内部でスレッドを立ててキューから音声を読み続けるが、発話ごとに貸し借りする
    使い方では止める・再開するの管理が増える。process() を呼び出し側のスレッドで直接呼ぶ。
    process() は 1 回に 1 フレーム分しか推論しないため、音声をフレームの長さに切って渡す。
    """

    def __init__(self, model: MaaiModel) -> None:
        """読み込み済みの Maai を受け取る。"""
        self._model = model
        self._pending = np.zeros(0, dtype=np.float32)
        self._frame_count = 0

    def reset(self) -> None:
        """前の発話の文脈と、フレームに満たない残りの音声を捨てる。"""
        self._model.reset_runtime_state()
        _drain(self._model.result_dict_queue)
        self._pending = np.zeros(0, dtype=np.float32)
        self._frame_count = 0

    def predict(self, samples: AudioSamples) -> list[BackchannelFrame]:
        """音声を追加し、フレームの長さに達した分を推論して結果を返す。"""
        self._pending = np.concatenate([self._pending, samples])
        frames: list[BackchannelFrame] = []
        while len(self._pending) >= SAMPLES_PER_FRAME:
            user_frame = self._pending[:SAMPLES_PER_FRAME]
            self._pending = self._pending[SAMPLES_PER_FRAME:]
            # ゲートウェイはシステム側（クライアントの読み上げ）の音声を持たないため無音を渡す。
            self._model.process(user_frame, np.zeros(SAMPLES_PER_FRAME, dtype=np.float32))
            frames.extend(self._collect_results())
        return frames

    def _collect_results(self) -> list[BackchannelFrame]:
        """process() が確定させた結果を取り出し、フレームの位置を付ける。"""
        frames: list[BackchannelFrame] = []
        for result in _drain(self._model.result_dict_queue):
            self._frame_count += 1
            probability = float(np.asarray(result["p_bc"]).reshape(-1)[0])
            frames.append(BackchannelFrame(self._frame_count * MILLISECONDS_PER_FRAME, probability))
        return frames


def _drain(results: queue.Queue[dict[str, object]]) -> list[dict[str, object]]:
    """キューに溜まっている結果をすべて取り出す。取り出すのは呼び出し元のスレッドだけなので empty() で判定できる。"""
    drained: list[dict[str, object]] = []
    while not results.empty():
        drained.append(results.get_nowait())
    return drained


def create_maai_predictor() -> MaaiBackchannelPredictor:
    """MaAI の相槌モデルを読み込む。extras `stream` が無ければ RuntimeError にする。

    初回はモデルの重みと CPC エンコーダをダウンロードするため時間がかかる。
    """
    try:
        from maai import Maai, MaaiInput
    except ImportError as error:
        raise RuntimeError(
            "/stream backchannel requires the 'stream' extra: run `uv sync --extra stream`"
        ) from error
    model = Maai(
        mode=MAAI_MODE,
        lang=MAAI_LANGUAGE,
        frame_rate=FRAME_RATE_HZ,
        context_len_sec=CONTEXT_SECONDS,
        audio_ch1=MaaiInput.Chunk(),
        audio_ch2=MaaiInput.Chunk(),
        device=MAAI_DEVICE,
    )
    return MaaiBackchannelPredictor(model)
