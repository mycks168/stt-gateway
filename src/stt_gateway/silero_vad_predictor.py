"""Silero VAD で、32ms ごとに話し手が話していない確率を予測する。

silero-vad は extras `stream` にあるため、このモジュールの import では読み込まず、
create_silero_predictor() の中で読み込む。
"""

from __future__ import annotations

from importlib import resources
from typing import Protocol

import numpy as np
import numpy.typing as npt

from .audio_decoding import AudioSamples
from .backchannel_policy import BackchannelFrame

SILERO_SAMPLE_RATE_HZ = 16000
# Silero VAD（16kHz）は 1 回に 512 サンプル（32ms）を判定し、直前の 64 サンプルを文脈として足す。
SAMPLES_PER_CHUNK = 512
CONTEXT_SAMPLES = 64
STATE_SHAPE = (2, 1, 128)
MILLISECONDS_PER_CHUNK = SAMPLES_PER_CHUNK * 1000 // SILERO_SAMPLE_RATE_HZ
SILERO_MODEL_FILE = "silero_vad.onnx"
Float32Array = npt.NDArray[np.float32]


class SileroSession(Protocol):
    """このモジュールが使う onnxruntime.InferenceSession の機能。"""

    def run(
        self, output_names: None, input_feed: dict[str, npt.NDArray[np.generic]]
    ) -> list[Float32Array]:
        """入力を推論し、出力を返す。"""
        ...


class SileroSilencePredictor:
    """Silero VAD を 32ms ごとに呼び、話していない確率をフレームとして返す。

    確率を「話していない」側にそろえるのは、相槌を出す規則がフレームの確率を
    「相槌を打ってよい度合い」として扱うため。RNN の状態と直前の音声を発話の中で引き継ぐ。
    """

    def __init__(self, session: SileroSession) -> None:
        """読み込み済みの推論セッションを受け取る。"""
        self._session = session
        self.reset()

    def reset(self) -> None:
        """前の発話の状態と、チャンクに満たない残りの音声を捨てる。"""
        self._state: Float32Array = np.zeros(STATE_SHAPE, dtype=np.float32)
        self._context: Float32Array = np.zeros(CONTEXT_SAMPLES, dtype=np.float32)
        self._pending: AudioSamples = np.zeros(0, dtype=np.float32)
        self._chunk_count = 0

    def predict(self, samples: AudioSamples) -> list[BackchannelFrame]:
        """音声を追加し、512 サンプルに達した分を判定して結果を返す。"""
        self._pending = np.concatenate([self._pending, samples])
        frames: list[BackchannelFrame] = []
        while len(self._pending) >= SAMPLES_PER_CHUNK:
            chunk = self._pending[:SAMPLES_PER_CHUNK]
            self._pending = self._pending[SAMPLES_PER_CHUNK:]
            frames.append(self._judge_chunk(chunk))
        return frames

    def _judge_chunk(self, chunk: AudioSamples) -> BackchannelFrame:
        """1 チャンクを判定し、チャンクの終わりの位置と話していない確率を返す。"""
        model_input = np.concatenate([self._context, chunk]).astype(np.float32)[np.newaxis, :]
        speech_probability, self._state = self._session.run(
            None,
            {
                "input": model_input,
                "state": self._state,
                "sr": np.array(SILERO_SAMPLE_RATE_HZ, dtype=np.int64),
            },
        )
        self._context = chunk[-CONTEXT_SAMPLES:].astype(np.float32)
        self._chunk_count += 1
        return BackchannelFrame(
            self._chunk_count * MILLISECONDS_PER_CHUNK,
            1.0 - float(speech_probability.reshape(-1)[0]),
        )


def create_silero_predictor() -> SileroSilencePredictor:
    """silero-vad に同梱の ONNX モデルを読み込む。extras `stream` が無ければ RuntimeError にする。"""
    try:
        import onnxruntime
    except ImportError as error:
        raise RuntimeError(
            "silence backchannel timing requires onnxruntime: run `uv sync --extra stream`"
        ) from error
    try:
        model_file = resources.files("silero_vad.data") / SILERO_MODEL_FILE
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "silence backchannel timing requires the 'stream' extra (silero-vad): "
            "run `uv sync --extra stream`"
        ) from error
    options = onnxruntime.SessionOptions()
    # 1 チャンクの判定は 0.1ms 程度で、スレッドを増やしても速くならず他の推論と CPU を取り合うため。
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session = onnxruntime.InferenceSession(
        model_file.read_bytes(), providers=["CPUExecutionProvider"], sess_options=options
    )
    return SileroSilencePredictor(session)
