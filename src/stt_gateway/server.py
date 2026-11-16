"""STT ゲートウェイの HTTP API。"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    HTTPException,
    Request,
    Security,
    UploadFile,
)
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .ack_generator import AckGenerator
from .auth import is_token_accepted
from .backchannel_emitters import SpeechJudging
from .backchannel_pool import BackchannelPool
from .backend_selection import (
    UnsupportedBackendError,
    partial_transcription_backend,
    select_backend,
    startup_backends,
)
from .backends import BackendDescription, describe_backend, load_transcriber
from .boundary_judge import BoundaryJudge
from .causal_lm_boundary import load_break_probability_model
from .config import settings
from .formats import UnsupportedAudioFormat, resolve_suffix
from .serial_transcriber import SerialTranscriber
from .silero_vad_predictor import create_silero_predictor
from .stream_api import router as stream_router
from .stream_session import StreamDependencies
from .transcription import BackendName, UnsupportedLanguageError
from .utterance_recording import UtteranceRecorder

logger = logging.getLogger(__name__)


def build_serial_transcriber() -> SerialTranscriber:
    """有効なエンジンごとに、モデルを読み込む関数を用意した文字起こしの窓口を作る。"""
    loaders = {
        backend: partial(load_transcriber, settings, backend)
        for backend in settings.backends.enabled
    }
    return SerialTranscriber(loaders)


def create_http_client() -> httpx.AsyncClient:
    """一時的な返事のために Ollama を呼ぶ HTTP クライアントを作る。

    待つ上限は呼び出し側（STT_ACK_TIMEOUT_MS）が決めるため、ここでは設定しない。
    """
    return httpx.AsyncClient(timeout=None)


def build_boundary_judge() -> BoundaryJudge | None:
    """言い終わりの区切りの判定を用意する。モデルが未指定、または読み込めなければ None を返す。

    読み込めなくても起動は続ける。区切りを判定しないだけで、相槌は出せるため。
    """
    model_name = settings.backchannel.boundary_model
    if not model_name:
        return None
    try:
        return BoundaryJudge(load_break_probability_model(model_name))
    except Exception:
        logger.exception("backchannel boundary judge is disabled (model=%s)", model_name)
        return None


def build_speech_judging(serial_transcriber: SerialTranscriber) -> SpeechJudging | None:
    """途中の文字起こしで相槌を判定する部品を用意する。区切りを判定できなければ None を返す。"""
    boundary_judge = build_boundary_judge()
    if boundary_judge is None:
        return None
    return SpeechJudging(
        serial_transcriber,
        settings.backchannel.judge_timeout_ms,
        boundary_judge,
        partial_transcription_backend(settings.backends),
    )


def build_stream_dependencies(
    serial_transcriber: SerialTranscriber, ack_generator: AckGenerator
) -> StreamDependencies:
    """このワーカーの /stream 用の部品を、現在の設定から組み立てる。"""
    return StreamDependencies(
        serial_transcriber=serial_transcriber,
        backchannel_pool=BackchannelPool(
            settings.backchannel.max_sessions, create_silero_predictor
        ),
        ack_generator=ack_generator,
        backends=settings.backends,
        backend_descriptions={
            backend: describe_backend(settings, backend) for backend in settings.backends.enabled
        },
        default_language=settings.language,
        max_utterance_seconds=settings.stream_max_utterance_seconds,
        backchannel=settings.backchannel,
        ack=settings.ack,
        speech_judging=build_speech_judging(serial_transcriber),
        recorder=None
        if settings.stream_recording_dir is None
        else UtteranceRecorder(settings.stream_recording_dir),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """ワーカーの起動時にモデルを読み込み、/stream 用の部品を用意する。

    一時的な返事の LLM は、起動を待たせないようバックグラウンドで GPU に読み込ませる。
    """
    serial_transcriber = build_serial_transcriber()
    app.state.serial_transcriber = serial_transcriber
    http_client = create_http_client()
    ack_generator = AckGenerator(settings.ack, http_client)
    stream_dependencies = build_stream_dependencies(serial_transcriber, ack_generator)
    app.state.stream_dependencies = stream_dependencies
    ack_generator.request_preload()
    await preload_transcribers(serial_transcriber, stream_dependencies.speech_judging is not None)
    try:
        yield
    finally:
        await ack_generator.close()
        await http_client.aclose()


async def preload_transcribers(
    serial_transcriber: SerialTranscriber, is_judging_enabled: bool
) -> None:
    """起動時に読み込むエンジンのモデルを読み込む。ほかのエンジンは初めて指定されたときに読み込む。

    使わないエンジンで GPU メモリを使わないため、すべては読み込まない。
    読み込みに失敗しても起動は続ける。失敗の理由を文字起こしの 500 で利用者に返せるよう、
    最初のリクエストで読み込みを再試行する。
    """
    for backend in startup_backends(settings.backends, is_judging_enabled):
        try:
            await serial_transcriber.preload(backend)
        except Exception:
            logger.exception(
                "failed to preload transcriber (backend=%s); retrying on first request", backend
            )


app = FastAPI(title="stt-gateway", version="0.4.0", lifespan=lifespan)
app.include_router(stream_router)
_bearer = HTTPBearer()


class BackendInfo(BaseModel):
    """エンジンとモデルの情報。"""

    backend: BackendName
    model: str
    device: str
    compute_type: str


class HealthResponse(BackendInfo):
    """GET /health の応答。

    backend などは既定のエンジン、enabled_backends はリクエストで選べるエンジン。
    """

    ok: bool
    enabled_backends: list[BackendInfo]


class TranscribeResponse(BaseModel):
    """POST /transcribe の応答。"""

    ok: bool
    text: str
    language: str
    duration: float | None
    latency_ms: int
    backend: BackendName
    model: str


def verify_token(
    credentials: Annotated[HTTPAuthorizationCredentials, Security(_bearer)],
) -> None:
    """Bearer トークンを検証する。STT_API_KEY が未設定の場合は値の検証をスキップする。"""
    if not is_token_accepted(credentials.credentials, settings.api_key):
        raise HTTPException(status_code=401, detail="invalid_token")


def get_serial_transcriber(request: Request) -> SerialTranscriber:
    """lifespan で用意した、このワーカーの文字起こし窓口を返す。"""
    serial_transcriber = getattr(request.app.state, "serial_transcriber", None)
    if not isinstance(serial_transcriber, SerialTranscriber):
        raise RuntimeError("serial_transcriber is not initialized: app started without lifespan")
    return serial_transcriber


@app.get("/health")
def health() -> HealthResponse:
    """モデルを読み込まずに、設定上のエンジンとモデルを返す。"""
    default = describe_backend(settings, settings.backends.default)
    return HealthResponse(
        ok=True,
        **_backend_info(default).model_dump(),
        enabled_backends=[
            _backend_info(describe_backend(settings, backend))
            for backend in settings.backends.enabled
        ],
    )


def _backend_info(description: BackendDescription) -> BackendInfo:
    """エンジンの情報を応答の型にする。"""
    return BackendInfo(
        backend=description.backend,
        model=description.model,
        device=description.device,
        compute_type=description.compute_type,
    )


@app.post("/transcribe")
async def transcribe(
    file: Annotated[UploadFile, File()],
    _: Annotated[None, Depends(verify_token)],
    serial_transcriber: Annotated[SerialTranscriber, Depends(get_serial_transcriber)],
    language: Annotated[str | None, Form()] = None,
    backend: Annotated[str | None, Form()] = None,
) -> TranscribeResponse:
    """アップロードされた音声を一時ファイルに書き出して、指定されたエンジンで文字起こしする。

    同じワーカーで処理中の文字起こしがあれば、それが終わるまで待つ。
    latency_ms はその待ち時間と、初めて使うエンジンのモデルの読み込みを含む。
    """
    try:
        suffix = resolve_suffix(file.filename)
        selected_backend = select_backend(backend, settings.backends)
    except (UnsupportedAudioFormat, UnsupportedBackendError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        audio_path = Path(temp_file.name)
        temp_file.write(await file.read())

    start = time.perf_counter()
    try:
        result = await serial_transcriber.transcribe(
            audio_path, language or settings.language, selected_backend
        )
    except UnsupportedLanguageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"transcribe_failed: {error}") from error
    finally:
        audio_path.unlink(missing_ok=True)

    latency_ms = int((time.perf_counter() - start) * 1000)
    description = describe_backend(settings, selected_backend)
    return TranscribeResponse(
        ok=True,
        text=result.text,
        language=result.language,
        duration=result.duration_seconds,
        latency_ms=latency_ms,
        backend=description.backend,
        model=description.model,
    )
