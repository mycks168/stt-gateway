"""STT ゲートウェイの HTTP API。"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import httpx
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .ack_generator import AckGenerator
from .auth import is_token_accepted
from .backchannel_pool import BackchannelPool
from .backends import describe_backend, load_transcriber
from .config import settings
from .formats import UnsupportedAudioFormat, resolve_suffix
from .maai_backchannel import create_maai_predictor
from .serial_transcriber import SerialTranscriber
from .stream_api import router as stream_router
from .stream_session import StreamDependencies
from .transcription import BackendName, Transcriber, UnsupportedLanguageError

logger = logging.getLogger(__name__)


def load_configured_transcriber() -> Transcriber:
    """現在の設定のバックエンドを読み込む。"""
    return load_transcriber(settings)


def create_ack_http_client() -> httpx.AsyncClient:
    """一時的な返事を作るために Ollama を呼ぶ HTTP クライアントを作る。

    タイムアウトは AckGenerator が STT_ACK_TIMEOUT_MS で全体にかけるため、ここでは設定しない。
    """
    return httpx.AsyncClient(timeout=None)


def build_stream_dependencies(
    serial_transcriber: SerialTranscriber, ack_generator: AckGenerator
) -> StreamDependencies:
    """このワーカーの /stream 用の部品を、現在の設定から組み立てる。"""
    return StreamDependencies(
        serial_transcriber=serial_transcriber,
        backchannel_pool=BackchannelPool(settings.backchannel.max_sessions, create_maai_predictor),
        ack_generator=ack_generator,
        backend=describe_backend(settings),
        default_language=settings.language,
        max_utterance_seconds=settings.stream_max_utterance_seconds,
        backchannel=settings.backchannel,
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """ワーカーの起動時にモデルを読み込み、/stream 用の部品を用意する。

    読み込みに失敗しても起動は続ける。失敗の理由を /transcribe の 500 で利用者に返せるよう、
    最初のリクエストで読み込みを再試行する。
    一時的な返事の LLM は、起動を待たせないようバックグラウンドで GPU に読み込ませる。
    """
    serial_transcriber = SerialTranscriber(load_configured_transcriber)
    app.state.serial_transcriber = serial_transcriber
    ack_http_client = create_ack_http_client()
    ack_generator = AckGenerator(settings.ack, ack_http_client)
    app.state.stream_dependencies = build_stream_dependencies(serial_transcriber, ack_generator)
    ack_warm_up = asyncio.create_task(ack_generator.warm_up())
    try:
        await serial_transcriber.preload()
    except Exception:
        logger.exception(
            "failed to preload transcriber (backend=%s); retrying on first request",
            settings.backend,
        )
    try:
        yield
    finally:
        ack_warm_up.cancel()
        await ack_http_client.aclose()


app = FastAPI(title="stt-gateway", version="0.4.0", lifespan=lifespan)
app.include_router(stream_router)
_bearer = HTTPBearer()


class HealthResponse(BaseModel):
    """GET /health の応答。"""

    ok: bool
    backend: BackendName
    model: str
    device: str
    compute_type: str


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
    """モデルを読み込まずに、設定上のバックエンドとモデルを返す。"""
    description = describe_backend(settings)
    return HealthResponse(
        ok=True,
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
) -> TranscribeResponse:
    """アップロードされた音声を一時ファイルに書き出して文字起こしする。

    同じワーカーで処理中の文字起こしがあれば、それが終わるまで待つ。
    latency_ms はその待ち時間を含む。
    """
    try:
        suffix = resolve_suffix(file.filename)
    except UnsupportedAudioFormat as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        audio_path = Path(temp_file.name)
        temp_file.write(await file.read())

    start = time.perf_counter()
    try:
        result = await serial_transcriber.transcribe(audio_path, language or settings.language)
    except UnsupportedLanguageError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"transcribe_failed: {error}") from error
    finally:
        audio_path.unlink(missing_ok=True)

    latency_ms = int((time.perf_counter() - start) * 1000)
    description = describe_backend(settings)
    return TranscribeResponse(
        ok=True,
        text=result.text,
        language=result.language,
        duration=result.duration_seconds,
        latency_ms=latency_ms,
        backend=description.backend,
        model=description.model,
    )
