"""STT ゲートウェイの HTTP API。"""

from __future__ import annotations

import logging
import tempfile
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .backends import describe_backend, load_transcriber
from .config import settings
from .formats import UnsupportedAudioFormat, resolve_suffix
from .serial_transcriber import SerialTranscriber
from .transcription import BackendName, Transcriber, UnsupportedLanguageError

logger = logging.getLogger(__name__)


def load_configured_transcriber() -> Transcriber:
    """現在の設定のバックエンドを読み込む。"""
    return load_transcriber(settings)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """ワーカーの起動時にモデルを読み込む。

    読み込みに失敗しても起動は続ける。失敗の理由を /transcribe の 500 で利用者に返せるよう、
    最初のリクエストで読み込みを再試行する。
    """
    serial_transcriber = SerialTranscriber(load_configured_transcriber)
    app.state.serial_transcriber = serial_transcriber
    try:
        await serial_transcriber.preload()
    except Exception:
        logger.exception(
            "failed to preload transcriber (backend=%s); retrying on first request",
            settings.backend,
        )
    yield


app = FastAPI(title="stt-gateway", version="0.3.0", lifespan=lifespan)
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
    if not settings.api_key:
        return
    if credentials.credentials != settings.api_key:
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
