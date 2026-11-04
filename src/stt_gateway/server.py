"""STT ゲートウェイの HTTP API。"""

from __future__ import annotations

import tempfile
import time
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from .backends import describe_backend, load_transcriber
from .config import settings
from .formats import UnsupportedAudioFormat, resolve_suffix
from .transcription import BackendName, Transcriber, UnsupportedLanguageError

app = FastAPI(title="stt-gateway", version="0.2.0")
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


@lru_cache(maxsize=1)
def get_transcriber() -> Transcriber:
    """最初のリクエストでモデルを読み込み、以降は使い回す。"""
    return load_transcriber(settings)


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
    language: Annotated[str | None, Form()] = None,
) -> TranscribeResponse:
    """アップロードされた音声を一時ファイルに書き出して文字起こしする。"""
    try:
        suffix = resolve_suffix(file.filename)
    except UnsupportedAudioFormat as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        audio_path = Path(temp_file.name)
        temp_file.write(await file.read())

    start = time.perf_counter()
    try:
        # モデルの読み込み失敗（extras 未導入など）も 500 の理由として返すため try の中で取得する
        result = get_transcriber().transcribe(audio_path, language or settings.language)
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
