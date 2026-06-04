from __future__ import annotations

import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Security, UploadFile
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from faster_whisper import WhisperModel

from .config import settings

app = FastAPI(title="stt-gateway", version="0.1.0")
_model: WhisperModel | None = None
_bearer = HTTPBearer()


def verify_token(credentials: HTTPAuthorizationCredentials = Security(_bearer)) -> None:
    """Bearer トークンを検証する。STT_API_KEY が未設定の場合は認証をスキップ。"""
    if not settings.api_key:
        return
    if credentials.credentials != settings.api_key:
        raise HTTPException(status_code=401, detail="invalid_token")


def get_model() -> WhisperModel:
    global _model
    if _model is None:
        _model = WhisperModel(
            settings.model,
            device=settings.device,
            compute_type=settings.compute_type,
        )
    return _model


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model": settings.model,
        "device": settings.device,
        "compute_type": settings.compute_type,
    }


@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
    _: None = Depends(verify_token),
) -> dict:
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tf:
        tmp = tf.name
        tf.write(await file.read())

    model = get_model()
    start = time.perf_counter()
    try:
        segments, info = model.transcribe(
            tmp,
            language=language or settings.language,
            vad_filter=True,
            vad_parameters=dict(threshold=0.3),
        )
        text = "".join(seg.text for seg in segments).strip()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"transcribe_failed: {e}")
    finally:
        Path(tmp).unlink(missing_ok=True)

    ms = int((time.perf_counter() - start) * 1000)
    return {
        "ok": True,
        "text": text,
        "language": getattr(info, "language", language or settings.language),
        "duration": getattr(info, "duration", None),
        "latency_ms": ms,
        "model": settings.model,
    }
