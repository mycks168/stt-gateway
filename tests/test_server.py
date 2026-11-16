from __future__ import annotations

import dataclasses
from collections.abc import Iterator
from http import HTTPStatus
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient

from stt_gateway import server
from stt_gateway.backend_selection import BackendSelection
from stt_gateway.config import Settings
from stt_gateway.transcription import (
    BackendName,
    Transcriber,
    Transcription,
    UnsupportedLanguageError,
)

_FAKE_DURATION_SECONDS = 1.23


class _FakeTranscriber:
    """transcribe() の呼び出しを記録するダミーのバックエンド。"""

    def __init__(self) -> None:
        self.calls: list[tuple[Path, str]] = []

    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """呼び出しを記録して固定の結果を返す。"""
        self.calls.append((audio_path, language))
        return Transcription(
            text="こんにちは", language=language, duration_seconds=_FAKE_DURATION_SECONDS
        )


@pytest.fixture(autouse=True)
def default_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    """ローカルの.envの値に依存せず、whisper バックエンド・認証無効の状態でテストする。"""
    monkeypatch.setattr(
        server,
        "settings",
        dataclasses.replace(
            server.settings, backends=BackendSelection("whisper", ("whisper",)), api_key=""
        ),
    )


def _use_backend(monkeypatch: pytest.MonkeyPatch, transcriber: Transcriber) -> None:
    """設定に関係なく、指定したダミーをバックエンドとして読み込ませる。"""
    monkeypatch.setattr(server, "load_transcriber", lambda _settings, _backend: transcriber)


@pytest.fixture(autouse=True)
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeTranscriber:
    """起動時の読み込みで実モデルを読まないよう、既定でバックエンドをダミーに差し替える。"""
    model = _FakeTranscriber()
    _use_backend(monkeypatch, model)
    return model


@pytest.fixture
def client() -> Iterator[TestClient]:
    """lifespan（起動時のモデル読み込み）を実行した状態のクライアント。"""
    with TestClient(server.app) as test_client:
        yield test_client


# HTTPBearer は STT_API_KEY 未設定でもヘッダ自体の有無は検証するため、
# 認証そのものを見るテスト以外はダミーのトークンを付与しておく。
_AUTH_HEADERS = {"Authorization": "Bearer dummy-token-for-tests"}


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == HTTPStatus.OK
    body = res.json()
    assert body["ok"] is True
    assert body["backend"] == "whisper"
    assert body["model"] == server.settings.model


def test_health_reports_haas_backend(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        server,
        "settings",
        dataclasses.replace(server.settings, backends=BackendSelection("haas", ("haas",))),
    )
    body = client.get("/health").json()
    assert body["backend"] == "haas"
    assert body["model"] == server.settings.haas_model
    assert body["device"] == "cuda"
    assert body["compute_type"] == "bfloat16"


def test_transcribe_wav_still_works(client: TestClient, fake_model: _FakeTranscriber) -> None:
    res = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == HTTPStatus.OK
    body = res.json()
    assert body["ok"] is True
    assert body["text"] == "こんにちは"
    assert fake_model.calls, "バックエンドの transcribe が呼ばれていること"


@pytest.mark.parametrize("filename", ["voice.ogg", "voice.opus", "voice.webm"])
def test_transcribe_accepts_compressed_opus_formats(
    client: TestClient, fake_model: _FakeTranscriber, filename: str
) -> None:
    res = client.post(
        "/transcribe",
        files={"file": (filename, b"OggS....", "application/octet-stream")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == HTTPStatus.OK
    assert res.json()["ok"] is True
    assert fake_model.calls


def test_transcribe_rejects_unsupported_format(
    client: TestClient, fake_model: _FakeTranscriber
) -> None:
    res = client.post(
        "/transcribe",
        files={"file": ("video.mp4", b"....", "video/mp4")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == HTTPStatus.BAD_REQUEST
    assert not fake_model.calls, "未許可フォーマットではバックエンドが呼ばれないこと"


def test_transcribe_requires_bearer_token_when_api_key_set(
    client: TestClient, fake_model: _FakeTranscriber, monkeypatch: pytest.MonkeyPatch
) -> None:
    patched_settings = dataclasses.replace(server.settings, api_key="secret-token")
    monkeypatch.setattr(server, "settings", patched_settings)

    res_no_auth = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
    )
    assert res_no_auth.status_code == HTTPStatus.UNAUTHORIZED

    res_ok = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert res_ok.status_code == HTTPStatus.OK

    res_wrong_token = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert res_wrong_token.status_code == HTTPStatus.UNAUTHORIZED


class _BrokenTranscriber:
    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """常に失敗する。"""
        raise RuntimeError("boom")


def test_transcribe_returns_500_when_model_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _use_backend(monkeypatch, _BrokenTranscriber())
    with TestClient(server.app) as client:
        res = client.post(
            "/transcribe",
            files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
            headers=_AUTH_HEADERS,
        )
    assert res.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "boom" in res.json()["detail"]


def test_transcribe_passes_language_and_reports_backend(
    client: TestClient, fake_model: _FakeTranscriber
) -> None:
    res = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        data={"language": "en"},
        headers=_AUTH_HEADERS,
    )
    body = res.json()
    assert body["language"] == "en"
    assert body["duration"] == _FAKE_DURATION_SECONDS
    assert body["backend"] == "whisper"
    assert fake_model.calls[0][1] == "en"


def test_transcribe_removes_temporary_file(
    client: TestClient, fake_model: _FakeTranscriber
) -> None:
    client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers=_AUTH_HEADERS,
    )
    audio_path, _ = fake_model.calls[0]
    assert audio_path.suffix == ".wav"
    assert not audio_path.exists()


def test_transcribe_returns_400_for_unsupported_language(monkeypatch: pytest.MonkeyPatch) -> None:
    class _JapaneseOnlyTranscriber:
        def transcribe(self, audio_path: Path, language: str) -> Transcription:
            """日本語以外を拒否する。"""
            raise UnsupportedLanguageError("haas", language, ("ja",))

    _use_backend(monkeypatch, _JapaneseOnlyTranscriber())
    with TestClient(server.app) as client:
        res = client.post(
            "/transcribe",
            files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
            data={"language": "en"},
            headers=_AUTH_HEADERS,
        )
    assert res.status_code == HTTPStatus.BAD_REQUEST
    assert "'en'" in res.json()["detail"]


def test_model_is_loaded_at_startup(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[BackendName] = []

    def _count_loads(_settings: Settings, backend: BackendName) -> _FakeTranscriber:
        """読み込み回数を数える。"""
        loaded.append(backend)
        return _FakeTranscriber()

    monkeypatch.setattr(server, "load_transcriber", _count_loads)
    with TestClient(server.app) as client:
        assert len(loaded) == 1, "リクエスト前の起動時に読み込まれること"
        client.post(
            "/transcribe",
            files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
            headers=_AUTH_HEADERS,
        )
    assert len(loaded) == 1, "リクエストのたびに読み込み直さないこと"


def test_transcribe_returns_500_when_backend_cannot_load(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def _fail_to_load(_settings: Settings, _backend: BackendName) -> Transcriber:
        """extras 未導入でモデルを読み込めない状況を再現する。"""
        raise RuntimeError("requires the 'haas' extra")

    monkeypatch.setattr(server, "load_transcriber", _fail_to_load)
    with TestClient(server.app) as client:
        assert "failed to preload transcriber" in caplog.text, "起動は続け、失敗をログに残すこと"
        res = client.post(
            "/transcribe",
            files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
            headers=_AUTH_HEADERS,
        )
    assert res.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "haas" in res.json()["detail"]


def test_transcribe_fails_without_lifespan(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(server.app.state, "serial_transcriber", raising=False)
    res = TestClient(server.app, raise_server_exceptions=False).post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == HTTPStatus.INTERNAL_SERVER_ERROR


def _enable_both_backends(monkeypatch: pytest.MonkeyPatch) -> list[BackendName]:
    """既定を haas にして whisper も選べるようにし、読み込んだエンジンの記録を返す。

    lifespan の前に呼ぶ。
    """
    loaded: list[BackendName] = []

    def _record_loads(_settings: Settings, backend: BackendName) -> _FakeTranscriber:
        """読み込んだエンジンを記録する。"""
        loaded.append(backend)
        return _FakeTranscriber()

    monkeypatch.setattr(
        server,
        "settings",
        dataclasses.replace(
            server.settings, backends=BackendSelection("haas", ("haas", "whisper"))
        ),
    )
    monkeypatch.setattr(server, "load_transcriber", _record_loads)
    return loaded


def _post_sample(client: TestClient, **form: str) -> httpx.Response:
    """WAV を 1 つ送って文字起こしさせる。"""
    return client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        data=form,
        headers=_AUTH_HEADERS,
    )


def test_requested_backend_is_loaded_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _enable_both_backends(monkeypatch)
    with TestClient(server.app) as client:
        assert loaded == ["haas"], "起動時は既定のエンジンだけを読み込むこと"
        first = _post_sample(client, backend="whisper")
        second = _post_sample(client, backend="whisper")
        default = _post_sample(client)
    assert loaded == ["haas", "whisper"], "指定されたエンジンは 1 回だけ読み込むこと"
    assert first.json()["backend"] == "whisper"
    assert first.json()["model"] == server.settings.model
    assert second.status_code == HTTPStatus.OK
    assert default.json()["backend"] == "haas"
    assert default.json()["model"] == server.settings.haas_model


@pytest.mark.parametrize("requested", ["haas", "Whisper", "kotoba"])
def test_transcribe_rejects_backend_that_is_not_enabled(
    client: TestClient, fake_model: _FakeTranscriber, requested: str
) -> None:
    res = _post_sample(client, backend=requested)
    assert res.status_code == HTTPStatus.BAD_REQUEST
    assert res.json()["detail"] == f"unsupported backend: '{requested}' (enabled: whisper)"
    assert not fake_model.calls


def test_health_lists_enabled_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    _enable_both_backends(monkeypatch)
    with TestClient(server.app) as client:
        body = client.get("/health").json()
    assert body["backend"] == "haas"
    assert [info["backend"] for info in body["enabled_backends"]] == ["haas", "whisper"]
    assert body["enabled_backends"][1] == {
        "backend": "whisper",
        "model": server.settings.model,
        "device": server.settings.device,
        "compute_type": server.settings.compute_type,
    }
