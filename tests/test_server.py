from __future__ import annotations

import dataclasses
from http import HTTPStatus
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from stt_gateway import server
from stt_gateway.transcription import Transcription, UnsupportedLanguageError

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
def no_auth_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """ローカルの.envの値に依存せず、デフォルトでは認証無効の状態でテストする。"""
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, api_key=""))


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeTranscriber:
    """バックエンドをダミーに差し替える。"""
    model = _FakeTranscriber()
    monkeypatch.setattr(server, "get_transcriber", lambda: model)
    return model


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


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
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, backend="haas"))
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


def test_transcribe_returns_500_when_model_raises(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenTranscriber:
        def transcribe(self, audio_path: Path, language: str) -> Transcription:
            """常に失敗する。"""
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "get_transcriber", lambda: _BrokenTranscriber())

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


def test_transcribe_returns_400_for_unsupported_language(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _JapaneseOnlyTranscriber:
        def transcribe(self, audio_path: Path, language: str) -> Transcription:
            """日本語以外を拒否する。"""
            raise UnsupportedLanguageError("haas", language, ("ja",))

    monkeypatch.setattr(server, "get_transcriber", lambda: _JapaneseOnlyTranscriber())

    res = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        data={"language": "en"},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == HTTPStatus.BAD_REQUEST
    assert "'en'" in res.json()["detail"]


def test_transcribe_returns_500_when_backend_cannot_load(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _fail_to_load() -> None:
        """extras 未導入でモデルを読み込めない状況を再現する。"""
        raise RuntimeError("requires the 'haas' extra")

    monkeypatch.setattr(server, "get_transcriber", _fail_to_load)

    res = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
    assert "haas" in res.json()["detail"]


def test_get_transcriber_loads_once(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded: list[object] = []

    def _fake_load(settings: object) -> _FakeTranscriber:
        """読み込み回数を数える。"""
        loaded.append(settings)
        return _FakeTranscriber()

    monkeypatch.setattr(server, "load_transcriber", _fake_load)
    server.get_transcriber.cache_clear()
    try:
        first = server.get_transcriber()
        assert server.get_transcriber() is first
        assert len(loaded) == 1
    finally:
        server.get_transcriber.cache_clear()
