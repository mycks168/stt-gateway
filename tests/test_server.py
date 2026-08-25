from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from stt_gateway import server


@dataclass
class _FakeSegment:
    text: str


@dataclass
class _FakeInfo:
    language: str
    duration: float


class _FakeModel:
    """model.transcribe() の呼び出しを記録するダミーモデル。"""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def transcribe(self, path: str, **kwargs):
        self.calls.append(path)
        return [_FakeSegment("こんにちは")], _FakeInfo(language="ja", duration=1.23)


@pytest.fixture(autouse=True)
def no_auth_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """ローカルの.envの値に依存せず、デフォルトでは認証無効の状態でテストする。"""
    monkeypatch.setattr(server, "settings", dataclasses.replace(server.settings, api_key=""))


@pytest.fixture
def fake_model(monkeypatch: pytest.MonkeyPatch) -> _FakeModel:
    model = _FakeModel()
    monkeypatch.setattr(server, "get_model", lambda: model)
    return model


@pytest.fixture
def client() -> TestClient:
    return TestClient(server.app)


# HTTPBearer は STT_API_KEY 未設定でもヘッダ自体の有無は検証するため、
# 認証そのものを見るテスト以外はダミーのトークンを付与しておく。
_AUTH_HEADERS = {"Authorization": "Bearer dummy-token-for-tests"}


def test_health(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json()["ok"] is True


def test_transcribe_wav_still_works(client: TestClient, fake_model: _FakeModel) -> None:
    res = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["ok"] is True
    assert body["text"] == "こんにちは"
    assert fake_model.calls, "model.transcribe が呼ばれていること"


@pytest.mark.parametrize("filename", ["voice.ogg", "voice.opus", "voice.webm"])
def test_transcribe_accepts_compressed_opus_formats(
    client: TestClient, fake_model: _FakeModel, filename: str
) -> None:
    res = client.post(
        "/transcribe",
        files={"file": (filename, b"OggS....", "application/octet-stream")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == 200
    assert res.json()["ok"] is True
    assert fake_model.calls


def test_transcribe_rejects_unsupported_format(
    client: TestClient, fake_model: _FakeModel
) -> None:
    res = client.post(
        "/transcribe",
        files={"file": ("video.mp4", b"....", "video/mp4")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == 400
    assert not fake_model.calls, "未許可フォーマットではmodel.transcribeが呼ばれないこと"


def test_transcribe_requires_bearer_token_when_api_key_set(
    client: TestClient, fake_model: _FakeModel, monkeypatch: pytest.MonkeyPatch
) -> None:
    patched_settings = dataclasses.replace(server.settings, api_key="secret-token")
    monkeypatch.setattr(server, "settings", patched_settings)

    res_no_auth = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
    )
    assert res_no_auth.status_code == 401

    res_ok = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers={"Authorization": "Bearer secret-token"},
    )
    assert res_ok.status_code == 200

    res_wrong_token = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers={"Authorization": "Bearer wrong-token"},
    )
    assert res_wrong_token.status_code == 401


def test_transcribe_returns_500_when_model_raises(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _BrokenModel:
        def transcribe(self, path: str, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(server, "get_model", lambda: _BrokenModel())

    res = client.post(
        "/transcribe",
        files={"file": ("sample.wav", b"RIFF....", "audio/wav")},
        headers=_AUTH_HEADERS,
    )
    assert res.status_code == 500
    assert "boom" in res.json()["detail"]
