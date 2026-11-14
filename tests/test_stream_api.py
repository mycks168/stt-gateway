from __future__ import annotations

import dataclasses
import json
import wave
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketTestSession
from starlette.websockets import WebSocketDisconnect
from stream_helpers import encode_opus_packets, sine_wave

from stt_gateway import server, stream_api
from stt_gateway.audio_decoding import AudioSamples
from stt_gateway.backchannel_policy import BackchannelFrame
from stt_gateway.config import Settings
from stt_gateway.config import settings as base_settings
from stt_gateway.maai_backchannel import MILLISECONDS_PER_FRAME, SAMPLES_PER_FRAME
from stt_gateway.pcm_audio import samples_to_pcm_s16le
from stt_gateway.transcription import Transcription

_AUTH_HEADERS = {"Authorization": "Bearer dummy-token-for-tests"}
_ACK_TEXT = "天気ね、確認するね。"


class _RecordingTranscriber:
    """受け取った WAV の長さを記録し、決まった文字起こし結果を返すダミー。"""

    def __init__(self) -> None:
        self.text = "明日の天気は？"
        self.error: Exception | None = None
        self.sample_counts: list[int] = []
        self.ollama_requests: list[dict[str, Any]] = []

    def transcribe(self, audio_path: Path, language: str) -> Transcription:
        """WAV のフレーム数を記録する。error が設定されていれば送出する。"""
        if self.error is not None:
            raise self.error
        with wave.open(str(audio_path), "rb") as wav_file:
            self.sample_counts.append(wav_file.getnframes())
        return Transcription(self.text, language, self.sample_counts[-1] / 16000)


class _ScriptedPredictor:
    """フレームごとに決まった確率を返すダミーの予測器。"""

    def __init__(self, probabilities: list[float]) -> None:
        self._probabilities = probabilities
        self._pending = 0
        self._frame_count = 0

    def reset(self) -> None:
        """フレームの位置を数え直す。"""
        self._pending = 0
        self._frame_count = 0

    def predict(self, samples: AudioSamples) -> list[BackchannelFrame]:
        """1 フレーム分の音声ごとに確率を 1 つ返す。"""
        self._pending += len(samples)
        frames = []
        while self._pending >= SAMPLES_PER_FRAME:
            self._pending -= SAMPLES_PER_FRAME
            probability = self._probabilities[self._frame_count % len(self._probabilities)]
            self._frame_count += 1
            frames.append(BackchannelFrame(self._frame_count * MILLISECONDS_PER_FRAME, probability))
        return frames


# ローカルの .env に依存しないよう、どのテストでも whisper バックエンド・認証無効・日本語にする。
_TEST_SETTING_CHANGES: dict[str, Any] = {"backend": "whisper", "api_key": "", "language": "ja"}


def _settings_with(**changes: Any) -> Settings:
    """テスト用の設定に changes を重ねた設定を作る。"""
    return dataclasses.replace(base_settings, **{**_TEST_SETTING_CHANGES, **changes})


def _patch_settings(monkeypatch: pytest.MonkeyPatch, **changes: Any) -> None:
    """server と stream_api の両方の設定を書き換える。lifespan の前に呼ぶ。"""
    patched = _settings_with(**changes)
    monkeypatch.setattr(server, "settings", patched)
    monkeypatch.setattr(stream_api, "settings", patched)


@pytest.fixture
def transcriber(monkeypatch: pytest.MonkeyPatch) -> _RecordingTranscriber:
    """設定に依存せず、whisper バックエンド・認証無効・ダミーの部品で動かす。"""
    fake = _RecordingTranscriber()
    _patch_settings(monkeypatch)
    monkeypatch.setattr(server, "load_transcriber", lambda _settings: fake)
    monkeypatch.setattr(server, "create_maai_predictor", lambda: _ScriptedPredictor([0.1, 0.9]))

    def ollama(request: httpx.Request) -> httpx.Response:
        """ダミーの Ollama。リクエストを記録し、常に同じ返事を返す。"""
        fake.ollama_requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": _ACK_TEXT}})

    monkeypatch.setattr(
        server,
        "create_ack_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(ollama)),
    )
    return fake


@pytest.fixture
def client(transcriber: _RecordingTranscriber) -> Iterator[TestClient]:
    """lifespan を実行した状態のクライアント。"""
    with TestClient(server.app) as test_client:
        yield test_client


def _pcm(duration_seconds: float) -> bytes:
    """正弦波の 16kHz PCM を作る。"""
    return samples_to_pcm_s16le(sine_wave(duration_seconds))


def _start(websocket: WebSocketTestSession, **fields: Any) -> dict[str, Any]:
    """発話を始め、started を受け取って返す。"""
    websocket.send_json({"type": "start", **fields})
    started: dict[str, Any] = websocket.receive_json()
    assert started["type"] == "started"
    return started


def test_stream_sends_backchannel_transcript_and_ack_in_order(
    client: TestClient, transcriber: _RecordingTranscriber
) -> None:
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        started = _start(websocket, encoding="pcm_s16le")
        assert started == {
            "type": "started",
            "utterance_id": 1,
            "backchannel": True,
            "backchannel_unavailable_reason": None,
        }

        websocket.send_bytes(_pcm(0.2))
        backchannel = websocket.receive_json()
        assert backchannel == {
            "type": "backchannel",
            "utterance_id": 1,
            "text": "うん",
            "offset_ms": 200,
            "probability": 0.9,
        }

        websocket.send_json({"type": "end"})
        transcript = websocket.receive_json()
        assert transcript["type"] == "transcript"
        assert transcript["utterance_id"] == 1
        assert transcript["text"] == "明日の天気は？"
        assert transcript["language"] == "ja"
        assert transcript["duration"] == pytest.approx(0.2)
        assert transcript["backend"] == "whisper"
        assert transcript["model"] == base_settings.model

        ack = websocket.receive_json()
        assert ack["type"] == "ack"
        assert ack["utterance_id"] == 1
        assert ack["text"] == _ACK_TEXT
        assert ack["source"] == "llm"
        assert isinstance(ack["latency_ms"], int)

    assert transcriber.sample_counts == [3200]


def test_stream_decodes_opus_packets_by_default(
    client: TestClient, transcriber: _RecordingTranscriber
) -> None:
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        _start(websocket)
        for packet in encode_opus_packets(0.5):
            websocket.send_bytes(packet)
        websocket.send_json({"type": "end"})
        events = [websocket.receive_json() for _ in range(2)]
        # 相槌のイベントは数が音声の長さで変わるため、transcript と ack だけを見る
        while events[-1]["type"] != "ack":
            events.append(websocket.receive_json())

    assert [event["type"] for event in events if event["type"] != "backchannel"] == [
        "transcript",
        "ack",
    ]
    assert 7000 <= transcriber.sample_counts[0] <= 9000, "0.5 秒ぶんの 16kHz の音声"


@pytest.mark.usefixtures("transcriber")
def test_stream_rejects_missing_or_wrong_token(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(monkeypatch, api_key="secret-token")
    with TestClient(server.app) as client:
        for headers in ({}, {"Authorization": "Bearer wrong"}):
            with pytest.raises(WebSocketDisconnect) as exc_info:
                client.websocket_connect("/stream", headers=headers).__enter__()
            assert exc_info.value.code == 1008
        with client.websocket_connect(
            "/stream", headers={"Authorization": "Bearer secret-token"}
        ) as websocket:
            assert _start(websocket)["utterance_id"] == 1


@pytest.mark.usefixtures("transcriber")
def test_stream_continues_without_backchannel_when_busy(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_settings(
        monkeypatch, backchannel=dataclasses.replace(base_settings.backchannel, max_sessions=1)
    )
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as first,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as second,
    ):
        assert _start(first, encoding="pcm_s16le")["backchannel"] is True
        started = _start(second, encoding="pcm_s16le")
        assert started["backchannel"] is False
        assert started["backchannel_unavailable_reason"] == "busy"

        second.send_bytes(_pcm(0.2))
        second.send_json({"type": "end"})
        assert second.receive_json()["type"] == "transcript", "相槌なしでも文字起こしは返す"
        assert second.receive_json()["type"] == "ack"


@pytest.mark.usefixtures("transcriber")
def test_stream_continues_without_backchannel_when_predictor_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail() -> _ScriptedPredictor:
        """extras が無い状態を再現する。"""
        raise RuntimeError("/stream backchannel requires the 'stream' extra")

    monkeypatch.setattr(server, "create_maai_predictor", fail)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket,
    ):
        started = _start(websocket, encoding="pcm_s16le")
        assert started["backchannel"] is False
        assert started["backchannel_unavailable_reason"] == "unavailable"


def test_stream_reports_protocol_errors_and_keeps_connection(client: TestClient) -> None:
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        websocket.send_text("not json")
        assert websocket.receive_json()["code"] == "invalid_message"
        websocket.send_json({"type": "pause"})
        assert websocket.receive_json()["code"] == "invalid_message"
        websocket.send_json({"type": "start", "encoding": "mp3"})
        assert websocket.receive_json()["code"] == "invalid_message"

        websocket.send_bytes(b"\x00\x00")
        error = websocket.receive_json()
        assert (error["code"], error["utterance_id"]) == ("invalid_state", None)
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["code"] == "invalid_state"

        websocket.send_json({"type": "start", "language": "en"})
        error = websocket.receive_json()
        assert error == {
            "type": "error",
            "utterance_id": None,
            "code": "unsupported_language",
            "detail": "unsupported language for stream: 'en' (supported: ja)",
        }

        _start(websocket, encoding="pcm_s16le")
        websocket.send_json({"type": "start"})
        error = websocket.receive_json()
        assert (error["code"], error["utterance_id"]) == ("invalid_state", 1)
        websocket.send_bytes(b"\x00")
        assert websocket.receive_json()["code"] == "invalid_audio"


def test_stream_discards_too_long_utterance(
    transcriber: _RecordingTranscriber, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_settings(monkeypatch, stream_max_utterance_seconds=0.15)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket,
    ):
        _start(websocket, encoding="pcm_s16le")
        websocket.send_bytes(_pcm(0.2))
        error = websocket.receive_json()
        assert (error["code"], error["utterance_id"]) == ("utterance_too_long", 1)
        assert "0.15 seconds" in error["detail"]
        websocket.send_bytes(_pcm(0.1))  # end までの音声は黙って捨てる
        websocket.send_json({"type": "end"})

        assert _start(websocket, encoding="pcm_s16le")["utterance_id"] == 2
    assert transcriber.sample_counts == [], "破棄した発話は文字起こししない"


def test_stream_reports_transcribe_failure_without_ack(
    client: TestClient, transcriber: _RecordingTranscriber
) -> None:
    transcriber.error = RuntimeError("CUDA out of memory")
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        _start(websocket, encoding="pcm_s16le")
        websocket.send_bytes(_pcm(0.05))
        websocket.send_json({"type": "end"})
        error = websocket.receive_json()
        assert error == {
            "type": "error",
            "utterance_id": 1,
            "code": "transcribe_failed",
            "detail": "transcribe_failed: CUDA out of memory",
        }
        # ack が送られていれば started より先に届く
        assert _start(websocket)["utterance_id"] == 2


def test_stream_skips_ack_for_empty_transcript(
    client: TestClient, transcriber: _RecordingTranscriber
) -> None:
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        _start(websocket, encoding="pcm_s16le")
        websocket.send_json({"type": "end"})
        transcript = websocket.receive_json()
        assert (transcript["text"], transcript["duration"]) == ("", 0.0)
        assert _start(websocket)["utterance_id"] == 2, "空の文字起こしには ack を送らない"
    assert transcriber.sample_counts == [], "音声の無い発話はバックエンドに渡さない"


def test_stream_requires_lifespan() -> None:
    app_without_lifespan = FastAPI()
    app_without_lifespan.include_router(stream_api.router)
    with pytest.raises(RuntimeError, match="stream_dependencies is not initialized"):
        stream_api.get_stream_dependencies(app_without_lifespan)


def test_pcm_helper_matches_frame_size() -> None:
    # 0.2 秒の PCM がちょうど 2 フレームであることを、上のテストの前提として確かめる
    assert len(_pcm(0.2)) // 2 == 2 * SAMPLES_PER_FRAME
    assert np.frombuffer(_pcm(0.1), dtype="<i2").size == SAMPLES_PER_FRAME


def test_stream_uses_client_ack_prompt(
    client: TestClient, transcriber: _RecordingTranscriber
) -> None:
    ack_options = {
        "system_prompt": "つなぎ役です。内容には触れません。",
        "examples": [{"user": "天気は？", "reply": "ちょっと見てみるね。"}],
    }
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        _start(websocket, encoding="pcm_s16le", ack=ack_options)
        websocket.send_bytes(_pcm(0.1))
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "transcript"
        assert websocket.receive_json()["text"] == _ACK_TEXT

    messages = transcriber.ollama_requests[-1]["messages"]
    assert messages == [
        {"role": "system", "content": "つなぎ役です。内容には触れません。"},
        {"role": "user", "content": "天気は？"},
        {"role": "assistant", "content": "ちょっと見てみるね。"},
        {"role": "user", "content": "明日の天気は？"},
    ]


def test_stream_skips_ack_when_disabled(client: TestClient) -> None:
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        _start(websocket, encoding="pcm_s16le", ack=False)
        websocket.send_bytes(_pcm(0.1))
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "transcript"
        assert _start(websocket)["utterance_id"] == 2, "ack: false なら ack を送らない"


def test_stream_uses_client_backchannel_rule(client: TestClient) -> None:
    # ダミーの予測器は 0.1 と 0.9 を交互に返す。しきい値 0.05 なら最初のフレームから超える
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        _start(
            websocket,
            encoding="pcm_s16le",
            backchannel={"phrases": ["はい", "ええ"], "threshold": 0.05, "min_interval_ms": 0},
        )
        websocket.send_bytes(_pcm(0.1))
        first = websocket.receive_json()
        assert (first["text"], first["offset_ms"]) == ("はい", 100)
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "transcript"
        assert websocket.receive_json()["type"] == "ack"

        # 文言が既定に戻ったら、既定の文言の先頭から使う
        _start(websocket, encoding="pcm_s16le")
        websocket.send_bytes(_pcm(0.2))
        assert websocket.receive_json()["text"] == "うん"


def test_stream_rejects_invalid_options_without_starting(client: TestClient) -> None:
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        websocket.send_json({"type": "start", "backchannel": {"threshold": 2}})
        error = websocket.receive_json()
        assert (error["code"], error["utterance_id"]) == ("invalid_message", None)
        assert _start(websocket)["utterance_id"] == 1, "不正な指定の発話は始まらない"
