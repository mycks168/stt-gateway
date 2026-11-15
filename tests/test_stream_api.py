from __future__ import annotations

import dataclasses
import json
import stat
import time
import wave
from collections.abc import Callable, Iterator
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
from stt_gateway.config import BackchannelSettings, Settings
from stt_gateway.config import settings as base_settings
from stt_gateway.pcm_audio import samples_to_pcm_s16le
from stt_gateway.transcription import Transcription

_AUTH_HEADERS = {"Authorization": "Bearer dummy-token-for-tests"}
_ACK_TEXT = "天気ね、確認するね。"
# ダミーの予測器は 100ms（1600 サンプル）ごとに 1 フレームを返す。
_SAMPLES_PER_FRAME = 1600
_MILLISECONDS_PER_FRAME = 100


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
        while self._pending >= _SAMPLES_PER_FRAME:
            self._pending -= _SAMPLES_PER_FRAME
            probability = self._probabilities[self._frame_count % len(self._probabilities)]
            self._frame_count += 1
            frames.append(
                BackchannelFrame(self._frame_count * _MILLISECONDS_PER_FRAME, probability)
            )
        return frames


# 100ms 話して 100ms 黙ったところで相槌を出す。区切りの判定は、使うテストだけで有効にする。
_TEST_BACKCHANNEL = BackchannelSettings(
    max_sessions=4,
    silence_ms=100,
    min_new_speech_ms=0,
    min_interval_ms=1500,
    interval_per_minute_ms=3000,
    interval_factor=1.1,
    max_interval_ms=8000,
    boundary_model="",
    boundary_threshold=0.35,
    judge_timeout_ms=1000,
    phrases=("うん", "うんうん", "はい", "なるほど"),
)
# ローカルの .env に依存しないよう、どのテストでも whisper バックエンド・認証無効・日本語にする。
_TEST_SETTING_CHANGES: dict[str, Any] = {
    "backend": "whisper",
    "api_key": "",
    "language": "ja",
    "backchannel": _TEST_BACKCHANNEL,
    "stream_recording_dir": None,
}


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
    # 本物の Silero VAD を読み込まず、100ms 話して 100ms 黙る音声として扱う。
    monkeypatch.setattr(server, "create_silero_predictor", lambda: _ScriptedPredictor([0.1, 0.9]))

    async def fake_ollama(request: httpx.Request) -> httpx.Response:
        """ダミーの Ollama。リクエストを記録し、決まった答えを返す。"""
        fake.ollama_requests.append(json.loads(request.content))
        return httpx.Response(200, json={"message": {"content": _ACK_TEXT}})

    monkeypatch.setattr(
        server,
        "create_http_client",
        lambda: httpx.AsyncClient(transport=httpx.MockTransport(fake_ollama)),
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


@pytest.mark.usefixtures("transcriber")
def test_stream_records_audio_and_probabilities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    recording_dir = tmp_path / "recordings"
    _patch_settings(monkeypatch, stream_recording_dir=recording_dir)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket,
    ):
        _start(websocket, encoding="pcm_s16le")
        websocket.send_bytes(_pcm(0.2))
        assert websocket.receive_json()["type"] == "backchannel"
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "transcript"
        assert websocket.receive_json()["type"] == "ack"
        _wait_until(lambda: len(list(recording_dir.glob("*.json"))) == 1)

    [json_path] = recording_dir.glob("*-u1.json")
    document = json.loads(json_path.read_text(encoding="utf-8"))
    assert document["duration_ms"] == 200
    assert (document["silence_ms"], document["min_new_speech_ms"]) == (100, 0)
    assert document["frames"] == [
        {"offset_ms": 100, "probability": 0.1},
        {"offset_ms": 200, "probability": 0.9},
    ]
    assert document["fires"] == [{"offset_ms": 200, "probability": 0.9, "has_prepared": False}]
    with wave.open(str(json_path.with_suffix(".wav"))) as wav_file:
        assert (wav_file.getframerate(), wav_file.getnframes()) == (16000, 3200)
    assert stat.S_IMODE(json_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(recording_dir.stat().st_mode) == 0o700


@pytest.mark.usefixtures("transcriber")
def test_stream_waits_for_client_silence_ms(monkeypatch: pytest.MonkeyPatch) -> None:
    # 100ms 話してから、無音（話していない確率 0.9）が続く。
    monkeypatch.setattr(
        server, "create_silero_predictor", lambda: _ScriptedPredictor([0.1, 0.9, 0.9, 0.9])
    )
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket,
    ):
        _start(websocket, encoding="pcm_s16le", backchannel={"silence_ms": 300})
        websocket.send_bytes(_pcm(0.4))
        backchannel = websocket.receive_json()
        assert (backchannel["offset_ms"], backchannel["probability"]) == (400, 0.9)


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

    monkeypatch.setattr(server, "create_silero_predictor", fail)
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
    assert len(_pcm(0.2)) // 2 == 2 * _SAMPLES_PER_FRAME
    assert np.frombuffer(_pcm(0.1), dtype="<i2").size == _SAMPLES_PER_FRAME


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
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        _start(
            websocket,
            encoding="pcm_s16le",
            backchannel={"phrases": ["はい", "ええ"], "min_interval_ms": 0},
        )
        websocket.send_bytes(_pcm(0.2))
        first = websocket.receive_json()
        assert (first["text"], first["offset_ms"]) == ("はい", 200)
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "transcript"
        assert websocket.receive_json()["type"] == "ack"

        # 文言が既定に戻ったら、既定の文言の先頭から使う
        _start(websocket, encoding="pcm_s16le")
        websocket.send_bytes(_pcm(0.2))
        assert websocket.receive_json()["text"] == "うん"


@pytest.mark.parametrize(
    "backchannel",
    [{"silence_ms": 10}, {"timing": "silence"}, {"phrases_by_kind": {"news": "へえ"}}],
    ids=["out-of-range", "unknown-timing-field", "unknown-phrases-field"],
)
def test_stream_rejects_invalid_options_without_starting(
    client: TestClient, backchannel: dict[str, Any]
) -> None:
    with client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket:
        websocket.send_json({"type": "start", "backchannel": backchannel})
        error = websocket.receive_json()
        assert (error["code"], error["utterance_id"]) == ("invalid_message", None)
        assert _start(websocket)["utterance_id"] == 1, "不正な指定の発話は始まらない"


def _wait_until(condition: Callable[[], bool], timeout_seconds: float = 5.0) -> None:
    """別スレッドで動くサーバ側の処理が condition を満たすまで待つ。"""
    deadline = time.monotonic() + timeout_seconds
    while not condition():
        assert time.monotonic() < deadline, "条件を満たさないまま時間切れになった"
        time.sleep(0.01)


def _enable_boundary_judge(
    monkeypatch: pytest.MonkeyPatch,
    boundary_model: Callable[[str], float] | None,
    recording_dir: Path | None = None,
    **backchannel_changes: Any,
) -> list[str]:
    """区切りの判定を有効にする。lifespan の前に呼ぶ。判定した文の記録を返す。

    boundary_model が None なら、モデルの読み込みに失敗させる。recording_dir を渡せば録音する。
    """
    judged_texts: list[str] = []

    def judge(text: str) -> float:
        """判定した文を記録してから、ダミーのモデルで判定する。"""
        judged_texts.append(text)
        assert boundary_model is not None
        return boundary_model(text)

    def load(model_name: str) -> Callable[[str], float]:
        """ダミーのモデルを読み込む。"""
        assert model_name == "test/boundary-model"
        if boundary_model is None:
            raise RuntimeError("model files are missing")
        return judge

    _patch_settings(
        monkeypatch,
        backchannel=dataclasses.replace(
            _TEST_BACKCHANNEL, boundary_model="test/boundary-model", **backchannel_changes
        ),
        stream_recording_dir=recording_dir,
    )
    monkeypatch.setattr(server, "load_break_probability_model", load)
    return judged_texts


def _boundary_stream_events(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    recording_dir: Path,
    boundary_probability: float | None,
) -> tuple[list[str], dict[str, Any]]:
    """区切りの判定を有効にして 1 発話を流し、届いたイベントの種類と録音の JSON を返す。

    boundary_probability が None なら、モデルの読み込みに失敗させる。
    """
    caplog.set_level("INFO", logger="stt_gateway")
    _enable_boundary_judge(
        monkeypatch,
        None if boundary_probability is None else lambda _text: boundary_probability,
        recording_dir,
    )
    event_types: list[str] = []
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket,
    ):
        _start(websocket, encoding="pcm_s16le", backchannel={"boundary_threshold": 0.35})
        websocket.send_bytes(_pcm(0.2))
        if boundary_probability is None:
            # 判定を使えないときは、文言を順に回すだけなので判定のログは出ない。
            event_types.append(websocket.receive_json()["type"])
        else:
            _wait_until(lambda: "backchannel decision" in caplog.text)
        websocket.send_json({"type": "end"})
        while not event_types or event_types[-1] != "ack":
            event_types.append(websocket.receive_json()["type"])
        _wait_until(lambda: len(list(recording_dir.glob("*.json"))) == 1)
    [json_path] = recording_dir.glob("*-u1.json")
    document: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
    return event_types, document


@pytest.mark.usefixtures("transcriber")
def test_stream_stays_silent_when_speaker_is_mid_phrase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    event_types, document = _boundary_stream_events(monkeypatch, caplog, tmp_path, 0.1)
    assert event_types == ["transcript", "ack"], "言いかけの途中では相槌を出さない"
    assert document["boundary_threshold"] == 0.35
    assert document["decisions"] == [
        {
            "offset_ms": 200,
            "reason": "not_boundary",
            "text": "明日の天気は？",
            "boundary_probability": 0.1,
        }
    ]


@pytest.mark.usefixtures("transcriber")
def test_stream_sends_backchannel_at_phrase_boundary(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    event_types, document = _boundary_stream_events(monkeypatch, caplog, tmp_path, 0.9)
    assert event_types == ["backchannel", "transcript", "ack"]
    assert [decision["reason"] for decision in document["decisions"]] == ["sent"]


@pytest.mark.usefixtures("transcriber")
def test_stream_sends_backchannel_when_boundary_model_cannot_be_loaded(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    event_types, document = _boundary_stream_events(monkeypatch, caplog, tmp_path, None)
    assert event_types == ["backchannel", "transcript", "ack"], "区切りを判定しないだけで相槌は出す"
    assert document["decisions"] == []
    assert "backchannel boundary judge is disabled (model=test/boundary-model)" in caplog.text


@pytest.mark.usefixtures("transcriber")
def test_stream_uses_judgment_started_at_silence_start(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="stt_gateway.backchannel_emitters")
    # 200ms の無音を待つ。黙り始めた 200ms で判定を始め、300ms で相槌を出す。
    judged_texts = _enable_boundary_judge(monkeypatch, lambda _text: 0.9, silence_ms=200)
    monkeypatch.setattr(
        server, "create_silero_predictor", lambda: _ScriptedPredictor([0.1, 0.9, 0.9])
    )
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket,
    ):
        _start(websocket, encoding="pcm_s16le")
        websocket.send_bytes(_pcm(0.3))
        assert websocket.receive_json()["offset_ms"] == 300
    assert judged_texts == ["明日の天気は"], "準備で始めた判定を使い、判定し直さない"
    assert "prepared=True" in caplog.text


@pytest.mark.usefixtures("transcriber")
def test_stream_skips_backchannel_when_judgment_times_out(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO", logger="stt_gateway")

    def slow_model(_text: str) -> float:
        """待つ上限より遅れて区切りと判定する。"""
        time.sleep(0.3)
        return 0.9

    _enable_boundary_judge(monkeypatch, slow_model, judge_timeout_ms=50)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket,
    ):
        _start(websocket, encoding="pcm_s16le")
        websocket.send_bytes(_pcm(0.2))
        _wait_until(lambda: "reason=timeout" in caplog.text)
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "transcript", "相槌は届かない"
        assert websocket.receive_json()["type"] == "ack"


@pytest.mark.usefixtures("transcriber")
def test_stream_does_not_judge_when_client_disables_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    judged_texts = _enable_boundary_judge(monkeypatch, lambda _text: 0.1)
    with (
        TestClient(server.app) as client,
        client.websocket_connect("/stream", headers=_AUTH_HEADERS) as websocket,
    ):
        _start(websocket, encoding="pcm_s16le", backchannel={"boundary_threshold": 0})
        websocket.send_bytes(_pcm(0.2))
        assert websocket.receive_json()["type"] == "backchannel"
    assert judged_texts == [], "0 を指定した発話では区切りを判定しない"
