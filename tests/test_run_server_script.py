from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_server.sh"
_SCRIPT_TIMEOUT_SECONDS = 10
# 起動スクリプトが使うコマンド（bash・grep など）だけを見せ、本物の uv・uvicorn は見せない。
_SYSTEM_PATH = "/usr/bin:/bin"


def _run_script(tmp_path: Path, environment: dict[str, str]) -> str:
    """偽の uv を置いて起動スクリプトを実行し、uvicorn を起動したときの uv の引数を返す。"""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake_uv = bin_dir / "uv"
    calls_path = tmp_path / "uv_calls.txt"
    fake_uv.write_text(f'#!/bin/bash\necho "$*" >> {calls_path}\n', encoding="utf-8")
    fake_uv.chmod(fake_uv.stat().st_mode | stat.S_IXUSR)
    completed = subprocess.run(
        ["bash", str(_SCRIPT_PATH)],
        cwd=tmp_path,
        env={"PATH": f"{bin_dir}:{_SYSTEM_PATH}", "HOME": os.environ["HOME"], **environment},
        capture_output=True,
        text=True,
        timeout=_SCRIPT_TIMEOUT_SECONDS,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    [uvicorn_call] = [
        line for line in calls_path.read_text(encoding="utf-8").splitlines() if "uvicorn" in line
    ]
    return uvicorn_call


def test_script_starts_without_enabled_backends_variable(tmp_path: Path) -> None:
    call = _run_script(tmp_path, {})
    assert call.startswith("run uvicorn stt_gateway.server:app")


@pytest.mark.parametrize(
    "environment",
    [
        {"STT_BACKEND": "haas"},
        {"STT_BACKEND": "whisper", "STT_ENABLED_BACKENDS": "whisper, haas"},
    ],
)
def test_script_adds_haas_extra_when_haas_is_used(
    tmp_path: Path, environment: dict[str, str]
) -> None:
    assert _run_script(tmp_path, environment).startswith("run --extra haas uvicorn")


def test_script_does_not_add_haas_extra_for_whisper_only(tmp_path: Path) -> None:
    call = _run_script(tmp_path, {"STT_BACKEND": "whisper", "STT_ENABLED_BACKENDS": "whisper"})
    assert call.startswith("run uvicorn")
