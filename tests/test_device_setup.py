from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiment.device_setup import (
    Device,
    _ensure_user_ffmpeg,
    _health_probe,
    _is_required_setup_checkout,
    _python_import_probe_code,
    _python_install_timeout_sec,
    _python_requirement_installs,
    _validate_apks,
)


def test_health_probe_uses_shared_reset_observe_contract(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_ready(adb, serial, **kwargs):
        captured.update(adb=adb, serial=serial, **kwargs)
        return {"ready": True, "repaired": False, "xml_chars": 42}

    monkeypatch.setattr(
        "src.experiment.device_setup.ensure_oob_device_ready", fake_ready
    )
    result = _health_probe(
        tmp_path / "adb",
        Device("standard45562", "emulator-45562", 45562, "avd", "small_phone"),
    )

    assert result["name"] == "oob_reset_observe_smoke"
    assert result["status"] == "ok"
    assert json.loads(result["detail"])["xml_chars"] == 42
    assert captured == {
        "adb": str(tmp_path / "adb"),
        "serial": "emulator-45562",
        "timeout_seconds": 30,
        "repair": True,
    }


def test_setup_rejects_stale_oob_apk_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oob_apk = tmp_path / "omnibot-oob.apk"
    mobilegpt_apk = tmp_path / "mobilegpt.apk"
    oob_apk.write_bytes(b"oob")
    mobilegpt_apk.write_bytes(b"mobilegpt")

    def fake_aapt(_aapt: Path, apk: Path, mode: str) -> str:
        if apk == mobilegpt_apk:
            return "package: name='com.example.MobileGPT'"
        if mode == "badging":
            return "package: name='cn.com.omnimind.bot.debug' versionName='0.5.6.1'"
        return " ".join(
            (
                ".DebugOmniFlowControlReceiver",
                "cn.com.omnimind.bot.debug.CONTROL_OMNIFLOW",
                ".DebugOmniFlowObserveReceiver",
                "cn.com.omnimind.bot.debug.OBSERVE_OMNIFLOW",
            )
        )

    monkeypatch.setattr("src.experiment.device_setup._aapt_text", fake_aapt)

    with pytest.raises(RuntimeError, match="oob_version"):
        _validate_apks(
            aapt=tmp_path / "aapt",
            oob_apk=oob_apk,
            mobilegpt_apk=mobilegpt_apk,
        )


def test_python_install_timeout_covers_official_mobilegpt_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("OMNIFLOW_SETUP_PYTHON_TIMEOUT_SEC", raising=False)

    assert _python_install_timeout_sec() == 7200.0


def test_python_install_timeout_is_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OMNIFLOW_SETUP_PYTHON_TIMEOUT_SEC", "5400")

    assert _python_install_timeout_sec() == 5400.0


@pytest.mark.parametrize("value", ["0", "-1", "nan", "invalid"])
def test_python_install_timeout_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    value: str,
) -> None:
    monkeypatch.setenv("OMNIFLOW_SETUP_PYTHON_TIMEOUT_SEC", value)

    with pytest.raises(ValueError, match="invalid setup Python timeout"):
        _python_install_timeout_sec()


def test_python_requirement_installs_skip_absent_optional_appagent_file(
    tmp_path,
) -> None:
    repo = tmp_path / "repo"
    android_world = tmp_path / "android_world"
    mobilegpt = tmp_path / "mobilegpt"
    appagent = tmp_path / "appagent"
    for root in (repo, android_world, mobilegpt / "Server", appagent):
        root.mkdir(parents=True)
    (android_world / "requirements.txt").write_text("requests\n", encoding="utf-8")
    (mobilegpt / "Server" / "requirements.txt").write_text(
        "openai\n", encoding="utf-8"
    )

    installs = _python_requirement_installs(
        repo=repo,
        android_world_root=android_world,
        mobilegpt_root=mobilegpt,
        appagent_root=appagent,
    )

    requirement_paths = {
        arguments[-1]
        for kind, arguments in installs
        if kind == "requirements"
    }
    assert str(android_world / "requirements.txt") in requirement_paths
    assert str(mobilegpt / "Server" / "requirements.txt") in requirement_paths
    assert str(appagent / "requirements.txt") not in requirement_paths
    packages = {
        arguments[-1]
        for kind, arguments in installs
        if kind == "package"
    }
    assert packages == {
        "colorama",
        "dashscope",
        "google-search-results==2.4.2",
        "imageio-ffmpeg>=0.6",
    }
    assert ("uninstall", ["-y", "serpapi"]) in installs


def test_python_import_probe_loads_mobilegpt_google_search_symbol() -> None:
    assert "from serpapi import GoogleSearch" in _python_import_probe_code()


def test_only_method_scoped_appagent_checkout_is_optional() -> None:
    assert _is_required_setup_checkout("appagent_checkout") is False
    assert _is_required_setup_checkout("androidworld_checkout") is True
    assert _is_required_setup_checkout("omnitransfer_checkout") is True
    assert _is_required_setup_checkout("mobilegpt_checkout") is True


def test_setup_installs_and_exposes_user_ffmpeg(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    python_bin = tmp_path / "venv" / "bin" / "python"
    python_bin.parent.mkdir(parents=True)
    python_bin.write_text("", encoding="utf-8")
    packaged_ffmpeg = tmp_path / "package" / "ffmpeg-linux-x86_64"
    packaged_ffmpeg.parent.mkdir(parents=True)
    packaged_ffmpeg.write_text("", encoding="utf-8")
    packaged_ffmpeg.chmod(0o755)
    commands: list[list[str]] = []

    def fake_run(command: list[str], **_: object):
        import subprocess

        commands.append(command)
        if command[-2:] == ["install", "imageio-ffmpeg>=0.6"]:
            return subprocess.CompletedProcess(command, 0, "installed")
        if len([item for item in commands if "get_ffmpeg_exe" in " ".join(item)]) == 1:
            return subprocess.CompletedProcess(command, 1, "missing")
        return subprocess.CompletedProcess(command, 0, str(packaged_ffmpeg) + "\n")

    monkeypatch.setattr("src.experiment.device_setup._run", fake_run)
    monkeypatch.setattr("src.experiment.device_setup.shutil.which", lambda _name: None)

    exposed = _ensure_user_ffmpeg(
        python_bin=python_bin,
        install_python=True,
        user_bin_dir=tmp_path / "user-bin",
    )

    assert exposed == tmp_path / "user-bin" / "ffmpeg"
    assert exposed.is_symlink()
    assert exposed.resolve() == packaged_ffmpeg
    assert [str(python_bin), "-m", "pip", "install", "imageio-ffmpeg>=0.6"] in commands
