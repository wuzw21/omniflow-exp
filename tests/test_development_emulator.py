from __future__ import annotations

from pathlib import Path
import subprocess

import pytest

from src.experiment.development_emulator import _adb_output, _console_port


def test_console_port_requires_exact_emulator_serial() -> None:
    assert _console_port("emulator-5554") == 5554
    with pytest.raises(ValueError, match="invalid emulator serial"):
        _console_port("device-5554")


def test_adb_timeout_is_treated_as_transient_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def timeout(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=["adb"], timeout=10)

    monkeypatch.setattr(subprocess, "run", timeout)

    assert _adb_output(Path("/opt/android/adb"), "get-state") == ""
