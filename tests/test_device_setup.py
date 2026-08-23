from __future__ import annotations

import pytest

from src.experiment.device_setup import _python_install_timeout_sec


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
