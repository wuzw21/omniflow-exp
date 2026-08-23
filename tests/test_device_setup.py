from __future__ import annotations

import pytest

from src.experiment.device_setup import (
    _is_required_setup_checkout,
    _python_import_probe_code,
    _python_install_timeout_sec,
    _python_requirement_installs,
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
    }
    assert ("uninstall", ["-y", "serpapi"]) in installs


def test_python_import_probe_loads_mobilegpt_google_search_symbol() -> None:
    assert "from serpapi import GoogleSearch" in _python_import_probe_code()


def test_only_method_scoped_appagent_checkout_is_optional() -> None:
    assert _is_required_setup_checkout("appagent_checkout") is False
    assert _is_required_setup_checkout("androidworld_checkout") is True
    assert _is_required_setup_checkout("omnitransfer_checkout") is True
    assert _is_required_setup_checkout("mobilegpt_checkout") is True
