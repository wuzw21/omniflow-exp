from types import SimpleNamespace

import pytest

from src.integrations.android_world import launch


@pytest.mark.parametrize(
    ("stdout", "returncode", "expected"),
    [
        (
            "package:/data/app/com.google.androidenv.accessibilityforwarder/base.apk\n",
            0,
            True,
        ),
        ("", 0, False),
        ("package:/data/app/forwarder/base.apk\n", 1, False),
    ],
)
def test_accessibility_forwarder_installation_probe(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
    expected: bool,
) -> None:
    monkeypatch.setattr(
        launch,
        "_run_adb_command",
        lambda **kwargs: {"returncode": returncode, "stdout": stdout},
    )

    assert launch._androidworld_accessibility_forwarder_installed(
        adb_serial="emulator-5560",
        adb_path="/sdk/adb",
    ) is expected


def test_native_a11y_runtime_retries_transient_state_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    refresh_count = 0
    state_count = 0

    def refresh_env() -> None:
        nonlocal refresh_count
        refresh_count += 1

    controller = SimpleNamespace(
        _a11y_method=SimpleNamespace(value="a11y_forwarder_app"),
        refresh_env=refresh_env,
    )

    class Environment:
        def __init__(self) -> None:
            self.controller = controller

        def get_state(self) -> SimpleNamespace:
            nonlocal state_count
            state_count += 1
            if state_count == 1:
                raise RuntimeError("Could not get a11y tree.")
            return SimpleNamespace(ui_elements=[object()])

    monkeypatch.setattr(
        launch,
        "_quiesce_androidworld_accessibility_forwarder",
        lambda **kwargs: {"removed": True, "remaining_services": []},
    )
    monkeypatch.setattr(
        launch,
        "_run_adb_command",
        lambda **kwargs: {
            "returncode": 0,
            "stdout": "0" if "settings" in kwargs["adb_args"] else "",
        },
    )
    monkeypatch.setattr(
        launch,
        "_close_android_system_dialogs",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(launch.time, "sleep", lambda seconds: None)

    result = launch._prepare_native_androidworld_a11y_runtime(
        Environment(),
        adb_serial="emulator-5560",
        adb_path="/sdk/adb",
    )

    assert result["ready"] is True
    assert result["ui_element_count"] == 1
    assert result["readiness_attempts"] == 2
    assert state_count == 2
    assert refresh_count == 2
