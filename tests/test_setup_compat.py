from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.integrations.android_world.setup_compat import (
    patch_androidworld_setup_click_retry,
    patch_androidworld_setup_fail_closed,
)


def _setup_element(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        package_name="com.android.chrome",
        text=text,
        content_description="",
    )


def test_androidworld_setup_clicks_visible_alias_before_legacy_label() -> None:
    click_calls: list[str] = []

    class Controller:
        def __init__(self) -> None:
            self._env = SimpleNamespace(
                get_ui_elements=lambda: [_setup_element("Use without an account")]
            )

        def click_element(self, element_text: str) -> None:
            click_calls.append(element_text)
            if element_text != "Use without an account":
                raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=Controller)
    patch_androidworld_setup_click_retry(
        tools_module,
        attempts=1,
        delay_seconds=0,
    )

    Controller().click_element("Accept & continue")

    assert click_calls == ["Use without an account"]


def test_androidworld_setup_skips_click_when_chrome_is_complete() -> None:
    click_calls: list[str] = []

    class Controller:
        def __init__(self) -> None:
            self._env = SimpleNamespace(
                get_ui_elements=lambda: [_setup_element("Search or type web address")]
            )

        def click_element(self, element_text: str) -> None:
            click_calls.append(element_text)
            raise ValueError(f'Target text "{element_text}" not found.')

    tools_module = SimpleNamespace(AndroidToolController=Controller)
    patch_androidworld_setup_click_retry(
        tools_module,
        attempts=1,
        delay_seconds=0,
    )

    assert Controller().click_element("No thanks") is None
    assert click_calls == []


def test_androidworld_setup_retries_before_saving_snapshot() -> None:
    setup_calls: list[object] = []
    snapshot_calls: list[tuple[str, object]] = []
    env = SimpleNamespace(controller=object())

    class App:
        app_name = "chrome"

        @classmethod
        def setup(cls, actual_env) -> None:
            setup_calls.append(actual_env)
            if len(setup_calls) == 1:
                raise ValueError("onboarding not ready")

    setup_module = SimpleNamespace(
        setup_app=lambda app, actual_env: None,
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, controller: snapshot_calls.append(
                (name, controller)
            )
        ),
    )

    patch_androidworld_setup_fail_closed(setup_module, attempts=2)
    setup_module.setup_app(App, env)

    assert setup_calls == [env, env]
    assert snapshot_calls == [("chrome", env.controller)]


def test_androidworld_setup_failure_does_not_save_snapshot() -> None:
    snapshot_calls: list[tuple[str, object]] = []
    env = SimpleNamespace(controller=object())

    class App:
        app_name = "chrome"

        @classmethod
        def setup(cls, actual_env) -> None:
            raise ValueError("onboarding never became ready")

    setup_module = SimpleNamespace(
        setup_app=lambda app, actual_env: None,
        app_snapshot=SimpleNamespace(
            save_snapshot=lambda name, controller: snapshot_calls.append(
                (name, controller)
            )
        ),
    )

    patch_androidworld_setup_fail_closed(setup_module, attempts=2)

    with pytest.raises(ValueError, match="onboarding never became ready"):
        setup_module.setup_app(App, env)

    assert snapshot_calls == []
