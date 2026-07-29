from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.integrations.android_world.setup_compat import (
    patch_androidworld_setup_fail_closed,
)


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
