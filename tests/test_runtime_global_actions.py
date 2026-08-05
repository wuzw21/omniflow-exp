from __future__ import annotations

import asyncio

from omniflow.core.config import PluginSet
from omniflow.core.model import (
    Action,
    CheckerContext,
    Observation,
    TransferResult,
)
from omniflow.runtime.checker import default_checker
from omniflow.runtime.execution import execute_action, prepare_action


def test_global_actions_skip_page_recovery_checker() -> None:
    source = Observation(package_name="com.oplus.battery")
    current = Observation(package_name="cn.com.omnimind.bot.debug")

    for action in (
        Action("open_app", {"package_name": "com.android.settings"}),
        Action("press_key", {"key": "back"}),
    ):
        assert default_checker(CheckerContext(source, current, action)) is None


def test_global_actions_skip_transfer_validation() -> None:
    transferred: list[str] = []

    async def transfer(action, observation, source_state):
        transferred.append(action.tool)
        return TransferResult(None, reason="unexpected_transfer")

    plugins = PluginSet(transfer=transfer)
    source = Observation(package_name="com.oplus.battery")
    current = Observation(package_name="cn.com.omnimind.bot.debug")

    for action in (
        Action("open_app", {"package_name": "com.android.settings"}),
        Action("press_key", {"key": "back"}),
    ):
        decision = asyncio.run(
            prepare_action(
                action,
                observation=current,
                plugins=plugins,
                source_state=source,
            )
        )
        assert decision.kind == "ready"
        assert decision.action == action

    assert transferred == []


def test_unlaunchable_checker_recovery_falls_back_to_transfer_failure() -> None:
    async def transfer(action, observation, source_state):
        return TransferResult(None, reason="transfer_failed")

    class Host:
        def act(self, action):
            raise AssertionError(f"unexpected action dispatch: {action}")

    result = asyncio.run(
        execute_action(
            Action("click", {"x": 77, "y": 83}),
            observation=Observation(package_name="cn.com.omnimind.bot.debug"),
            source_state=Observation(package_name="com.oplus.battery"),
            host=Host(),
            plugins=PluginSet(checker=default_checker, transfer=transfer),
            installed_packages=frozenset({"cn.com.omnimind.bot.debug"}),
        )
    )

    assert result.success is False
    assert result.error == "transfer_failed"
    assert result.actions_executed == 0
