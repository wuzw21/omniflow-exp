from __future__ import annotations

import asyncio

from omniflow import Action, ActionResult, Observation, PluginSet
from omniflow.core.model import TransferResult
from omniflow.runtime import core


class RecordingHost:
    def __init__(self, after: Observation) -> None:
        self.after = after
        self.actions: list[Action] = []
        self.observe_requests: list[dict[str, object]] = []

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        return ActionResult(True)

    def observe(self, **kwargs: object) -> Observation:
        self.observe_requests.append(dict(kwargs))
        return self.after


def test_core_executes_one_transferred_action_and_observes_once(monkeypatch) -> None:
    settle_calls: list[float] = []

    async def settle(seconds: float) -> None:
        settle_calls.append(seconds)

    monkeypatch.setattr(core.asyncio, "sleep", settle)
    before = Observation(xml="<before/>", package_name="com.example")
    source = Observation(xml="<source/>", package_name="com.example")
    after = Observation(xml="<after/>", package_name="com.example")
    host = RecordingHost(after)
    transfer_calls: list[tuple[Action, Observation, Observation]] = []
    recorded = Action("click", {"x": 100, "y": 200})
    mapped = Action("click", {"x": 300, "y": 400})

    async def transfer(
        action: Action,
        observation: Observation,
        source_state: Observation,
    ) -> TransferResult:
        transfer_calls.append((action, observation, source_state))
        return TransferResult(
            mapped,
            reason="omnitransfer_mapped",
            detail={"score": 1.0},
        )

    result = asyncio.run(
        core.execute_action(
            recorded,
            observation=before,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=source,
        )
    )

    assert result.success is True
    assert result.action == mapped
    assert result.before == before
    assert result.after == after
    assert result.detail == {"score": 1.0}
    assert transfer_calls == [(recorded, before, source)]
    assert host.actions == [mapped]
    assert settle_calls == [1.0]
    assert host.observe_requests == [
        {"xml": True, "screenshot": True, "app_info": True}
    ]


def test_core_reports_transfer_failure_without_dispatch(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(xml="<before/>", package_name="com.example")
    source = Observation(xml="<source/>", package_name="com.example")
    host = RecordingHost(Observation(xml="<after/>"))

    async def transfer(
        _action: Action,
        _observation: Observation,
        _source_state: Observation,
    ) -> TransferResult:
        return TransferResult(None, reason="omnitransfer_failed")

    result = asyncio.run(
        core.execute_action(
            Action("click", {"x": 100, "y": 200}),
            observation=before,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=source,
        )
    )

    assert result.success is False
    assert result.error == "omnitransfer_failed"
    assert host.actions == []
    assert host.observe_requests == []


def test_core_has_no_open_app_catalog_gate(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(package_name="cn.com.omnimind.bot.debug")
    after = Observation(package_name="com.sankuai.meituan")
    host = RecordingHost(after)
    action = Action("open_app", {"package_name": "com.sankuai.meituan"})

    result = asyncio.run(
        core.execute_action(
            action,
            observation=before,
            host=host,
            plugins=PluginSet(),
        )
    )

    assert result.success is True
    assert host.actions == [action]
