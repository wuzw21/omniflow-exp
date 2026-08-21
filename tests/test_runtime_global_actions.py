from __future__ import annotations

import asyncio

import pytest

from omniflow.core.config import PluginSet
from omniflow.core.model import (
    Action,
    Function,
    FunctionStep,
    Observation,
    TransferResult,
)
from omniflow.runtime.execution import (
    execute_function,
    execute_robust_action,
    prepare_action,
)


def test_function_uses_catalog_state_when_host_state_is_missing(monkeypatch) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    source = Observation(
        xml='<hierarchy><node text="Search" bounds="[0,0][100,100]"/></hierarchy>',
        package_name="com.example",
    )
    current = Observation(
        xml='<hierarchy><node text="Search" bounds="[0,0][100,100]"/></hierarchy>',
        package_name="com.example",
    )
    transferred_sources: list[Observation] = []

    async def transfer(action, observation, source_state):
        transferred_sources.append(source_state)
        return TransferResult(action, reason="mapped")

    class Host:
        def get_state(self, _source_state_id):
            return None

        def act(self, _action):
            return {"success": True}

        def observe(self, **_kwargs):
            return current

    function = Function(
        function_id="catalog_function",
        name="catalog function",
        description="catalog state fallback",
        steps=(
            FunctionStep(0, Action("click", {"x": 50, "y": 50}), "source-1"),
        ),
        schema_version="omniflow.function.v2",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        agent_visible=True,
    )

    result = asyncio.run(
        execute_function(
            function,
            host=Host(),
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda state_id: source if state_id == "source-1" else None,
        )
    )

    assert result.success is True
    assert transferred_sources == [source]


def test_payment_text_does_not_create_hidden_runtime_policy(monkeypatch) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)

    class Host:
        def __init__(self) -> None:
            self.actions: list[Action] = []

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **_kwargs):
            return Observation(package_name="com.example")

    host = Host()
    action = Action("click", {"x": 900, "y": 2200})
    result = asyncio.run(
        execute_robust_action(
            action,
            observation=Observation(
                xml='<hierarchy><node text="立即支付" bounds="[0,0][100,100]"/></hierarchy>',
                package_name="com.example",
            ),
            host=host,
            plugins=PluginSet(),
        )
    )

    assert result.success is True
    assert host.actions == [action]


def test_runtime_dispatches_semantically_grounded_vlm_action(monkeypatch) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(
        xml=(
            '<hierarchy><node id="64:5" text="Turn off" '
            'resource-id="android:id/button1" bounds="[730,1310][926,1436]" '
            'clickable="true" enabled="true"/></hierarchy>'
        ),
        extra={"display": {"width": 1080, "height": 2400}},
    )

    class Host:
        def __init__(self) -> None:
            self.actions = []

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **_kwargs):
            return current

    host = Host()
    result = asyncio.run(
        execute_robust_action(
            Action(
                "click",
                {"target_description": "Turn off", "x": 572, "y": 572},
            ),
            observation=current,
            host=host,
            plugins=PluginSet(),
        )
    )

    assert result.success is True
    assert host.actions[0].args["x"] == pytest.approx(828 / 1080 * 1000)
    assert host.actions[0].args["y"] == pytest.approx(1373 / 2400 * 1000)
    assert result.detail["semantic_grounding"]["status"] == "resolved"


def test_runtime_dispatches_semantically_grounded_function_action(monkeypatch) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(
        xml=(
            '<hierarchy><node bounds="[174,358][309,442]" clickable="true" '
            'enabled="true"><node text="Income" bounds="[190,370][290,430]"/>'
            "</node></hierarchy>"
        ),
        extra={"display": {"width": 720, "height": 1280}},
    )
    source = Observation(
        xml=(
            '<hierarchy><node bounds="[56,358][158,442]" clickable="true" '
            'enabled="true"><node text="Food" bounds="[70,370][140,430]"/>'
            "</node></hierarchy>"
        ),
        extra={"display": {"width": 720, "height": 1280}},
    )

    class Host:
        def __init__(self) -> None:
            self.actions = []

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **_kwargs):
            return current

    host = Host()
    result = asyncio.run(
        execute_robust_action(
            Action(
                "click",
                {
                    "target_description": "Income",
                    "x": 148.61111111111111,
                    "y": 312.5,
                },
            ),
            observation=current,
            host=host,
            plugins=PluginSet(),
            source_state=source,
        )
    )

    assert result.success is True
    assert host.actions[0].args["x"] == pytest.approx(241.5 / 720 * 1000)
    assert host.actions[0].args["y"] == pytest.approx(400 / 1280 * 1000)
    assert result.detail["semantic_grounding"]["status"] == "resolved"


def test_function_semantic_target_falls_back_to_transfer_when_missing(
    monkeypatch,
) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(
        xml='<hierarchy><node text="Food" bounds="[0,0][100,100]"/></hierarchy>',
        extra={"display": {"width": 720, "height": 1280}},
    )
    source = Observation(
        xml='<hierarchy><node text="Food" bounds="[0,0][100,100]"/></hierarchy>',
        extra={"display": {"width": 720, "height": 1280}},
    )
    mapped = Action("click", {"x": 600, "y": 700})
    transfer_calls = []

    async def transfer(action, observation, source_state):
        transfer_calls.append((action, observation, source_state))
        return TransferResult(mapped, reason="mapped")

    class Host:
        def __init__(self) -> None:
            self.actions = []

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **_kwargs):
            return current

    host = Host()
    result = asyncio.run(
        execute_robust_action(
            Action(
                "click",
                {
                    "target_description": "Income",
                    "x": 148.61111111111111,
                    "y": 312.5,
                },
            ),
            observation=current,
            host=host,
            plugins=PluginSet(transfer=transfer),
            source_state=source,
        )
    )

    assert result.success is True
    assert len(transfer_calls) == 1
    assert host.actions == [mapped]
    assert result.detail["semantic_grounding"]["status"] == "fallback"


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


def test_direction_swipe_skips_element_transfer_validation() -> None:
    transferred: list[str] = []

    async def transfer(action, observation, source_state):
        transferred.append(action.tool)
        return TransferResult(None, reason="unexpected_transfer")

    action = Action(
        "swipe",
        {
            "direction": "right",
            "x1": 1000,
            "y1": 500,
            "x2": 0,
            "y2": 500,
        },
    )
    decision = asyncio.run(
        prepare_action(
            action,
            observation=Observation(package_name="com.android.camera2"),
            plugins=PluginSet(transfer=transfer),
            source_state=Observation(package_name="com.android.camera2"),
        )
    )

    assert decision.kind == "ready"
    assert decision.action == action
    assert transferred == []


def test_open_app_waits_for_cold_launch_target_package(monkeypatch) -> None:
    import omniflow.runtime.core as core
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_OPEN_APP_READY_POLL_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_OPEN_APP_READY_MAX_ATTEMPTS", 3)

    class Host:
        def __init__(self) -> None:
            self.observations = [
                Observation(package_name="com.android.launcher"),
                Observation(package_name="com.sankuai.meituan"),
            ]
            self.observe_calls = 0

        def act(self, action):
            assert action == Action(
                "open_app",
                {"package_name": "com.sankuai.meituan"},
            )
            return {"success": True}

        def observe(self, **kwargs):
            self.observe_calls += 1
            return self.observations.pop(0)

    host = Host()
    result = asyncio.run(
        execute_robust_action(
            Action("open_app", {"package_name": "com.sankuai.meituan"}),
            observation=Observation(package_name="com.android.launcher"),
            host=host,
            plugins=PluginSet(),
            installed_packages=frozenset({"com.sankuai.meituan"}),
        )
    )

    assert result.success is True
    assert result.after is not None
    assert result.after.package_name == "com.sankuai.meituan"
    assert host.observe_calls == 2


def test_open_app_reports_not_ready_after_retry_budget(monkeypatch) -> None:
    import omniflow.runtime.core as core
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_OPEN_APP_READY_POLL_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_OPEN_APP_READY_MAX_ATTEMPTS", 3)

    class Host:
        def __init__(self) -> None:
            self.observe_calls = 0

        def act(self, action):
            return {"success": True}

        def observe(self, **kwargs):
            self.observe_calls += 1
            return Observation(package_name="com.android.launcher")

    host = Host()
    result = asyncio.run(
        execute_robust_action(
            Action("open_app", {"package_name": "com.sankuai.meituan"}),
            observation=Observation(package_name="com.android.launcher"),
            host=host,
            plugins=PluginSet(),
            installed_packages=frozenset({"com.sankuai.meituan"}),
        )
    )

    assert result.success is False
    assert result.error == (
        "open_app_target_not_ready:"
        "expected=com.sankuai.meituan:observed=com.android.launcher"
    )
    assert host.observe_calls == 3


def test_open_app_dispatches_when_installed_package_inventory_is_incomplete(
    monkeypatch,
) -> None:
    import omniflow.runtime.core as core
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_OPEN_APP_READY_POLL_SECONDS", 0.0)

    action = Action("open_app", {"package_name": "com.android.settings"})

    class Host:
        def __init__(self) -> None:
            self.actions: list[Action] = []

        def act(self, dispatched_action):
            self.actions.append(dispatched_action)
            return {"success": True}

        def observe(self, **_kwargs):
            return Observation(package_name="com.android.settings")

    host = Host()
    result = asyncio.run(
        execute_robust_action(
            action,
            observation=Observation(package_name="cn.com.omnimind.bot.debug"),
            host=host,
            plugins=PluginSet(),
            installed_packages=frozenset({"cn.com.omnimind.bot.debug"}),
        )
    )

    assert result.success is True
    assert host.actions == [action]


def test_function_global_package_preflight_opens_target_before_recorded_step(
    monkeypatch,
) -> None:
    import omniflow.runtime.core as core
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_OPEN_APP_READY_POLL_SECONDS", 0.0)

    target_package = "com.example.target"
    source = Observation(package_name=target_package)
    current = Observation(package_name="com.android.launcher")

    async def transfer(action, observation, source_state):
        return TransferResult(action, reason="mapped")

    class Host:
        def __init__(self) -> None:
            self.actions: list[Action] = []
            self.observations = [
                Observation(package_name=target_package),
                Observation(package_name=target_package),
            ]

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **_kwargs):
            if self.observations:
                return self.observations.pop(0)
            return Observation(package_name=target_package)

    function = Function(
        function_id="global_package_preflight_function",
        name="global package preflight function",
        description="opens the target app before a recorded action",
        steps=(
            FunctionStep(
                0,
                Action("click", {"x": 500, "y": 500}),
                "source-1",
            ),
        ),
        schema_version="omniflow.function.v2",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        agent_visible=True,
    )

    host = Host()
    result = asyncio.run(
        execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda state_id: source if state_id == "source-1" else None,
        )
    )

    assert result.success is True
    assert result.actions_executed == 2
    assert host.actions == [
        Action("open_app", {"package_name": target_package}),
        Action("click", {"x": 500, "y": 500}),
    ]
    assert result.detail["checker_decisions"][0] == {
        "function_id": "global_package_preflight_function",
        "checker_kind": "global_package_preflight",
        "source_state_id": "source-1",
        "before_function_step": 0,
        "target_package": target_package,
        "observed_package": "com.android.launcher",
        "status": "executed",
        "reason": "package_mismatch",
    }


def test_action_waits_for_transition_window_to_enter_display(monkeypatch) -> None:
    import omniflow.runtime.core as core
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_OBSERVATION_READY_POLL_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_OBSERVATION_READY_MAX_ATTEMPTS", 3)

    transient_xml = (
        '<hierarchy><node bounds="[361,0][1441,2376]" '
        'package="com.sankuai.meituan"/></hierarchy>'
    )
    settled_xml = (
        '<hierarchy><node bounds="[0,0][1080,2376]" '
        'package="com.sankuai.meituan"/></hierarchy>'
    )

    class Host:
        def __init__(self) -> None:
            self.observations = [
                Observation(
                    package_name="com.sankuai.meituan",
                    xml=transient_xml,
                    extra={"display": {"width": 1080, "height": 2376}},
                ),
                Observation(
                    package_name="com.sankuai.meituan",
                    xml=settled_xml,
                    extra={"display": {"width": 1080, "height": 2376}},
                ),
            ]
            self.observe_calls = 0

        def act(self, action):
            return {"success": True}

        def observe(self, **kwargs):
            self.observe_calls += 1
            return self.observations.pop(0)

    host = Host()
    result = asyncio.run(
        execute_robust_action(
            Action("click", {"x": 500, "y": 500}),
            observation=Observation(package_name="com.sankuai.meituan"),
            host=host,
            plugins=PluginSet(),
        )
    )

    assert result.success is True
    assert result.after is not None
    assert result.after.xml == settled_xml
    assert host.observe_calls == 2
