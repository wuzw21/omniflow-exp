from __future__ import annotations

import asyncio

from omniflow.core.config import PluginSet
from omniflow.core.model import (
    Action,
    CheckerContext,
    Function,
    FunctionStep,
    Observation,
    TransferResult,
)
from omniflow.runtime.checker import (
    CheckerLibrary,
    checker_rule_matches,
    default_checker,
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
        xml=(
            '<hierarchy><node text="Search" bounds="[0,0][100,100]" '
            'clickable="true" enabled="true" /></hierarchy>'
        ),
        package_name="com.example",
        extra={"display": {"width": 100, "height": 100}},
    )
    current = Observation(
        xml=(
            '<hierarchy><node text="Search" bounds="[0,0][100,100]" '
            'clickable="true" enabled="true" /></hierarchy>'
        ),
        package_name="com.example",
        extra={"display": {"width": 100, "height": 100}},
    )
    transferred_sources: list[Observation] = []

    async def transfer(action, observation, source_state):
        transferred_sources.append(source_state)
        return TransferResult(action, reason="mapped", detail={"score": 0.9})

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


def test_function_execution_is_gated_by_transfer_not_page_embedding(monkeypatch) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    source = Observation(
        xml=(
            '<hierarchy><node class="android.view.SurfaceView" '
            'bounds="[0,0][1000,850]" /></hierarchy>'
        ),
        package_name="com.example",
    )
    current = Observation(
        xml=(
            '<hierarchy><node class="android.widget.Button" text="Continue" '
            'bounds="[650,800][950,920]" clickable="true" enabled="true" />'
            "</hierarchy>"
        ),
        package_name="com.example",
        extra={"display": {"width": 1000, "height": 1000}},
    )
    actions: list[Action] = []

    class Host:
        def act(self, action: Action):
            actions.append(action)
            return {"success": True}

        def observe(self, **_kwargs: object):
            return current

    async def transfer(
        action: Action,
        _observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        return TransferResult(
            Action(action.tool, {"x": 800.0, "y": 860.0}),
            reason="omnitransfer_unified_association_v1",
            detail={"absolute_contextual_confidence": 0.95},
        )

    function = Function(
        function_id="continue_form",
        name="Continue form",
        description="Continue the current form.",
        steps=(
            FunctionStep(0, Action("click", {"x": 500, "y": 500}), "source"),
        ),
    )

    result = asyncio.run(
        execute_function(
            function,
            host=Host(),
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert actions == [Action("click", {"x": 800.0, "y": 860.0})]


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


def test_default_overlay_checker_ignores_normal_chrome_close_controls() -> None:
    rule = next(
        rule
        for rule in CheckerLibrary.load().rules
        if rule["id"] == "dismiss_transient_overlay"
    )
    action = Action("click", {"x": 500, "y": 500})
    source = Observation(package_name="com.android.chrome")
    normal_chrome = Observation(
        package_name="com.android.chrome",
        xml=(
            '<hierarchy>'
            '<node clickable="true" content-desc="Switch or close tabs" '
            'resource-id="com.android.chrome:id/tab_switcher_button"/>'
            '<node clickable="true" content-desc="Close Memory Task tab"/>'
            '</hierarchy>'
        ),
    )
    explicit_overlay = Observation(
        package_name="com.android.chrome",
        xml=(
            '<hierarchy><node clickable="true" text="Close" '
            'bounds="[10,10][50,50]"/></hierarchy>'
        ),
    )

    assert checker_rule_matches(
        rule,
        current=normal_chrome,
        source=source,
        function_id="click_button_5_times",
        step_index=0,
        action=action,
    ) is False
    assert checker_rule_matches(
        rule,
        current=explicit_overlay,
        source=source,
        function_id="click_button_5_times",
        step_index=0,
        action=action,
    ) is True


def test_checker_drains_consecutive_explicit_obstructions_before_function_action(
    monkeypatch,
) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    obstruction = Observation(
        xml=(
            '<hierarchy><node text="Not now" class="android.widget.Button" '
            'clickable="true" enabled="true" bounds="[100,100][300,200]"/>'
            "</hierarchy>"
        ),
        package_name="com.example",
        extra={"display": {"width": 1000, "height": 1000}},
    )
    target = Observation(
        xml=(
            '<hierarchy><node text="Target" bounds="[0,0][1000,1000]" '
            'clickable="true" enabled="true" /></hierarchy>'
        ),
        package_name="com.example",
        extra={"display": {"width": 1000, "height": 1000}},
    )

    async def transfer(action, observation, source_state):
        return TransferResult(action, reason="mapped", detail={"score": 0.9})

    class Host:
        def __init__(self) -> None:
            self.observations = [
                obstruction,
                obstruction,
                obstruction,
                target,
                target,
            ]
            self.actions: list[Action] = []

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **kwargs):
            return self.observations.pop(0)

    host = Host()
    original_action = Action("click", {"x": 500, "y": 500})
    result = asyncio.run(
        execute_robust_action(
            original_action,
            observation=obstruction,
            host=host,
            plugins=PluginSet(checker=default_checker, transfer=transfer),
            source_state=target,
        )
    )

    assert result.success is True
    assert all(step.origin == "checker" for step in result.executed_steps[:-1])
    assert result.executed_steps[-1].origin == "action"
    assert host.actions[-1] == original_action


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


def test_unlaunchable_checker_recovery_falls_back_to_transfer_failure() -> None:
    async def transfer(action, observation, source_state):
        return TransferResult(None, reason="transfer_failed")

    class Host:
        def act(self, action):
            raise AssertionError(f"unexpected action dispatch: {action}")

    result = asyncio.run(
        execute_robust_action(
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
