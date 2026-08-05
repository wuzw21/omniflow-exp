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
from omniflow.runtime.checker import default_checker
from omniflow.runtime.execution import execute_action, execute_function, prepare_action


def test_function_uses_catalog_state_when_host_state_is_missing(monkeypatch) -> None:
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(execution, "_ACTION_SETTLE_SECONDS", 0.0)
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


def test_delayed_checker_recovery_retries_the_same_function_step(monkeypatch) -> None:
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(execution, "_ACTION_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_TRANSFER_REOBSERVE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_TRANSFER_REOBSERVE_MAX_ATTEMPTS", 1)
    source = Observation(
        xml='<hierarchy><node text="购买" bounds="[0,0][100,100]"/></hierarchy>',
        package_name="com.example",
    )
    initial = Observation(
        xml='<hierarchy><node text="加载中" bounds="[0,0][100,100]"/></hierarchy>',
        package_name="com.example",
    )
    advertisement = Observation(
        xml='<hierarchy><node text="跳过广告" bounds="[0,0][100,100]"/></hierarchy>',
        package_name="com.example",
    )
    target = Observation(
        xml='<hierarchy><node text="购买" bounds="[0,0][100,100]"/></hierarchy>',
        package_name="com.example",
    )
    transfer_calls = 0

    async def transfer(action, observation, _source_state):
        nonlocal transfer_calls
        transfer_calls += 1
        if "购买" not in str(observation.xml):
            return TransferResult(
                None,
                reason="omnitransfer_target_semantic_missing",
            )
        return TransferResult(action, reason="mapped")

    class Host:
        def __init__(self) -> None:
            self.observations = [advertisement, target, target]
            self.actions: list[Action] = []

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **_kwargs):
            return self.observations.pop(0)

    function = Function(
        function_id="recovery_function",
        name="recovery function",
        description="retry after delayed ad",
        steps=(),
        schema_version="omniflow.function.v2",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        checker_rules=(
            {
                "schema_version": "omniflow.checker_rule.v1",
                "trigger": 'text_contains("跳过广告")',
                "source_state_id": "checker-anchor",
                "action": {"tool": "press_key", "args": {"key": "back"}},
            },
        ),
        agent_visible=True,
    )
    host = Host()

    result = asyncio.run(
        execute_action(
            Action("click", {"x": 50, "y": 50}),
            observation=initial,
            host=host,
            plugins=PluginSet(transfer=transfer),
            function=function,
            source_state=source,
        )
    )

    assert result.success is True
    assert host.actions == [
        Action("press_key", {"key": "back"}),
        Action("click", {"x": 50, "y": 50}),
    ]
    assert transfer_calls == 3
    assert result.executed_steps[0].origin == "checker"


def test_payment_confirmation_screen_blocks_interactive_action() -> None:
    class Host:
        def act(self, _action):
            raise AssertionError("payment screen action must not be dispatched")

    result = asyncio.run(
        execute_action(
            Action("click", {"x": 900, "y": 2200}),
            observation=Observation(
                xml='<hierarchy><node text="立即支付" bounds="[0,0][100,100]"/></hierarchy>',
                package_name="com.example",
            ),
            host=Host(),
            plugins=PluginSet(),
        )
    )

    assert result.success is False
    assert result.error == "payment_confirmation_blocked"
    assert result.origin == "blocked"


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


def test_open_app_waits_for_cold_launch_target_package(monkeypatch) -> None:
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(execution, "_ACTION_SETTLE_SECONDS", 0.0)
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
        execute_action(
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
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(execution, "_ACTION_SETTLE_SECONDS", 0.0)
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
        execute_action(
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
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(execution, "_ACTION_SETTLE_SECONDS", 0.0)
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
        execute_action(
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


def test_transfer_reobserves_until_target_semantic_appears(monkeypatch) -> None:
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(execution, "_ACTION_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_TRANSFER_REOBSERVE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_TRANSFER_REOBSERVE_MAX_ATTEMPTS", 3)

    source = Observation(package_name="com.sankuai.meituan", xml="<source/>")
    loading = Observation(
        package_name="com.sankuai.meituan",
        xml="<hierarchy><node text='拿铁'/></hierarchy>",
    )
    ready = Observation(
        package_name="com.sankuai.meituan",
        xml="<hierarchy><node text='24小时营业'/></hierarchy>",
    )
    after = Observation(package_name="com.sankuai.meituan", xml="<merchant/>")
    transfer_observations: list[Observation] = []

    async def transfer(action, observation, source_state):
        assert source_state == source
        transfer_observations.append(observation)
        if observation == loading:
            return TransferResult(
                None,
                reason="omnitransfer_target_semantic_missing",
                detail={"source_title": "24小时营业", "target_titles": []},
            )
        assert observation == ready
        return TransferResult(
            Action("click", {"x": 500, "y": 800}),
            reason="equivalent_ui_graph",
            detail={"score": 1.0},
        )

    class Host:
        def __init__(self) -> None:
            self.observations = [ready, after]
            self.observe_calls = 0
            self.actions: list[Action] = []

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **kwargs):
            self.observe_calls += 1
            return self.observations.pop(0)

    host = Host()
    result = asyncio.run(
        execute_action(
            Action("click", {"x": 472, "y": 336}),
            observation=loading,
            source_state=source,
            host=host,
            plugins=PluginSet(transfer=transfer),
        )
    )

    assert result.success is True
    assert host.actions == [Action("click", {"x": 500, "y": 800})]
    assert host.observe_calls == 2
    assert transfer_observations == [loading, ready]
    assert result.detail["observation_retry"] == {
        "attempts": 1,
        "initial_reason": "omnitransfer_target_semantic_missing",
    }


def test_transfer_reobserves_before_executing_coordinate_stretch_fallback(
    monkeypatch,
) -> None:
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(execution, "_ACTION_SETTLE_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_TRANSFER_REOBSERVE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_TRANSFER_REOBSERVE_MAX_ATTEMPTS", 2)

    loading = Observation(package_name="com.sankuai.meituan", xml="<loading/>")
    ready = Observation(package_name="com.sankuai.meituan", xml="<results/>")
    after = Observation(package_name="com.sankuai.meituan", xml="<merchant/>")

    async def transfer(action, observation, source_state):
        if observation == loading:
            return TransferResult(
                Action("click", {"x": 472, "y": 336}),
                reason="mutual_graph_matcher_no_null_v3_coordinate_stretch_fallback",
                detail={
                    "mapping_mode": (
                        "mutual_graph_matcher_no_null_v3_coordinate_stretch_fallback"
                    ),
                    "score": 0.0068,
                },
            )
        assert observation == ready
        return TransferResult(
            Action("click", {"x": 500, "y": 800}),
            reason="equivalent_ui_graph",
            detail={"mapping_mode": "equivalent_ui_graph", "score": 1.0},
        )

    class Host:
        def __init__(self) -> None:
            self.observations = [ready, after]
            self.actions: list[Action] = []

        def act(self, action):
            self.actions.append(action)
            return {"success": True}

        def observe(self, **kwargs):
            return self.observations.pop(0)

    host = Host()
    result = asyncio.run(
        execute_action(
            Action("click", {"x": 472, "y": 336}),
            observation=loading,
            source_state=Observation(
                package_name="com.sankuai.meituan",
                xml="<source-results/>",
            ),
            host=host,
            plugins=PluginSet(transfer=transfer),
        )
    )

    assert result.success is True
    assert host.actions == [Action("click", {"x": 500, "y": 800})]
    assert result.detail["mapping_mode"] == "equivalent_ui_graph"
    assert result.detail["observation_retry"] == {
        "attempts": 1,
        "initial_reason": (
            "mutual_graph_matcher_no_null_v3_coordinate_stretch_fallback"
        ),
    }


def test_transfer_falls_back_only_after_reobservation_budget(monkeypatch) -> None:
    import omniflow.runtime.execution as execution

    monkeypatch.setattr(execution, "_TRANSFER_REOBSERVE_POLL_SECONDS", 0.0)
    monkeypatch.setattr(execution, "_TRANSFER_REOBSERVE_MAX_ATTEMPTS", 2)

    transfer_observations: list[Observation] = []

    async def transfer(action, observation, source_state):
        transfer_observations.append(observation)
        return TransferResult(
            None,
            reason="omnitransfer_target_semantic_missing",
            detail={"source_title": "商家搜索", "target_titles": []},
        )

    class Host:
        def __init__(self) -> None:
            self.observe_calls = 0

        def act(self, action):
            raise AssertionError(f"unexpected action dispatch: {action}")

        def observe(self, **kwargs):
            self.observe_calls += 1
            return Observation(
                package_name="com.sankuai.meituan",
                xml=f"<loading attempt='{self.observe_calls}'/>",
            )

    host = Host()
    initial = Observation(
        package_name="com.sankuai.meituan",
        xml="<loading attempt='0'/>",
    )
    result = asyncio.run(
        execute_action(
            Action("click", {"x": 400, "y": 200}),
            observation=initial,
            source_state=Observation(
                package_name="com.sankuai.meituan",
                xml="<source/>",
            ),
            host=host,
            plugins=PluginSet(transfer=transfer),
        )
    )

    assert result.success is False
    assert result.error == "omnitransfer_target_semantic_missing"
    assert host.observe_calls == 2
    assert len(transfer_observations) == 3
    assert result.before == transfer_observations[-1]
    assert result.detail["observation_retry"] == {
        "attempts": 2,
        "initial_reason": "omnitransfer_target_semantic_missing",
    }


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
