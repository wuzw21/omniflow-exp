from __future__ import annotations

import asyncio

from omniflow import Action, ActionResult, Function, Observation, PluginSet
from omniflow.core.config import ANDROIDWORLD_PROTOCOL, RuntimeSettings
from omniflow.core.model import FunctionStep, StepResult, TransferResult
from omniflow.runtime import core, engine, execution


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


def test_checker_thresholds_come_from_the_single_protocol_config() -> None:
    checker = ANDROIDWORLD_PROTOCOL["checker"]
    settings = RuntimeSettings()

    assert "checker_action_confidence" not in ANDROIDWORLD_PROTOCOL
    assert set(checker) == {"target_probability_threshold"}
    assert settings.max_steps == ANDROIDWORLD_PROTOCOL["max_steps"]
    assert settings.max_fallback_steps == ANDROIDWORLD_PROTOCOL["max_fallback_steps"]
    assert settings.max_function_tools == ANDROIDWORLD_PROTOCOL["max_function_tools"]
    assert settings.checker_target_threshold == checker[
        "target_probability_threshold"
    ]


def test_bound_function_text_renders_into_source_observation_xml() -> None:
    source_function = Function(
        function_id="copy_text",
        name="Copy text",
        description="Copy caller text.",
        steps=(
            FunctionStep(
                0,
                Action("input_text", {"text": "source text"}),
                "source-before-input",
            ),
            FunctionStep(
                1,
                Action("long_press", {"x": 500, "y": 100}),
                "source-after-input",
            ),
        ),
        bindings=(
            {
                "source": "$.arguments.text",
                "target": "$.steps[0].action.args.text",
            },
        ),
    )
    bound_function = Function(
        function_id=source_function.function_id,
        name=source_function.name,
        description=source_function.description,
        steps=(
            FunctionStep(
                0,
                Action("input_text", {"text": "target text"}),
                "source-before-input",
            ),
            source_function.steps[1],
        ),
        bindings=source_function.bindings,
    )

    substitutions = engine._function_source_text_substitutions(
        source_function,
        bound_function,
    )
    rendered = execution._render_source_observation_text(
        Observation(
            package_name="com.example",
            xml=(
                '<hierarchy><node text="source text" '
                'content-desc="source text result" '
                'resource-id="com.example:id/input" /></hierarchy>'
            ),
        ),
        substitutions,
    )

    assert substitutions == {"source text": "target text"}
    assert rendered is not None
    assert 'text="target text"' in rendered.xml
    assert 'content-desc="target text result"' in rendered.xml
    assert 'resource-id="com.example:id/input"' in rendered.xml


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


def test_core_observes_after_dispatched_action_failure(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(
        xml='<hierarchy><node text="Old screen" /></hierarchy>',
        package_name="com.example",
        extra={"state_id": "before"},
    )
    after = Observation(
        xml='<hierarchy><node text="Current screen" /></hierarchy>',
        package_name="com.example",
        extra={"state_id": "after"},
    )

    class FailingHost(RecordingHost):
        def act(self, action: Action) -> ActionResult:
            self.actions.append(action)
            return ActionResult(False, "input_target_not_found")

    host = FailingHost(after)
    action = Action("input_text", {"text": "Alice", "x": 500, "y": 300})

    result = asyncio.run(
        core.execute_action(
            action,
            observation=before,
            host=host,
            plugins=PluginSet(),
        )
    )

    assert result.success is False
    assert result.error == "input_target_not_found"
    assert result.before == before
    assert result.after == after
    assert result.actions_executed == 1
    assert host.actions == [action]
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


def test_function_execution_counts_all_auxiliary_actions(monkeypatch) -> None:
    before = Observation(extra={"state_id": "before"})
    host = RecordingHost(before)
    function = Function(
        function_id="two_steps",
        name="Two steps",
        description="Two semantic steps with one auxiliary action each.",
        steps=(
            FunctionStep(0, Action("wait", {"duration_ms": 0}), ""),
            FunctionStep(1, Action("wait", {"duration_ms": 0}), ""),
        ),
    )

    async def execute_with_auxiliary(action, **kwargs):
        auxiliary = StepResult(
            True,
            action=Action("press_back", {}),
            before=kwargs["observation"],
            after=kwargs["observation"],
            result=ActionResult(True),
            actions_executed=1,
            origin="page_gate",
        )
        primary = StepResult(
            True,
            action=action,
            before=kwargs["observation"],
            after=kwargs["observation"],
            result=ActionResult(True),
            actions_executed=1,
        )
        return StepResult(
            True,
            action=action,
            before=kwargs["observation"],
            after=kwargs["observation"],
            result=ActionResult(True),
            actions_executed=2,
            executed_steps=(primary, auxiliary),
        )

    monkeypatch.setattr(execution, "execute_robust_action", execute_with_auxiliary)

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(),
            observation=before,
        )
    )

    assert result.success is True
    assert result.actions_executed == 4
    assert [step["metadata"]["function_step_index"] for step in result.detail["trace"]] == [
        0,
        0,
        1,
        1,
    ]


def test_checker_step_executes_when_omnitransfer_target_is_present(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(xml="<target/>", package_name="com.example")
    source = Observation(xml="<source/>", package_name="com.example")
    host = RecordingHost(before)
    checker_action = Action("click", {"x": 100, "y": 200})
    mapped_action = Action("click", {"x": 300, "y": 400})
    function = Function(
        function_id="optional_checker",
        name="Optional checker",
        description="Dismiss an optional page before continuing.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main-source"),),
        checker_rules=(
            {
                "source_state_id": "checker-source",
                "action": checker_action.to_dict(),
            },
        ),
    )

    async def transfer(_action, _observation, _source_state):
        if _action.tool == "wait":
            return TransferResult(_action)
        return TransferResult(
            mapped_action,
            reason="omnitransfer_mapped",
            detail={
                "score": 0.99,
                "margin": 0.1,
                "source": {"resource_id": "com.example:id/optional"},
                "candidates": [
                    {"resource_id": "com.example:id/optional", "score": 0.9}
                ],
            },
        )

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=before,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert result.actions_executed == 2
    assert host.actions[0] == mapped_action
    assert result.detail["checker_decisions"][0]["status"] == "executed"
    assert result.execution_summary["checker_decisions"] == result.detail[
        "checker_decisions"
    ]
    assert result.detail["checker_decisions"][0]["function_id"] == (
        "optional_checker"
    )
    assert result.detail["checker_decisions"][0]["source_state_id"] == (
        "checker-source"
    )
    assert "before_function_step" not in result.detail["checker_decisions"][0]
    assert "checker_rule_index" not in result.detail["checker_decisions"][0]
    assert result.detail["trace"][0]["metadata"]["origin"] == "checker"


def test_checker_policy_redirects_before_function_action_without_checking_action(
    monkeypatch,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    target_package = "com.example.settings"
    current = Observation(package_name="com.android.launcher")
    source = Observation(package_name=target_package)

    class Host:
        def __init__(self) -> None:
            self.actions: list[Action] = []
            self.after = Observation(package_name=target_package)

        def act(self, action: Action) -> ActionResult:
            self.actions.append(action)
            return ActionResult(True)

        def observe(self, **_kwargs: object) -> Observation:
            return self.after

    function = Function(
        function_id="policy_function",
        name="Policy function",
        description="Run a Function only inside its target app.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main"),),
        checker_rules=(
            {
                "source_state_id": "checker-source",
                "if": {"package_name": target_package},
                "then": {
                    "tool": "open_app",
                    "args": {"package_name": target_package},
                },
            },
        ),
    )

    host = Host()
    async def transfer(action, _observation, _source_state):
        return TransferResult(action)

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert host.actions == [
        Action("open_app", {"package_name": target_package}),
        Action("wait", {"duration_ms": 0}),
    ]
    assert result.detail["checker_decisions"][0]["status"] == "redirected"
    assert result.detail["checker_decisions"][0]["checker_kind"] == "policy"
    assert result.detail["trace"][0]["metadata"]["origin"] == "action"


def test_global_app_policy_redirects_before_every_transfer_action(
    monkeypatch,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    target_package = "com.example.settings"
    current = Observation(package_name="com.android.launcher")
    source = Observation(package_name=target_package)

    class Host:
        def __init__(self) -> None:
            self.actions: list[Action] = []
            self.after = Observation(package_name=target_package)

        def act(self, action: Action) -> ActionResult:
            self.actions.append(action)
            return ActionResult(True)

        def observe(self, **_kwargs: object) -> Observation:
            return self.after

    function = Function(
        function_id="global_app_policy_function",
        name="Global app policy function",
        description="Run the mapped action inside the source application.",
        steps=(
            FunctionStep(
                0,
                Action("click", {"x": 100, "y": 200}),
                "source-step",
            ),
        ),
    )
    mapped_action = Action("click", {"x": 300, "y": 400})

    async def transfer(_action, _observation, _source_state):
        return TransferResult(mapped_action)

    host = Host()
    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert host.actions == [
        Action("open_app", {"package_name": target_package}),
        mapped_action,
    ]
    assert result.detail["checker_decisions"][0]["checker_kind"] == (
        "global_app_policy"
    )
    assert result.detail["checker_decisions"][0]["status"] == "redirected"


def test_global_app_policy_uses_home_for_launcher_redirect(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    launcher = "com.google.android.apps.nexuslauncher"
    current = Observation(package_name="com.example.app")
    source = Observation(package_name=launcher)

    class Host:
        def __init__(self) -> None:
            self.actions: list[Action] = []

        def act(self, action: Action) -> ActionResult:
            self.actions.append(action)
            return ActionResult(True)

        def observe(self, **_kwargs: object) -> Observation:
            return Observation(package_name=launcher)

    function = Function(
        function_id="launcher_global_policy",
        name="Launcher global policy",
        description="Run a mapped action on the launcher.",
        steps=(
            FunctionStep(0, Action("click", {"x": 100, "y": 200}), "source"),
        ),
    )

    async def transfer(action, _observation, _source_state):
        return TransferResult(action)

    host = Host()
    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert host.actions[0] == Action("press_key", {"key": "home"})
    assert result.detail["checker_decisions"][0]["status"] == "redirected"


def test_global_app_policy_uses_corresponding_observation_xml_package(
    monkeypatch,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    launcher = "com.google.android.apps.nexuslauncher"
    source = Observation(
        xml=f'<hierarchy><node package="{launcher}" bounds="[0,0][720,1280]" /></hierarchy>'
    )
    current = Observation(
        package_name="android",
        xml=(
            '<hierarchy><node package="android" bounds="[0,0][720,1280]" />'
            f'<node package="{launcher}" bounds="[0,0][720,1280]" /></hierarchy>'
        ),
    )

    assert execution._observation_expected_package(source) == launcher
    assert execution._checker_policy_condition_matches(
        {"package_name": launcher}, current
    )


def test_checker_policy_passes_without_executing_then_when_condition_matches(
    monkeypatch,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    target_package = "com.example.settings"
    current = Observation(package_name=target_package)
    source = Observation(package_name=target_package)

    class Host:
        def __init__(self) -> None:
            self.actions: list[Action] = []

        def act(self, action: Action) -> ActionResult:
            self.actions.append(action)
            return ActionResult(True)

        def observe(self, **_kwargs: object) -> Observation:
            return current

    function = Function(
        function_id="policy_pass_function",
        name="Policy pass function",
        description="Do not navigate when already in the target app.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main"),),
        checker_rules=(
            {
                "source_state_id": "checker-source",
                "if": {"package_name": target_package},
                "then": {
                    "tool": "open_app",
                    "args": {"package_name": "com.example.other"},
                },
            },
        ),
    )

    host = Host()
    async def transfer(action, _observation, _source_state):
        return TransferResult(action)

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert host.actions == [Action("wait", {"duration_ms": 0})]
    assert result.detail["checker_decisions"][0]["status"] == "passed"
    assert result.detail["trace"][0]["metadata"]["origin"] == "action"


def test_checker_uses_the_configured_target_probability_threshold(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(
        xml="<target/>",
        package_name="com.example",
        activity_name="MainActivity",
    )
    source = Observation(
        xml="<source/>",
        package_name="com.example",
        activity_name="MainActivity",
    )
    host = RecordingHost(current)
    transfer_calls: list[Action] = []

    async def transfer(action, _observation, _source_state):
        transfer_calls.append(action)
        return TransferResult(
            action,
            detail={"score": 0.999, "candidates": [{"score": 0.95}]},
        )

    function = Function(
        function_id="strict_checker",
        name="Strict checker",
        description="Use the configured action-mapping threshold.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main"),),
        checker_rules=(
            {
                "source_state_id": "checker",
                "action": {"tool": "click", "args": {"x": 100, "y": 200}},
            },
        ),
    )

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
            checker_target_threshold=0.96,
        )
    )

    assert result.success is True
    assert transfer_calls == [
        Action("click", {"x": 100, "y": 200}),
        Action("wait", {"duration_ms": 0}),
    ]
    assert host.actions == [Action("wait", {"duration_ms": 0})]
    assert result.detail["checker_decisions"][0]["reason"] == (
        "checker_target_probability_too_low"
    )


def test_checker_step_skips_when_omnitransfer_target_is_not_present(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    before = Observation(xml="<target/>", package_name="com.example")
    source = Observation(xml="<source/>", package_name="com.example")
    host = RecordingHost(before)
    function = Function(
        function_id="optional_checker",
        name="Optional checker",
        description="Dismiss an optional page before continuing.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main-source"),),
        checker_rules=(
            {
                "source_state_id": "checker-source",
                "action": {
                    "tool": "click",
                    "args": {"x": 100, "y": 200},
                },
            },
        ),
    )

    async def transfer(_action, _observation, _source_state):
        if _action.tool == "wait":
            return TransferResult(_action)
        return TransferResult(None, reason="omnitransfer_failed")

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=before,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert result.actions_executed == 1
    assert host.actions == [Action("wait", {"duration_ms": 0})]
    assert result.detail["checker_decisions"][0]["status"] == "skipped"
    assert result.detail["checker_decisions"][0]["reason"] == (
        "omnitransfer_failed"
    )


def test_checker_does_not_execute_a_low_confidence_mapping(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(xml="<wrong/>", package_name="com.other")
    source = Observation(xml="<source/>", package_name="com.example")
    host = RecordingHost(current)
    transfer_calls: list[Action] = []

    async def transfer(action, _observation, _source_state):
        transfer_calls.append(action)
        if action.tool == "wait":
            return TransferResult(action)
        return TransferResult(
            action,
            detail={"score": 0.999, "candidates": [{"score": 0.2}]},
        )

    function = Function(
        function_id="scoped_checker",
        name="Scoped checker",
        description="Run a checker only when its registered action maps strongly.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main"),),
        checker_rules=(
            {
                "source_state_id": "checker",
                "action": {"tool": "click", "args": {"x": 100, "y": 200}},
            },
        ),
    )

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert transfer_calls == [
        Action("click", {"x": 100, "y": 200}),
        Action("wait", {"duration_ms": 0}),
    ]
    assert result.detail["checker_decisions"][0]["reason"] == (
        "checker_target_probability_too_low"
    )


def test_low_confidence_checker_is_checked_again_before_the_next_action(
    monkeypatch,
) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    checker_page = Observation(xml="<checker/>", package_name="com.example")
    source = Observation(xml="<source/>", package_name="com.example")
    host = RecordingHost(checker_page)
    checker_action = Action("click", {"x": 100, "y": 200})
    mapped_checker_action = Action("click", {"x": 300, "y": 400})
    checker_calls = 0

    async def transfer(action, _observation, _source_state):
        nonlocal checker_calls
        if action == checker_action:
            checker_calls += 1
            return TransferResult(
                mapped_checker_action,
                detail={
                    "score": 0.999,
                    "candidates": [
                        {"score": 0.2 if checker_calls == 1 else 0.99}
                    ],
                },
            )
        return TransferResult(action)

    function = Function(
        function_id="rechecked_checker",
        name="Rechecked checker",
        description="Keep a skipped checker eligible for the next action.",
        steps=(
            FunctionStep(0, Action("wait", {"duration_ms": 0}), "main-0"),
            FunctionStep(1, Action("wait", {"duration_ms": 0}), "main-1"),
        ),
        checker_rules=(
            {"source_state_id": "checker", "action": checker_action.to_dict()},
        ),
    )

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=checker_page,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert [item["status"] for item in result.detail["checker_decisions"]] == [
        "skipped",
        "executed",
    ]
    assert checker_calls == 2
    assert host.actions.count(mapped_checker_action) == 1


def test_executed_checker_is_not_repeated_when_a_function_resumes(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(xml="<page/>", package_name="com.example")
    source = Observation(xml="<source/>", package_name="com.example")
    host = RecordingHost(current)
    checker_action = Action("click", {"x": 100, "y": 200})
    mapped_checker_action = Action("click", {"x": 300, "y": 400})
    executed_checker_rules: set[int] = set()

    async def transfer(action, _observation, _source_state):
        if action == checker_action:
            return TransferResult(
                mapped_checker_action,
                detail={"candidates": [{"score": 0.99}]},
            )
        return TransferResult(action)

    function = Function(
        function_id="resumed_checker",
        name="Resumed checker",
        description="Do not repeat a checker after a later formal action fails.",
        steps=(
            FunctionStep(0, Action("wait", {"duration_ms": 0}), "main-0"),
            FunctionStep(1, Action("wait", {"duration_ms": 0}), "main-1"),
        ),
        checker_rules=(
            {"source_state_id": "checker", "action": checker_action.to_dict()},
        ),
    )

    first = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
            executed_checker_rules=executed_checker_rules,
        )
    )
    resumed = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=first.final_state,
            start_step_index=1,
            state_loader=lambda _state_id: source,
            executed_checker_rules=executed_checker_rules,
        )
    )

    assert first.success is True
    assert resumed.success is True
    assert executed_checker_rules == {0}
    assert resumed.detail["checker_decisions"] == []
    assert host.actions.count(mapped_checker_action) == 1


def test_checker_uses_high_target_mapping_without_a_page_gate(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(xml="<unrelated/>", package_name="com.example")
    source = Observation(xml="<dialog/>", package_name="com.example")
    host = RecordingHost(current)
    transfer_calls: list[Action] = []
    checker_action = Action("click", {"x": 100, "y": 200})
    mapped_checker_action = Action("click", {"x": 300, "y": 400})

    async def transfer(action, _observation, _source_state):
        transfer_calls.append(action)
        return TransferResult(
            mapped_checker_action if action == checker_action else action,
            detail={"score": 0.999, "candidates": [{"score": 0.99}]},
        )

    function = Function(
        function_id="action_scoped_checker",
        name="Action-scoped checker",
        description="Use only high-confidence action transfer evidence.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main"),),
        checker_rules=(
            {
                "source_state_id": "dialog",
                "action": checker_action.to_dict(),
            },
        ),
    )

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert transfer_calls == [
        Action("click", {"x": 100, "y": 200}),
        Action("wait", {"duration_ms": 0}),
    ]
    assert host.actions == [mapped_checker_action, Action("wait", {"duration_ms": 0})]
    assert result.detail["checker_decisions"][0]["status"] == "executed"
    assert "page" not in result.detail["checker_decisions"][0]


def test_blocking_overlay_dismisses_android_legacy_app_dialog_only() -> None:
    dialog = Observation(
        xml=(
            '<hierarchy><node class="android.widget.FrameLayout" bounds="[0,0][720,1280]">'
            '<node resource-id="android:id/parentPanel" class="android.widget.LinearLayout" '
            'bounds="[50,452][670,827]">'
            '<node resource-id="android:id/alertTitle" text="Clipper" '
            'bounds="[98,488][622,542]" />'
            '<node text="This app was built for an older version of Android and may not work properly." '
            'bounds="[60,400][660,700]" />'
            '<node text="OK" resource-id="android:id/button1" clickable="true" '
            'bounds="[540,700][640,800]" />'
            "</node></node></hierarchy>"
        )
    )
    ordinary_ok = Observation(
        xml=(
            '<hierarchy><node class="android.app.Dialog" bounds="[0,0][720,1280]">'
            '<node text="Delete this item?" bounds="[60,400][660,700]" />'
            '<node text="OK" clickable="true" bounds="[540,700][640,800]" />'
            "</node></hierarchy>"
        )
    )

    assert execution._blocking_overlay_action(dialog) == Action(
        "click", {"x": 590.0, "y": 750.0}
    )
    assert execution._blocking_overlay_action(ordinary_ok) is None


def test_checker_uses_target_rank_probability_not_pair_confidence(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(xml="<wrong/>", package_name="com.other")
    source = Observation(xml="<source/>", package_name="com.example")
    host = RecordingHost(current)

    async def transfer(action, _observation, _source_state):
        if action.tool == "wait":
            return TransferResult(action)
        return TransferResult(
            action,
            detail={
                "score": 0.99999,
                "candidates": [
                    {"score": 0.338},
                    {"score": 0.329},
                ],
            },
        )

    function = Function(
        function_id="strict_target_checker",
        name="Strict target checker",
        description="Reject an ambiguous target despite strong pair alignment.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main"),),
        checker_rules=(
            {
                "source_state_id": "checker",
                "action": {"tool": "click", "args": {"x": 100, "y": 200}},
            },
        ),
    )

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert host.actions == [Action("wait", {"duration_ms": 0})]
    decision = result.detail["checker_decisions"][0]
    assert decision["reason"] == "checker_target_probability_too_low"
    assert decision["target"] == {
        "probability": 0.338,
        "minimum_probability": 0.9,
    }


def test_function_without_registered_checker_executes_no_checker(monkeypatch) -> None:
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    current = Observation(xml="<page/>", package_name="com.example")
    host = RecordingHost(current)
    function = Function(
        function_id="no_checker",
        name="No checker",
        description="A Function receives only its own registered checkers.",
        steps=(FunctionStep(0, Action("wait", {"duration_ms": 0}), "main"),),
        checker_rules=(),
    )

    async def transfer(action, _observation, _source_state):
        return TransferResult(action)

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
        )
    )

    assert result.success is True
    assert result.detail["checker_decisions"] == []
    assert result.execution_summary["checker_decisions"] == []
    assert host.actions == [Action("wait", {"duration_ms": 0})]


def test_function_action_relies_on_omnitransfer_without_a_page_gate() -> None:
    current = Observation(xml="<wrong-page/>", package_name="com.example")
    source = Observation(xml="<source-page/>", package_name="com.example")
    host = RecordingHost(current)
    function = Function(
        function_id="page_bound_action",
        name="Page-bound action",
        description="Execute one action through OmniTransfer target mapping.",
        steps=(
            FunctionStep(
                0,
                Action("click", {"x": 100, "y": 200}),
                "source-page",
            ),
        ),
    )

    source_action = Action("click", {"x": 100, "y": 200})
    mapped_action = Action("click", {"x": 300, "y": 400})
    transfer_calls: list[Action] = []

    async def transfer(action, _observation, _source_state):
        transfer_calls.append(action)
        return TransferResult(
            mapped_action,
            detail={"candidates": [{"score": 0.99}]},
        )

    result = asyncio.run(
        execution.execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=transfer),
            observation=current,
            state_loader=lambda _state_id: source,
        )
    )

    assert result.success is True
    assert result.actions_executed == 1
    assert transfer_calls == [source_action]
    assert host.actions == [mapped_action]


def test_checker_rules_reject_actions_without_transferable_targets() -> None:
    from omniflow.runtime.checker import validate_checker_rule

    try:
        validate_checker_rule(
            {
                "source_state_id": "drawer-source",
                "action": {
                    "tool": "swipe",
                    "args": {
                        "direction": "up",
                        "x1": 540,
                        "y1": 900,
                        "x2": 540,
                        "y2": 400,
                    },
                },
            }
        )
    except ValueError as error:
        assert str(error) == "checker_action_requires_transfer_target:swipe"
    else:
        raise AssertionError("unanchored checker action must be rejected")


def test_checker_policy_uses_explicit_if_then_navigation() -> None:
    from omniflow.runtime.checker import validate_checker_rule

    rule = validate_checker_rule(
        {
            "source_state_id": "app-source",
            "if": {"package_name": "com.android.settings"},
            "then": {
                "tool": "open_app",
                "args": {"package_name": "com.android.settings"},
            },
        }
    )

    assert rule == {
        "source_state_id": "app-source",
        "policy": {
            "if": {"package_name": "com.android.settings"},
            "then": {
                "tool": "open_app",
                "args": {"package_name": "com.android.settings"},
            },
        },
    }


def test_checker_rule_contains_only_source_state_and_source_action() -> None:
    from omniflow.runtime.checker import validate_checker_rule

    rule = {
        "source_state_id": "dialog-source",
        "action": {"tool": "click", "args": {"x": 100, "y": 200}},
    }

    assert validate_checker_rule(rule) == rule

    try:
        validate_checker_rule({**rule, "when": {"step": 3}})
    except ValueError as error:
        assert str(error) == "checker_rule_contract_invalid"
    else:
        raise AssertionError("checker rules must not contain trigger DSL")
