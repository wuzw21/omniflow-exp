from __future__ import annotations

import asyncio

from omniflow.core.config import OmniFlowConfig, PluginSet, RuntimeSettings
from omniflow.core.model import (
    Action,
    ActionResult,
    Observation,
    ToolCall,
    TransferResult,
)
from omniflow.functions.artifact import parse_function_artifact
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import OmniFlow
from omniflow.runtime.checker import CheckerLibrary, checker_rule_matches
from omniflow.runtime.execution import _transfer_page_package, execute_robust_action


def _observation(name: str, *, keyboard_visible: bool = False) -> Observation:
    keyboard = (
        '<node package="com.google.android.inputmethod.latin" '
        'resource-id="key_pos_0" bounds="[0,700][720,1280]" />'
        if keyboard_visible
        else ""
    )
    return Observation(
        xml=(
            '<hierarchy width="720" height="1280">'
            f'<node text="{name}" bounds="[0,0][720,1280]" />{keyboard}'
            "</hierarchy>"
        ),
        package_name="net.gsantner.markor",
        extra={
            "display": {"width": 720, "height": 1280},
            **({"keyboard_visible": True} if keyboard_visible else {}),
        },
    )


def _obscured_source_observation() -> Observation:
    return Observation(
        xml=(
            '<hierarchy width="720" height="1280">'
            '<node package="net.gsantner.markor" resource-id="save" '
            'bounds="[300,760][420,840]" clickable="true" />'
            '<node package="com.google.android.inputmethod.latin" '
            'resource-id="key_pos_0" bounds="[0,700][720,1280]" />'
            "</hierarchy>"
        ),
        package_name="net.gsantner.markor",
        extra={"display": {"width": 720, "height": 1280}},
    )


def test_transfer_identity_uses_app_under_ime_overlay() -> None:
    source = Observation(
        xml=(
            '<hierarchy width="720" height="1280">'
            '<node package="net.gsantner.markor" bounds="[0,0][720,700]" />'
            '<node package="com.google.android.inputmethod.latin" '
            'bounds="[0,700][720,1280]" />'
            "</hierarchy>"
        ),
        package_name="com.google.android.inputmethod.latin",
        activity_name="android.inputmethodservice.SoftInputWindow",
    )
    target = Observation(
        xml=(
            '<hierarchy width="720" height="1280">'
            '<node package="net.gsantner.markor" bounds="[0,0][720,1280]" />'
            "</hierarchy>"
        ),
        package_name="net.gsantner.markor",
    )

    assert _transfer_page_package(source) == "net.gsantner.markor"
    assert _transfer_page_package(target) == "net.gsantner.markor"


class _CheckerHost:
    def __init__(self) -> None:
        self.state = "keyboard"
        self.actions: list[Action] = []
        self.observations = 0

    async def observe(self, **_: object) -> Observation:
        self.observations += 1
        return _observation(self.state, keyboard_visible=self.state == "keyboard")

    async def get_state(self, _state_id: str) -> Observation:
        return _obscured_source_observation()

    async def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        if action == Action("press_key", {"key": "back"}):
            self.state = "dialog"
        return ActionResult(True)


def test_plugin_checker_retries_original_action_after_fresh_observation() -> None:
    host = _CheckerHost()

    result = asyncio.run(
        execute_robust_action(
            Action("wait", {"duration_ms": 1}),
            observation=_observation("keyboard", keyboard_visible=True),
            host=host,
            plugins=PluginSet(
                checker=lambda context: (
                    Action("press_key", {"key": "back"})
                    if context.current.extra.get("keyboard_visible") is True
                    else None
                )
            ),
        )
    )

    assert result.success is True
    assert result.origin == "action"
    assert result.after == _observation("dialog")
    assert host.actions == [
        Action("press_key", {"key": "back"}),
        Action("wait", {"duration_ms": 1}),
    ]
    assert result.actions_executed == 2
    assert host.observations >= 1


def test_planner_receives_noop_action_result_before_next_decision(tmp_path) -> None:
    class Host:
        def __init__(self) -> None:
            self.actions: list[Action] = []

        async def observe(self, **_: object) -> Observation:
            return _observation("unchanged")

        async def act(self, action: Action) -> ActionResult:
            self.actions.append(action)
            return ActionResult(True)

    class Planner:
        def __init__(self) -> None:
            self.observations: list[Observation] = []

        async def one_step_tool_call(self, _goal: str, observation: Observation, *_: object) -> ToolCall:
            self.observations.append(observation)
            if len(self.observations) == 1:
                return ToolCall("click", {"x": 500, "y": 500})
            return ToolCall("finished", {"content": "task completed"})

    store_path = tmp_path / "store.json"
    FunctionStore(store_path).save()
    host = Host()
    planner = Planner()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_steps=3),
            plugins=PluginSet(transfer=lambda action, *_: TransferResult(action)),
        ),
    )

    result = asyncio.run(flow.arun("Complete the requested change."))

    assert result.success is True
    assert host.actions == [Action("click", {"x": 500, "y": 500})]
    assert planner.observations[1].extra["previous_action_error"] == (
        "action_completed_without_observed_state_change"
    )


def test_shared_checker_runs_before_function_action_without_planner_handoff(tmp_path) -> None:
    from omniflow.runtime.execution import execute_function

    host = _CheckerHost()
    function = parse_function_artifact(
        {
            "schema_version": "omniflow.function.v2",
            "function_id": "shared_checker_function",
            "name": "Shared checker function",
            "description": "Perform the requested action.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "bindings": [],
            "render_bindings": [],
            "steps": [
                {
                    "step_index": 0,
                    "source_state_id": "state-1",
                    "action": {"tool": "click", "args": {"x": 500, "y": 625}},
                }
            ],
            "agent_visible": True,
        }
    )

    result = asyncio.run(
        execute_function(
            function,
            host=host,
            plugins=PluginSet(transfer=lambda action, *_: TransferResult(action)),
            observation=_observation("keyboard", keyboard_visible=True),
            state_loader=lambda _state_id: _observation("source"),
            checker_rules=(
                {
                    "schema_version": "omniflow.checker_rule.v1",
                    "id": "hide_keyboard_test",
                    "enabled": True,
                    "phase": "pre_action",
                    "condition": {"keyboard_obscuring": True},
                    "action": {"action": "hide_keyboard"},
                    "budget": {"max_triggers_per_run": 1},
                },
            ),
        )
    )

    assert result.success is True
    assert host.actions == [
        Action("press_key", {"key": "back"}),
        Action("click", {"x": 500, "y": 625}),
    ]


def test_shared_checker_continues_function_without_planner_handoff(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    store = FunctionStore(store_path)
    store.put_function(
        parse_function_artifact(
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "continue_after_recovery",
                "name": "Continue after recovery",
                "description": "Complete the requested change.",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "bindings": [],
                "render_bindings": [],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "state-1",
                        "action": {"tool": "click", "args": {"x": 500, "y": 625}},
                    }
                ],
                "agent_visible": True,
            }
        )
    )

    class Planner:
        def __init__(self) -> None:
            self.observations: list[Observation] = []

        async def one_step_tool_call(self, _goal: str, observation: Observation, *_: object) -> ToolCall:
            self.observations.append(observation)
            if len(self.observations) == 1:
                return ToolCall("continue_after_recovery", {})
            return ToolCall("finished", {"content": "task completed"})

    host = _CheckerHost()
    planner = Planner()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_steps=4),
            plugins=PluginSet(transfer=lambda action, *_: TransferResult(action)),
        ),
    )

    result = asyncio.run(flow.arun("Complete the requested change."))

    assert result.success is True
    assert host.actions == [
        Action("press_key", {"key": "back"}),
        Action("click", {"x": 500, "y": 625}),
    ]
    assert len(planner.observations) == 2
    history = str(planner.observations[1].extra["execution_history"])
    assert "Action: continue_after_recovery | result: succeeded" in history


def test_keyboard_checker_does_not_override_planner_action(tmp_path) -> None:
    class Planner:
        def __init__(self) -> None:
            self.calls = 0

        async def one_step_tool_call(self, *_: object) -> ToolCall:
            self.calls += 1
            if self.calls == 1:
                return ToolCall("wait", {"duration_ms": 1})
            return ToolCall("finished", {"content": "task completed"})

    store_path = tmp_path / "store.json"
    FunctionStore(store_path).save()
    host = _CheckerHost()
    planner = Planner()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    )
    flow.checker_library = CheckerLibrary(
        (
            {
                "schema_version": "omniflow.checker_rule.v1",
                "id": "hide_keyboard_before_planner_action",
                "enabled": True,
                "phase": "pre_action",
                "condition": {"keyboard_obscuring": True},
                "action": {"action": "hide_keyboard"},
                "budget": {"max_triggers_per_run": 1},
            },
        )
    )

    result = asyncio.run(flow.arun("Wait for the requested UI."))

    assert result.success is True
    assert planner.calls == 2
    assert host.actions == [Action("wait", {"duration_ms": 1})]


def test_shared_checker_respects_run_budget_after_fresh_observations() -> None:
    host = _CheckerHost()
    rules = (
        {
            "schema_version": "omniflow.checker_rule.v1",
            "id": "hide_keyboard_once",
            "enabled": True,
            "phase": "pre_action",
            "condition": {"keyboard_obscuring": True},
            "action": {"action": "hide_keyboard"},
            "budget": {"max_triggers_per_run": 1},
        },
    )
    trigger_counts: dict[str, int] = {}

    first = asyncio.run(
        execute_robust_action(
            Action("click", {"x": 500, "y": 625}),
            observation=_observation("keyboard", keyboard_visible=True),
            host=host,
            plugins=PluginSet(transfer=lambda action, *_: TransferResult(action)),
            checker_rules=rules,
            checker_trigger_counts=trigger_counts,
            source_state=_obscured_source_observation(),
        )
    )
    second = asyncio.run(
        execute_robust_action(
            Action("click", {"x": 500, "y": 625}),
            observation=_observation("keyboard", keyboard_visible=True),
            host=host,
            plugins=PluginSet(transfer=lambda action, *_: TransferResult(action)),
            checker_rules=rules,
            checker_trigger_counts=trigger_counts,
            source_state=_obscured_source_observation(),
        )
    )

    assert first.success is True
    assert second.success is True
    assert trigger_counts == {"hide_keyboard_once": 1}
    assert host.actions == [
        Action("press_key", {"key": "back"}),
        Action("click", {"x": 500, "y": 625}),
        Action("click", {"x": 500, "y": 625}),
    ]


def test_keyboard_checker_requires_current_ime_and_function_source_target() -> None:
    rule = {
        "schema_version": "omniflow.checker_rule.v1",
        "id": "hide_keyboard_safely",
        "enabled": True,
        "phase": "pre_action",
        "condition": {"keyboard_obscuring": True},
        "action": {"action": "hide_keyboard"},
    }
    action = Action("input_text", {"x": 500, "y": 625, "text": "name"})
    source = _obscured_source_observation()

    assert checker_rule_matches(
        rule,
        current=_observation("keyboard", keyboard_visible=True),
        source=source,
        function_id="enter_name",
        step_index=0,
        action=action,
    )
    assert not checker_rule_matches(
        rule,
        current=_observation("dialog"),
        source=source,
        function_id="enter_name",
        step_index=0,
        action=action,
    )
    assert not checker_rule_matches(
        rule,
        current=_observation("keyboard", keyboard_visible=True),
        source=None,
        function_id="",
        step_index=0,
        action=action,
    )
