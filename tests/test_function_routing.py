from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from runlog_fixtures import androidworld_state

from omniflow import (
    Action,
    ActionResult,
    Function,
    Observation,
    OmniFlow,
    ToolCall,
)
from omniflow.core.config import OmniFlowConfig, PluginSet, RuntimeSettings
from omniflow.core.model import FunctionStep, TransferResult
from omniflow.core.trajectory import state_id
from omniflow.functions.assets import FUNCTION_ARTIFACT_VERSION, FunctionStore
from omniflow.vlm.planner import (
    SYSTEM_PROMPT,
    VLMPlanner,
    adapt_tool_arguments,
    build_model_turn_request,
    function_tools,
    parse_model_turn_response,
)
from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm_coordinates import canonical_action_to_screen_pixels
from src.integrations.android_world import launch as androidworld_launch
from src.integrations.android_world.agent import (
    _TaskHost,
    build_agent,
)


class RecordingHost:
    def __init__(self) -> None:
        self.package_name = "com.android.launcher"
        self.actions: list[Action] = []
        self.observe_requests: list[dict[str, object]] = []

    def observe(self, **kwargs: object) -> Observation:
        self.observe_requests.append(dict(kwargs))
        return Observation(
            package_name=self.package_name,
            activity_name="MainActivity",
            image_base64=(
                "final-screenshot" if kwargs.get("screenshot") is True else None
            ),
            extra={"state_id": f"state_{len(self.actions)}"},
        )

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        if action.tool == "open_app":
            self.package_name = str(action.args["package_name"])
        return ActionResult(True)

    def get_state(self, _source_state_id: str) -> None:
        return None


class ResumableHost(RecordingHost):
    def observe(self, **kwargs: object) -> Observation:
        self.observe_requests.append(dict(kwargs))
        state = f"state_{len(self.actions)}"
        package = self.package_name
        xml = (
            '<hierarchy><node package="%s" class="android.widget.TextView" '
            'text="Target" bounds="[0,0][1000,1000]" clickable="true" /></hierarchy>'
            % package
        )
        return Observation(
            xml=xml,
            package_name=package,
            activity_name="MainActivity",
            image_base64=(
                "final-screenshot" if kwargs.get("screenshot") is True else None
            ),
            extra={"state_id": state, "display": {"width": 1000, "height": 1000}},
        )

    def get_state(self, source_state_id: str) -> Observation:
        package = (
            "com.android.launcher"
            if source_state_id == "source_home"
            else "com.android.settings"
        )
        xml = (
            '<hierarchy><node package="%s" class="android.widget.TextView" '
            'text="Target" bounds="[0,0][1000,1000]" clickable="true" /></hierarchy>'
            % package
        )
        return Observation(
            xml=xml,
            package_name=package,
            activity_name="MainActivity",
            extra={"state_id": source_state_id, "display": {"width": 1000, "height": 1000}},
        )


def test_androidworld_host_keeps_the_captured_transfer_state() -> None:
    official_state = androidworld_state(
        "ignored-derived-id",
        forest={"source": "official"},
        ui_elements=[{"text": "Settings"}],
        with_pixels=True,
    )
    official_state["auxiliaries"].pop("state_id")
    identifier = state_id(official_state)
    raw_host = SimpleNamespace(
        observe=lambda **_: Observation(
            xml="<hierarchy />",
            package_name="com.android.settings",
            extra={
                "androidworld_state": official_state,
                "display": {"width": 1000, "height": 1000},
            },
        ),
        act=lambda action: ActionResult(True),
    )
    runtime_state = {
        "captured_transfer_states": {},
    }
    host = _TaskHost(raw_host, runtime_state, {})

    observation = host.observe(xml=True, screenshot=True, app_info=True)

    assert observation.extra["state_id"] == identifier
    assert set(runtime_state["captured_transfer_states"]) == {identifier}


class FinishingPlanner:
    def __init__(self) -> None:
        self.visible_function_ids: list[tuple[str, ...]] = []
        self.observations: list[Observation] = []

    def one_step_tool_call(
        self,
        _goal: str,
        _observation: Observation,
        functions: tuple[Function, ...],
        _installed_apps: dict[str, str],
    ) -> ToolCall:
        self.visible_function_ids.append(tuple(function.id for function in functions))
        self.observations.append(_observation)
        return ToolCall("finished", {"content": ""})


class SequencePlanner(FinishingPlanner):
    def __init__(self, responses: list[ToolCall]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.goals: list[str] = []
        self.previous_action_errors: list[str | None] = []

    def one_step_tool_call(
        self,
        _goal: str,
        observation: Observation,
        functions: tuple[Function, ...],
        _installed_apps: dict[str, str],
    ) -> ToolCall:
        self.goals.append(_goal)
        self.visible_function_ids.append(tuple(function.id for function in functions))
        self.observations.append(observation)
        self.previous_action_errors.append(
            observation.extra.get("previous_action_error")
        )
        return self.responses.pop(0)


def _store_with_open_settings_function(path: object) -> str:
    function = Function(
        function_id="complete_run_turn_bluetooth_on",
        name="Turn bluetooth on",
        description="Complete the exact goal: turn bluetooth on.",
        steps=(
            FunctionStep(
                step_index=0,
                source_state_id="source_home",
                action=Action(
                    "open_app",
                    {"package_name": "com.android.settings"},
                ),
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        agent_visible=True,
    )
    store = FunctionStore(path)
    store.put_function(function)
    return function.id


def test_function_tools_preserve_router_order_without_visibility_filter() -> None:
    def function(function_id: str, *, agent_visible: bool) -> Function:
        return Function(
            function_id=function_id,
            name=function_id,
            description=function_id,
            steps=(
                FunctionStep(
                    step_index=0,
                    source_state_id=f"source_{function_id}",
                    action=Action("press_key", {"key": "back"}),
                ),
            ),
            schema_version=FUNCTION_ARTIFACT_VERSION,
            input_schema={
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            agent_visible=agent_visible,
        )

    ranked = (
        function("z_top_ranked", agent_visible=False),
        function("a_second_ranked", agent_visible=True),
    )

    tools = function_tools(ranked, include_summary=False)

    assert [tool["function"]["name"] for tool in tools] == [
        "z_top_ranked",
        "a_second_ranked",
    ]


def _store_with_long_function(path: object) -> str:
    function = Function(
        function_id="complete_run_long_function",
        name="Complete a long replay",
        description="Execute every recorded Function action before planning.",
        steps=tuple(
            FunctionStep(
                step_index=step_index,
                source_state_id=f"source_{step_index}",
                action=Action("press_key", {"key": "back"}),
            )
            for step_index in range(3)
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        agent_visible=True,
    )
    store = FunctionStore(path)
    store.put_function(function)
    return function.id


def _store_with_untransferable_click_function(path: object) -> str:
    function = Function(
        function_id="complete_run_untransferable_click",
        name="Turn bluetooth on",
        description="Complete the exact goal: turn bluetooth on.",
        steps=(
            FunctionStep(
                step_index=0,
                source_state_id="missing_source_state",
                action=Action("click", {"x": 500.0, "y": 500.0}),
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        agent_visible=True,
    )
    store = FunctionStore(path)
    store.put_function(function)
    return function.id


def _store_with_resumable_click_function(path: object) -> str:
    function = Function(
        function_id="complete_run_resumable_click",
        name="Open settings and choose the target",
        description="Open Settings and choose the requested target.",
        steps=(
            FunctionStep(
                step_index=0,
                source_state_id="source_home",
                action=Action(
                    "open_app",
                    {"package_name": "com.android.settings"},
                ),
            ),
            FunctionStep(
                step_index=1,
                source_state_id="source_target_page",
                action=Action("click", {"x": 500.0, "y": 500.0}),
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        agent_visible=True,
    )
    store = FunctionStore(path)
    store.put_function(function)
    return function.id


class ResumableTransfer:
    def __init__(self) -> None:
        self.target_state_ids: list[str] = []

    def __call__(
        self,
        action: Action,
        observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        target_state_id = str(observation.extra.get("state_id") or "")
        self.target_state_ids.append(target_state_id)
        if target_state_id == "state_2":
            return TransferResult(
                Action(action.tool, {"x": 700.0, "y": 700.0}),
                reason="omnitransfer_test_match",
                detail={"score": 0.9},
            )
        return TransferResult(
            None,
            reason="omnitransfer_test_unaligned",
            detail={"score": 0.1},
        )


class CompletionRecoveryTransfer:
    def __call__(
        self,
        action: Action,
        observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        target_state_id = str(observation.extra.get("state_id") or "")
        if target_state_id == "state_3":
            return TransferResult(
                Action(action.tool, {"x": 700.0, "y": 700.0}),
                reason="omnitransfer_test_match",
                detail={"score": 0.9},
            )
        return TransferResult(
            Action(action.tool, dict(action.args)),
            reason="omnitransfer_test_initial_match",
            detail={"score": 0.9},
        )


def test_planner_selects_recalled_function_as_one_peer_tool(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_open_settings_function(store_path)
    host = RecordingHost()
    planner = SequencePlanner(
        [
            ToolCall(function_id, {}),
            ToolCall("finished", {"content": ""}),
        ]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == ["open_app"]
    assert planner.visible_function_ids == [(function_id,), (function_id,)]
    assert planner.observations[0].image_base64 == "final-screenshot"
    assert [request.get("screenshot") for request in host.observe_requests] == [
        False,
        True,
        True,
    ]
    assert planner.observations[1].extra["function_execution"] == {
        "schema_version": "omniflow.function-execution-evidence.v1",
        "function_id": function_id,
        "function_name": "Turn bluetooth on",
        "function_description": "Complete the exact goal: turn bluetooth on.",
        "replay_status": "actions_succeeded",
        "official_validator_status": "pending",
        "steps": [
            {
                "step_index": 0,
                "before_state_id": "state_0",
                "after_state_id": "state_1",
                "tool": "open_app",
                "success": True,
            }
        ],
        "final_observation": {
            "state_id": "state_1",
            "package_name": "com.android.settings",
            "activity_name": "MainActivity",
        },
    }
    function_resolution = result.detail["function_resolution"]
    assert function_resolution["candidate_count"] == 1
    assert function_resolution["candidate_function_ids"] == [function_id]
    assert function_resolution["status"] == "planner_tool_space"
    assert [
        event["candidate_function_ids"]
        for event in function_resolution["recall"]["events"]
    ] == [[function_id], [function_id]]
    assert result.detail["runtime_limits"] == {
        "max_steps": 20,
        "max_fallback_steps": None,
    }


def test_zero_fallback_budget_allows_task_planning_after_function(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_open_settings_function(store_path)
    host = RecordingHost()
    planner = SequencePlanner(
        [ToolCall(function_id, {}), ToolCall("finished", {"content": ""})]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_steps=20, max_fallback_steps=0),
        ),
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert result.error is None
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == ["open_app"]
    assert planner.visible_function_ids == [(function_id,), (function_id,)]
    assert result.fallback_steps == 0
    assert result.detail["completion_review_calls"] == 0
    assert result.execution_summary["completion_review_calls"] == 0
    assert result.detail["function_resolution"]["status"] == "planner_tool_space"
    assert result.detail["runtime_limits"] == {
        "max_steps": 20,
        "max_fallback_steps": 0,
    }


def test_max_steps_limits_planner_calls_not_function_actions(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_long_function(store_path)
    host = RecordingHost()
    planner = FinishingPlanner()
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=1)),
    )

    result = flow.call_tool(ToolCall(function_id, {}))

    assert result.success is True
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == [
        "press_key",
        "press_key",
        "press_key",
    ]
    assert planner.visible_function_ids == []
    assert result.actions_executed == 3
    assert result.detail["runtime_limits"]["max_steps"] == 1


def test_task_planner_receives_recalled_functions(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    _store_with_open_settings_function(store_path)
    planner = FinishingPlanner()
    flow = OmniFlow(
        store_path,
        host=RecordingHost(),
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert result.function_id is None
    assert planner.visible_function_ids == [
        ("complete_run_turn_bluetooth_on",),
    ]


def test_transfer_failure_falls_back_without_replaying_source_coordinates(
    tmp_path,
) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_untransferable_click_function(store_path)
    host = RecordingHost()
    planner = SequencePlanner(
        [
            ToolCall("open_app", {"package_name": "com.android.settings"}),
            ToolCall("finished", {"content": ""}),
        ]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.call_tool(ToolCall(function_id, {}))

    assert result.success is True
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == ["open_app"]
    assert all(action.tool != "click" for action in host.actions)
    assert planner.visible_function_ids == [(function_id,), (function_id,)]
    assert planner.previous_action_errors[0] == "omnitransfer_missing_target_page"
    assert planner.observations[0].extra["function_execution"] == {
        "schema_version": "omniflow.function-execution-evidence.v1",
        "function_id": function_id,
        "function_name": "Turn bluetooth on",
        "function_description": "Complete the exact goal: turn bluetooth on.",
        "replay_status": "actions_failed",
        "official_validator_status": "pending",
        "steps": [
            {
                "step_index": 0,
                "before_state_id": "state_0",
                "after_state_id": "state_0",
                "tool": "click",
                "success": False,
                "error": "omnitransfer_missing_target_page",
            }
        ],
        "final_observation": {
            "state_id": "state_0",
            "package_name": "com.android.launcher",
            "activity_name": "MainActivity",
        },
    }


def test_direct_function_transfer_failure_continues_with_gui_planner(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_untransferable_click_function(store_path)
    host = RecordingHost()
    planner = SequencePlanner(
        [
            ToolCall("open_app", {"package_name": "com.android.settings"}),
            ToolCall("finished", {"content": ""}),
        ]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.call_tool(ToolCall(function_id, {}))

    assert result.success is True
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == ["open_app"]
    assert all(action.tool != "click" for action in host.actions)
    assert planner.visible_function_ids == [(function_id,), (function_id,)]
    assert planner.previous_action_errors[0] == "omnitransfer_missing_target_page"
    assert "Continue Function" in planner.goals[0]
    assert "Do not repeat actions that already succeeded" in planner.goals[0]
    assert result.detail["function_resolution"]["status"] == "direct"
    assert result.detail["function_resolution"]["replay_status"] == "failed"


def test_function_failure_retries_failed_step_only_after_explicit_function_call(
    tmp_path,
) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_resumable_click_function(store_path)
    host = ResumableHost()
    transfer = ResumableTransfer()
    planner = SequencePlanner(
        [
            ToolCall(function_id, {}),
            ToolCall("click", {"x": 100.0, "y": 100.0}),
            ToolCall(function_id, {}),
            ToolCall("finished", {"content": ""}),
        ]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_steps=10, max_fallback_steps=5),
            plugins=PluginSet(transfer=transfer),
        ),
    )

    result = flow.run("Open settings and choose the target")

    assert result.success is True
    assert result.function_id == function_id
    assert result.fallback_steps == 2
    assert [action.to_dict() for action in host.actions] == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.android.settings"},
        },
        {"tool": "click", "args": {"x": 100.0, "y": 100.0}},
        {"tool": "click", "args": {"x": 700.0, "y": 700.0}},
    ]
    assert planner.previous_action_errors[1] == "omnitransfer_test_unaligned"
    assert planner.observations[1].extra["function_execution"]["replay_status"] == (
        "actions_failed"
    )
    retried_steps = [
        step
        for step in result.detail["trace"]
        if step.get("metadata", {}).get("function_alignment")
    ]
    assert len(retried_steps) == 1
    assert retried_steps[0]["metadata"]["function_alignment"] == {
        "protocol": "explicit_function_retry_v1",
        "start_step_index": 1,
        "resume_step_index": 1,
        "source_state_id": "source_target_page",
    }
    assert result.detail["function_resume"] == {
        "schema_version": "omniflow.function-resume-audit.v1",
        "events": [
            {
                "start_step_index": 1,
                "status": "succeeded",
                "trigger": "explicit_function_call",
                "resume_step_index": 1,
                "source_state_id": "source_target_page",
            }
        ],
        "attempt_count": 1,
        "success_count": 1,
    }
    assert result.execution_summary["fallback_steps"] == 2


def test_planner_recovery_does_not_automatically_resume_function(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_resumable_click_function(store_path)
    host = ResumableHost()
    planner = SequencePlanner(
        [
            ToolCall(function_id, {}),
            ToolCall("click", {"x": 100.0, "y": 100.0}),
            ToolCall("finished", {"content": ""}),
        ]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_steps=10, max_fallback_steps=5),
            plugins=PluginSet(transfer=ResumableTransfer()),
        ),
    )

    result = flow.run("Open settings and choose the target")

    assert result.success is True
    assert [action.to_dict() for action in host.actions] == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.android.settings"},
        },
        {"tool": "click", "args": {"x": 100.0, "y": 100.0}},
    ]
    assert "function_resume" not in result.detail


def test_successful_function_can_be_called_again_without_resume(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_resumable_click_function(store_path)
    host = ResumableHost()
    planner = SequencePlanner(
        [
            ToolCall(function_id, {}),
            ToolCall(function_id, {}),
            ToolCall("finished", {"content": ""}),
        ]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_steps=10, max_fallback_steps=5),
            plugins=PluginSet(transfer=CompletionRecoveryTransfer()),
        ),
    )

    result = flow.run("Open settings and choose the target")

    assert result.success is True
    assert result.function_id == function_id
    assert result.fallback_steps == 0
    assert [action.to_dict() for action in host.actions] == [
        {
            "tool": "open_app",
            "args": {"package_name": "com.android.settings"},
        },
        {"tool": "click", "args": {"x": 500.0, "y": 500.0}},
        {
            "tool": "open_app",
            "args": {"package_name": "com.android.settings"},
        },
        {"tool": "click", "args": {"x": 700.0, "y": 700.0}},
    ]
    assert planner.visible_function_ids == [
        (function_id,),
        (function_id,),
        (function_id,),
    ]
    assert "function_resume" not in result.detail
    assert result.detail["completion_review_calls"] == 0


def test_planner_can_repeat_action_on_same_logical_ui_state(
    tmp_path,
    monkeypatch,
) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    host = RecordingHost()
    repeated_click = ToolCall("click", {"x": 120, "y": 240})
    planner = SequencePlanner(
        [repeated_click, repeated_click, ToolCall("finished", {})]
    )
    flow = OmniFlow(tmp_path / "store.json", host=host, planner=planner)

    result = flow.run("Add this item to the cart")

    assert result.success is True
    assert host.actions == [
        Action("click", {"x": 120, "y": 240}),
        Action("click", {"x": 120, "y": 240}),
    ]
    assert planner.previous_action_errors[0] is None
    assert "action_already_succeeded_on_current_state" not in planner.previous_action_errors
    assert len(planner.previous_action_errors) <= 3


class CapturingCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return self.response


def test_qwen_adapter_preserves_normalized_scalar_coordinates() -> None:
    adapted, metadata = adapt_tool_arguments(
        tool="click",
        arguments={"x": 876, "y": 869},
        requested_model="qwen3-vl-plus",
        resolved_model="qwen3-vl-plus",
        display={"width": 720, "height": 1280},
    )

    assert adapted == {"x": 876, "y": 869}
    assert metadata is None


def test_planner_exposes_normalized_coordinates_independent_of_display() -> None:
    request = build_model_turn_request(
        goal="Tap add",
        model="test-model",
        state={"display": {"width": 720, "height": 1280}},
        max_steps=8,
        turn_index=1,
    )

    click_tool = next(
        tool for tool in request["tools"] if tool["function"]["name"] == "click"
    )
    properties = click_tool["function"]["parameters"]["properties"]
    for field in ("x", "y"):
        assert properties[field]["minimum"] == 0
        assert properties[field]["maximum"] == 1000
        assert "0..1000 relative" in properties[field]["description"]
        assert "Raw" not in properties[field]["description"]


def test_planner_parses_normalized_coordinates_without_conversion() -> None:
    tool_call, metadata = parse_model_turn_response(
        {
            "requested_model": "test-model",
            "resolved_model": "test-model",
            "tool_calls": [
                {
                    "function": {
                        "name": "click",
                        "arguments": json.dumps(
                            {"summary": "Tap add", "x": 876, "y": 869}
                        ),
                    }
                }
            ],
        },
        requested_model="test-model",
        turn_index=1,
        display={"width": 720, "height": 1280},
    )

    assert tool_call == ToolCall("click", {"x": 876, "y": 869})
    assert metadata == {"summary": "Tap add"}


def test_runtime_maps_normalized_coordinates_to_screen_pixels() -> None:
    assert canonical_action_to_screen_pixels(
        {"tool": "click", "args": {"x": 876, "y": 869}},
        {"width": 720, "height": 1280},
    ) == {
        "tool": "click",
        "args": {"x": pytest.approx(630.72), "y": pytest.approx(1112.32)},
    }


def test_vlm_planner_exposes_packages_only_through_open_app_tool() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="finished",
                                arguments='{"summary":"Goal complete"}',
                            )
                        )
                    ]
                )
            )
        ],
        usage=None,
    )
    completions = CapturingCompletions(response)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    planner = VLMPlanner(model="test-model", client=client)
    installed_apps = {
        "Chrome": "com.android.chrome",
        "Settings": "com.android.settings",
    }

    planned = asyncio.run(
        planner.one_step_tool_call(
            "Turn bluetooth on",
            Observation(
                extra={
                    "display": {"width": 720, "height": 1280},
                    "installed_apps": installed_apps,
                }
            ),
            (),
            installed_apps,
        )
    )

    assert planned == ToolCall("finished", {})
    request = completions.requests[0]
    message_text = request["messages"][1]["content"][0]["text"]
    assert "installed_apps" not in message_text
    assert "com.android.chrome" not in message_text
    assert "com.android.settings" not in message_text
    for tool in request["tools"]:
        function = tool["function"]
        serialized = str(function)
        if function["name"] == "open_app":
            assert function["parameters"]["properties"]["package_name"]["enum"] == [
                "com.android.chrome",
                "com.android.settings",
            ]
        else:
            assert "com.android.chrome" not in serialized
            assert "com.android.settings" not in serialized


def test_vlm_planner_uses_only_compact_current_runtime_context() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="finished",
                                arguments='{"summary":"Goal complete"}',
                            )
                        )
                    ]
                )
            )
        ],
        usage=None,
    )
    completions = CapturingCompletions(response)
    planner = VLMPlanner(
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    asyncio.run(
        planner.one_step_tool_call(
            "Add the item to the cart",
            Observation(
                extra={
                    "display": {"width": 720, "height": 1280},
                    "previous_action_error": "action_completed_without_state_change",
                    "recent_actions": [
                        {
                            "tool": "click",
                            "args": {"x": 120, "y": 240},
                            "success": True,
                        }
                    ],
                    "execution_history": "1. [Planner] Clicked the item successfully.",
                    "function_execution": {
                        "schema_version": (
                            "omniflow.function-execution-evidence.v1"
                        ),
                        "function_id": "add_item",
                        "replay_status": "actions_succeeded",
                        "official_validator_status": "pending",
                        "steps": [],
                        "final_observation": {
                            "state_id": "state_after",
                            "package_name": "com.example.shop",
                            "activity_name": "CartActivity",
                        },
                    },
                }
            ),
            (),
            {},
        )
    )
    payload = completions.requests[0]["messages"][1]["content"][0]["text"]
    assert "Previous action error: action_completed_without_state_change" in payload
    assert "Previous Function result: add_item succeeded" in payload
    assert '"tool":"click"' not in payload
    assert "1. [Planner] Clicked the item successfully." not in payload
    assert '"official_validator_status":"pending"' not in payload
    assert '"state_id":"state_after"' not in payload


def test_vlm_planner_omits_androidworld_internal_state_from_prompt() -> None:
    official_state = {
        "pixels": {"path": "/evidence/current.png", "sha256": "abc"},
        "forest": {"windows": [{"tree": {"nodes": [{"text": "bulk"}]}}]},
        "ui_elements": [{"text": "duplicate bulk"}],
        "auxiliaries": {"package_name": "com.example"},
    }

    request = build_model_turn_request(
        goal="Open the target",
        model="test-model",
        state={
            "xml": '<hierarchy><node text="Target" bounds="[0,0][10,10]" /></hierarchy>',
            "display": {"width": 100, "height": 200},
            "extra": {"androidworld_state": official_state},
        },
        max_steps=8,
        turn_index=1,
    )

    payload = request["messages"][1]["content"][0]["text"]
    assert '"pixels"' not in payload
    assert '"auxiliaries"' not in payload
    assert '"forest"' not in payload
    assert '"ui_elements"' not in payload
    assert "bulk" not in payload
    assert official_state["forest"]["windows"]
    assert official_state["ui_elements"]


def test_vlm_planner_keeps_ui_tars_style_action_history() -> None:
    requests: list[dict[str, object]] = []
    responses = iter(
        [
            {
                "requested_model": "test-model",
                "resolved_model": "test-model",
                "tool_calls": [
                    {
                        "function": {
                            "name": "open_app",
                            "arguments": json.dumps(
                                {
                                    "summary": (
                                        "Read Theater Show, Museum Tickets, and "
                                        "Household Items; open target app."
                                    ),
                                    "package_name": "com.arduia.expense",
                                }
                            ),
                        }
                    }
                ],
            },
            {
                "requested_model": "test-model",
                "resolved_model": "test-model",
                "tool_calls": [
                    {
                        "function": {
                            "name": "finished",
                            "arguments": json.dumps(
                                {"summary": "All three expenses added"}
                            ),
                        }
                    }
                ],
            },
        ]
    )

    def transport(envelope: dict[str, object]) -> dict[str, object]:
        request = envelope["request"]
        assert isinstance(request, dict)
        requests.append(request)
        return next(responses)

    planner = VLMPlanner(model="test-model", transport=transport)
    first = Observation(
        image_base64="gallery-image",
        package_name="com.simplemobiletools.gallery.pro",
        extra={"display": {"width": 720, "height": 1280}},
    )
    second = Observation(
        image_base64="expense-image",
        package_name="com.arduia.expense",
        extra={"display": {"width": 720, "height": 1280}},
    )

    asyncio.run(planner.one_step_tool_call("Add three expenses", first))
    asyncio.run(planner.one_step_tool_call("Add three expenses", second))

    messages = requests[1]["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == [
        "system",
        "assistant",
        "tool",
        "user",
    ]
    assistant_call = messages[1]["tool_calls"][0]["function"]
    assert "Theater Show" in assistant_call["arguments"]
    assert "Museum Tickets" in assistant_call["arguments"]
    assert "Household Items" in assistant_call["arguments"]
    assert messages[2]["content"] == "Action dispatched; inspect the current screenshot."

def test_bridge_planner_exposes_packages_only_through_open_app_tool() -> None:
    installed_apps = {
        "Chrome": "com.android.chrome",
        "Settings": "com.android.settings",
    }

    request = build_model_turn_request(
        goal="Turn bluetooth on",
        model="test-model",
        state={
            "xml": "",
            "display": {"width": 720, "height": 1280},
            "extra": {"installed_apps": installed_apps},
        },
        max_steps=8,
        turn_index=0,
        installed_apps=installed_apps,
    )

    message_text = request["messages"][1]["content"][0]["text"]
    assert "Installed app candidates" not in message_text
    assert "com.android.chrome" not in message_text
    assert "com.android.settings" not in message_text
    for tool in request["tools"]:
        function = tool["function"]
        serialized = str(function)
        if function["name"] == "open_app":
            assert function["parameters"]["properties"]["package_name"]["enum"] == [
                "com.android.chrome",
                "com.android.settings",
            ]
        else:
            assert "com.android.chrome" not in serialized
            assert "com.android.settings" not in serialized


def test_bridge_planner_uses_unified_short_decision_policy() -> None:
    request = build_model_turn_request(
        goal="Search for a contact",
        model="test-model",
        state={"xml": "", "display": {"width": 720, "height": 1280}},
        max_steps=8,
        turn_index=0,
    )

    assert request["max_completion_tokens"] == 512
    assert request["reasoning_effort"] == "none"
    assert request["enable_thinking"] is False
    assert "You are an Android GUI agent" in SYSTEM_PROMPT
    assert "task, your action history, and the current screenshot" in SYSTEM_PROMPT
    assert "A recalled Function is an action API" in SYSTEM_PROMPT
    assert "normalized 0..1000 coordinates" in SYSTEM_PROMPT
    assert "raw-pixel" not in SYSTEM_PROMPT
    assert "Never infer image content from a cropped thumbnail" in SYSTEM_PROMPT
    assert "provides search" not in SYSTEM_PROMPT
    assert "RunLog" not in SYSTEM_PROMPT
    assert len(SYSTEM_PROMPT.split()) < 100


def test_planner_adds_recalled_function_as_a_peer_action_api() -> None:
    function = Function(
        function_id="add_expense",
        name="Add one expense",
        description="Add one expense using the provided values.",
        steps=(),
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )

    request = build_model_turn_request(
        goal="Add three expenses",
        model="test-model",
        state={"xml": "", "display": {"width": 720, "height": 1280}},
        max_steps=8,
        turn_index=0,
        functions=(function,),
    )

    tool_names = [tool["function"]["name"] for tool in request["tools"]]
    assert tool_names[0] == "add_expense"
    assert "A recalled Function is an action API" in SYSTEM_PROMPT
    assert "Prefer it when it directly performs the next part of the task" in SYSTEM_PROMPT
    summaries = [
        tool["function"]["parameters"]["properties"]["summary"]["description"]
        for tool in request["tools"]
    ]
    assert all(text == "Brief plan and reason for the next action." for text in summaries)


def test_vlm_planner_retries_empty_required_function_string() -> None:
    requests: list[dict[str, object]] = []
    responses = iter(
        [
            {
                "requested_model": "test-model",
                "resolved_model": "test-model",
                "tool_calls": [
                    {
                        "function": {
                            "name": "finish_expense",
                            "arguments": json.dumps(
                                {"summary": "Save the expense.", "note": "   "}
                            ),
                        }
                    }
                ],
            },
            {
                "requested_model": "test-model",
                "resolved_model": "test-model",
                "tool_calls": [
                    {
                        "function": {
                            "name": "finish_expense",
                            "arguments": json.dumps(
                                {
                                    "summary": "Save the expense with its note.",
                                    "note": "Paid by card",
                                }
                            ),
                        }
                    }
                ],
            },
        ]
    )

    def transport(envelope: dict[str, object]) -> dict[str, object]:
        request = envelope["request"]
        assert isinstance(request, dict)
        requests.append(request)
        return next(responses)

    function = Function(
        function_id="finish_expense",
        name="Finish expense",
        description="Enter the note and save the current expense.",
        steps=(),
        input_schema={
            "type": "object",
            "properties": {"note": {"type": "string", "minLength": 1}},
            "required": ["note"],
            "additionalProperties": False,
        },
    )
    planner = VLMPlanner(model="test-model", transport=transport)

    call = asyncio.run(
        planner.one_step_tool_call(
            "Add the expense",
            Observation(extra={"display": {"width": 720, "height": 1280}}),
            (function,),
        )
    )

    assert call == ToolCall("finish_expense", {"note": "Paid by card"})
    assert len(requests) == 2
    assert [tool["function"]["name"] for tool in requests[1]["tools"]] == [
        "finish_expense"
    ]
    retry_text = requests[1]["messages"][-1]["content"][0]["text"]
    assert "function_arguments_invalid:minLength:note" in retry_text


def test_function_completion_review_keeps_current_screenshot_and_result() -> None:
    request = build_model_turn_request(
        goal="Turn bluetooth off",
        model="test-model",
        state={
            "xml": (
                '<hierarchy><node text="Use Bluetooth" checkable="true" '
                'checked="false" bounds="[0,406][720,620]" /></hierarchy>'
            ),
            "image_base64": "final-screenshot",
            "display": {"width": 720, "height": 1280},
            "extra": {
                "function_execution": {
                    "function_id": "complete_run_turn_bluetooth_off",
                    "replay_status": "actions_succeeded",
                },
            },
        },
        max_steps=8,
        turn_index=0,
    )

    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]
    assert "Previous Function result: complete_run_turn_bluetooth_off succeeded" in content[0]["text"]
    assert '"checked":false' not in content[0]["text"]


def test_vlm_planner_function_completion_review_uses_current_screenshot() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="finished",
                                arguments='{"summary":"Goal complete"}',
                            )
                        )
                    ]
                )
            )
        ],
        usage=None,
    )
    completions = CapturingCompletions(response)
    planner = VLMPlanner(
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    planned = asyncio.run(
        planner.one_step_tool_call(
            "Turn bluetooth off",
            Observation(
                xml=(
                    '<hierarchy><node text="Use Bluetooth" checkable="true" '
                    'checked="false" bounds="[0,406][720,620]" /></hierarchy>'
                ),
                image_base64="final-screenshot",
                extra={
                    "display": {"width": 720, "height": 1280},
                    "function_execution": {
                        "function_id": "complete_run_turn_bluetooth_off",
                        "replay_status": "actions_succeeded",
                    },
                },
            ),
        )
    )

    assert planned == ToolCall("finished", {})
    request = completions.requests[0]
    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]
    turn_payload = content[0]["text"]
    assert "Previous Function result: complete_run_turn_bluetooth_off succeeded" in turn_payload
    assert '"checked":false' not in turn_payload


def test_androidworld_launcher_configures_one_unified_planner(
    monkeypatch,
) -> None:
    planner_options: dict[str, object] = {}

    class CapturingPlanner:
        def __init__(self, **options: object) -> None:
            planner_options.update(options)

    monkeypatch.setattr("omniflow.vlm.planner.VLMPlanner", CapturingPlanner)
    monkeypatch.setattr(
        androidworld_launch,
        "build_agent",
        lambda **options: SimpleNamespace(**options),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "not-required")
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.setenv("LLMTHU_KEY", "unified-key")
    monkeypatch.setenv("LLMTHU_BASE_URL", "https://llmapi.example/v1")

    flow = androidworld_launch._build_launch_agent(
        agent="omniflow",
        env=SimpleNamespace(),
        store_path="store.json",
        adb_serial="emulator-5554",
        planner_provider="openai",
        planner_model="test-model",
    )

    assert planner_options["api_key"] == "unified-key"
    assert planner_options["base_url"] == "https://llmapi.example/v1"
    assert flow.planner is not None


def test_llmthu_endpoint_profile_ignores_conflicting_openai_variables() -> None:
    api_key, base_url = resolve_openai_compatible_config(
        environment={
            "OPENAI_API_KEY": "dashscope-key",
            "OPENAI_BASE_URL": "https://dashscope.example/v1",
            "LLMTHU_KEY": "llmthu-key",
            "LLMTHU_BASE_URL": "https://llmthu.example/v1",
        },
        profile="llmthu",
    )

    assert api_key == "llmthu-key"
    assert base_url == "https://llmthu.example/v1"


def test_llmthu_endpoint_profile_does_not_fall_back_to_openai() -> None:
    with pytest.raises(
        ValueError,
        match="model_endpoint_profile_incomplete:llmthu",
    ):
        resolve_openai_compatible_config(
            environment={
                "OPENAI_API_KEY": "dashscope-key",
                "OPENAI_BASE_URL": "https://dashscope.example/v1",
            },
            profile="llmthu",
        )


def test_model_endpoint_profile_rejects_unknown_accounts() -> None:
    with pytest.raises(ValueError, match="model_endpoint_profile_invalid:unknown"):
        resolve_openai_compatible_config(profile="unknown", environment={})


def test_androidworld_agent_exposes_target_states_when_source_catalog_exists(
    tmp_path,
    monkeypatch,
) -> None:
    store_path = tmp_path / "store.json"
    source_catalog = tmp_path / "transfer_states.json"
    source_catalog.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "source-run",
                "states": {},
            }
        ),
        encoding="utf-8",
    )
    original_source_catalog = source_catalog.read_bytes()
    monkeypatch.setattr(
        "src.integrations.android_world.host.AndroidWorldHost.installed_packages",
        lambda _host: set(),
    )
    flow = build_agent(env=SimpleNamespace(), store_path=str(store_path))
    flow.host.state.update(
        last_result=SimpleNamespace(success=True),
        captured_transfer_states={
            "target-before": {
                "state_id": "target-before",
                "xml": "<hierarchy />",
            }
        },
    )

    captured = flow.get_captured_transfer_states()

    assert source_catalog.read_bytes() == original_source_catalog
    assert captured == {
        "target-before": {
            "state_id": "target-before",
            "xml": "<hierarchy />",
        }
    }
