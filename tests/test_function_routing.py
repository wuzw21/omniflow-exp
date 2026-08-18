from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from runlog_fixtures import androidworld_state, write_function_store

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
from omniflow.functions.assets import FUNCTION_ARTIFACT_VERSION
from omniflow.vlm.model_config import resolve_openai_compatible_config
from omniflow.vlm.planner import (
    SYSTEM_PROMPT,
    ModelToolCallError,
    VLMPlanner,
    adapt_tool_arguments,
    build_model_turn_request,
    function_tools,
    parse_model_turn_response,
)
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


def test_task_host_exposes_the_native_androidworld_environment() -> None:
    environment = object()
    host = _TaskHost(
        SimpleNamespace(env=environment),
        {"captured_transfer_states": {}},
        {},
    )

    assert host.env is environment


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


def test_ui_tars_mobile_prompt_keeps_structured_peer_tools() -> None:
    function = Function(
        function_id="create_contact",
        name="Create contact",
        description="Create one contact with the provided name and phone number.",
        steps=(),
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "phone": {"type": "string"},
            },
            "required": ["name", "phone"],
            "additionalProperties": False,
        },
    )

    request = build_model_turn_request(
        goal="Add Daniel Mohammed as 5550100",
        model="test-model",
        state={
            "image_base64": "full-image",
            "display": {"width": 720, "height": 1280},
        },
        turn_index=1,
        functions=(function,),
    )

    assert "tools" in request
    assert request["messages"][0]["content"].startswith("You are a GUI agent.")
    prompt = request["messages"][0]["content"]
    assert "task and your action history, with screenshots" in prompt
    assert "summary" not in prompt.casefold()
    assert request["tool_choice"] == "required"
    assert request["parallel_tool_calls"] is False
    function_tool = next(
        tool
        for tool in request["tools"]
        if tool["function"]["name"] == "create_contact"
    )
    assert function_tool["function"]["parameters"]["required"] == [
        "name",
        "phone",
    ]
    assert request["messages"][-1]["content"][0]["image_url"]["url"].endswith(
        "full-image"
    )


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
    write_function_store(path, (function,))
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

    tools = function_tools(ranked)

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
                action=Action("wait", {"duration_ms": 0}),
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
    write_function_store(path, (function,))
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
    write_function_store(path, (function,))
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
    write_function_store(path, (function,))
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
        "max_fallback_steps": 5,
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
    assert [action.tool for action in host.actions] == ["wait", "wait", "wait"]
    assert planner.visible_function_ids == []
    assert result.actions_executed == 3
    assert result.detail["planner_steps"] == 0
    assert result.detail["runtime_limits"]["max_steps"] == 1


def test_one_function_call_is_one_planner_step_with_multiple_actions(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_long_function(store_path)
    host = RecordingHost()
    planner = SequencePlanner([ToolCall(function_id, {})])
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=1)),
    )

    result = flow.run("Complete the replay")

    assert result.success is False
    assert result.error is None
    assert result.detail["done_reason"] == "step_completed"
    assert result.detail["planner_steps"] == 1
    assert result.actions_executed == 3


def test_one_native_action_is_one_nonterminal_planner_step(tmp_path) -> None:
    planner = SequencePlanner(
        [ToolCall("open_app", {"package_name": "com.android.settings"})]
    )
    flow = OmniFlow(
        tmp_path / "store.json",
        host=RecordingHost(),
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=1)),
    )

    result = flow.run("Open Settings")

    assert result.success is False
    assert result.error is None
    assert result.detail["done_reason"] == "step_completed"
    assert result.detail["planner_steps"] == 1
    assert result.actions_executed == 1


def test_finished_is_one_terminal_planner_step(tmp_path) -> None:
    flow = OmniFlow(
        tmp_path / "store.json",
        host=RecordingHost(),
        planner=FinishingPlanner(),
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=1)),
    )

    result = flow.run("Nothing remains")

    assert result.success is True
    assert result.error is None
    assert result.detail["done_reason"] == "finished"
    assert result.detail["planner_steps"] == 1
    assert result.actions_executed == 0


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
    assert all(observation.image_base64 for observation in planner.observations)
    assert [request.get("screenshot") for request in host.observe_requests] == [
        True,
        True,
        True,
    ]


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


def test_planner_adapter_normalizes_raw_pixel_coordinates() -> None:
    adapted, metadata = adapt_tool_arguments(
        tool="click",
        arguments={"x": 632, "y": 1112},
        requested_model="GLM-5.1",
        resolved_model="GLM-5.1",
        display={"width": 720, "height": 1280},
    )

    assert adapted == {
        "x": pytest.approx(632 / 720 * 1000),
        "y": pytest.approx(1112 / 1280 * 1000),
    }
    assert metadata is not None
    assert metadata["name"] == "planner_coordinate_adapter.v1"
    assert metadata["model"] == "GLM-5.1"


def test_planner_adapter_preserves_canonical_coordinates() -> None:
    adapted, metadata = adapt_tool_arguments(
        tool="click",
        arguments={"x": 878, "y": 869},
        requested_model="GLM-5.1",
        resolved_model="GLM-5.1",
        display={"width": 720, "height": 1280},
    )

    assert adapted == {"x": 878, "y": 869}
    assert metadata is None


def test_planner_adapter_normalizes_mixed_coordinate_axes() -> None:
    adapted, metadata = adapt_tool_arguments(
        tool="click",
        arguments={"x": 878, "y": 1112},
        requested_model="GLM-5.1",
        resolved_model="GLM-5.1",
        display={"width": 720, "height": 1280},
    )

    assert adapted == {"x": 878, "y": pytest.approx(1112 / 1280 * 1000)}
    assert metadata is not None
    assert metadata["name"] == "planner_coordinate_adapter.v1"


def test_planner_adapter_rejects_raw_pixel_axis_outside_display() -> None:
    with pytest.raises(ValueError, match="canonical_action_arg_range_invalid:y"):
        adapt_tool_arguments(
            tool="click",
            arguments={"x": 632, "y": 1281},
            requested_model="GLM-5.1",
            resolved_model="GLM-5.1",
            display={"width": 720, "height": 1280},
        )


def test_qwen36_adapter_normalizes_swipe_coordinate_arrays() -> None:
    adapted, metadata = adapt_tool_arguments(
        tool="swipe",
        arguments={
            "direction": "left",
            "x1": [875, 449],
            "y1": [875, 449],
            "x2": [125, 449],
            "y2": [125, 449],
        },
        requested_model="Qwen3.6-Plus",
        resolved_model="Qwen3.6-Plus",
        display={"width": 720, "height": 1280},
    )

    assert adapted == {
        "direction": "left",
        "x1": 875,
        "y1": 449,
        "x2": 125,
        "y2": 449,
    }
    assert metadata is not None
    assert metadata["model"] == "Qwen3.6-Plus"


def test_planner_exposes_normalized_coordinates_independent_of_display() -> None:
    request = build_model_turn_request(
        goal="Tap add",
        model="test-model",
        state={"display": {"width": 720, "height": 1280}},
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
                        "arguments": json.dumps({"x": 876, "y": 869}),
                    }
                }
            ],
        },
        requested_model="test-model",
        turn_index=1,
        display={"width": 720, "height": 1280},
    )

    assert tool_call == ToolCall("click", {"x": 876, "y": 869})
    assert metadata == {}


def test_model_turn_uses_only_generic_planning_context() -> None:
    request = build_model_turn_request(
        goal="Copy every record into another app",
        model="test-model",
        state={
            "image_base64": "full-image",
            "package_name": "com.example.source",
            "display": {"width": 720, "height": 1280},
        },
        turn_index=1,
    )

    turn_text = next(
        item["text"]
        for item in request["messages"][-1]["content"]
        if item["type"] == "text"
    )
    assert "Task: Copy every record into another app" in turn_text
    assert "Current screen: package=com.example.source" in turn_text
    assert "file" not in turn_text.casefold()
    assert "search" not in turn_text.casefold()


def test_runtime_maps_normalized_coordinates_to_screen_pixels() -> None:
    assert canonical_action_to_screen_pixels(
        {"tool": "click", "args": {"x": 876, "y": 869}},
        {"width": 720, "height": 1280},
    ) == {
        "tool": "click",
        "args": {"x": pytest.approx(630.72), "y": pytest.approx(1112.32)},
    }


def test_vlm_planner_exposes_installed_apps_only_through_open_app() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                                function=SimpleNamespace(
                                    name="finished",
                                    arguments='{}',
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
            package_schema = function["parameters"]["properties"]["package_name"]
            assert package_schema["enum"] == [
                "com.android.chrome",
                "com.android.settings",
            ]
            assert "Chrome -> com.android.chrome" in package_schema["description"]
            assert "Settings -> com.android.settings" in package_schema["description"]
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
                                    arguments='{}',
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


def test_vlm_planner_sends_only_current_observation() -> None:
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
                                {"package_name": "com.arduia.expense"}
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
                            "arguments": "{}",
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
    installed_apps = {
        "Gallery": "com.simplemobiletools.gallery.pro",
        "Expense Manager": "com.arduia.expense",
    }

    asyncio.run(
        planner.one_step_tool_call("Add three expenses", first, installed_apps=installed_apps)
    )
    asyncio.run(
        planner.one_step_tool_call("Add three expenses", second, installed_apps=installed_apps)
    )

    messages = requests[1]["messages"]
    assert isinstance(messages, list)
    assert [message["role"] for message in messages] == ["system", "user"]
    assert [tool["function"]["name"] for tool in requests[1]["tools"]] == [
        "click",
        "input_text",
        "swipe",
        "open_app",
        "press_key",
        "finished",
    ]

def test_bridge_planner_exposes_installed_apps_only_through_open_app() -> None:
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
            package_schema = function["parameters"]["properties"]["package_name"]
            assert package_schema["enum"] == [
                "com.android.chrome",
                "com.android.settings",
            ]
            assert "Chrome -> com.android.chrome" in package_schema["description"]
            assert "Settings -> com.android.settings" in package_schema["description"]
        else:
            assert "com.android.chrome" not in serialized
            assert "com.android.settings" not in serialized


def test_planner_uses_compact_xml_as_primary_grounding_with_screenshot() -> None:
    request = build_model_turn_request(
        goal="Enter a name",
        model="test-model",
        state={
            "xml": (
                '<hierarchy><node class="android.widget.EditText" text="Name" '
                'resource-id="contacts:id/name" editable="true" focused="true" '
                'bounds="[72,240][648,360]" /></hierarchy>'
            ),
            "image_base64": "screen-image",
            "package_name": "com.android.contacts",
            "display": {"width": 720, "height": 1280},
        },
        turn_index=1,
    )

    content = request["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert content[0]["image_url"]["url"].endswith("screen-image")
    assert "Primary UI grounding evidence" in content[1]["text"]
    assert 'text="Name"' in content[1]["text"]
    assert 'id="contacts:id/name"' in content[1]["text"]
    assert "bounds=[100,188][900,281]" in content[1]["text"]
    assert "flags=editable,focused" in content[1]["text"]
    assert "<hierarchy" not in content[1]["text"]
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "click",
        "input_text",
        "swipe",
        "open_app",
        "press_key",
        "finished",
    ]


def test_xml_only_state_does_not_expose_a_second_observation_tool() -> None:
    request = build_model_turn_request(
        goal="Tap the unlabeled icon",
        model="test-model",
        state={
            "xml": (
                '<hierarchy><node clickable="true" '
                'bounds="[640,80][720,160]" /></hierarchy>'
            ),
            "package_name": "com.example.app",
            "display": {"width": 720, "height": 1280},
        },
        turn_index=1,
    )

    content = request["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["text"]
    assert "Primary UI grounding evidence" in content[0]["text"]
    assert "bounds=[889,62][1000,125]" in content[0]["text"]
    assert "flags=clickable" in content[0]["text"]
    assert "get_state" not in {
        tool["function"]["name"] for tool in request["tools"]
    }
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "click",
        "input_text",
        "swipe",
        "open_app",
        "press_key",
        "finished",
    ]


def test_screenshot_state_hides_get_state_fallback() -> None:
    request = build_model_turn_request(
        goal="Tap the icon",
        model="test-model",
        state={
            "xml": "",
            "image_base64": "screen-image",
            "package_name": "com.example.app",
            "display": {"width": 720, "height": 1280},
        },
        turn_index=2,
    )

    assert [item["type"] for item in request["messages"][-1]["content"]] == [
        "image_url",
        "text",
    ]
    assert "get_state" not in {
        tool["function"]["name"] for tool in request["tools"]
    }
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "click",
        "input_text",
        "swipe",
        "open_app",
        "press_key",
        "finished",
    ]


def test_visual_surface_keeps_screenshot_and_full_gui_fallback_tools() -> None:
    request = build_model_turn_request(
        goal="Tap the game icon",
        model="test-model",
        state={
            "xml": (
                '<hierarchy><node class="android.view.SurfaceView" '
                'bounds="[0,0][720,1280]" /></hierarchy>'
            ),
            "image_base64": "screen-image",
            "package_name": "com.example.game",
            "display": {"width": 720, "height": 1280},
        },
        turn_index=1,
    )

    content = request["messages"][-1]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "click",
        "input_text",
        "swipe",
        "open_app",
        "press_key",
        "finished",
    ]


def test_labeled_controls_keep_image_and_full_tool_set() -> None:
    request = build_model_turn_request(
        goal="Open Bluetooth",
        model="test-model",
        state={
            "xml": (
                '<hierarchy><node text="Settings" bounds="[0,0][720,100]" />'
                '<node text="Bluetooth" clickable="true" bounds="[0,100][720,220]" />'
                '<node text="Wi-Fi" clickable="true" bounds="[0,220][720,340]" />'
                '<node clickable="true" bounds="[640,0][720,100]" />'
                '<node clickable="true" bounds="[560,0][640,100]" />'
                '<node clickable="true" bounds="[480,0][560,100]" /></hierarchy>'
            ),
            "image_base64": "screen-image",
            "package_name": "com.android.settings",
            "display": {"width": 720, "height": 1280},
        },
        turn_index=1,
    )

    assert [item["type"] for item in request["messages"][-1]["content"]] == [
        "image_url",
        "text",
    ]
    turn_text = request["messages"][-1]["content"][1]["text"]
    assert 'text="Bluetooth"' in turn_text
    assert "bounds=[0,78][1000,172]" in turn_text
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "click",
        "input_text",
        "swipe",
        "open_app",
        "press_key",
        "finished",
    ]


def test_compact_xml_prioritizes_actionable_navigation_after_long_content() -> None:
    call_rows = "".join(
        f'<node id="{index}" text="Recent call {index}" '
        f'bounds="[0,{index * 40}][1080,{index * 40 + 40}]" />'
        for index in range(40)
    )
    request = build_model_turn_request(
        goal="Add a contact",
        model="test-model",
        state={
            "xml": (
                f'<hierarchy width="1080" height="2376">{call_rows}'
                '<node id="116" content-desc="联系人" clickable="true" '
                'bounds="[372,2160][708,2328]" />'
                '<node id="122" content-desc="营业厅" clickable="true" '
                'bounds="[708,2160][1044,2328]" /></hierarchy>'
            ),
            "image_base64": "screen-image",
            "package_name": "com.android.contacts",
            "display": {"width": 1080, "height": 2376},
        },
        turn_index=1,
    )

    turn_text = request["messages"][-1]["content"][1]["text"]
    assert 'node_id=116 desc="联系人"' in turn_text
    assert "bounds=[344,909][656,980]" in turn_text
    assert 'node_id=122 desc="营业厅"' in turn_text
    assert "Recent call 23" not in turn_text


def test_open_app_remains_available_in_current_app() -> None:
    request = build_model_turn_request(
        goal="Open network settings",
        model="test-model",
        state={
            "xml": (
                '<hierarchy><node text="Network &amp; internet" clickable="true" '
                'bounds="[40,200][680,320]" /></hierarchy>'
            ),
            "package_name": "com.android.settings",
            "display": {"width": 720, "height": 1280},
        },
        installed_apps={"Settings": "com.android.settings"},
        turn_index=1,
    )

    names = [tool["function"]["name"] for tool in request["tools"]]
    assert names == [
        "click",
        "input_text",
        "swipe",
        "open_app",
        "press_key",
        "finished",
    ]


def test_open_app_exposes_complete_installed_app_vocabulary() -> None:
    request = build_model_turn_request(
        goal="Open contacts",
        model="test-model",
        state={
            "package_name": "com.android.launcher",
            "display": {"width": 720, "height": 1280},
        },
        installed_apps={
            "Contacts": "com.android.contacts",
            "Settings": "com.android.settings",
            "Chrome": "com.android.chrome",
        },
        turn_index=1,
    )

    open_app = next(
        tool for tool in request["tools"] if tool["function"]["name"] == "open_app"
    )
    package_schema = open_app["function"]["parameters"]["properties"]["package_name"]
    assert package_schema["enum"] == [
        "com.android.chrome",
        "com.android.contacts",
        "com.android.settings",
    ]
    assert "Contacts -> com.android.contacts" in package_schema["description"]
    assert "Settings -> com.android.settings" in package_schema["description"]
    assert "Chrome -> com.android.chrome" in package_schema["description"]


def test_parameterized_app_function_exposes_installed_app_vocabulary() -> None:
    function = Function(
        function_id="open_requested_app",
        name="Open requested app",
        description="Verified path that opens the requested installed app.",
        steps=(),
        input_schema={
            "type": "object",
            "properties": {
                "package_name": {
                    "type": "string",
                    "description": "Installed Android package to open.",
                }
            },
            "required": ["package_name"],
            "additionalProperties": False,
        },
    )
    request = build_model_turn_request(
        goal="Open contacts",
        model="test-model",
        state={
            "package_name": "com.android.launcher",
            "display": {"width": 720, "height": 1280},
        },
        installed_apps={
            "Contacts": "com.android.contacts",
            "Settings": "com.android.settings",
        },
        functions=(function,),
        turn_index=1,
    )

    function_tool = next(
        tool
        for tool in request["tools"]
        if tool["function"]["name"] == "open_requested_app"
    )
    package_schema = function_tool["function"]["parameters"]["properties"][
        "package_name"
    ]
    assert package_schema["enum"] == [
        "com.android.contacts",
        "com.android.settings",
    ]
    assert package_schema["description"].startswith(
        "Installed Android package to open."
    )
    assert "Contacts -> com.android.contacts" in package_schema["description"]
    assert "Settings -> com.android.settings" in package_schema["description"]


def test_bridge_planner_uses_unified_short_decision_policy() -> None:
    request = build_model_turn_request(
        goal="Search for a contact",
        model="test-model",
        state={"xml": "", "display": {"width": 720, "height": 1280}},
        turn_index=0,
    )

    assert request["max_completion_tokens"] == 512
    assert request["reasoning_effort"] == "none"
    assert request["enable_thinking"] is False
    assert "You are a GUI agent" in SYSTEM_PROMPT
    assert "task and your action history, with screenshots" in SYSTEM_PROMPT
    assert "Choose exactly one provided tool call" in SYSTEM_PROMPT
    assert "Functions are verified multi-step action paths" in SYSTEM_PROMPT
    assert "When a Function matches the task, prefer it" in SYSTEM_PROMPT
    assert "if it fails" in SYSTEM_PROMPT
    assert "normalized 0..1000 coordinates" in SYSTEM_PROMPT
    assert "Accessibility XML is primary evidence" in SYSTEM_PROMPT
    assert "vision only supplements" in SYSTEM_PROMPT
    assert "never guess future layout" in SYSTEM_PROMPT


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
        turn_index=0,
        functions=(function,),
    )

    tool_names = [tool["function"]["name"] for tool in request["tools"]]
    assert tool_names[0] == "add_expense"
    assert all(
        "summary" not in tool["function"]["parameters"]["properties"]
        for tool in request["tools"]
    )


def test_vlm_planner_rejects_invalid_function_once_without_retry() -> None:
    requests: list[dict[str, object]] = []
    response = {
        "requested_model": "test-model",
        "resolved_model": "test-model",
        "tool_calls": [
            {
                "function": {
                    "name": "finish_expense",
                    "arguments": json.dumps({"note": "   "}),
                }
            }
        ],
    }

    def transport(envelope: dict[str, object]) -> dict[str, object]:
        request = envelope["request"]
        assert isinstance(request, dict)
        requests.append(request)
        return response

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

    with pytest.raises(ModelToolCallError, match="minLength:note"):
        asyncio.run(
            planner.one_step_tool_call(
                "Add the expense",
                Observation(extra={"display": {"width": 720, "height": 1280}}),
                (function,),
            )
        )

    assert len(requests) == 1


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
        turn_index=0,
    )

    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["image_url", "text"]
    assert "Previous Function result: complete_run_turn_bluetooth_off succeeded" in content[1]["text"]
    assert '"checked":false' not in content[1]["text"]


def test_vlm_planner_function_completion_review_uses_current_screenshot() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="finished",
                                arguments="{}",
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
    assert [item["type"] for item in content] == ["image_url", "text"]
    turn_payload = content[1]["text"]
    assert "Previous Function result: complete_run_turn_bluetooth_off succeeded" in turn_payload
    assert '"checked":false' not in turn_payload


def test_androidworld_launcher_configures_one_unified_planner(
    monkeypatch,
) -> None:
    planner_options: dict[str, object] = {}
    performance_metrics = object()

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
    monkeypatch.setenv("LLMTHU_API_KEY", "unified-key")

    flow = androidworld_launch._build_launch_agent(
        agent="omniflow",
        env=SimpleNamespace(),
        store_path="store.json",
        adb_serial="emulator-5554",
        planner_provider="openai",
        planner_model="test-model",
        performance_metrics=performance_metrics,
    )

    assert planner_options["api_key"] == "unified-key"
    assert planner_options["base_url"] == "https://llmapi.paratera.com/v1"
    assert flow.planner is not None
    assert flow.performance_metrics is performance_metrics


def test_llmthu_endpoint_profile_ignores_conflicting_openai_variables() -> None:
    api_key, base_url = resolve_openai_compatible_config(
        base_url="https://llmapi.paratera.com/v1",
        environment={
            "OPENAI_API_KEY": "dashscope-key",
            "OPENAI_BASE_URL": "https://dashscope.example/v1",
            "LLMTHU_API_KEY": "llmthu-key",
        },
        profile="llmthu",
    )

    assert api_key == "llmthu-key"
    assert base_url == "https://llmapi.paratera.com/v1"


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
