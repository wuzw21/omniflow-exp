from __future__ import annotations

import asyncio
import json
import sys
from types import SimpleNamespace

from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow import (
    Action,
    ActionResult,
    Function,
    Observation,
    OmniFlow,
    ToolCall,
)
from omniflow.core.config import OmniFlowConfig, RuntimeSettings
from omniflow.core.model import FunctionStep
from omniflow.core.trajectory import state_id
from omniflow.functions.artifact import FUNCTION_ARTIFACT_VERSION
from omniflow.functions.store import FunctionStore
from omniflow.vlm.function_router import VLMFunctionRouter
from omniflow.vlm.gui import SYSTEM_PROMPT, build_model_turn_request
from omniflow.vlm.planner import VLMPlanner
from src.integrations.android_world.agent import (
    _androidworld_run_log_steps,
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


class AcceptingRouter:
    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Function, ...]]] = []

    def route_function(
        self,
        goal: str,
        functions: tuple[Function, ...],
    ) -> ToolCall:
        self.calls.append((goal, functions))
        return ToolCall(functions[0].id, {})


class RejectingRouter(AcceptingRouter):
    def route_function(
        self,
        goal: str,
        functions: tuple[Function, ...],
    ) -> None:
        self.calls.append((goal, functions))
        return None


class FailingRouter(AcceptingRouter):
    def route_function(
        self,
        goal: str,
        functions: tuple[Function, ...],
    ) -> None:
        self.calls.append((goal, functions))
        raise RuntimeError("router unavailable")


def test_androidworld_trace_keeps_the_captured_official_state() -> None:
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
        "captured_androidworld_states": {},
    }
    host = _TaskHost(raw_host, runtime_state, {})

    observation = host.observe(xml=True, screenshot=True, app_info=True)
    steps = _androidworld_run_log_steps(
        [
            {
                "before_state_id": identifier,
                "after_state_id": identifier,
                "action": {"tool": "wait", "args": {"duration_ms": 1000}},
                "result": {"success": True},
            }
        ],
        runtime_state["captured_androidworld_states"],
    )

    assert observation.extra["state_id"] == identifier
    assert runtime_state["captured_androidworld_states"] == {
        identifier: official_state
    }
    assert steps[0]["observation"] == official_state


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


def test_run_routes_recalled_function_before_gui_planner(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_open_settings_function(store_path)
    host = RecordingHost()
    router = AcceptingRouter()
    planner = FinishingPlanner()
    flow = OmniFlow(
        store_path,
        host=host,
        function_router=router,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == ["open_app"]
    assert len(router.calls) == 1
    assert planner.visible_function_ids == [()]
    assert planner.observations[0].image_base64 == "final-screenshot"
    assert [request.get("screenshot") for request in host.observe_requests] == [
        False,
        True,
    ]
    assert "Function `complete_run_turn_bluetooth_on`" in str(
        planner.observations[0].extra.get("execution_history")
    )
    assert planner.observations[0].extra["function_execution"] == {
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
    assert result.detail["function_resolution"] == {
        "candidate_count": 1,
        "candidate_function_ids": [function_id],
        "router_configured": True,
        "status": "selected",
        "selected_function_id": function_id,
        "arguments": {},
        "binding_status": "succeeded",
        "binding_error": None,
        "replay_status": "succeeded",
        "replay_error": None,
        "failed_step_index": None,
    }
    assert result.detail["runtime_limits"] == {
        "max_steps": 20,
        "max_fallback_steps": None,
    }


def test_zero_fallback_budget_never_calls_gui_planner(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_open_settings_function(store_path)
    host = RecordingHost()
    router = AcceptingRouter()
    planner = FinishingPlanner()
    flow = OmniFlow(
        store_path,
        host=host,
        function_router=router,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_steps=20, max_fallback_steps=0),
        ),
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is False
    assert result.error == "fallback_budget_exhausted"
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == ["open_app"]
    assert len(router.calls) == 1
    assert planner.visible_function_ids == []
    assert result.fallback_steps == 0
    assert result.detail["function_resolution"]["status"] == "selected"
    assert result.detail["function_resolution"]["binding_status"] == "succeeded"
    assert result.detail["function_resolution"]["replay_status"] == "succeeded"
    assert result.detail["runtime_limits"] == {
        "max_steps": 20,
        "max_fallback_steps": 0,
    }


def test_rejected_function_enters_gui_planner_without_function_tools(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    _store_with_open_settings_function(store_path)
    host = RecordingHost()
    router = RejectingRouter()
    planner = FinishingPlanner()
    flow = OmniFlow(
        store_path,
        host=host,
        function_router=router,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert result.function_id is None
    assert host.actions == []
    assert len(router.calls) == 1
    assert planner.visible_function_ids == [()]
    assert result.detail["function_resolution"]["status"] == "rejected"
    assert result.detail["function_resolution"]["selected_function_id"] is None
    assert result.detail["function_resolution"]["arguments"] == {}


def test_function_router_error_is_preserved_for_result_audit(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_open_settings_function(store_path)
    router = FailingRouter()
    flow = OmniFlow(
        store_path,
        host=RecordingHost(),
        function_router=router,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is False
    assert result.detail["function_resolution"] == {
        "candidate_count": 1,
        "candidate_function_ids": [function_id],
        "router_configured": True,
        "status": "error",
        "selected_function_id": None,
        "arguments": {},
        "binding_status": "not_attempted",
        "binding_error": None,
        "replay_status": "not_started",
        "replay_error": None,
        "failed_step_index": None,
        "router_error": "RuntimeError:router unavailable",
    }


def test_gui_planner_never_receives_function_tools_without_router(tmp_path) -> None:
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
    assert planner.visible_function_ids == [()]


def test_transfer_failure_falls_back_without_replaying_source_coordinates(
    tmp_path,
) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_untransferable_click_function(store_path)
    host = RecordingHost()
    router = AcceptingRouter()
    planner = SequencePlanner(
        [
            ToolCall("open_app", {"package_name": "com.android.settings"}),
            ToolCall("finished", {"content": ""}),
        ]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        function_router=router,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == ["open_app"]
    assert all(action.tool != "click" for action in host.actions)
    assert planner.visible_function_ids == [(), ()]
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
    assert planner.visible_function_ids == [(), ()]
    assert planner.previous_action_errors[0] == "omnitransfer_missing_target_page"
    assert "Continue Function" in planner.goals[0]
    assert "Do not repeat actions that already succeeded" in planner.goals[0]
    assert result.detail["function_resolution"]["status"] == "direct"
    assert result.detail["function_resolution"]["replay_status"] == "failed"


def test_vlm_history_blocks_successful_repeat_on_same_logical_ui_state(
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
    assert host.actions == [Action("click", {"x": 120, "y": 240})]
    assert planner.previous_action_errors[0] is None
    assert planner.previous_action_errors[-1] == (
        "action_already_succeeded_on_current_state"
    )
    assert len(planner.previous_action_errors) <= 3


class CapturingCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return self.response


def test_vlm_function_router_only_exposes_candidates_and_reject() -> None:
    function = Function(
        function_id="connect_device",
        name="Connect a Bluetooth device",
        description="Connect the named Bluetooth device from system settings.",
        steps=(
            FunctionStep(
                step_index=0,
                source_state_id="source_settings",
                action=Action("press_key", {"key": "ENTER"}),
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {
                "device_name": {
                    "type": "string",
                    "description": "Exact Bluetooth device name from the user goal.",
                }
            },
            "required": ["device_name"],
            "additionalProperties": False,
        },
        agent_visible=True,
    )
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name=function.id,
                                arguments='{"device_name":"Headphones"}',
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
    router = VLMFunctionRouter(model="test-model", client=client)

    routed = asyncio.run(
        router.route_function(
            "Connect Headphones over Bluetooth",
            (function,),
        )
    )

    assert routed == ToolCall(function.id, {"device_name": "Headphones"})
    request = completions.requests[0]
    tool_names = [tool["function"]["name"] for tool in request["tools"]]
    assert tool_names == [function.id, "reject_recalled_function"]
    candidate_tool = request["tools"][0]["function"]
    assert function.name in candidate_tool["description"]
    assert function.description in candidate_tool["description"]
    assert candidate_tool["parameters"] == function.input_schema
    assert request["messages"][1]["content"] == (
        '{"goal":"Connect Headphones over Bluetooth"}'
    )


def test_vlm_function_router_disables_sdk_retries(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def openai_client(**options: object) -> object:
        captured.update(options)
        return object()

    monkeypatch.setitem(
        sys.modules,
        "openai",
        SimpleNamespace(OpenAI=openai_client),
    )

    VLMFunctionRouter(model="test-model")._build_client()

    assert captured["max_retries"] == 0


def test_vlm_planner_exposes_packages_only_through_open_app_tool() -> None:
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


def test_vlm_planner_sends_execution_history_to_model() -> None:
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

    payload = json.loads(completions.requests[0]["messages"][1]["content"][0]["text"])
    assert payload["screen_context"]["recent_actions"][0]["tool"] == "click"
    assert payload["screen_context"]["execution_history"].startswith("1.")
    assert payload["screen_context"]["function_execution"][
        "official_validator_status"
    ] == "pending"
    assert payload["screen_context"]["function_execution"]["final_observation"] == {
        "state_id": "state_after",
        "package_name": "com.example.shop",
        "activity_name": "CartActivity",
    }
    assert "must not be issued again" in payload["history_policy"]


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
    assert "provides search" in SYSTEM_PROMPT
    assert "history, recent, suggestion" in SYSTEM_PROMPT


def test_transient_obstruction_fast_path_runs_before_planner(
    tmp_path,
    monkeypatch,
) -> None:
    import omniflow.runtime.core as core
    import omniflow.runtime.engine as engine

    host = RecordingHost()
    planner = FinishingPlanner()
    recovery = Action("click", {"x": 500.0, "y": 500.0})
    calls = 0

    def recover(_observation: Observation) -> Action | None:
        nonlocal calls
        calls += 1
        return recovery if calls == 1 else None

    monkeypatch.setattr(engine, "transient_obstruction_recovery", recover)
    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    flow = OmniFlow(tmp_path / "store.json", host=host, planner=planner)

    result = flow.run("Dismiss the blocker and finish")

    assert result.success is True
    assert host.actions == [recovery]
    assert planner.visible_function_ids == [()]
    assert result.detail["trace"][0]["metadata"]["decision_origin"] == (
        "harness_fast_path"
    )


def test_function_completion_review_keeps_final_screenshot_and_checked_state() -> None:
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
                "recent_actions": [
                    {
                        "tool": "click",
                        "args": {"x": 500, "y": 400},
                        "success": True,
                        "function_id": "complete_run_turn_bluetooth_off",
                    }
                ],
                "execution_history": (
                    "Function `complete_run_turn_bluetooth_off` completed successfully."
                ),
            },
        },
        max_steps=8,
        turn_index=0,
    )

    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]
    assert '"checked":false' in content[0]["text"]
    assert "Those actions are already applied" in content[0]["text"]
    assert "Never repeat or toggle" in content[0]["text"]


def test_vlm_planner_function_completion_review_uses_final_screenshot() -> None:
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
                    "recent_actions": [
                        {
                            "tool": "click",
                            "args": {"x": 500, "y": 400},
                            "success": True,
                            "function_id": "complete_run_turn_bluetooth_off",
                        }
                    ],
                    "execution_history": (
                        "Function `complete_run_turn_bluetooth_off` completed successfully."
                    ),
                },
            ),
        )
    )

    assert planned == ToolCall("finished", {})
    request = completions.requests[0]
    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]
    turn_payload = json.loads(content[0]["text"])
    assert '"checked":false' in turn_payload["relevant_ui_elements"]
    assert "Never repeat or toggle" in turn_payload["completion_review"]


def test_androidworld_agent_installs_function_router(tmp_path) -> None:
    router = RejectingRouter()

    flow = build_agent(
        env=SimpleNamespace(),
        store_path=str(tmp_path / "empty-store.json"),
        function_router=router,
    )

    assert flow.function_router is router


def test_androidworld_agent_returns_target_states_when_source_catalog_exists(
    tmp_path,
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
    flow = build_agent(env=SimpleNamespace(), store_path=str(store_path))
    target_run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "com.android.settings"}],
        observations=[androidworld_state("target-before")],
        task_name="OpenSettings",
        run_id="target-run",
        goal="Open Settings.",
    )
    target_run_log["steps"][0]["next_observation"] = androidworld_state(
        "target-after"
    )
    flow.host.state.update(
        last_result=SimpleNamespace(success=True),
        last_run_log=target_run_log,
        captured_transfer_states={
            "target-before": {
                "state_id": "target-before",
                "xml": "<hierarchy />",
            }
        },
    )

    payload = flow.save_run_log(success=True)

    assert source_catalog.read_bytes() == original_source_catalog
    assert payload["captured_transfer_states"] == {
        "target-before": {
            "state_id": "target-before",
            "xml": "<hierarchy />",
        }
    }
    assert payload["transfer_state_audit"] == {
        "referenced_state_ids": ["target-after", "target-before"],
        "captured_state_ids": ["target-before"],
        "missing_state_ids": ["target-after"],
        "referenced_state_count": 2,
        "captured_state_count": 1,
        "missing_state_count": 1,
        "complete": False,
    }
