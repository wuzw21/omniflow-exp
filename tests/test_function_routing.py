from __future__ import annotations

import asyncio
from types import SimpleNamespace

from omniflow import (
    Action,
    ActionResult,
    Function,
    Observation,
    OmniFlow,
    ToolCall,
)
from omniflow.core.model import FunctionStep
from omniflow.functions.artifact import FUNCTION_ARTIFACT_VERSION
from omniflow.functions.store import FunctionStore
from omniflow.vlm.completion_checker import VLMCompletionChecker
from omniflow.vlm.function_router import VLMFunctionRouter
from omniflow.vlm.gui import build_model_turn_request
from omniflow.vlm.planner import VLMPlanner
from src.integrations.android_world.agent import build_agent


class RecordingHost:
    def __init__(self) -> None:
        self.package_name = "com.android.launcher"
        self.actions: list[Action] = []

    def observe(self, **kwargs: object) -> Observation:
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


class FinishingPlanner:
    def __init__(self) -> None:
        self.visible_function_ids: list[tuple[str, ...]] = []

    def one_step_tool_call(
        self,
        _goal: str,
        _observation: Observation,
        functions: tuple[Function, ...],
        _installed_apps: dict[str, str],
    ) -> ToolCall:
        self.visible_function_ids.append(tuple(function.id for function in functions))
        return ToolCall("finished", {"content": ""})


class AcceptingCompletionChecker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Observation, str]] = []

    def check_completion(
        self,
        goal: str,
        observation: Observation,
        action_summary: str,
    ) -> bool:
        self.calls.append((goal, observation, action_summary))
        return True


class RejectingCompletionChecker(AcceptingCompletionChecker):
    def check_completion(
        self,
        goal: str,
        observation: Observation,
        action_summary: str,
    ) -> bool:
        self.calls.append((goal, observation, action_summary))
        return False


class SequencePlanner(FinishingPlanner):
    def __init__(self, responses: list[ToolCall]) -> None:
        super().__init__()
        self.responses = list(responses)
        self.previous_action_errors: list[str | None] = []

    def one_step_tool_call(
        self,
        _goal: str,
        observation: Observation,
        functions: tuple[Function, ...],
        _installed_apps: dict[str, str],
    ) -> ToolCall:
        self.visible_function_ids.append(tuple(function.id for function in functions))
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
    completion_checker = AcceptingCompletionChecker()
    flow = OmniFlow(
        store_path,
        host=host,
        function_router=router,
        planner=planner,
        completion_checker=completion_checker,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert result.function_id == function_id
    assert [action.tool for action in host.actions] == ["open_app"]
    assert len(router.calls) == 1
    assert planner.visible_function_ids == []
    assert len(completion_checker.calls) == 1
    checked_goal, checked_observation, action_summary = completion_checker.calls[0]
    assert checked_goal == "Turn bluetooth on"
    assert checked_observation.image_base64 == "final-screenshot"
    assert "Turn bluetooth on" in action_summary


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


def test_rejected_completion_check_enters_gui_planner(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function_id = _store_with_open_settings_function(store_path)
    host = RecordingHost()
    router = AcceptingRouter()
    completion_checker = RejectingCompletionChecker()
    planner = FinishingPlanner()
    flow = OmniFlow(
        store_path,
        host=host,
        function_router=router,
        completion_checker=completion_checker,
        planner=planner,
        installed_apps={"Settings": "com.android.settings"},
    )

    result = flow.run("Turn bluetooth on")

    assert result.success is True
    assert result.function_id == function_id
    assert len(completion_checker.calls) == 1
    assert planner.visible_function_ids == [()]
    assert result.fallback_steps == 1


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


def test_vlm_completion_checker_only_receives_summary_and_final_screenshot() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="confirm_goal_finished",
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
    checker = VLMCompletionChecker(model="test-model", client=client)

    confirmed = asyncio.run(
        checker.check_completion(
            "Turn bluetooth on",
            Observation(
                xml="<hierarchy><node text='must-not-leak'/></hierarchy>",
                package_name="com.android.settings",
                image_base64="final-screenshot-base64",
                extra={
                    "installed_apps": {"Settings": "com.android.settings"},
                    "recent_actions": [{"tool": "click"}],
                },
            ),
            "Turn Bluetooth on ran successfully (open_app x1, click x4).",
        )
    )

    assert confirmed is True
    request = completions.requests[0]
    assert [tool["function"]["name"] for tool in request["tools"]] == [
        "confirm_goal_finished",
        "reject_goal_finished",
    ]
    request_text = request["messages"][1]["content"][0]["text"]
    assert request_text == (
        '{"goal":"Turn bluetooth on","action_summary":'
        '"Turn Bluetooth on ran successfully (open_app x1, click x4)."}'
    )
    assert request["messages"][1]["content"][1] == {
        "type": "image_url",
        "image_url": {
            "url": "data:image/png;base64,final-screenshot-base64"
        },
    }
    serialized_request = str(request)
    assert "installed_apps" not in serialized_request
    assert "must-not-leak" not in serialized_request
    assert "recent_actions" not in serialized_request
    assert "open_app" not in {
        tool["function"]["name"] for tool in request["tools"]
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


def test_androidworld_agent_installs_function_router(tmp_path) -> None:
    router = RejectingRouter()
    completion_checker = AcceptingCompletionChecker()

    flow = build_agent(
        env=SimpleNamespace(),
        store_path=str(tmp_path / "empty-store.json"),
        function_router=router,
        completion_checker=completion_checker,
    )

    assert flow.function_router is router
    assert flow.completion_checker is completion_checker
