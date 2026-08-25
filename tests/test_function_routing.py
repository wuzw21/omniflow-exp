from __future__ import annotations

import asyncio
import base64
from io import BytesIO
import json
import sys
from types import SimpleNamespace

from PIL import Image
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
from omniflow.functions.artifact import FUNCTION_ARTIFACT_VERSION
from omniflow.functions.store import FunctionStore
from omniflow.runtime.engine import _same_entry_observation
from omniflow.vlm.gui import (
    SYSTEM_PROMPT,
    ModelToolCallError,
    build_model_turn_request,
    function_tools,
    parse_model_turn_response,
)
from omniflow.vlm.planner import VLMPlanner, _configured_http_proxy
from src.integrations.android_world.agent import (
    _TaskHost,
    build_agent,
)


def test_planner_disables_hidden_openai_transport_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: dict[str, object] = {}

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            options.update(kwargs)

    monkeypatch.setitem(sys.modules, "openai", SimpleNamespace(OpenAI=FakeOpenAI))
    planner = VLMPlanner(
        model="test-model",
        api_key="test-key",
        base_url="https://example.test/v1",
    )

    planner._build_client()

    assert options["max_retries"] == 0


def test_planner_uses_http_proxy_instead_of_ambient_socks_proxy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ALL_PROXY", "socks5://127.0.0.1:7891")
    monkeypatch.setenv("all_proxy", "socks5://127.0.0.1:7891")
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:7890")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:7890")

    assert _configured_http_proxy() == "http://127.0.0.1:7890"


def test_malformed_planner_tool_payload_continues_with_live_context(tmp_path) -> None:
    class MalformedThenFinishPlanner(FinishingPlanner):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def one_step_tool_call(
            self,
            _goal: str,
            observation: Observation,
            functions: tuple[Function, ...],
            installed_apps: dict[str, str],
        ) -> ToolCall:
            self.calls += 1
            self.observations.append(observation)
            if self.calls == 1:
                raise ValueError(
                    "canonical_action_args_unknown:input_text:text"
                )
            return ToolCall("finished", {"content": "done"})

    planner = MalformedThenFinishPlanner()
    result = OmniFlow(
        tmp_path / "store.json",
        host=RecordingHost(),
        planner=planner,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    ).run("Finish the task")

    assert result.success is True
    assert planner.calls == 2
    assert planner.observations[1].extra["previous_action_error"] == (
        "vlm_planner_failed:canonical_action_args_unknown:input_text:text"
    )


def test_empty_model_tool_name_continues_with_live_context(tmp_path) -> None:
    class EmptyToolThenFinishPlanner(FinishingPlanner):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def one_step_tool_call(
            self,
            _goal: str,
            observation: Observation,
            _functions: tuple[Function, ...],
            _installed_apps: dict[str, str],
        ) -> ToolCall:
            self.calls += 1
            self.observations.append(observation)
            if self.calls == 1:
                raise ModelToolCallError("model_turn_tool_not_visible:")
            return ToolCall("finished", {"content": "done"})

    planner = EmptyToolThenFinishPlanner()
    result = OmniFlow(
        tmp_path / "store.json",
        host=RecordingHost(),
        planner=planner,
        config=OmniFlowConfig(runtime=RuntimeSettings(max_steps=3)),
    ).run("Finish the task")

    assert result.success is True
    assert planner.calls == 2
    assert planner.observations[1].extra["previous_action_error"] == (
        "vlm_planner_failed:model_turn_tool_not_visible:"
    )


class RecordingHost:
    def __init__(self) -> None:
        self.package_name = "com.android.launcher"
        self.actions: list[Action] = []
        self.observe_requests: list[dict[str, object]] = []

    def observe(self, **kwargs: object) -> Observation:
        self.observe_requests.append(dict(kwargs))
        xml = (
            '<hierarchy><node package="%s" class="android.widget.TextView" '
            'text="Current page" bounds="[0,0][1000,1000]" /></hierarchy>'
            % self.package_name
        )
        return Observation(
            xml=xml,
            package_name=self.package_name,
            activity_name="MainActivity",
            image_base64=(
                "final-screenshot" if kwargs.get("screenshot") is True else None
            ),
            extra={
                "state_id": f"state_{len(self.actions)}",
                "display": {"width": 1000, "height": 1000},
            },
        )

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        if action.tool == "open_app":
            self.package_name = str(action.args["package_name"])
        return ActionResult(True)

    def get_state(self, source_state_id: str) -> Observation | None:
        if source_state_id == "missing_source_state":
            return None
        package = (
            "com.android.launcher"
            if source_state_id == "source_home"
            or source_state_id.startswith("source_")
            else self.package_name
        )
        return Observation(
            xml=(
                '<hierarchy><node package="%s" class="android.widget.TextView" '
                'text="Current page" bounds="[0,0][1000,1000]" /></hierarchy>'
                % package
            ),
            package_name=package,
            activity_name="MainActivity",
            extra={
                "state_id": source_state_id,
                "display": {"width": 1000, "height": 1000},
            },
        )


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


class _OfflineMultiplyHost:
    numbers = (2, 3, 4, 5, 1)

    def __init__(self) -> None:
        self.opened = False
        self.click_count = 0
        self.entered_product: str | None = None
        self.submitted = False
        self.actions: list[Action] = []

    def observe(self, **kwargs: object) -> Observation:
        if not self.opened:
            xml = (
                '<hierarchy><node text="Home" bounds="[0,0][1000,1000]" />'
                "</hierarchy>"
            )
            package = "com.android.launcher"
        elif self.click_count < len(self.numbers):
            visible_number = (
                "Ready"
                if self.click_count == 0
                else str(self.numbers[self.click_count - 1])
            )
            xml = (
                '<hierarchy><node text="Click Me" bounds="[350,200][650,400]" '
                'clickable="true" enabled="true" />'
                f'<node text="{visible_number}" bounds="[350,450][650,550]" />'
                "</hierarchy>"
            )
            package = "com.android.chrome"
        elif self.submitted:
            xml = (
                '<hierarchy><node text="Success" bounds="[0,0][1000,1000]" />'
                "</hierarchy>"
            )
            package = "com.android.chrome"
        else:
            xml = (
                '<hierarchy><node class="android.widget.EditText" '
                'resource-id="product" bounds="[200,500][800,650]" '
                'editable="true" enabled="true" focused="true" />'
                '<node text="Submit" bounds="[350,700][650,900]" '
                'clickable="true" enabled="true" /></hierarchy>'
            )
            package = "com.android.chrome"
        return Observation(
            xml=xml,
            package_name=package,
            activity_name="MainActivity",
            image_base64="offline" if kwargs.get("screenshot") else None,
            extra={
                "state_id": (
                    f"target_{self.click_count}_{self.entered_product}_{self.submitted}"
                ),
                "display": {"width": 1000, "height": 1000},
            },
        )

    def act(self, action: Action) -> ActionResult:
        self.actions.append(action)
        if action.tool == "open_app":
            self.opened = True
        elif action.tool == "click" and self.click_count < len(self.numbers):
            self.click_count += 1
        elif action.tool == "input_text":
            self.entered_product = str(action.args["text"])
        elif action.tool == "click" and self.entered_product is not None:
            self.submitted = True
        return ActionResult(True)

    def get_state(self, source_state_id: str) -> Observation:
        if source_state_id == "source_start":
            return Observation(
                xml='<hierarchy><node text="Home" /></hierarchy>',
                package_name="com.android.launcher",
                extra={"state_id": source_state_id},
            )
        if source_state_id == "source_form":
            xml = (
                '<hierarchy><node class="android.widget.EditText" '
                'resource-id="product" bounds="[200,500][800,650]" '
                'editable="true" enabled="true" focused="true" /></hierarchy>'
            )
        elif source_state_id == "source_submit":
            xml = (
                '<hierarchy><node text="Submit" bounds="[350,700][650,900]" '
                'clickable="true" enabled="true" /></hierarchy>'
            )
        else:
            xml = (
                '<hierarchy><node text="Click Me" bounds="[350,200][650,400]" '
                'clickable="true" enabled="true" /></hierarchy>'
            )
        return Observation(
            xml=xml,
            package_name="com.android.chrome",
            extra={
                "state_id": source_state_id,
                "display": {"width": 1000, "height": 1000},
            },
        )


class _OfflineMultiplyPlanner(FinishingPlanner):
    def __init__(self, host: _OfflineMultiplyHost) -> None:
        super().__init__()
        self.host = host
        self.global_selected = False

    def one_step_tool_call(
        self,
        _goal: str,
        observation: Observation,
        functions: tuple[Function, ...],
        _installed_apps: dict[str, str],
    ) -> ToolCall:
        ids = tuple(function.id for function in functions)
        self.visible_function_ids.append(ids)
        self.observations.append(observation)
        if not self.global_selected:
            assert "complete_browser_multiply" in ids
            self.global_selected = True
            return ToolCall("complete_browser_multiply", {})
        if self.host.click_count < len(self.host.numbers):
            assert "click_number" in ids, ids
            return ToolCall("click_number", {})
        if self.host.entered_product is None:
            assert "enter_product" in ids, ids
            return ToolCall("enter_product", {"product": "120"})
        if not self.host.submitted:
            assert "submit_product" in ids, ids
            return ToolCall("submit_product", {})
        return ToolCall("finished", {"content": "120"})


def test_offline_browser_multiply_completes_with_global_then_local_functions(
    tmp_path,
    monkeypatch,
) -> None:
    import omniflow.runtime.core as core

    monkeypatch.setattr(core, "_ACTION_SETTLE_SECONDS", 0.0)
    store_path = tmp_path / "store.json"
    store = FunctionStore(store_path)
    click = Action("click", {"x": 500.0, "y": 300.0})
    empty_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }
    store.put_function(
        Function(
            function_id="complete_browser_multiply",
            name="Complete BrowserMultiply task",
            description="Open the task, collect numbers, enter their product, and submit.",
            steps=(
                FunctionStep(
                    0,
                    Action("open_app", {"package_name": "com.android.chrome"}),
                    "source_start",
                ),
                *tuple(
                    FunctionStep(index + 1, click, f"source_number_{index}")
                    for index in range(5)
                ),
                FunctionStep(
                    6,
                    Action("click", {"x": 500.0, "y": 800.0}),
                    "source_submit",
                ),
            ),
            schema_version=FUNCTION_ARTIFACT_VERSION,
            input_schema=empty_schema,
            agent_visible=True,
        )
    )
    store.put_function(
        Function(
            function_id="click_number",
            name="Click number button",
            description="Click the visible button once to reveal one number.",
            steps=(FunctionStep(0, click, "source_number_0"),),
            schema_version=FUNCTION_ARTIFACT_VERSION,
            input_schema=empty_schema,
            agent_visible=True,
        )
    )
    store.put_function(
        Function(
            function_id="enter_product",
            name="Enter product",
            description="Enter the computed product in the visible field.",
            steps=(
                FunctionStep(
                    0,
                    Action(
                        "input_text",
                        {"text": "", "target_description": "product"},
                    ),
                    "source_form",
                ),
            ),
            schema_version=FUNCTION_ARTIFACT_VERSION,
            input_schema={
                "type": "object",
                "properties": {"product": {"type": "string"}},
                "required": ["product"],
                "additionalProperties": False,
            },
            bindings=(
                {
                    "source": "$.arguments.product",
                    "target": "$.steps[0].action.args.text",
                },
            ),
            agent_visible=True,
        )
    )
    store.put_function(
        Function(
            function_id="submit_product",
            name="Submit product",
            description="Submit the entered product.",
            steps=(
                FunctionStep(
                    0,
                    Action("click", {"x": 500.0, "y": 800.0}),
                    "source_submit",
                ),
            ),
            schema_version=FUNCTION_ARTIFACT_VERSION,
            input_schema=empty_schema,
            agent_visible=True,
        )
    )

    host = _OfflineMultiplyHost()

    async def transfer(
        action: Action,
        _observation: Observation,
        source_state: Observation | None,
    ) -> TransferResult:
        source_id = str((source_state.extra if source_state else {}).get("state_id") or "")
        if source_id.startswith("source_number_") and host.click_count < 5:
            return TransferResult(
                Action("click", {"x": 500.0, "y": 300.0}),
                detail={"absolute_contextual_confidence": 0.95},
            )
        if source_id == "source_form" and host.click_count == 5:
            return TransferResult(
                Action(
                    "input_text",
                    {
                        "text": action.args.get("text", ""),
                        "target_description": "product",
                        "x": 500.0,
                        "y": 575.0,
                    },
                ),
                detail={"absolute_contextual_confidence": 0.95},
            )
        if (
            source_id == "source_submit"
            and host.click_count == 5
            and host.entered_product is not None
        ):
            return TransferResult(
                Action("click", {"x": 500.0, "y": 800.0}),
                detail={"absolute_contextual_confidence": 0.95},
            )
        return TransferResult(None, reason="offline_state_mismatch")

    planner = _OfflineMultiplyPlanner(host)
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Chrome": "com.android.chrome"},
        config=OmniFlowConfig(
            runtime=RuntimeSettings(max_steps=20, max_fallback_steps=10),
            plugins=PluginSet(transfer=transfer),
        ),
    )

    result = flow.run("Click five numbers, multiply them, and submit the product.")

    assert result.success is True, json.dumps(
        result.detail["function_resolution"]["recall"]["events"][-1]["decisions"],
        indent=2,
    )
    assert host.click_count == 5
    assert host.entered_product == "120"
    assert host.submitted is True
    assert result.fallback_steps == 1
    assert planner.visible_function_ids[0] == ("complete_browser_multiply",)
    assert planner.visible_function_ids.count(("click_number",)) == 5


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
    assert planner.visible_function_ids == [(function_id,), ()]
    assert planner.observations[0].image_base64 == "final-screenshot"
    assert [request.get("screenshot") for request in host.observe_requests] == [
        False,
        True,
        True,
        True,
    ]
    assert "Function `complete_run_turn_bluetooth_on`" in str(
        planner.observations[1].extra.get("execution_history")
    )
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
    assert function_resolution["candidate_count"] == 0
    assert function_resolution["candidate_function_ids"] == []
    assert function_resolution["status"] == "planner_tool_space"
    assert [
        event["candidate_function_ids"]
        for event in function_resolution["recall"]["events"]
    ] == [[function_id], []]
    assert result.detail["runtime_limits"] == {
        "max_steps": 20,
        "max_fallback_steps": None,
    }


def test_planner_fills_function_schema_arguments_in_e2e_loop(tmp_path) -> None:
    store_path = tmp_path / "store.json"
    function = Function(
        function_id="open_requested_app",
        name="Open requested app",
        description="Open the app requested by the current goal.",
        steps=(
            FunctionStep(
                step_index=0,
                source_state_id="source_home",
                action=Action("open_app", {"package_name": "source.package"}),
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {"package_name": {"type": "string"}},
            "required": ["package_name"],
            "additionalProperties": False,
        },
        bindings=(
            {
                "source": "$.arguments.package_name",
                "target": "$.steps[0].action.args.package_name",
            },
        ),
        agent_visible=True,
    )
    FunctionStore(store_path).put_function(function)
    host = RecordingHost()
    planner = SequencePlanner(
        [
            ToolCall(function.id, {"package_name": "target.package"}),
            ToolCall("finished", {"content": ""}),
        ]
    )
    flow = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        installed_apps={"Target": "target.package"},
    )

    result = flow.run("Open the target app")

    assert result.success is True
    assert planner.visible_function_ids == [(function.id,), ()]
    assert host.actions == [Action("open_app", {"package_name": "target.package"})]


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
    assert planner.visible_function_ids == [(function_id,), ()]
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


def test_selected_function_is_not_executed_after_entry_state_changes(tmp_path) -> None:
    class ChangingHost:
        def __init__(self) -> None:
            self.page = "entry"
            self.actions: list[Action] = []

        def observe(self, **_kwargs: object) -> Observation:
            if self.page == "entry":
                xml = (
                    '<hierarchy><node class="android.widget.Button" text="Click Me" '
                    'bounds="[100,100][300,200]" clickable="true" enabled="true" />'
                    "</hierarchy>"
                )
            else:
                xml = (
                    '<hierarchy><node class="android.widget.TextView" text="Changed" '
                    'bounds="[0,0][400,800]" /></hierarchy>'
                )
            return Observation(
                xml=xml,
                package_name="com.example",
                extra={
                    "state_id": self.page,
                    "display": {"width": 400, "height": 800},
                },
            )

        def get_state(self, _source_state_id: str) -> Observation:
            return self.observe()

        def act(self, action: Action) -> ActionResult:
            self.actions.append(action)
            return ActionResult(True)

    class StateChangingPlanner(SequencePlanner):
        def __init__(self, host: ChangingHost, function_id: str) -> None:
            super().__init__([ToolCall(function_id, {}), ToolCall("finished", {})])
            self.host = host

        def one_step_tool_call(self, *args: object, **kwargs: object) -> ToolCall:
            call = super().one_step_tool_call(*args, **kwargs)
            if len(self.goals) == 1:
                self.host.page = "changed"
            return call

    function = Function(
        function_id="click_me",
        name="Click me",
        description="Click the visible Click Me button.",
        steps=(
            FunctionStep(0, Action("click", {"x": 500, "y": 187.5}), "source"),
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
    store_path = tmp_path / "store.json"
    FunctionStore(store_path).put_function(function)
    host = ChangingHost()
    planner = StateChangingPlanner(host, function.id)

    def transfer(
        action: Action,
        _observation: Observation,
        _source_state: Observation | None,
    ) -> TransferResult:
        return TransferResult(
            Action(action.tool, {"x": 500.0, "y": 187.5}),
            reason="omnitransfer_unified_association_v1",
            detail={"absolute_contextual_confidence": 0.99},
        )

    result = OmniFlow(
        store_path,
        host=host,
        planner=planner,
        config=OmniFlowConfig(plugins=PluginSet(transfer=transfer)),
    ).run("Click the Click Me button")

    assert result.success is True
    assert host.actions == []
    assert planner.previous_action_errors == [
        None,
        "function_entry_state_changed_after_mapping",
    ]


def test_function_entry_gate_uses_canonical_state_id_over_screenshot_bytes() -> None:
    before = Observation(
        xml='<hierarchy><node text="Launcher" /></hierarchy>',
        package_name="com.example",
        activity_name="MainActivity",
        image_base64="screenshot-before",
        extra={"state_id": "state_same"},
    )
    after = Observation(
        xml=before.xml,
        package_name=before.package_name,
        activity_name=before.activity_name,
        image_base64="screenshot-after",
        extra={"state_id": "state_same"},
    )

    assert _same_entry_observation(before, after) is True


def test_function_entry_gate_rejects_changed_canonical_state_id() -> None:
    before = Observation(
        xml='<hierarchy><node text="Launcher" /></hierarchy>',
        package_name="com.example",
        activity_name="MainActivity",
        image_base64="same-screenshot",
        extra={"state_id": "state_before"},
    )
    after = Observation(
        xml=before.xml,
        package_name=before.package_name,
        activity_name=before.activity_name,
        image_base64=before.image_base64,
        extra={"state_id": "state_after"},
    )

    assert _same_entry_observation(before, after) is False


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
    assert planner.visible_function_ids == [(), ()]
    assert planner.previous_action_errors[0] == "omnitransfer_source_state_missing"


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
    assert planner.previous_action_errors[0] == "omnitransfer_source_state_missing"
    assert "Continue Function" in planner.goals[0]
    assert "Do not repeat actions that already succeeded" in planner.goals[0]
    assert result.detail["function_resolution"]["status"] == "direct"
    assert result.detail["function_resolution"]["replay_status"] == "failed"


def test_function_failure_returns_to_offline_resume_after_planner_recovery(
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
    assert result.fallback_steps == 1
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
    resumed_steps = [
        step
        for step in result.detail["trace"]
        if step.get("metadata", {}).get("function_alignment")
    ]
    assert len(resumed_steps) == 1
    assert resumed_steps[0]["metadata"]["function_alignment"] == {
        "protocol": "weighted_lcs_v1",
        "start_step_index": 1,
        "resume_step_index": 1,
        "probability": 0.9,
        "score": pytest.approx(2.1972245773362196),
        "minimum_probability": 0.8,
        "source_skip_penalty": pytest.approx(1.0986122886681098),
        "target_observation_count": 2,
        "path": [
            {
                "function_step_index": 1,
                "target_observation_index": 1,
                "probability": 0.9,
            }
        ],
    }
    assert result.detail["function_resume"] == {
        "schema_version": "omniflow.function-resume-audit.v1",
        "events": [
                {
                    "start_step_index": 1,
                    "status": "succeeded",
                    "trigger": "function_replay_failure",
                    "resume_step_index": 1,
                "probability": 0.9,
                "score": pytest.approx(2.1972245773362196),
            }
        ],
        "attempt_count": 1,
        "success_count": 1,
    }
    assert result.execution_summary["fallback_steps"] == 1


def test_completed_function_cannot_be_recalled_from_the_wrong_entry_page(
    tmp_path,
) -> None:
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
    ]
    assert planner.visible_function_ids == [
        (function_id,),
        (),
        (),
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


class SequenceCompletions:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.requests: list[dict[str, object]] = []

    def create(self, **request: object) -> object:
        self.requests.append(request)
        return self.responses.pop(0)


def _planner_response(tool: str, arguments: dict[str, object]) -> object:
    arguments = {"summary": f"Use {tool}", **arguments}
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name=tool,
                                arguments=json.dumps(arguments),
                            )
                        )
                    ]
                )
            )
        ],
        usage=None,
    )


def test_vlm_planner_does_not_retry_invalid_coordinates() -> None:
    completions = SequenceCompletions(
        [
            _planner_response("click", {"x": [361, 1136]}),
            _planner_response("click", {"x": 361, "y": 1136}),
        ]
    )
    planner = VLMPlanner(
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )

    with pytest.raises(ModelToolCallError, match="canonical_action_arg_type_invalid:x"):
        asyncio.run(
            planner.one_step_tool_call(
                "Open Downloads",
                Observation(extra={"display": {"width": 720, "height": 1280}}),
            )
        )
    assert len(completions.requests) == 1
    assert planner.take_metadata()["rejected_tool_calls"] == [
        {
            "turn_index": 1,
            "tool": "click",
            "error": "canonical_action_arg_type_invalid:x",
            "arguments": {"summary": "Use click", "x": [361, 1136]},
        }
    ]


def test_vlm_planner_does_not_retry_blank_tool() -> None:
    completions = SequenceCompletions(
        [
            _planner_response("", {}),
            _planner_response("click", {"x": 200, "y": 150}),
        ]
    )
    planner = VLMPlanner(
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    observation = Observation(
        xml=(
            '<hierarchy><node class="android.widget.Button" text="Click Me" '
            'bounds="[100,100][300,200]" clickable="true" /></hierarchy>'
        ),
        image_base64="screen-evidence",
        extra={"display": {"width": 400, "height": 800}},
    )

    with pytest.raises(ModelToolCallError, match="model_turn_tool_not_visible"):
        asyncio.run(
            planner.one_step_tool_call("Click the Click Me button", observation)
        )
    assert len(completions.requests) == 1


def test_vlm_planner_binds_blank_name_to_single_global_function() -> None:
    function = Function(
        function_id="launch_task",
        name="Launch task",
        description="Open the app and begin the task.",
        steps=(
            FunctionStep(
                0,
                Action("open_app", {"package_name": "com.example.app"}),
                "source-start",
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )
    tool_call, _ = parse_model_turn_response(
        {
            "requested_model": "test-model",
            "resolved_model": "test-model",
            "tool_calls": [
                {"function": {"name": "", "arguments": "{}"}},
            ],
        },
        requested_model="test-model",
        turn_index=1,
        functions=(function,),
    )

    assert tool_call.name == "launch_task"


def test_vlm_planner_does_not_retry_invalid_open_app_package() -> None:
    completions = SequenceCompletions(
        [
            _planner_response(
                "open_app",
                {"package_name": "com.android.filemanager"},
            ),
            _planner_response(
                "open_app",
                {"package_name": "com.google.android.documentsui"},
            ),
        ]
    )
    planner = VLMPlanner(
        model="test-model",
        client=SimpleNamespace(chat=SimpleNamespace(completions=completions)),
    )
    installed_apps = {"Files": "com.google.android.documentsui"}

    with pytest.raises(
        ModelToolCallError,
        match="planner_open_app_package_not_installed",
    ):
        asyncio.run(
            planner.one_step_tool_call(
                "Open Downloads in Files",
                Observation(extra={"display": {"width": 720, "height": 1280}}),
                installed_apps=installed_apps,
            )
        )
    assert len(completions.requests) == 1
    assert planner.take_metadata()["rejected_tool_calls"][0]["arguments"] == {
        "summary": "Use open_app",
        "package_name": "com.android.filemanager",
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
                                arguments='{"summary":"Finish"}',
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
    assert request["max_tokens"] == 512
    assert "max_completion_tokens" not in request
    assert request["stream"] is True
    assert "stream_options" not in request
    assert request["reasoning_effort"] == "none"
    assert request["parallel_tool_calls"] is False
    assert request["extra_body"] == {
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    }
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
            assert (
                function["parameters"]["properties"]["package_name"]["description"]
                == "Exact installed launchable package. Runtime app mapping: "
                "Chrome=com.android.chrome, Settings=com.android.settings"
            )
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
                                arguments='{"summary":"Finish"}',
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

    turn_text = completions.requests[0]["messages"][1]["content"][0]["text"]
    assert "Completed tool-call history:" in turn_text
    assert "1. [Planner] Clicked the item successfully." in turn_text
    assert '"tool":"click"' in turn_text
    assert '"official_validator_status":"pending"' in turn_text
    assert '"state_id":"state_after"' in turn_text
    assert "Do not repeat the same action" in turn_text


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


def test_global_startup_function_owns_open_app_tool_slot() -> None:
    function = Function(
        function_id="complete_task",
        name="Complete task",
        description="Open the app and complete the stable workflow prefix.",
        steps=(
            FunctionStep(
                0,
                Action("open_app", {"package_name": "com.android.documentsui"}),
                "source-start",
            ),
        ),
        schema_version=FUNCTION_ARTIFACT_VERSION,
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
    )
    request = build_model_turn_request(
        goal="Open the task and continue",
        model="test-model",
        state={"xml": "", "display": {"width": 720, "height": 1280}},
        max_steps=8,
        turn_index=0,
        installed_apps={"Files": "com.android.documentsui"},
        functions=(function,),
    )

    names = [tool["function"]["name"] for tool in request["tools"]]
    assert names == ["complete_task"]
    assert request["tool_choice"] == {
        "type": "function",
        "function": {"name": "complete_task"},
    }


def test_planner_defaults_to_xml_without_uploading_native_screenshot() -> None:
    request = build_model_turn_request(
        goal="Open the Timer tab",
        model="test-model",
        state={
            "xml": (
                '<hierarchy><node text="Timer" clickable="true" '
                'bounds="[288,1072][432,1232]" /></hierarchy>'
            ),
            "image_base64": "should-not-be-uploaded",
            "display": {"width": 720, "height": 1280},
        },
        max_steps=8,
        turn_index=0,
    )

    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text"]
    assert "Screenshot upload is omitted" in content[0]["text"]


def test_planner_keeps_screenshot_for_webview_grounding() -> None:
    request = build_model_turn_request(
        goal="Click Chrome in the web page",
        model="test-model",
        state={
            "xml": (
                '<hierarchy><node class="android.webkit.WebView" '
                'bounds="[0,0][720,1280]"><node class="android.view.View" '
                'text="Chrome" clickable="true" bounds="[0,766][720,878]" />'
                "</node></hierarchy>"
            ),
            "image_base64": "must-be-uploaded",
            "display": {"width": 720, "height": 1280},
        },
        max_steps=8,
        turn_index=0,
    )

    content = request["messages"][1]["content"]
    assert [item["type"] for item in content] == ["text", "image_url"]


def test_bridge_planner_uses_unified_short_decision_policy() -> None:
    request = build_model_turn_request(
        goal="Search for a contact",
        model="test-model",
        state={"xml": "", "display": {"width": 720, "height": 1280}},
        max_steps=8,
        turn_index=0,
    )

    assert request["max_tokens"] == 512
    assert "max_completion_tokens" not in request
    assert request["stream"] is True
    assert "stream_options" not in request
    assert request["reasoning_effort"] == "none"
    assert request["enable_thinking"] is False
    assert request["thinking"] == {"type": "disabled"}
    assert "summary of at most 12 words" in SYSTEM_PROMPT
    assert "Do not emit analysis, chain-of-thought" in SYSTEM_PROMPT
    assert "return only the tool call" in SYSTEM_PROMPT
    assert "Never call a recalled Function merely because it matches" in SYSTEM_PROMPT
    assert "finish onboarding and navigation" in SYSTEM_PROMPT
    assert "provides search" in SYSTEM_PROMPT
    assert "history, recent, suggestion" in SYSTEM_PROMPT
    assert "not claim that a RunLog or reusable Function was registered" in SYSTEM_PROMPT
    assert "return the answer through finished(content)" in SYSTEM_PROMPT
    assert "When you choose a projected native XML node" in SYSTEM_PROMPT
    assert "does not apply\nto WebView or screenshot-only visual targets" in SYSTEM_PROMPT
    assert "do not repeat the same coordinates" in SYSTEM_PROMPT


def test_clicking_unique_projected_native_node_uses_bounds_center() -> None:
    state = {
        "xml": (
            '<hierarchy><node class="android.widget.LinearLayout" '
            'bounds="[0,766][720,878]">'
            '<node class="android.widget.TextView" text="Chrome" '
            'bounds="[96,800][209,843]" /></node></hierarchy>'
        ),
        "display": {"width": 720, "height": 1280},
    }
    response = {
        "requested_model": "test-model",
        "resolved_model": "test-model",
        "tool_calls": [
            {
                "function": {
                    "name": "click",
                    "arguments": json.dumps(
                        {
                            "summary": "Open Chrome",
                            "target_description": "Chrome",
                            "x": 154,
                            "y": 640,
                        }
                    ),
                }
            }
        ],
    }

    tool_call, metadata = parse_model_turn_response(
        response,
        requested_model="test-model",
        turn_index=1,
        display=state["display"],
        state=state,
        goal="Open the file with Chrome",
    )

    assert tool_call.name == "click"
    assert tool_call.arguments["x"] == pytest.approx(211.8055555556)
    assert tool_call.arguments["y"] == pytest.approx(641.796875)
    assert metadata["node_grounding"] == {
        "name": "projected_node_center.v1",
        "reference": "",
        "target_description": "Chrome",
        "bounds": [96, 800, 209, 843],
        "original_raw_pixels": {"x": 154, "y": 640},
        "grounded_raw_pixels": {"x": 152.5, "y": 821.5},
    }


def test_projected_node_center_does_not_override_webview_click() -> None:
    state = {
        "xml": (
            '<hierarchy><node class="android.webkit.WebView" '
            'bounds="[0,0][720,1280]">'
            '<node class="android.view.View" text="Chrome" clickable="true" '
            'bounds="[0,766][720,878]" /></node></hierarchy>'
        ),
        "display": {"width": 720, "height": 1280},
    }
    response = {
        "requested_model": "test-model",
        "resolved_model": "test-model",
        "tool_calls": [
            {
                "function": {
                    "name": "click",
                    "arguments": json.dumps(
                        {
                            "summary": "Click visual target",
                            "target_description": "Chrome",
                            "x": 154,
                            "y": 640,
                        }
                    ),
                }
            }
        ],
    }

    tool_call, metadata = parse_model_turn_response(
        response,
        requested_model="test-model",
        turn_index=1,
        display=state["display"],
        state=state,
        goal="Click Chrome in the web page",
    )

    assert tool_call.arguments["x"] == pytest.approx(213.8888888889)
    assert tool_call.arguments["y"] == 500
    assert "node_grounding" not in metadata


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
    assert content[1]["image_url"]["detail"] == "low"
    assert '"checked":false' in content[0]["text"]
    assert "Those actions are already applied" in content[0]["text"]
    assert "Never repeat or toggle" in content[0]["text"]


def test_planner_compacts_large_screenshot_before_upload() -> None:
    image = Image.new("RGB", (720, 1280), color="white")
    output = BytesIO()
    image.save(output, format="PNG")
    encoded = base64.b64encode(output.getvalue()).decode("ascii")

    request = build_model_turn_request(
        goal="Inspect the current page",
        model="test-model",
        state={
            "xml": "<hierarchy />",
            "image_base64": encoded,
            "display": {"width": 720, "height": 1280},
        },
        max_steps=8,
        turn_index=0,
    )

    image_url = request["messages"][1]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/jpeg;base64,")
    compact = Image.open(BytesIO(base64.b64decode(image_url.split(",", 1)[1])))
    assert compact.size == (360, 640)


def test_vlm_planner_function_completion_review_uses_final_screenshot() -> None:
    response = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    tool_calls=[
                        SimpleNamespace(
                            function=SimpleNamespace(
                                name="finished",
                                arguments='{"summary":"Finish"}',
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
    assert content[1]["image_url"]["detail"] == "low"
    turn_text = content[0]["text"]
    assert '"checked":false' in turn_text
    assert "Never repeat or toggle" in turn_text


@pytest.mark.skip(reason="current experiment lifecycle owns method construction")
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


@pytest.mark.skip(reason="covered by the current AndroidWorld adapter tests")
def test_androidworld_agent_exposes_target_states_when_source_catalog_exists(
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
