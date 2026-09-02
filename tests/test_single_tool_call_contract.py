from __future__ import annotations

import asyncio
from types import SimpleNamespace

from omniflow.core.model import Action, Function, Observation, StepResult
from omniflow.runtime.execution import step_fact
from omniflow.vlm.gui import (
    build_model_turn_request,
    parse_model_turn_response,
    project_planner_context,
)
from omniflow.vlm.function_router import VLMFunctionRouter


def _function() -> Function:
    return Function.from_dict(
        {
            "schema_version": "omniflow.function.v2",
            "function_id": "edit_note_in_markor",
            "name": "Edit note",
            "description": "Edit the requested note.",
            "input_schema": {
                "type": "object",
                "properties": {"file_name": {"type": "string"}},
                "required": ["file_name"],
                "additionalProperties": False,
            },
            "bindings": [],
            "steps": [
                {
                    "step_index": 0,
                    "source_state_id": "state-1",
                    "action": {"tool": "wait", "args": {"duration_ms": 1}},
                }
            ],
            "agent_visible": True,
        }
    )


def test_planner_request_disables_parallel_tool_calls() -> None:
    request = build_model_turn_request(
        goal="Edit the note.",
        model="gpt-5.5",
        state={"xml": "<hierarchy />", "display": {"width": 720, "height": 1280}},
        max_steps=10,
        turn_index=1,
    )
    assert request["parallel_tool_calls"] is False


def test_planner_prompt_distinguishes_drags_from_scrolls() -> None:
    request = build_model_turn_request(
        goal="Set the value.",
        model="gpt-5.5",
        state={"xml": "<hierarchy />", "display": {"width": 720, "height": 1280}},
        max_steps=10,
        turn_index=1,
    )
    system_prompt = request["messages"][0]["content"]
    assert "physical drag as well as a scroll" in system_prompt
    assert "verify the requested value or state" in system_prompt
    assert "state_changed=false" in system_prompt
    assert "never repeat the same ineffective click or gesture" in system_prompt


def test_action_history_ignores_refreshed_screenshot_for_noop_detection() -> None:
    before = Observation(
        xml='<hierarchy><node text="Type" /></hierarchy>',
        package_name="net.gsantner.markor",
        activity_name="Markor",
        extra={"screenshot_path": "/tmp/before.png"},
    )
    after = Observation(
        xml=before.xml,
        package_name=before.package_name,
        activity_name=before.activity_name,
        extra={"screenshot_path": "/tmp/after.png"},
    )

    fact = step_fact(
        StepResult(True, action=Action("click", {"x": 500, "y": 500}), before=before, after=after)
    )

    assert fact["before_state_id"] != fact["after_state_id"]
    assert fact["metadata"]["action_effect"]["state_changed"] is False


def test_planner_context_exposes_native_slider_as_swipe_target() -> None:
    state = project_planner_context(
        {
            "package_name": "com.android.systemui",
            "display": {"width": 720, "height": 1280},
            "xml": (
                '<hierarchy width="720" height="1280">'
                '<node class="android.widget.SeekBar" '
                'resource-id="com.android.systemui:id/slider" '
                'bounds="[32,272][688,368]" scrollable="false" '
                'clickable="false" enabled="true" visible-to-user="true" />'
                "</hierarchy>"
            ),
        }
    )
    assert '"label":"slider"' in state["xml"]
    assert '"actions":["swipe"]' in state["xml"]


def test_parser_consumes_one_call_when_gateway_returns_multiple() -> None:
    function = _function()
    response = {
        "requested_model": "gpt-5.5",
        "resolved_model": "gpt-5.5",
        "tool_calls": [
            {
                "function": {
                    "name": "edit_note_in_markor",
                    "arguments": '{"file_name":"note.md"}',
                }
            },
            {
                "function": {
                    "name": "edit_note_in_markor",
                    "arguments": '{"file_name":"other.md"}',
                }
            },
        ],
    }

    tool_call, metadata = parse_model_turn_response(
        response,
        requested_model="gpt-5.5",
        turn_index=1,
        functions=(function,),
    )

    assert tool_call.name == "edit_note_in_markor"
    assert tool_call.arguments == {"file_name": "note.md"}
    assert metadata["discarded_extra_tool_calls"] == 1


def test_function_router_consumes_one_call_and_disables_parallel_calls() -> None:
    class Client:
        def __init__(self) -> None:
            self.kwargs = {}
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kwargs):
            self.kwargs = kwargs
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            tool_calls=[
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name="edit_note_in_markor",
                                        arguments='{"file_name":"note.md"}',
                                    )
                                ),
                                SimpleNamespace(
                                    function=SimpleNamespace(
                                        name="reject_recalled_function",
                                        arguments="{}",
                                    )
                                ),
                            ]
                        )
                    )
                ],
                usage=None,
            )

    client = Client()
    router = VLMFunctionRouter(model="gpt-5.5", client=client)
    call = asyncio.run(router.route_function("Edit the note.", (_function(),)))

    assert call is not None
    assert call.name == "edit_note_in_markor"
    assert client.kwargs["parallel_tool_calls"] is False
    assert client.kwargs["extra_body"] == {"parallel_tool_calls": False}
