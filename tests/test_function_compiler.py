from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.functions.compiler import compile_runlog_to_store


def _run_log(step_count: int) -> dict:
    actions = [
        (
            {"action_type": "open_app", "app_name": "com.android.settings"}
            if index == 0
            else {"action_type": "wait"}
        )
        for index in range(step_count)
    ]
    return androidworld_run_log(
        actions,
        observations=[
            androidworld_state(f"state_{index}")
            for index in range(step_count)
        ],
        goal="Open Settings and wait.",
    )


def test_compiler_requires_agent_authoring(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="function_author_agent_required"):
        compile_runlog_to_store(
            _run_log(1),
            tmp_path / "output",
            source_states={"state_0": {"state_id": "state_0"}},
        )


def test_compiler_freezes_only_function_referenced_states(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    bundle = {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {"open_settings_and_wait": {}},
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "open_settings_and_wait",
                "name": "Open Settings and wait once",
                "description": (
                    "Open the recorded Settings package and wait once for its "
                    "initial page. This does not change any setting or verify a task."
                ),
                "input_schema": {
                    "type": "object",
                    "properties": {},
                    "required": [],
                    "additionalProperties": False,
                },
                "bindings": [],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "state_0",
                        "action": {
                            "tool": "open_app",
                            "args": {"package_name": "com.android.settings"},
                        },
                    },
                    {
                        "step_index": 1,
                        "source_state_id": "state_1",
                        "action": {"tool": "wait", "args": {"duration_ms": 1000}},
                    },
                ],
                "checker_rules": [],
                "agent_visible": True,
            }
        ],
    }
    result = compile_runlog_to_store(
        _run_log(2),
        output,
        function_bundle=bundle,
        source_states={
            "state_0": {"state_id": "state_0"},
            "state_1": {"state_id": "state_1"},
            "unused": {"state_id": "unused"},
        },
    )

    catalog = json.loads(
        Path(result["transfer_state_catalog"]).read_text(encoding="utf-8")
    )
    assert set(catalog["states"]) == {"state_0", "state_1"}
    assert result["transfer_state_count"] == 2
    assert result["source_arguments"] == {"open_settings_and_wait": {}}
    assert result["model_calls"] == 0
    assert result["prompt_tokens"] == 0
    assert result["completion_tokens"] == 0
    assert result["total_tokens"] == 0


def _single_input_bundle(parameter_name: str, *, text: str = "Paid by card") -> dict:
    return {
        "schema_version": "omniflow.function-bundle.v2",
        "run_id": "source-run",
        "arguments": {"enter_expense_field": {parameter_name: text}},
        "functions": [
            {
                "schema_version": "omniflow.function.v2",
                "function_id": "enter_expense_field",
                "name": "Enter expense field",
                "description": "Enter a value in the expense form.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        parameter_name: {
                            "type": "string",
                            "description": f"Expense {parameter_name.replace('_', ' ')}",
                        }
                    },
                    "required": [parameter_name],
                    "additionalProperties": False,
                },
                "bindings": [
                    {
                        "source": f"$.arguments.{parameter_name}",
                        "target": "$.steps[0].action.args.text",
                    }
                ],
                "steps": [
                    {
                        "step_index": 0,
                        "source_state_id": "note-state",
                        "action": {"tool": "input_text", "args": {"text": ""}},
                    }
                ],
                "checker_rules": [],
                "agent_visible": True,
            }
        ],
    }


def _note_input_run_log() -> dict:
    return androidworld_run_log(
        [{"action_type": "input_text", "text": "Paid by card"}],
        observations=[
            androidworld_state(
                "note-state",
                forest=(
                    '<hierarchy><node text="Note" resource-id="app:id/note" '
                    'class="android.widget.EditText" editable="true" '
                    'focused="true" bounds="[0,0][100,100]" /></hierarchy>'
                ),
            )
        ],
        goal="Enter an expense note.",
    )


def test_compiler_rejects_agent_action_not_grounded_in_runlog(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="function_action_not_grounded:enter_expense_field:0",
    ):
        compile_runlog_to_store(
            _note_input_run_log(),
            tmp_path / "output",
            function_bundle=_single_input_bundle("note", text="invented"),
            source_states={
                "note-state": {
                    "state_id": "note-state",
                    "xml": (
                        '<hierarchy><node text="Note" resource-id="app:id/note" '
                        'class="android.widget.EditText" editable="true" '
                        'focused="true" bounds="[0,0][100,100]" /></hierarchy>'
                    ),
                }
            },
        )


def test_agent_prompt_uses_actions_without_source_page_and_explains_transfer(
    tmp_path: Path,
) -> None:
    captured: dict = {}

    class Completions:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {
                                    "reason": "Step 0 enters the visible Note field.",
                                    "bundle": _single_input_bundle("note"),
                                }
                            )
                        )
                    )
                ],
                usage=SimpleNamespace(
                    prompt_tokens=10,
                    completion_tokens=5,
                    total_tokens=15,
                ),
            )

    client = SimpleNamespace(chat=SimpleNamespace(completions=Completions()))
    result = compile_runlog_to_store(
        _note_input_run_log(),
        tmp_path / "output",
        model="offline-agent-model",
        client=client,
        source_states={
            "note-state": {
                "state_id": "note-state",
                "xml": (
                    '<hierarchy><node text="Note" resource-id="app:id/note" '
                    'class="android.widget.EditText" editable="true" '
                    'focused="true" bounds="[0,0][100,100]" /></hierarchy>'
                ),
            }
        },
    )

    assert result["function_ids"] == ["enter_expense_field"]
    system_prompt = captured["messages"][0]["content"]
    user_payload = json.loads(captured["messages"][1]["content"])
    assert "best-effort target-element ranker" in system_prompt
    assert "does not infer what a Function parameter means" in system_prompt
    assert "what adjacent, repeated, dynamic" in system_prompt
    assert "Do not request or\nre-read screenshots" in system_prompt
    authoring_step = user_payload["run_log"]["steps"][0]
    assert "source_observation" not in authoring_step
    assert authoring_step["action"] == {
        "tool": "input_text",
        "args": {"text": "Paid by card"},
    }
