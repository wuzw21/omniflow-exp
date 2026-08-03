from __future__ import annotations

import json

from omniflow.bridge import JsonLineBridge
from omniflow.functions.management import enhance_function


def _function() -> dict:
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": "open_settings",
        "name": "Open settings",
        "description": "Open Android settings.",
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
                "source_state_id": "state-1",
                "action": {
                    "tool": "open_app",
                    "args": {"package_name": "com.android.settings"},
                },
            }
        ],
        "checker_rules": [],
        "agent_visible": True,
    }


def test_enhancement_instruction_is_included_in_prompt() -> None:
    prompts: list[str] = []

    enhance_function(
        _function(),
        {},
        lambda prompt: prompts.append(prompt) or "{}",
        instruction="Prefer a reusable search-first workflow.",
    )

    assert len(prompts) == 1
    assert "Prefer a reusable search-first workflow." in prompts[0]


def test_enhancement_uses_default_guidance_when_instruction_is_empty() -> None:
    prompts: list[str] = []

    enhance_function(
        _function(),
        {},
        lambda prompt: prompts.append(prompt) or "{}",
    )

    assert len(prompts) == 1
    assert '"user_instruction":""' in prompts[0]


def test_update_function_forwards_optional_instruction(tmp_path) -> None:
    captured: dict = {}

    class Bridge(JsonLineBridge):
        def _enhance(self, request_id, body):
            captured.update(body)
            return {"success": True}

    bridge = Bridge(tmp_path / "functions.json")
    bridge.flow.store.put_function(_function())

    result = bridge._update_function(
        "request-1",
        {
            "function_id": "open_settings",
            "mode": "enhance",
            "instruction": "Prefer search-first behavior.",
        },
    )

    assert result["success"] is True
    assert captured["instruction"] == "Prefer search-first behavior."


def test_agent_enhancement_generates_checker_from_runlog_evidence() -> None:
    checker_action = {
        "tool": "click",
        "args": {
            "target_description": "Dismiss the temporary prompt",
            "x": 500,
            "y": 500,
        },
    }
    checker_rule = {
        "schema_version": "omniflow.checker_rule.v1",
        "trigger": 'xml_contains("not now")',
        "source_state_id": "state-checker",
        "action": checker_action,
    }
    run_log = {
        "run_id": "run-with-checker",
        "steps": [
            {
                "before_state_id": "state-checker",
                "action": checker_action,
                "result": {"success": True},
                "metadata": {
                    "origin": "checker",
                    "checker_trigger": 'xml_contains("not now")',
                },
            }
        ],
    }

    enhanced, changes, status = enhance_function(
        _function(),
        run_log,
        lambda _prompt: json.dumps({"checker_rules": [checker_rule]}),
        instruction="Add only evidence-backed recovery conditions.",
    )

    assert enhanced["checker_rules"] == [
        {
            **checker_rule,
            "action": {"tool": "click", "args": {"x": 500, "y": 500}},
        }
    ]
    assert {"part": "function", "field": "checker_rules"} in changes
    assert status == "enhanced"
