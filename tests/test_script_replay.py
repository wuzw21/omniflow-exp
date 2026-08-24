from __future__ import annotations

import json
from pathlib import Path

from omniflow.core.model import Action, Observation, TransferResult
from src.integrations.script_replay import run_script_replay


class _Host:
    def __init__(self, source_states: dict[str, Observation]) -> None:
        self.source_states = source_states
        self.actions: list[Action] = []

    def observe(self, **_: object) -> Observation:
        xml = (
            '<page text="Continue" clickable="true" bounds="[100,100][300,200]"/>'
            if not self.actions
            else "<page/>"
        )
        return Observation(
            xml=xml,
            package_name="com.example",
            extra={"display": {"width": 1000, "height": 1000}},
        )

    def get_state(self, state_id: str) -> Observation | None:
        return self.source_states.get(state_id)

    def act(self, action: Action) -> dict[str, object]:
        self.actions.append(action)
        return {"success": True}


def _function(*, function_id: str, steps: int) -> dict[str, object]:
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
        "name": function_id.replace("_", " ").title(),
        "description": f"Execute {steps} reusable actions.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        "bindings": [],
        "agent_visible": True,
        "steps": [
            {
                "step_index": index,
                "source_state_id": f"source-{index}",
                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
            }
            for index in range(steps)
        ],
    }


def _store(
    path: Path,
    *functions: dict[str, object],
    source_calls: list[dict[str, object]] | None = None,
    checker_rules: list[dict[str, object]] | None = None,
) -> Path:
    calls = source_calls or []
    path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.store.v2",
                "functions": {
                    str(function["function_id"]): function
                    for function in functions
                },
            }
        ),
        encoding="utf-8",
    )
    path.with_name("compile_report.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.androidworld.function-gate.v2",
                "source_calls": calls,
                "source_arguments": {
                    str(call["function_id"]): dict(call["arguments"])
                    for call in calls
                },
            }
        ),
        encoding="utf-8",
    )
    path.with_name("checker_store.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.checker_store.v1",
                "checker_rules": checker_rules or [],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_script_replay_selects_full_function_and_uses_core_transfer(
    tmp_path: Path, monkeypatch
) -> None:
    complete = _function(function_id="complete_task", steps=2)
    complete["input_schema"] = {
        "type": "object",
        "properties": {"x": {"type": "integer"}},
        "required": ["x"],
        "additionalProperties": False,
    }
    complete["bindings"] = [
        {
            "source": "$.arguments.x",
            "target": "$.steps[0].action.args.x",
        }
    ]
    store_path = _store(
        tmp_path / "store.json",
        complete,
        source_calls=[
            {"function_id": "complete_task", "arguments": {"x": 650}}
        ],
        checker_rules=[
            {
                "schema_version": "omniflow.checker_rule.v1",
                "id": "continue_prompt",
                "phase": "pre_transfer",
                "condition": {"xpath_exists": "//*[@text='Continue']"},
                "action": {
                    "action": "click",
                    "target_xpath": "//*[@text='Continue']",
                },
                "budget": {"max_triggers_per_run": 1},
                "priority": 200,
            }
        ],
    )
    source_states = {
        f"source-{index}": Observation(xml="<page/>", package_name="com.example")
        for index in range(2)
    }
    host = _Host(source_states)
    transferred_sources: list[Observation | None] = []
    transferred_actions: list[Action] = []

    async def transfer(action, observation, source_state):
        transferred_actions.append(action)
        transferred_sources.append(source_state)
        return TransferResult(
            Action("click", {"x": 100 + len(transferred_sources), "y": 200}),
            reason="omnitransfer_mapped",
            detail={"score": 0.999, "candidates": [{"score": 0.99}]},
        )

    monkeypatch.setattr(
        "omniflow.runtime.execution.default_transfer",
        transfer,
    )

    result = run_script_replay(store_path=store_path, host=host)

    assert result.success is True
    assert result.function_id == "complete_task"
    assert result.model_calls == 0
    assert result.fallback_steps == 0
    assert transferred_sources == [
        source_states["source-0"],
        source_states["source-1"],
    ]
    assert transferred_actions[0].args["x"] == 650
    assert host.actions == [
        Action("click", {"x": 200.0, "y": 150.0}),
        Action("click", {"x": 101, "y": 200}),
        Action("click", {"x": 102, "y": 200}),
    ]


def test_script_replay_executes_multiple_functions_in_source_order(
    tmp_path: Path, monkeypatch
) -> None:
    first = _function(function_id="complete_a", steps=1)
    second = _function(function_id="complete_b", steps=1)
    store_path = _store(
        tmp_path / "store.json",
        first,
        second,
        source_calls=[
            {"function_id": "complete_a", "arguments": {}},
            {"function_id": "complete_b", "arguments": {}},
        ],
        checker_rules=[
            {
                "schema_version": "omniflow.checker_rule.v1",
                "id": "shared_continue_prompt",
                "phase": "pre_transfer",
                "condition": {"xpath_exists": "//*[@text='Continue']"},
                "action": {
                    "action": "click",
                    "target_xpath": "//*[@text='Continue']",
                },
                "budget": {"max_triggers_per_run": 1},
                "priority": 200,
            }
        ],
    )
    source = Observation(xml="<page/>", package_name="com.example")

    class AlwaysObstructedHost(_Host):
        def observe(self, **_: object) -> Observation:
            return Observation(
                xml=(
                    '<page text="Continue" clickable="true" '
                    'bounds="[100,100][300,200]"/>'
                ),
                package_name="com.example",
                extra={"display": {"width": 1000, "height": 1000}},
            )

    host = AlwaysObstructedHost({"source-0": source})

    async def transfer(action, observation, source_state):
        return TransferResult(
            Action(action.tool, {"x": 200.0, "y": 150.0}),
            reason="omnitransfer_mapped",
            detail={"score": 0.999},
        )

    monkeypatch.setattr("omniflow.runtime.execution.default_transfer", transfer)

    result = run_script_replay(store_path=store_path, host=host)

    assert result.success is True
    assert result.detail["function_ids"] == ["complete_a", "complete_b"]
    assert [item["function_id"] for item in result.detail["function_sequence"]] == [
        "complete_a",
        "complete_b",
    ]
    assert len(host.actions) == 3
    assert result.detail["checker_trigger_counts"] == {
        "shared_continue_prompt": 1
    }


def test_script_replay_contains_no_private_action_mapping_implementation() -> None:
    source = Path("src/integrations/script_replay.py").read_text(encoding="utf-8")

    assert "ElementTree" not in source
    assert "resource-id" not in source
    assert "content-desc" not in source
    assert "source_states" not in source
    assert "run_function_sequence(" in source
