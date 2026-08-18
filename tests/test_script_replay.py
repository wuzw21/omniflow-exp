from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from omniflow.core.model import Action, Observation, TransferResult
from omniflow.transfer.page_embedding import PageEmbedding
from src.integrations.script_replay import run_script_replay


class _Host:
    def __init__(self, source_states: dict[str, Observation]) -> None:
        self.source_states = source_states
        self.actions: list[Action] = []

    def observe(self, **_: object) -> Observation:
        return Observation(xml="<page/>", package_name="com.example")

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
        "checker_rules": [],
    }


def _store(
    path: Path,
    *functions: dict[str, object],
    source_calls: list[dict[str, object]] | None = None,
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.store.v2",
                "functions": {
                    str(function["function_id"]): function
                    for function in functions
                },
                "source_calls": source_calls or [],
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
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
        "additionalProperties": False,
    }
    complete["bindings"] = [
        {
            "source": "$.arguments.target",
            "target": "$.steps[0].action.args.target_description",
        }
    ]
    complete["steps"][0]["action"]["args"]["target_description"] = ""
    complete["checker_rules"] = [
        {
            "source_state_id": "checker-source",
            "action": {"tool": "click", "args": {"x": 300, "y": 400}},
        }
    ]
    store_path = _store(
        tmp_path / "store.json",
        complete,
        _function(function_id="reusable_part", steps=2),
        source_calls=[
            {"function_id": "complete_task", "arguments": {"target": "Alarm"}}
        ],
    )
    source_states = {
        f"source-{index}": Observation(xml="<page/>", package_name="com.example")
        for index in range(2)
    }
    source_states["checker-source"] = Observation(
        xml="<page/>", package_name="com.example"
    )
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

    class PageEncoder:
        def embed(self, _value: Observation) -> PageEmbedding:
            return PageEmbedding(
                vector=np.asarray((1.0, 0.0), dtype=np.float32),
                element_count=1,
                encoder_version="test",
                checkpoint_path="/test/checkpoint.pt",
                checkpoint_sha256="test",
            )

    monkeypatch.setattr(
        "omniflow.runtime.engine.OmniTransferPageEncoder",
        PageEncoder,
    )

    result = run_script_replay(store_path=store_path, host=host)

    assert result.success is True
    assert result.function_id == "complete_task"
    assert result.model_calls == 0
    assert result.fallback_steps == 0
    assert transferred_sources == [
        source_states["checker-source"],
        source_states["source-0"],
        source_states["source-1"],
    ]
    assert transferred_actions[1].args["target_description"] == "Alarm"
    assert host.actions == [
        Action("click", {"x": 101, "y": 200}),
        Action("click", {"x": 102, "y": 200}),
        Action("click", {"x": 103, "y": 200}),
    ]


def test_script_replay_rejects_ambiguous_complete_function(tmp_path: Path) -> None:
    store_path = _store(
        tmp_path / "store.json",
        _function(function_id="complete_a", steps=2),
        _function(function_id="complete_b", steps=2),
    )

    try:
        run_script_replay(store_path=store_path, host=_Host({}))
    except ValueError as error:
        assert str(error) == (
            "script_replay_full_function_ambiguous:complete_a,complete_b"
        )
    else:
        raise AssertionError("ambiguous complete Functions must fail closed")


def test_script_replay_contains_no_private_action_mapping_implementation() -> None:
    source = Path("src/integrations/script_replay.py").read_text(encoding="utf-8")

    assert "ElementTree" not in source
    assert "resource-id" not in source
    assert "content-desc" not in source
    assert "source_states" not in source
    assert ".call_tool(" in source
