from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.integrations.script_replay import run_script_replay


class _Host:
    def __init__(self, xml: str) -> None:
        self.xml = xml
        self.actions: list[dict[str, object]] = []

    def observe(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            xml=self.xml,
            package_name="com.example",
            extra={"display": {"width": 1000, "height": 1000}},
        )

    def act(self, action: dict[str, object]) -> SimpleNamespace:
        self.actions.append(action)
        return SimpleNamespace(success=True, error=None)


def _store(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.store.v2",
                "functions": {
                    "demo": {
                        "schema_version": "omniflow.function.v2",
                        "function_id": "demo",
                        "name": "Demo",
                        "description": "Replay one demo action.",
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
                                "step_index": 0,
                                "source_state_id": "source",
                                "action": {"tool": "click", "args": {"x": 500, "y": 500}},
                            }
                        ],
                        "checker_rules": [
                            {
                                "source_state_id": "checker",
                                "action": {"tool": "click", "args": {"x": 1, "y": 1}},
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_script_replay_uses_formal_steps_and_unique_child_semantics(tmp_path: Path) -> None:
    source = (
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" class="android.widget.LinearLayout" '
        'resource-id="unstable:id/source" clickable="true" bounds="[400,400][600,600]">'
        '<node package="com.example" class="android.widget.TextView" text="Sound" '
        'bounds="[400,400][600,450]" /></node></hierarchy>'
    )
    target = (
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" class="android.widget.LinearLayout" '
        'resource-id="changed:id/target" clickable="true" bounds="[100,200][300,400]">'
        '<node package="com.example" class="android.widget.TextView" text="Sound" '
        'bounds="[100,200][300,250]" /></node></hierarchy>'
    )
    host = _Host(target)

    result = run_script_replay(
        store_path=_store(tmp_path / "store.json"),
        source_states={"source": {"xml": source}},
        host=host,
        stability_wait_seconds=0,
    )

    assert result.success is True
    assert result.actions_executed == 1
    assert result.execution_summary["model_calls"] == 0
    assert result.execution_summary["fallback_steps"] == 0
    assert result.trace[0]["selector"]["kind"] == "children"
    assert host.actions == [{"tool": "click", "args": {"x": 200.0, "y": 300.0}}]


def test_script_replay_does_not_use_resource_id_or_source_coordinate(tmp_path: Path) -> None:
    source = (
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" resource-id="stable:id/target" text="Source" '
        'clickable="true" bounds="[400,400][600,600]" /></hierarchy>'
    )
    target = (
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" resource-id="stable:id/target" text="Changed" '
        'clickable="true" bounds="[100,200][300,400]" /></hierarchy>'
    )
    host = _Host(target)

    result = run_script_replay(
        store_path=_store(tmp_path / "store.json"),
        source_states={"source": {"xml": source}},
        host=host,
        stability_wait_seconds=0,
    )

    assert result.success is False
    assert result.actions_executed == 0
    assert "script_replay_semantic_locator_absent" in str(result.error)
    assert host.actions == []


def test_script_replay_rejects_ambiguous_semantics(tmp_path: Path) -> None:
    source = (
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" text="Open" clickable="true" '
        'bounds="[400,400][600,600]" /></hierarchy>'
    )
    target = (
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" text="Open" bounds="[100,200][300,400]" />'
        '<node package="com.example" text="Open" bounds="[500,200][700,400]" />'
        '</hierarchy>'
    )
    host = _Host(target)

    result = run_script_replay(
        store_path=_store(tmp_path / "store.json"),
        source_states={"source": {"xml": source}},
        host=host,
        stability_wait_seconds=0,
    )

    assert result.success is False
    assert "script_replay_semantic_locator_ambiguous" in str(result.error)
    assert host.actions == []
