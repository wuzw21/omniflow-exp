from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.integrations.script_replay import run_script_replay


class _Host:
    def __init__(self, target_xml: str) -> None:
        self.target_xml = target_xml
        self.actions: list[dict[str, object]] = []

    def observe(self, **_: object) -> SimpleNamespace:
        return SimpleNamespace(
            xml=self.target_xml,
            package_name="com.example",
            extra={"display": {"width": 1000, "height": 1000}},
        )

    def act(self, action: dict[str, object]) -> SimpleNamespace:
        self.actions.append(action)
        return SimpleNamespace(success=True, error=None, extra={})


def _write_store(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.store.v2",
                "functions": {
                    "example": {
                        "schema_version": "omniflow.function.v2",
                        "function_id": "example",
                        "agent_visible": True,
                        "steps": [
                            {
                                "step_index": 0,
                                "source_state_id": "source",
                                "action": {
                                    "tool": "click",
                                    "args": {"x": 500, "y": 500},
                                },
                            }
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_script_replay_clicks_the_unique_exact_resource_id(tmp_path: Path) -> None:
    store_path = _write_store(tmp_path / "store.json")
    source_states = {
        "source": {
            "xml": (
                '<hierarchy width="1000" height="1000">'
                '<node package="com.example" resource-id="com.example:id/target" '
                'text="Source text" clickable="true" enabled="true" displayed="true" '
                'bounds="[400,400][600,600]" />'
                "</hierarchy>"
            )
        }
    }
    host = _Host(
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" resource-id="com.example:id/target" '
        'text="Changed text" clickable="true" enabled="true" displayed="true" '
        'bounds="[100,200][300,400]" />'
        "</hierarchy>"
    )

    result = run_script_replay(
        store_path=store_path,
        source_states=source_states,
        host=host,
    )

    assert result.success is True
    assert result.actions_executed == 1
    assert result.execution_summary["model_calls"] == 0
    assert result.execution_summary["fallback_steps"] == 0
    assert host.actions == [{"tool": "click", "args": {"x": 200.0, "y": 300.0}}]
    assert result.trace[0]["selector"] == {
        "kind": "resource-id",
        "value": "com.example:id/target",
    }


def test_script_replay_rejects_ambiguous_exact_selectors_without_clicking(
    tmp_path: Path,
) -> None:
    store_path = _write_store(tmp_path / "store.json")
    source_states = {
        "source": {
            "xml": (
                '<hierarchy width="1000" height="1000">'
                '<node package="com.example" resource-id="com.example:id/target" '
                'clickable="true" bounds="[400,400][600,600]" />'
                "</hierarchy>"
            )
        }
    }
    host = _Host(
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" resource-id="com.example:id/target" '
        'bounds="[100,200][300,400]" />'
        '<node package="com.example" resource-id="com.example:id/target" '
        'bounds="[500,200][700,400]" />'
        "</hierarchy>"
    )

    result = run_script_replay(
        store_path=store_path,
        source_states=source_states,
        host=host,
    )

    assert result.success is False
    assert result.actions_executed == 0
    assert "script_replay_selector_ambiguous" in str(result.error)
    assert host.actions == []


def test_script_replay_ignores_checkers_and_uses_unique_exact_text(
    tmp_path: Path,
) -> None:
    store_path = _write_store(tmp_path / "store.json")
    store = json.loads(store_path.read_text(encoding="utf-8"))
    steps = store["functions"]["example"]["steps"]
    steps.insert(
        0,
        {
            "step_index": -1,
            "source_state_id": "checker-state-is-intentionally-absent",
            "role": "checker",
            "action": {"tool": "click", "args": {"x": 1, "y": 1}},
        },
    )
    store_path.write_text(json.dumps(store), encoding="utf-8")
    source_states = {
        "source": {
            "xml": (
                '<hierarchy width="1000" height="1000">'
                '<node package="com.example" text="History" '
                'bounds="[400,400][600,600]" />'
                "</hierarchy>"
            )
        }
    }
    host = _Host(
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" text="History" '
        'bounds="[100,200][300,400]" />'
        "</hierarchy>"
    )

    result = run_script_replay(
        store_path=store_path,
        source_states=source_states,
        host=host,
    )

    assert result.success is True
    assert result.actions_executed == 1
    assert result.trace[0]["status"] == "ignored_checker"
    assert result.trace[1]["selector"] == {"kind": "text", "value": "History"}
