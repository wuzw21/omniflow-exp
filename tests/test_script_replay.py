from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.integrations.script_replay import (
    enhance_prepared_function_store,
    prepare_script_replay_store,
    run_script_replay,
)


class _Host:
    def __init__(self, target_xml: str | list[str]) -> None:
        self.target_xmls = (
            list(target_xml) if isinstance(target_xml, list) else [target_xml]
        )
        self.actions: list[dict[str, object]] = []
        self.observation_count = 0

    def observe(self, **_: object) -> SimpleNamespace:
        target_xml = self.target_xmls[
            min(self.observation_count, len(self.target_xmls) - 1)
        ]
        self.observation_count += 1
        return SimpleNamespace(
            xml=target_xml,
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
                        "name": "Example",
                        "description": "Run the example action.",
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
                                "action": {
                                    "tool": "click",
                                    "args": {"x": 500, "y": 500},
                                },
                            }
                        ],
                        "checker_rules": [],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


def test_prepare_script_replay_store_from_canonical_runlog(tmp_path: Path) -> None:
    runlog_path = tmp_path / "runlog.json"
    states_path = tmp_path / "transfer_states.json"
    runlog_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "source-run",
                "goal": "open Chrome",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "source",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500, "y": 500},
                        },
                        "result": {"success": True},
                        "after_state_id": "target",
                    }
                ],
                "diagnostics": {"task_id": "chrome/open_Chrome"},
                "final_state_id": "target",
            }
        ),
        encoding="utf-8",
    )
    states_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "source-run",
                "states": {
                    "source": {
                        "state_id": "source",
                        "xml": '<hierarchy width="1000" height="1000" />',
                    },
                    "target": {
                        "state_id": "target",
                        "xml": '<hierarchy width="1000" height="1000" />',
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = prepare_script_replay_store(
        runlog_path=runlog_path,
        source_states_path=states_path,
        output_root=tmp_path / "prepared",
        expected_task_id="chrome/open_Chrome",
    )

    store = json.loads(Path(report["store_path"]).read_text(encoding="utf-8"))
    function = next(iter(store["functions"].values()))
    assert function["steps"] == [
        {
            "step_index": 0,
            "source_state_id": "source",
            "action": {"tool": "click", "args": {"x": 500, "y": 500}},
        }
    ]
    assert function["checker_rules"] == []
    assert report["model_calls"] == 0
    assert report["step_count"] == 1
    assert (tmp_path / "prepared" / "source.runlog.json").is_file()
    assert (tmp_path / "prepared" / "transfer_states.json").is_file()


def test_prepare_script_replay_normalizes_launcher_click_to_open_app(
    tmp_path: Path,
) -> None:
    runlog_path = tmp_path / "runlog.json"
    states_path = tmp_path / "transfer_states.json"
    runlog_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "source-run",
                "goal": "create an alarm",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "launcher",
                        "action": {
                            "tool": "click",
                            "args": {"x": 695, "y": 619},
                        },
                        "result": {"success": True},
                        "after_state_id": "clock",
                    }
                ],
                "diagnostics": {"task_id": "clock/create_alarm"},
                "final_state_id": "clock",
            }
        ),
        encoding="utf-8",
    )
    states_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "source-run",
                "states": {
                    "launcher": {
                        "state_id": "launcher",
                        "package_name": "com.google.android.apps.nexuslauncher",
                        "activity_name": ".NexusLauncherActivity",
                        "xml": '<hierarchy width="1080" height="1920" />',
                    },
                    "clock": {
                        "state_id": "clock",
                        "package_name": "com.google.android.deskclock",
                        "activity_name": ".DeskClock",
                        "xml": '<hierarchy width="1080" height="1920" />',
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    report = prepare_script_replay_store(
        runlog_path=runlog_path,
        source_states_path=states_path,
        output_root=tmp_path / "prepared",
        expected_task_id="clock/create_alarm",
    )

    store = json.loads(Path(report["store_path"]).read_text(encoding="utf-8"))
    function = next(iter(store["functions"].values()))
    assert function["steps"] == [
        {
            "step_index": 0,
            "source_state_id": "launcher",
            "action": {
                "tool": "open_app",
                "args": {"package_name": "com.google.android.deskclock"},
            },
        }
    ]


def test_enhance_prepared_function_store_persists_checker_role(
    tmp_path: Path,
) -> None:
    store_path = _write_store(tmp_path / "store.json")
    runlog_path = tmp_path / "runlog.json"
    states_path = tmp_path / "transfer_states.json"
    runlog_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "source-run",
                "goal": "dismiss optional setup",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "source",
                        "after_state_id": "after",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500, "y": 500},
                        },
                        "result": {"success": True},
                        "metadata": {"origin": "action"},
                    }
                ],
                "final_state_id": "after",
            }
        ),
        encoding="utf-8",
    )
    states_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "source-run",
                "states": {
                    "source": {
                        "state_id": "source",
                        "package_name": "com.example",
                        "activity_name": ".SetupActivity",
                        "xml": '<hierarchy><node text="Skip" /></hierarchy>',
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    report = enhance_prepared_function_store(
        store_path=store_path,
        runlog_path=runlog_path,
        source_states_path=states_path,
        complete_json=lambda _: json.dumps(
            {
                "step_decisions": [
                    {
                        "step": 0,
                        "role": "checker",
                        "reason": "Optional setup dialog",
                    }
                ]
            }
        ),
    )

    stored = json.loads(store_path.read_text(encoding="utf-8"))
    function = stored["functions"]["example"]
    assert function["steps"][0]["role"] == "checker"
    assert report["model_calls"] == 1
    assert report["role_counts"] == {"function": 0, "checker": 1}
    assert Path(report["report_path"]).is_file()


def test_script_replay_never_uses_resource_id(tmp_path: Path) -> None:
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

    assert result.success is False
    assert result.actions_executed == 0
    assert result.execution_summary["model_calls"] == 0
    assert result.execution_summary["fallback_steps"] == 0
    assert "script_replay_semantic_locator_absent" in str(result.error)
    assert host.actions == []


def test_script_replay_rejects_ambiguous_exact_selectors_without_clicking(
    tmp_path: Path,
) -> None:
    store_path = _write_store(tmp_path / "store.json")
    source_states = {
        "source": {
            "xml": (
                '<hierarchy width="1000" height="1000">'
                '<node package="com.example" text="Open" '
                'clickable="true" bounds="[400,400][600,600]" />'
                "</hierarchy>"
            )
        }
    }
    host = _Host(
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" text="Open" '
        'bounds="[100,200][300,400]" />'
        '<node package="com.example" text="Open" '
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
    assert "script_replay_semantic_locator_ambiguous" in str(result.error)
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


def test_script_replay_uses_mobilegpt_child_depth_and_rank_for_anonymous_node(
    tmp_path: Path,
) -> None:
    store_path = _write_store(tmp_path / "store.json")
    source_states = {
        "source": {
            "xml": (
                '<hierarchy width="1000" height="1000">'
                '<node package="com.example" class="android.widget.LinearLayout" '
                'clickable="true" bounds="[400,400][600,600]">'
                '<node package="com.example" class="android.widget.TextView" '
                'text="Sound" bounds="[400,400][600,450]" />'
                "</node></hierarchy>"
            )
        }
    }
    host = _Host(
        '<hierarchy width="1000" height="1000">'
        '<node package="com.example" class="android.widget.LinearLayout" '
        'clickable="true" bounds="[100,200][300,400]">'
        '<node package="com.example" class="android.widget.TextView" '
        'text="Sound" bounds="[100,200][300,250]" />'
        "</node></hierarchy>"
    )

    result = run_script_replay(
        store_path=store_path,
        source_states=source_states,
        host=host,
        stability_wait_seconds=0,
    )

    assert result.success is True
    assert host.actions == [{"tool": "click", "args": {"x": 200.0, "y": 300.0}}]
    assert result.trace[0]["selector"]["kind"] == "children"


def test_script_replay_rejects_loading_or_changing_page(tmp_path: Path) -> None:
    store_path = _write_store(tmp_path / "store.json")
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
        [
            '<hierarchy width="1000" height="1000">'
            '<node package="com.example" class="android.widget.ProgressBar" '
            'bounds="[400,400][600,600]" />'
            "</hierarchy>",
            '<hierarchy width="1000" height="1000">'
            '<node package="com.example" text="History" '
            'bounds="[100,200][300,400]" />'
            "</hierarchy>",
        ]
    )

    result = run_script_replay(
        store_path=store_path,
        source_states=source_states,
        host=host,
        stability_wait_seconds=0,
    )

    assert result.success is False
    assert result.actions_executed == 0
    assert "script_replay_page_loading" in str(result.error)
    assert host.actions == []
