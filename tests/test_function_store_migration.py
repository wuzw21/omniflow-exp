from __future__ import annotations

import json
from pathlib import Path

import pytest

from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.functions.assets import STORE_VERSION, FunctionStore
from omniflow.functions.migrate_store import migrate_function_store


def _function(function_id: str, source_state_id: str) -> dict:
    return {
        "schema_version": "omniflow.function.v2",
        "function_id": function_id,
        "name": function_id,
        "description": f"Run {function_id}.",
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
                "source_state_id": source_state_id,
                "action": {"tool": "click", "args": {"x": 50, "y": 50}},
            }
        ],
        "checker_rules": [],
        "agent_visible": True,
    }


def _write(path: Path, value: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_migration_splits_old_multi_function_store_and_states(tmp_path: Path) -> None:
    old_store = _write(
        tmp_path / "old" / "function_store.json",
        {
            "schema_version": STORE_VERSION,
            "functions": {
                "first": _function("first", "state-first"),
                "second": _function("second", "state-second"),
            },
            "source_calls": [],
        },
    )
    _write(
        old_store.with_name("transfer_states.json"),
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "run_id": "old-run",
            "states": {
                "state-first": {"state_id": "state-first"},
                "state-second": {"state_id": "state-second"},
            },
        },
    )

    report = migrate_function_store(old_store, tmp_path / "new")

    assert [item["function_id"] for item in report["stores"]] == ["first", "second"]
    for function_id in ("first", "second"):
        destination = tmp_path / "new" / function_id / "function_store.json"
        store = FunctionStore(destination)
        assert [item.id for item in store.list_functions()] == [function_id]
        states = json.loads(
            destination.with_name("transfer_states.json").read_text(encoding="utf-8")
        )
        assert sorted(states["states"]) == [f"state-{function_id}"]


def test_migration_converts_legacy_bundle_through_current_writer(tmp_path: Path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "click", "x": 50, "y": 50}],
        observations=[androidworld_state("state-0")],
        run_id="legacy-source",
    )
    run_log_path = _write(tmp_path / "source" / "run_log.json", run_log)
    bundle = _write(
        tmp_path / "old" / "codex_function_bundle.json",
        {
            "schema_version": "omniflow.function-bundle.v2",
            "source_run_id": "legacy-source",
            "source_success": True,
            "source_arguments": {"tap": {}},
            "functions": [
                {
                    "id": "tap",
                    "name": "Tap",
                    "description": "Tap the source point.",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": [],
                        "additionalProperties": False,
                    },
                    "bindings": [],
                    "actions": [
                        {"tool": "click", "arguments": {"x": 50, "y": 50}}
                    ],
                    "checker_rules": [],
                }
            ],
        },
    )

    report = migrate_function_store(
        bundle,
        tmp_path / "new" / "function_store.json",
        source_run_log=run_log_path,
    )

    assert report["stores"][0]["function_id"] == "tap"
    store_path = tmp_path / "new" / "function_store.json"
    store = FunctionStore(store_path)
    assert store.get_function("tap") is not None
    assert json.loads(store_path.read_text(encoding="utf-8"))["source_calls"] == [
        {"function_id": "tap", "arguments": {}}
    ]
    assert store_path.with_name("transfer_states.json").is_file()


def test_migration_does_not_treat_old_catalog_as_a_store(tmp_path: Path) -> None:
    catalog = _write(
        tmp_path / "function_catalog.json",
        {"schema_version": "omniflow.function-asset-catalog.v1", "tasks": {}},
    )

    with pytest.raises(ValueError, match="unsupported_function_json_version"):
        migrate_function_store(catalog, tmp_path / "new.json")
