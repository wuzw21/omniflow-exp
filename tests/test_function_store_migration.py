from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest

from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.functions.assets import STORE_VERSION, FunctionStore
from omniflow.functions.migrate_store import (
    migrate_function_catalog,
    migrate_function_store,
)


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
            "source_calls": [
                {"function_id": "first", "arguments": {}},
                {"function_id": "second", "arguments": {}},
            ],
        },
    )
    _write(old_store.with_name("run_log.json"), {"schema_version": "old.runlog"})
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
        assert destination.with_name("run_log.json").is_file()


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
    assert store_path.with_name("run_log.json").read_text(
        encoding="utf-8"
    ) == run_log_path.read_text(encoding="utf-8")


def test_migration_catalog_writes_canonical_task_path_and_reports_dry_run(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    store = _write(
        old_root / "function_store.json",
        {
            "schema_version": STORE_VERSION,
            "functions": {"tap": _function("tap", "state-0")},
            "source_calls": [{"function_id": "tap", "arguments": {}}],
        },
    )
    run_log = _write(old_root / "source.json", {"schema_version": "old.runlog"})
    transfer = _write(
        old_root / "states.json",
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "states": {"state-0": {"state_id": "state-0"}},
        },
    )
    catalog = _write(
        old_root / "function_catalog.json",
        {
            "schema_version": "omniflow.function-asset-catalog.v1",
            "tasks": {
                "DemoTask": {
                    "store_path": str(store),
                    "source_run_log": str(run_log),
                    "transfer_states_path": str(transfer),
                }
            },
        },
    )

    report = migrate_function_catalog(catalog, tmp_path / "new", dry_run=True)

    assert report["counts"] == {"converted": 1, "blocked": 0, "stores": 1}
    expected = (
        tmp_path / "new" / "androidworld" / "DemoTask" / "source5554"
        / "function" / "function_authoring"
    )
    assert report["tasks"][0]["stores"][0]["store_path"].startswith(str(expected))


def test_catalog_splits_multi_function_store_into_registerable_attempts(
    tmp_path: Path,
) -> None:
    old_root = tmp_path / "old"
    store = _write(
        old_root / "function_store.json",
        {
            "schema_version": STORE_VERSION,
            "functions": {
                "first": _function("first", "state-first"),
                "second": _function("second", "state-second"),
            },
            "source_calls": [
                {"function_id": "first", "arguments": {}},
                {"function_id": "second", "arguments": {}},
            ],
        },
    )
    run_log = _write(old_root / "run_log.json", {"schema_version": "old.runlog"})
    transfer = _write(
        old_root / "transfer_states.json",
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "states": {
                "state-first": {"state_id": "state-first"},
                "state-second": {"state_id": "state-second"},
            },
        },
    )
    catalog = _write(
        old_root / "catalog.json",
        {
            "schema_version": "omniflow.function-asset-catalog.v1",
            "tasks": {
                "DemoTask": {
                    "store_path": str(store),
                    "source_run_log": str(run_log),
                    "transfer_states_path": str(transfer),
                }
            },
        },
    )

    report = migrate_function_catalog(catalog, tmp_path / "new")

    assert report["counts"] == {"converted": 1, "blocked": 0, "stores": 2}
    paths = [Path(item["store_path"]) for item in report["tasks"][0]["stores"]]
    digest = hashlib.sha256(store.read_bytes()).hexdigest()[:12]
    assert {path.parent.name for path in paths} == {
        f"migration_{digest}_first",
        f"migration_{digest}_second",
    }
    for path in paths:
        assert len(FunctionStore(path).list_functions()) == 1


def test_migration_requires_source_call_evidence(tmp_path: Path) -> None:
    old_store = _write(
        tmp_path / "old" / "function_store.json",
        {
            "schema_version": STORE_VERSION,
            "functions": {"tap": _function("tap", "state-0")},
            "source_calls": [],
        },
    )
    _write(old_store.with_name("run_log.json"), {"schema_version": "old.runlog"})
    _write(
        old_store.with_name("transfer_states.json"),
        {
            "schema_version": "omniflow.transfer-state-catalog.v1",
            "states": {"state-0": {"state_id": "state-0"}},
        },
    )

    with pytest.raises(ValueError, match="function_migration_source_call_required"):
        migrate_function_store(old_store, tmp_path / "new")


def test_migration_does_not_treat_old_catalog_as_a_store(tmp_path: Path) -> None:
    catalog = _write(
        tmp_path / "function_catalog.json",
        {"schema_version": "omniflow.function-asset-catalog.v1", "tasks": {}},
    )

    with pytest.raises(ValueError, match="unsupported_function_json_version"):
        migrate_function_store(catalog, tmp_path / "new.json")
