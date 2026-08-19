from __future__ import annotations

import json
from pathlib import Path

from src.experiment.data_migration import build_migration_plan, stage_migration


def test_authoritative_migration_collects_json_dependencies_and_rewrites_paths(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-data"
    target = Path("/remote/authoritative-data")
    source.mkdir()
    dependency = source / "objects" / "state.json"
    dependency.parent.mkdir()
    screenshot = source / "objects" / "screen.png"
    screenshot.write_bytes(b"png")
    dependency.write_text(
        json.dumps({"screenshot": str(screenshot)}), encoding="utf-8"
    )
    store_dir = source / "androidworld" / "Task" / "function"
    store_dir.mkdir(parents=True)
    (store_dir / "function_store.json").write_text(
        json.dumps({"dependency": str(dependency)}), encoding="utf-8"
    )
    (store_dir / "transfer_states.json").write_text("{}", encoding="utf-8")
    (source / "current.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.data-index.v2",
                "canonical": {
                    "source_run_logs": {"Task": {"object_path": str(dependency)}},
                    "function_stores": {
                        "Task": {"store_path": str(store_dir / "function_store.json")}
                    },
                    "prepared_memories": {},
                    "result_cells": {},
                },
            }
        ),
        encoding="utf-8",
    )

    plan = build_migration_plan(source, target)
    assert plan.missing_references == ()
    assert {path.relative_to(source) for path in plan.files} == {
        Path("current.json"),
        Path("objects/state.json"),
        Path("objects/screen.png"),
        Path("androidworld/Task/function/function_store.json"),
        Path("androidworld/Task/function/transfer_states.json"),
    }

    stage = stage_migration(plan, tmp_path / "stage")
    migrated = json.loads((stage / "current.json").read_text(encoding="utf-8"))
    assert migrated["canonical"]["source_run_logs"]["Task"]["object_path"] == (
        "/remote/authoritative-data/objects/state.json"
    )
    assert "current.json" not in (stage / "files.txt").read_text(encoding="utf-8")
