from __future__ import annotations

import hashlib
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


def test_authoritative_migration_resolves_stale_paths_to_content_addressed_objects(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source-data"
    target = Path("/remote/authoritative-data")
    source.mkdir()
    payload = b"canonical screenshot"
    digest = hashlib.sha256(payload).hexdigest()
    object_path = source / "objects" / "sha256" / digest[:2] / f"{digest}.png"
    object_path.parent.mkdir(parents=True)
    object_path.write_bytes(payload)
    stale_path = source / "old" / "missing" / object_path.name
    run_log = source / "run_log.json"
    run_log.write_text(json.dumps({"screenshot": str(stale_path)}), encoding="utf-8")
    (source / "current.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.data-index.v2",
                "canonical": {
                    "source_run_logs": {"Task": {"object_path": str(run_log)}}
                },
            }
        ),
        encoding="utf-8",
    )

    plan = build_migration_plan(source, target)
    assert plan.missing_references == ()
    assert object_path in plan.files

    stage = stage_migration(plan, tmp_path / "stage")
    staged_run_log = json.loads(
        (stage / "data" / "run_log.json").read_text(encoding="utf-8")
    )
    assert staged_run_log["screenshot"] == (
        f"/remote/authoritative-data/objects/sha256/{digest[:2]}/{digest}.png"
    )
    assert (stage / "data" / object_path.relative_to(source)).read_bytes() == payload
