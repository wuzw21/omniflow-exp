from __future__ import annotations

import hashlib
from pathlib import Path

import json

from src.experiment import data_index
from src.experiment.data_index import (
    CURRENT_SCHEMA,
    _require_qualified_source_run_log,
    load_data_index,
    register_source_run_log_success,
)
from runlog_fixtures import androidworld_run_log, androidworld_state


def test_complete_source_run_log_seed_is_provenance_only(tmp_path: Path) -> None:
    before_path = tmp_path / "before.png"
    after_path = tmp_path / "after.png"
    before_path.write_bytes(b"before")
    after_path.write_bytes(b"after")

    def state(name: str, screenshot: Path) -> dict[str, object]:
        value = androidworld_state(name, with_pixels=False)
        value["pixels"] = {
            "path": str(screenshot),
            "sha256": hashlib.sha256(screenshot.read_bytes()).hexdigest(),
            "width": 720,
            "height": 1280,
            "mime_type": "image/png",
        }
        return value

    before = state("before", before_path)
    after = state("after", after_path)
    run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "com.example.app"}],
        observations=[before],
        task_name="Task",
        seed=987654321,
        success=True,
    )
    run_log["steps"][0]["next_observation"] = after
    run_log["steps"][0]["metadata"] = {
        "reasoning": "Open the requested app.",
        "screenshot_path": str(after_path),
    }

    qualified = _require_qualified_source_run_log(
        run_log,
        task="Task",
        source_metadata=None,
    )

    assert qualified["seed"] == 987654321


def test_register_source_success_replaces_stale_task_parameters(
    tmp_path: Path, monkeypatch
) -> None:
    run_log_path = tmp_path / "source.run_log.json"
    run_log_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        data_index,
        "_require_qualified_source_run_log",
        lambda payload, *, task, source_metadata: {
            "task_name": "Task",
            "success": True,
            "validator": {"official": True, "success": True},
            "steps": [{"step_index": 0}],
        },
    )
    current_path = tmp_path / "current.json"
    current_path.write_text(
        json.dumps(
            {
                "schema_version": CURRENT_SCHEMA,
                "canonical": {},
                "source_index": {
                    "Task": {
                        "task": "Task",
                        "params": {"hours": 0, "minutes": 1, "seconds": 15},
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    register_source_run_log_success(
        memory_index=current_path,
        task="Task",
        run_log_path=run_log_path,
        task_parameters={"hours": 1, "minutes": 15, "seconds": 30},
    )

    registered = load_data_index(current_path)
    assert registered["source_index"]["Task"]["params"] == {
        "hours": 1,
        "minutes": 15,
        "seconds": 30,
    }
    assert registered["canonical"]["source_run_logs"]["Task"][
        "object_path"
    ] == str(run_log_path.resolve())


def test_refresh_from_pointer_publishes_to_custom_pointer(
    tmp_path: Path, monkeypatch
) -> None:
    pointer = tmp_path / "index.json"
    pointer.write_text("{}", encoding="utf-8")
    registry = {
        "inputs": {
            "runlog_roots": [],
            "result_roots": [],
            "prepared_memory_roots": [],
            "baseline_batch_reports": [],
            "source_screenshot_roots": [],
            "source_index": str(pointer),
        },
        "canonical": {"function_stores": {}},
    }
    captured: dict[str, object] = {}

    monkeypatch.setattr(data_index, "load_data_index", lambda _path: registry)

    def refresh(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(data_index, "_refresh_data_index_unlocked", refresh)

    data_index.refresh_data_index_from_pointer(memory_index=pointer)

    assert captured["output_path"] == pointer.resolve()
