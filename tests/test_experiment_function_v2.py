from __future__ import annotations

import json
from pathlib import Path

from runlog_fixtures import androidworld_run_log, androidworld_state

from omniflow.core.trajectory import canonicalize_run_log
from src.experiment.observation_evidence import canonicalize_run_log_observation
from src.experiment.function_v2 import compile_function_v2, load_v2_source_calls
from src.experiment.run_task import validate_omniflow_transfer_assets


def test_experiment_compiles_native_v2_bundle(tmp_path: Path) -> None:
    run_log = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.android.settings"},
            {"action_type": "wait"},
        ],
        observations=[
            androidworld_state("state_0", with_pixels=True),
            androidworld_state("state_1", with_pixels=True),
        ],
        goal="Open Settings and wait.",
    )
    source = tmp_path / "run_log.json"
    source.write_text(json.dumps(run_log), encoding="utf-8")
    report = compile_function_v2(source, tmp_path / "memory", enhance=False)
    store_path = Path(report["store_path"])
    store = json.loads(store_path.read_text(encoding="utf-8"))

    assert store_path.name == "store.json"
    assert set(store) == {"schema_version", "functions"}
    assert store["schema_version"] == "omniflow.store.v2"
    assert {
        function["schema_version"] for function in store["functions"].values()
    } == {"omniflow.function.v2"}
    assert load_v2_source_calls(store_path) == [
        {"function_id": report["function_ids"][0], "arguments": {}}
    ]
    transfer_states = json.loads(
        store_path.with_name("transfer_states.json").read_text(encoding="utf-8")
    )
    assert transfer_states["states"]["state_0"]["screenshot_path"] == (
        "/tmp/state_0.png"
    )
    assert validate_omniflow_transfer_assets(
        store_path,
        require_action_transfer=False,
    )["complete"] is True


def test_runlog_accepts_new_and_legacy_collector_observations(
    tmp_path: Path,
) -> None:
    screenshot = {
        "path": "/tmp/source.png",
        "sha256": "0" * 64,
        "width": 720,
        "height": 1280,
        "mime_type": "image/png",
    }
    legacy = {
        "pixels": screenshot,
        "xml": "<hierarchy />",
        "auxiliaries": {"state_id": "legacy"},
    }
    new = canonicalize_run_log_observation(legacy)

    assert set(new) == {"screenshot", "xml"}
    assert "sha256" not in new["screenshot"]

    run_log = androidworld_run_log(
        [
            {"action_type": "open_app", "app_name": "com.android.settings"},
            {"action_type": "wait"},
        ],
        observations=[
            legacy,
            {
                **legacy,
                "auxiliaries": {"state_id": "legacy_after_open"},
            },
        ],
    )
    assert canonicalize_run_log(run_log)["steps"][0]["observation"] == {
        "pixels": {
            key: value for key, value in screenshot.items() if key != "sha256"
        },
        "xml": "<hierarchy />",
        "auxiliaries": {"state_id": "legacy"},
    }

    source = tmp_path / "legacy_run_log.json"
    source.write_text(json.dumps(run_log), encoding="utf-8")
    report = compile_function_v2(source, tmp_path / "legacy_memory", enhance=False)
    transfer_states = json.loads(
        Path(report["store_path"])
        .with_name("transfer_states.json")
        .read_text(encoding="utf-8")
    )
    assert transfer_states["states"]["legacy"]["screenshot_path"] == (
        "/tmp/source.png"
    )

    in_memory_report = compile_function_v2(
        run_log,
        tmp_path / "legacy_memory_from_dict",
        enhance=False,
    )
    assert Path(in_memory_report["store_path"]).name == "store.json"
