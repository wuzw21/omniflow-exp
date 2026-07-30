from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.experiment import androidworld as pipeline
from src.experiment.source_assets import (
    build_grounded_teacher_run_log,
    build_grounded_teacher_run_log_from_item,
    select_source_asset_revision,
)
from src.integrations.appagent_adapter import build_appagent_teacher_source
from src.integrations.mobilegpt_teacher import (
    preflight_teacher_source_run_log,
)
from runlog_fixtures import androidworld_run_log, androidworld_state


def _write_source_bundle(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {"action_type": "click", "x": 50, "y": 50},
                    {"action_type": "input_text", "text": "Example"},
                ],
                observations=[
                    androidworld_state("button-state", width=100, height=100),
                    androidworld_state("input-state", width=100, height=100),
                ],
                task_name="RecordWithName",
                goal="Save the recording as Example.",
            )
        ),
        encoding="utf-8",
    )
    xml = (
        '<hierarchy><node class="android.widget.Button" text="Save" '
        'resource-id="app:id/save" clickable="true" bounds="[0,0][100,100]" />'
        '<node class="android.widget.EditText" text="" '
        'resource-id="app:id/name" editable="true" focused="true" '
        'bounds="[0,100][100,200]" />'
        '<node class="android.widget.EditText" text="Address" '
        'resource-id="browser:id/url" editable="true" '
        'bounds="[0,200][100,300]" /></hierarchy>'
    )
    states = root / "transfer_states.json"
    states.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "source-run",
                "states": {
                    "button-state": {
                        "state_id": "button-state",
                        "xml": xml,
                    },
                    "input-state": {
                        "state_id": "input-state",
                        "xml": xml,
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    provenance = root / "provenance_manifest.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.source-replay-transfer-store.v1",
                "source_target_audit": {
                    "source_target_audit_complete": True,
                    "source_targets": [
                        {
                            "step_index": 0,
                            "source_state_id": "button-state",
                            "target": {
                                "text": "Save",
                                "resource_id": "app:id/save",
                            },
                        }
                    ],
                },
            }
        ),
        encoding="utf-8",
    )
    return source, states, provenance


def test_frozen_source_evidence_grounds_both_baseline_teachers(
    tmp_path: Path,
) -> None:
    source, states, provenance = _write_source_bundle(tmp_path)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

    grounded, audit = build_grounded_teacher_run_log(
        source_run_log=source,
        source_state_catalog=states,
        provenance_manifest=provenance,
        expected_source_run_log_sha256=source_sha256,
        expected_source_state_catalog_sha256=hashlib.sha256(
            states.read_bytes()
        ).hexdigest(),
        expected_provenance_sha256=hashlib.sha256(
            provenance.read_bytes()
        ).hexdigest(),
    )
    grounded_path = tmp_path / "grounded.teacher.run_log.json"
    grounded_path.write_text(json.dumps(grounded), encoding="utf-8")

    mobilegpt = preflight_teacher_source_run_log(grounded_path)
    mobilegpt_artifact = pipeline.build_mobilegpt_teacher_source(
        grounded_path,
        task_name="RecordWithName",
        provenance_source_run_log=source,
    )
    appagent = build_appagent_teacher_source(
        grounded_path,
        task_name="RecordWithName",
        provenance_source_run_log=source,
    )

    assert mobilegpt["teacher_action_count"] == 2
    assert mobilegpt["groundable_action_count"] == 2
    assert mobilegpt_artifact["source_run_log"] == str(source)
    assert mobilegpt_artifact["source_run_log_sha256"] == source_sha256
    assert (
        mobilegpt_artifact["grounded_teacher_run_log_sha256"]
        == hashlib.sha256(grounded_path.read_bytes()).hexdigest()
    )
    assert appagent["action_count"] == 2
    assert appagent["source_run_log"] == str(source)
    assert appagent["source_run_log_sha256"] == source_sha256
    assert audit["target_inputs_read"] is False
    assert audit["target_observations_read"] is False
    assert hashlib.sha256(source.read_bytes()).hexdigest() == source_sha256


def test_frozen_historical_source_is_rejected_before_grounding(
    tmp_path: Path,
) -> None:
    source, states, provenance = _write_source_bundle(tmp_path)
    source.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.run_log.v1",
                "run_id": "historical-source",
                "goal": "Save the recording as Example.",
                "completed": True,
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "observation_before_act": {
                            "state_id": "button-state",
                            "width": 100,
                            "height": 100,
                        },
                        "executed_actions": [
                            {
                                "type": "click",
                                "params": {"x": 50, "y": 50},
                            }
                        ],
                        "success": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="run_log_schema_invalid"):
        build_grounded_teacher_run_log(
            source_run_log=source,
            source_state_catalog=states,
            provenance_manifest=provenance,
            expected_source_run_log_sha256=hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            expected_source_state_catalog_sha256=hashlib.sha256(
                states.read_bytes()
            ).hexdigest(),
            expected_provenance_sha256=hashlib.sha256(
                provenance.read_bytes()
            ).hexdigest(),
        )


def test_grounding_rejects_changed_frozen_catalog(tmp_path: Path) -> None:
    source, states, provenance = _write_source_bundle(tmp_path)

    with pytest.raises(ValueError, match="source_state_catalog_hash_mismatch"):
        build_grounded_teacher_run_log(
            source_run_log=source,
            source_state_catalog=states,
            provenance_manifest=provenance,
            expected_source_run_log_sha256=hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            expected_source_state_catalog_sha256="0" * 64,
            expected_provenance_sha256=hashlib.sha256(
                provenance.read_bytes()
            ).hexdigest(),
        )


def test_source_and_store_indexes_join_without_rewriting_frozen_assets(
    tmp_path: Path,
) -> None:
    source, states, provenance = _write_source_bundle(tmp_path)
    store_index = tmp_path / "store_index.json"
    store_index.write_text(
        json.dumps(
            {
                "RecordWithName": {
                    "source_run_log_path": str(source),
                    "source_run_log_sha256": hashlib.sha256(
                        source.read_bytes()
                    ).hexdigest(),
                    "transfer_states_path": str(states),
                    "transfer_states_sha256": hashlib.sha256(
                        states.read_bytes()
                    ).hexdigest(),
                    "provenance_path": str(provenance),
                    "provenance_sha256": hashlib.sha256(
                        provenance.read_bytes()
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    item = SimpleNamespace(
        task="RecordWithName",
        source_run_log=source,
        meta={
            "source_run_log_sha256": hashlib.sha256(
                source.read_bytes()
            ).hexdigest(),
            "store_provenance": str(provenance),
            "store_provenance_sha256": hashlib.sha256(
                provenance.read_bytes()
            ).hexdigest(),
        },
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
        store_index_path=store_index,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"][
        "text"
    ] == "Save"
    assert audit["source_state_catalog"] == str(source)
    assert audit["source_state_catalog_source"] == "embedded_source_run_log"


def test_baseline_grounding_uses_complete_states_embedded_in_source_runlog(
    tmp_path: Path,
) -> None:
    xml = (
        '<hierarchy><node class="android.widget.Button" text="Continue" '
        'resource-id="app:id/continue" clickable="true" '
        'bounds="[0,0][100,100]" /></hierarchy>'
    )
    source = tmp_path / "source.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {"action_type": "open_app", "app_name": "com.example.app"},
                    {"action_type": "click", "x": 50, "y": 50},
                ],
                observations=[
                    androidworld_state(
                        "launcher-state",
                        forest=xml,
                        package_name="com.android.launcher",
                        width=100,
                        height=100,
                    ),
                    androidworld_state(
                        "complete-state",
                        forest=xml,
                        package_name="com.example.app",
                        width=100,
                        height=100,
                    ),
                ],
                task_name="CompleteTask",
                run_id="complete-source",
                goal="Open the app and continue.",
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    function_states = tmp_path / "function_transfer_states.json"
    function_states.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.transfer-state-catalog.v1",
                "run_id": "complete-source",
                "states": {
                    "launcher-state": {
                        "state_id": "launcher-state",
                        "xml": xml,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    provenance = tmp_path / "provenance_manifest.json"
    provenance.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.source-replay-transfer-store.v1",
                "source_run_log_sha256": "0" * 64,
                "output_source_run_log_sha256": source_sha256,
                "source_target_audit": {
                    "source_target_audit_complete": True,
                    "source_targets": [],
                },
            }
        ),
        encoding="utf-8",
    )
    store_index = tmp_path / "store_index.json"
    store_index.write_text(
        json.dumps(
            {
                "CompleteTask": {
                    "source_run_log_path": str(source),
                    "source_run_log_sha256": source_sha256,
                    "transfer_states_path": str(function_states),
                    "transfer_states_sha256": hashlib.sha256(
                        function_states.read_bytes()
                    ).hexdigest(),
                    "provenance_path": str(provenance),
                    "provenance_sha256": hashlib.sha256(
                        provenance.read_bytes()
                    ).hexdigest(),
                }
            }
        ),
        encoding="utf-8",
    )
    item = SimpleNamespace(
        task="CompleteTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
        store_index_path=store_index,
    )

    assert len(grounded["steps"]) == 2
    assert grounded["steps"][1]["metadata"]["source_context"]["element"] == {
        "text": "Continue",
        "resource_id": "app:id/continue",
    }
    assert audit["source_state_catalog_source"] == "embedded_source_run_log"
    assert audit["source_state_count"] == 2


def test_source_revision_reuses_frozen_asset_or_advances_past_failures(
    tmp_path: Path,
) -> None:
    base = tmp_path / "appagent_demo"
    failed = base / "native_source_r3"
    failed.mkdir(parents=True)
    (failed / "prep_failure.json").write_text("{}", encoding="utf-8")

    assert select_source_asset_revision(
        base,
        manifest_name="appagent_demo_manifest.json",
    ) == base / "native_source_r4"

    frozen = base / "native_source_r4"
    frozen.mkdir()
    (frozen / "appagent_demo_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    incomplete = base / "native_source_r5"
    incomplete.mkdir()

    assert select_source_asset_revision(
        base,
        manifest_name="appagent_demo_manifest.json",
    ) == frozen


def test_source_revision_is_stable_for_one_exact_source_hash(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt_offline_retrieval"
    old = base / "native_source_r3"
    old.mkdir(parents=True)
    (old / "cold_memory_manifest.json").write_text(
        json.dumps(
            {
                "source_run_log": {
                    "sha256": "1" * 64,
                }
            }
        ),
        encoding="utf-8",
    )
    expected = "2" * 64
    selected = base / f"source_{expected[:12]}"

    assert select_source_asset_revision(
        base,
        manifest_name="cold_memory_manifest.json",
        expected_source_sha256=expected,
    ) == selected

    selected.mkdir()
    (selected / "generation_failure.json").write_text("{}", encoding="utf-8")
    revision_two = base / f"source_{expected[:12]}_r2"
    assert select_source_asset_revision(
        base,
        manifest_name="cold_memory_manifest.json",
        expected_source_sha256=expected,
    ) == revision_two

    revision_two.mkdir()
    (revision_two / "cold_memory_manifest.json").write_text(
        json.dumps({"source_run_log_sha256": expected}),
        encoding="utf-8",
    )
    assert select_source_asset_revision(
        base,
        manifest_name="cold_memory_manifest.json",
        expected_source_sha256=expected,
    ) == revision_two


def test_source_revision_rejects_terminal_failure_for_exact_source_hash(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt_offline_retrieval"
    expected = "2" * 64
    failed = base / f"source_{expected[:12]}"
    failed.mkdir(parents=True)
    (failed / "prep_failure.json").write_text(
        json.dumps(
            {
                "error": "mobilegpt_cold_memory_official_source_failed",
                "retry_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="source_asset_retry_forbidden:.*"
        "mobilegpt_cold_memory_official_source_failed",
    ):
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=expected,
        )


def test_source_revision_allows_audited_environment_repair(
    tmp_path: Path,
) -> None:
    base = tmp_path / "appagent_demo"
    expected = "2" * 64
    failed = base / f"source_{expected[:12]}"
    failed.mkdir(parents=True)
    (failed / "prep_failure.json").write_text(
        json.dumps(
            {
                "error": "appagent_official_source_success_required",
                "retry_allowed": False,
            }
        ),
        encoding="utf-8",
    )

    assert select_source_asset_revision(
        base,
        manifest_name="appagent_demo_manifest.json",
        expected_source_sha256=expected,
        environment_repair_reason=(
            "launch_source_app_package_from_teacher_contract@7709f60"
        ),
    ) == base / f"source_{expected[:12]}_r2"


def test_source_revision_advances_beyond_two_digit_failure_revision(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt_offline_retrieval"
    expected = "2" * 64
    prefix = f"source_{expected[:12]}"
    for revision in range(1, 11):
        suffix = "" if revision == 1 else f"_r{revision}"
        attempt = base / f"{prefix}{suffix}"
        attempt.mkdir(parents=True)
        (attempt / "prep_failure.json").write_text("{}", encoding="utf-8")

    assert select_source_asset_revision(
        base,
        manifest_name="cold_memory_manifest.json",
        expected_source_sha256=expected,
    ) == base / f"{prefix}_r11"
