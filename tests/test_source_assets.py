from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from src.experiment import androidworld as pipeline
from src.experiment.source_assets import build_grounded_teacher_run_log
from src.integrations.appagent_adapter import build_appagent_teacher_source
from src.integrations.mobilegpt_teacher import (
    preflight_teacher_source_run_log,
)


def _write_source_bundle(root: Path) -> tuple[Path, Path, Path]:
    source = root / "source.run_log.json"
    source.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.canonical_run_log.v1",
                "run_id": "source-run",
                "goal": "Save the recording as Example.",
                "status": "succeeded",
                "success": True,
                "steps": [
                    {
                        "step_index": 0,
                        "before_state_id": "button-state",
                        "action": {
                            "tool": "click",
                            "args": {"x": 500.0, "y": 500.0},
                        },
                        "result": {"success": True},
                        "after_state_id": "input-state",
                    },
                    {
                        "step_index": 1,
                        "before_state_id": "input-state",
                        "action": {
                            "tool": "input_text",
                            "args": {"text": "Example"},
                        },
                        "result": {"success": True},
                        "after_state_id": "done-state",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    xml = (
        '<hierarchy><node class="android.widget.Button" text="Save" '
        'resource-id="app:id/save" clickable="true" bounds="[0,0][100,100]" />'
        '<node class="android.widget.EditText" text="" '
        'resource-id="app:id/name" editable="true" '
        'bounds="[0,100][100,200]" /></hierarchy>'
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
