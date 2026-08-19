from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment.source_evidence import (
    build_grounded_teacher_run_log,
    build_grounded_teacher_run_log_from_item,
    select_source_asset_revision,
)
from src.integrations.appagent import (
    build_appagent_teacher_source,
    ground_appagent_teacher_action,
)
from src.integrations.mobilegpt import preflight_runlog_conversion


def _source(report: dict) -> dict:
    return report["source"]


def _grounding(report: dict) -> dict:
    return report["grounding"]


def _safety(report: dict) -> dict:
    return report["safety"]
from src.integrations.runlog import convert_legacy_run_log


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


def test_frozen_source_evidence_grounds_appagent_teacher(
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
        expected_provenance_sha256=hashlib.sha256(provenance.read_bytes()).hexdigest(),
    )
    grounded_path = tmp_path / "grounded.teacher.run_log.json"
    grounded_path.write_text(json.dumps(grounded), encoding="utf-8")

    appagent = build_appagent_teacher_source(
        grounded_path,
        task_name="RecordWithName",
        provenance_source_run_log=source,
    )

    assert audit["schema_version"] == "omniflow.source.evidence.v2"
    assert appagent["action_count"] == 2
    assert appagent["source_run_log"] == str(source)
    assert appagent["source_run_log_sha256"] == source_sha256
    assert _safety(audit)["target_inputs_read"] is False
    assert _safety(audit)["target_observations_read"] is False
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
    source, states, _provenance = _write_source_bundle(tmp_path)
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
                }
            }
        ),
        encoding="utf-8",
    )
    item = SimpleNamespace(
        task="RecordWithName",
        source_run_log=source,
        meta={
            "source_run_log_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        },
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
        store_index_path=store_index,
    )

    assert (
        grounded["steps"][0]["metadata"]["source_context"]["element"]["text"] == "Save"
    )
    assert _source(audit)["state_catalog"] == str(states)
    assert _source(audit)["catalog_source"] == "frozen_catalog"
    assert "provenance_manifest" not in audit


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
    assert _source(audit)["catalog_source"] == "embedded_source_run_log"
    assert _source(audit)["state_count"] == 2


def test_canonical_runlog_grounds_mobilegpt_without_omniflow_store(
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
                        forest="",
                        package_name="com.android.launcher",
                        width=100,
                        height=100,
                    ),
                    androidworld_state(
                        "state-0",
                        forest=xml,
                        package_name="com.example.app",
                        width=100,
                        height=100,
                    ),
                ],
                task_name="CompleteTask",
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="CompleteTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][1]["metadata"]["source_context"]["element"] == {
        "text": "Continue",
        "resource_id": "app:id/continue",
    }
    assert _grounding(audit)["source"] == "canonical_androidworld_run_log"
    assert "provenance_manifest" not in audit


def _write_legacy_backed_canonical_source(
    root: Path,
    *,
    xml: str,
    target_description: str,
) -> tuple[Path, Path, SimpleNamespace, dict[str, object]]:
    raw_path = root / "raw.run_log.json"
    raw = {
        "run_id": "legacy-source",
        "goal": "Tap Continue.",
        "success": True,
        "steps": [
            {
                "observation_before_act": {
                    "xml": xml,
                    "width": 100,
                    "height": 100,
                },
                "action": {
                    "type": "click",
                    "params": {
                        "x": 90,
                        "y": 90,
                        "target_description": target_description,
                    },
                },
                "success": True,
            }
        ],
    }
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    canonical = convert_legacy_run_log(
        raw,
        task_name="LegacyTargetTask",
        task_parameters={},
        seed=111,
        source_path=raw_path,
        require_screenshots=False,
    )
    canonical["steps"][0].pop("metadata", None)
    canonical_path = root / "canonical.run_log.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    canonical_sha256 = hashlib.sha256(canonical_path.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="LegacyTargetTask",
        source_run_log=canonical_path,
        meta={"retained_source_run_log_sha256": canonical_sha256},
    )
    return raw_path, canonical_path, item, raw


def test_canonical_grounding_recovers_unique_verified_legacy_target(
    tmp_path: Path,
) -> None:
    xml = (
        '<hierarchy><node text="Continue" resource-id="app:id/continue" '
        'clickable="true" bounds="[0,0][20,20]" /></hierarchy>'
    )
    _, _, item, _ = _write_legacy_backed_canonical_source(
        tmp_path,
        xml=xml,
        target_description="Continue",
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "text": "Continue",
        "resource_id": "app:id/continue",
    }
    assert _grounding(audit)["source_target_evidence_source"] == ("verified_legacy_provenance")
    assert _grounding(audit)["source_target_evidence_count"] == 1
    assert _grounding(audit)["verified_source_target_count"] == 1


def test_canonical_grounding_reuses_observations_for_state_id_provenance(
    tmp_path: Path,
) -> None:
    raw_path = tmp_path / "raw-state-id.run_log.json"
    raw = {
        "schema_version": "omniflow.canonical_run_log.v1",
        "run_id": "legacy-state-id",
        "goal": "Tap Continue.",
        "success": True,
        "steps": [
            {
                "step_index": 0,
                "before_state_id": "state-before",
                "after_state_id": "state-after",
                "action": {
                    "tool": "click",
                    "args": {
                        "x": 500,
                        "y": 500,
                        "target_description": "Continue",
                    },
                },
                "result": {"success": True},
            }
        ],
    }
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    source_states = {
        state_id: {
            "state_id": state_id,
            "xml": (
                '<hierarchy><node text="Continue" '
                'resource-id="app:id/continue" clickable="true" '
                'bounds="[40,40][60,60]" /></hierarchy>'
            ),
            "package_name": "com.example.app",
            "activity_name": ".MainActivity",
            "display": {"width": 100, "height": 100},
        }
        for state_id in ("state-before", "state-after")
    }
    canonical = convert_legacy_run_log(
        raw,
        task_name="LegacyTargetTask",
        task_parameters={},
        seed=111,
        source_path=raw_path,
        source_states=source_states,
        require_screenshots=False,
    )
    canonical["steps"][0].pop("metadata", None)
    canonical_path = tmp_path / "canonical-state-id.run_log.json"
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    item = SimpleNamespace(
        task="LegacyTargetTask",
        source_run_log=canonical_path,
        meta={
            "retained_source_run_log_sha256": hashlib.sha256(
                canonical_path.read_bytes()
            ).hexdigest()
        },
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "text": "Continue",
        "resource_id": "app:id/continue",
    }
    assert _grounding(audit)["source_target_evidence_source"] == "verified_legacy_provenance"


def test_canonical_grounding_rejects_legacy_provenance_hash_mismatch(
    tmp_path: Path,
) -> None:
    raw_path, _, item, raw = _write_legacy_backed_canonical_source(
        tmp_path,
        xml='<hierarchy><node text="Continue" /></hierarchy>',
        target_description="Continue",
    )
    raw["goal"] = "Changed after conversion."
    raw_path.write_text(json.dumps(raw), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="source_legacy_provenance_hash_mismatch",
    ):
        build_grounded_teacher_run_log_from_item(
            index_path=tmp_path / "source_index.json",
            item=item,
        )


def test_canonical_grounding_rejects_legacy_action_mismatch(
    tmp_path: Path,
) -> None:
    raw_path, canonical_path, _, raw = _write_legacy_backed_canonical_source(
        tmp_path,
        xml='<hierarchy><node text="Continue" /></hierarchy>',
        target_description="Continue",
    )
    raw["steps"][0]["action"]["params"]["x"] = 80
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["provenance"]["source_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    item = SimpleNamespace(
        task="LegacyTargetTask",
        source_run_log=canonical_path,
        meta={
            "retained_source_run_log_sha256": hashlib.sha256(
                canonical_path.read_bytes()
            ).hexdigest()
        },
    )

    with pytest.raises(
        ValueError,
        match="source_legacy_provenance_action_mismatch:0",
    ):
        build_grounded_teacher_run_log_from_item(
            index_path=tmp_path / "source_index.json",
            item=item,
        )


def test_canonical_grounding_accepts_removed_legacy_wait_steps(
    tmp_path: Path,
) -> None:
    raw_path, canonical_path, _, raw = _write_legacy_backed_canonical_source(
        tmp_path,
        xml=(
            '<hierarchy><node text="Continue" resource-id="app:id/continue" '
            'clickable="true" bounds="[0,0][20,20]" /></hierarchy>'
        ),
        target_description="Continue",
    )
    raw["steps"].insert(
        0,
        {
            "observation_before_act": {
                "xml": '<hierarchy><node text="Ready" /></hierarchy>',
                "width": 100,
                "height": 100,
            },
            "action": {"type": "wait", "params": {}},
            "success": True,
        },
    )
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["provenance"]["source_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    item = SimpleNamespace(
        task="LegacyTargetTask",
        source_run_log=canonical_path,
        meta={
            "retained_source_run_log_sha256": hashlib.sha256(
                canonical_path.read_bytes()
            ).hexdigest()
        },
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "text": "Continue",
        "resource_id": "app:id/continue",
    }
    assert _grounding(audit)["source_target_evidence_source"] == "verified_legacy_provenance"


def test_canonical_grounding_rejects_removed_non_wait_step(
    tmp_path: Path,
) -> None:
    raw_path, canonical_path, _, raw = _write_legacy_backed_canonical_source(
        tmp_path,
        xml='<hierarchy><node text="Continue" /></hierarchy>',
        target_description="Continue",
    )
    raw["steps"].insert(0, dict(raw["steps"][0]))
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    canonical["provenance"]["source_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    canonical_path.write_text(json.dumps(canonical), encoding="utf-8")
    item = SimpleNamespace(
        task="LegacyTargetTask",
        source_run_log=canonical_path,
        meta={
            "retained_source_run_log_sha256": hashlib.sha256(
                canonical_path.read_bytes()
            ).hexdigest()
        },
    )

    with pytest.raises(
        ValueError,
        match="source_legacy_provenance_step_count_mismatch",
    ):
        build_grounded_teacher_run_log_from_item(
            index_path=tmp_path / "source_index.json",
            item=item,
        )


def test_canonical_grounding_does_not_use_ambiguous_legacy_target(
    tmp_path: Path,
) -> None:
    xml = (
        '<hierarchy><node text="Continue" bounds="[0,0][20,20]" />'
        '<node text="Continue" bounds="[20,0][40,20]" /></hierarchy>'
    )
    _, _, item, _ = _write_legacy_backed_canonical_source(
        tmp_path,
        xml=xml,
        target_description="Continue",
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert "element" not in grounded["steps"][0]["metadata"]["source_context"]
    assert _grounding(audit)["source_target_evidence_count"] == 1
    assert _grounding(audit)["verified_source_target_count"] == 0
    assert _grounding(audit)["semantic_action_count"] == 0


@pytest.mark.parametrize("action_type", ["answer", "status", "unknown"])
def test_canonical_grounding_preserves_non_ui_terminal_actions(
    tmp_path: Path,
    action_type: str,
) -> None:
    action = {"action_type": action_type}
    if action_type == "answer":
        action["text"] = "Done"
    elif action_type == "status":
        action["goal_status"] = "complete"
    source = tmp_path / f"{action_type}.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [action],
                observations=[androidworld_state("terminal-state", forest="")],
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="TerminalTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["action"] == action
    assert grounded["steps"][0]["metadata"]["source_context"] == {
        "package_name": "com.example.app"
    }
    assert _grounding(audit)["semantic_action_count"] == 0


def test_canonical_grounding_uses_input_text_action_point(
    tmp_path: Path,
) -> None:
    xml = (
        '<hierarchy><node class="android.widget.EditText" text="First" '
        'resource-id="app:id/first" editable="true" bounds="[0,0][50,100]" />'
        '<node class="android.widget.EditText" text="Second" '
        'resource-id="app:id/second" editable="true" bounds="[50,0][100,100]" />'
        "</hierarchy>"
    )
    source = tmp_path / "input-text.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "input_text", "text": "Example", "x": 75, "y": 50}],
                observations=[
                    androidworld_state(
                        "input-state",
                        forest=xml,
                        width=100,
                        height=100,
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="InputTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "text": "Second",
        "resource_id": "app:id/second",
    }
    assert _grounding(audit)["semantic_action_count"] == 1


def test_canonical_grounding_uses_unique_structural_child_target(
    tmp_path: Path,
) -> None:
    xml = (
        '<hierarchy><node clickable="true" bounds="[0,0][100,50]">'
        '<node text="Dreamer&apos;s Awake" bounds="[0,0][70,50]" />'
        '<node clickable="true" bounds="[70,0][100,50]" />'
        "</node></hierarchy>"
    )
    source = tmp_path / "anonymous-child.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 85, "y": 25}],
                observations=[
                    androidworld_state(
                        "anonymous-child",
                        forest=xml,
                        width=100,
                        height=50,
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="AnonymousChildTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "relation": "unique_actionable_descendant",
        "container_anchor": {"text": "Dreamer's Awake"},
    }
    grounded_path = tmp_path / "grounded-anonymous-child.run_log.json"
    grounded_path.write_text(json.dumps(grounded), encoding="utf-8")
    preflight = preflight_runlog_conversion(grounded_path)
    assert preflight["ready"] is True
    assert preflight["transition_count"] == _grounding(audit)["semantic_action_count"]
    assert _grounding(audit)["semantic_action_count"] == 1


def test_canonical_grounding_uses_unique_anonymous_editable_role(
    tmp_path: Path,
) -> None:
    xml = (
        '<hierarchy><node bounds="[0,0][100,100]">'
        '<node text="Select time" bounds="[0,0][100,50]" />'
        '<node class="android.widget.EditText" editable="true" '
        'focused="true" bounds="[0,50][100,100]" />'
        "</node></hierarchy>"
    )
    source = tmp_path / "anonymous-input.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {
                        "action_type": "input_text",
                        "text": "Example",
                        "x": 50,
                        "y": 75,
                    }
                ],
                observations=[
                    androidworld_state(
                        "anonymous-input",
                        forest=xml,
                        width=100,
                        height=100,
                    )
                ],
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="AnonymousInputTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "role": "editable"
    }
    grounded_path = tmp_path / "grounded-anonymous-input.run_log.json"
    grounded_path.write_text(json.dumps(grounded), encoding="utf-8")
    preflight = preflight_runlog_conversion(grounded_path)
    assert preflight["ready"] is True
    assert preflight["transition_count"] == _grounding(audit)["semantic_action_count"]
    assert _grounding(audit)["semantic_action_count"] == 1


def test_canonical_grounding_inherits_adjacent_unique_editable_target(
    tmp_path: Path,
) -> None:
    full_xml = (
        '<hierarchy><node bounds="[0,0][720,1280]">'
        '<node text="Name" bounds="[32,80][688,120]" />'
        '<node text=".md" editable="true" clickable="true" '
        'bounds="[433,124][592,207]" />'
        "</node></hierarchy>"
    )
    degraded_xml = '<hierarchy><node bounds="[492,185][572,265]" /></hierarchy>'
    source = tmp_path / "adjacent-editable.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {"action_type": "click", "x": 512, "y": 165},
                    {
                        "action_type": "input_text",
                        "clear_text": True,
                        "text": ".txt",
                    },
                ],
                observations=[
                    androidworld_state(
                        "editable-before",
                        forest=full_xml,
                        package_name="net.example.editor",
                        width=720,
                        height=1280,
                    ),
                    androidworld_state(
                        "editable-degraded",
                        forest=degraded_xml,
                        package_name="net.example.editor",
                        width=720,
                        height=1280,
                    ),
                ],
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="AdjacentEditableTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    for step in grounded["steps"]:
        assert step["metadata"]["source_context"]["element"] == {"text": ".md"}
    grounded_path = tmp_path / "grounded-adjacent-editable.run_log.json"
    grounded_path.write_text(json.dumps(grounded), encoding="utf-8")
    preflight = preflight_runlog_conversion(grounded_path)
    assert preflight["ready"] is True
    assert preflight["transition_count"] == _grounding(audit)["semantic_action_count"]
    assert _grounding(audit)["semantic_action_count"] == 2


def test_canonical_grounding_uses_verified_input_text_change(
    tmp_path: Path,
) -> None:
    before_xml = (
        '<hierarchy><node id="12" class="android.widget.EditText" '
        'text="my_note" clickable="true" bounds="[0,0][100,100]" />'
        '<node id="13" class="android.widget.EditText" text=".md" '
        'clickable="true" bounds="[100,0][200,100]" /></hierarchy>'
    )
    after_xml = before_xml.replace('text="my_note"', 'text="copy_warm_tree"')
    source = tmp_path / "changed-input.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {"action_type": "input_text", "text": "copy_warm_tree"},
                    {"action_type": "click", "x": 150, "y": 50},
                ],
                observations=[
                    androidworld_state("before-input", forest=before_xml),
                    androidworld_state("after-input", forest=after_xml),
                ],
            )
        ),
        encoding="utf-8",
    )
    item = SimpleNamespace(
        task="ChangedInputTask",
        source_run_log=source,
        meta={
            "retained_source_run_log_sha256": hashlib.sha256(
                source.read_bytes()
            ).hexdigest()
        },
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "text": "my_note"
    }
    grounded_path = tmp_path / "grounded-changed-input.run_log.json"
    grounded_path.write_text(json.dumps(grounded), encoding="utf-8")
    teacher = build_appagent_teacher_source(
        grounded_path,
        task_name="ChangedInputTask",
        provenance_source_run_log=source,
    )
    appagent_target = ground_appagent_teacher_action(
        before_xml,
        teacher["actions"][0]["action"],
        min_dist=30.0,
    )
    assert appagent_target.tag == 1
    assert appagent_target.match_reason == "exact_visible_identity"
    assert _grounding(audit)["semantic_action_count"] == 2


def test_canonical_grounding_does_not_inherit_editable_across_packages(
    tmp_path: Path,
) -> None:
    full_xml = (
        '<hierarchy><node bounds="[0,0][100,100]">'
        '<node text="Name" editable="true" clickable="true" '
        'bounds="[0,0][100,100]" />'
        "</node></hierarchy>"
    )
    source = tmp_path / "cross-package-editable.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {"action_type": "click", "x": 50, "y": 50},
                    {"action_type": "input_text", "text": "unsafe"},
                ],
                observations=[
                    androidworld_state(
                        "package-a",
                        forest=full_xml,
                        package_name="com.example.a",
                        width=100,
                        height=100,
                    ),
                    androidworld_state(
                        "package-b",
                        forest="<hierarchy />",
                        package_name="com.example.b",
                        width=100,
                        height=100,
                    ),
                ],
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="CrossPackageEditableTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert "element" not in grounded["steps"][1]["metadata"]["source_context"]
    assert _grounding(audit)["semantic_action_count"] == 1


def test_canonical_grounding_recovers_source_display_from_xml(
    tmp_path: Path,
) -> None:
    xml = (
        '<hierarchy><node bounds="[0,0][1080,2400]">'
        '<node text="Create folder" resource-id="app:id/create_folder" '
        'clickable="true" bounds="[900,2000][1040,2200]" />'
        "</node></hierarchy>"
    )
    observation = androidworld_state(
        "missing-display",
        forest=xml,
        width=1080,
        height=2400,
    )
    observation["auxiliaries"].pop("display")
    source = tmp_path / "missing-display.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "click", "x": 964, "y": 2074}],
                observations=[observation],
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="MissingDisplayTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "text": "Create folder",
        "resource_id": "app:id/create_folder",
    }
    assert grounded["steps"][0]["observation"]["auxiliaries"]["display"] == {
        "width": 1080,
        "height": 2400,
    }
    grounded_path = tmp_path / "grounded-missing-display.run_log.json"
    grounded_path.write_text(json.dumps(grounded), encoding="utf-8")
    preflight = preflight_runlog_conversion(grounded_path)
    assert preflight["ready"] is True
    assert preflight["transition_count"] == _grounding(audit)["semantic_action_count"]
    assert _grounding(audit)["semantic_action_count"] == 1


def test_canonical_grounding_reuses_unique_source_display_for_dialog_xml(
    tmp_path: Path,
) -> None:
    full_xml = '<hierarchy><node bounds="[0,0][1080,2400]" /></hierarchy>'
    dialog_xml = (
        '<hierarchy><node bounds="[100,500][980,1700]">'
        '<node text="Create folder" resource-id="app:id/create_folder" '
        'clickable="true" bounds="[700,1300][900,1500]" />'
        "</node></hierarchy>"
    )
    observations = [
        androidworld_state("full-state", forest=full_xml, width=1080, height=2400),
        androidworld_state("dialog-state", forest=dialog_xml, width=1080, height=2400),
    ]
    for observation in observations:
        observation["auxiliaries"].pop("display")
    source = tmp_path / "dialog-display.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [
                    {"action_type": "open_app", "app_name": "com.example.app"},
                    {"action_type": "click", "x": 800, "y": 1400},
                ],
                observations=observations,
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="DialogDisplayTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][1]["metadata"]["source_context"]["element"] == {
        "text": "Create folder",
        "resource_id": "app:id/create_folder",
    }
    assert _grounding(audit)["semantic_action_count"] == 1


def test_canonical_grounding_distinguishes_page_input_from_browser_chrome(
    tmp_path: Path,
) -> None:
    xml = (
        '<hierarchy><node class="android.widget.EditText" '
        'text="Enter the product" resource-id="answer" editable="true" '
        'bounds="[200,200][500,300]" />'
        '<node class="android.widget.EditText" text="https://example.test" '
        'resource-id="com.android.chrome:id/url_bar" editable="true" '
        'bounds="[100,50][600,150]" /></hierarchy>'
    )
    source = tmp_path / "browser-input.run_log.json"
    source.write_text(
        json.dumps(
            androidworld_run_log(
                [{"action_type": "input_text", "text": "1400"}],
                observations=[androidworld_state("browser-input", forest=xml)],
            )
        ),
        encoding="utf-8",
    )
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    item = SimpleNamespace(
        task="BrowserInputTask",
        source_run_log=source,
        meta={"retained_source_run_log_sha256": source_sha256},
    )

    grounded, audit = build_grounded_teacher_run_log_from_item(
        index_path=tmp_path / "source_index.json",
        item=item,
    )

    assert grounded["steps"][0]["metadata"]["source_context"]["element"] == {
        "text": "Enter the product",
        "resource_id": "answer",
    }
    assert _grounding(audit)["semantic_action_count"] == 1


def test_source_revision_reuses_frozen_asset_or_advances_past_failures(
    tmp_path: Path,
) -> None:
    base = tmp_path / "appagent"
    failed = base / "native_source_r3"
    failed.mkdir(parents=True)
    (failed / "prep_failure.json").write_text("{}", encoding="utf-8")

    assert (
        select_source_asset_revision(
            base,
            manifest_name="appagent_manifest.json",
        )
        == base / "native_source_r4"
    )

    frozen = base / "native_source_r4"
    frozen.mkdir()
    (frozen / "appagent_manifest.json").write_text(
        "{}",
        encoding="utf-8",
    )
    incomplete = base / "native_source_r5"
    incomplete.mkdir()

    assert (
        select_source_asset_revision(
            base,
            manifest_name="appagent_manifest.json",
        )
        == frozen
    )


def test_source_revision_is_stable_for_one_exact_source_hash(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt"
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

    assert (
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=expected,
        )
        == selected
    )

    selected.mkdir()
    (selected / "generation_failure.json").write_text("{}", encoding="utf-8")
    revision_two = base / f"source_{expected[:12]}_r2"
    assert (
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=expected,
        )
        == revision_two
    )

    revision_two.mkdir()
    (revision_two / "cold_memory_manifest.json").write_text(
        json.dumps({"source_run_log_sha256": expected}),
        encoding="utf-8",
    )
    assert (
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=expected,
        )
        == revision_two
    )


def test_source_revision_reuses_explicit_conversion_lineage_hash(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt"
    canonical = "2" * 64
    legacy = "1" * 64
    frozen = base / f"source_{legacy[:12]}"
    frozen.mkdir(parents=True)
    (frozen / "cold_memory_manifest.json").write_text(
        json.dumps({"source_run_log": {"sha256": legacy}}),
        encoding="utf-8",
    )

    assert (
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=canonical,
            compatible_source_sha256s=(legacy,),
        )
        == frozen
    )


def test_source_revision_skips_frozen_asset_from_wrong_model(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt"
    expected = "2" * 64
    legacy = "1" * 64
    frozen = base / "native_source_r2"
    frozen.mkdir(parents=True)
    (frozen / "cold_memory_manifest.json").write_text(
        json.dumps(
            {
                "source_run_log": {"sha256": legacy},
                "source_model": "qwen-plus",
            }
        ),
        encoding="utf-8",
    )
    selected = base / f"source_{expected[:12]}"

    assert (
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=expected,
            compatible_source_sha256s=(legacy,),
            expected_source_model="qwen3-vl-plus",
        )
        == selected
    )

    selected.mkdir()
    (selected / "cold_memory_manifest.json").write_text(
        json.dumps(
            {
                "source_run_log": {"sha256": legacy},
                "source_model": "qwen3-vl-plus",
            }
        ),
        encoding="utf-8",
    )
    assert (
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=expected,
            compatible_source_sha256s=(legacy,),
            expected_source_model="qwen3-vl-plus",
        )
        == selected
    )


def test_source_revision_skips_incompatible_mobilegpt_memory_contract(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt"
    expected = "2" * 64
    old = base / f"source_{expected[:12]}"
    old.mkdir(parents=True)
    (old / "cold_memory_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "omniflow.mobilegpt-cold-memory.v1",
                "source_method": "fixed_replay",
                "source_model": "qwen3-vl-plus",
                "source_run_log": {"sha256": expected},
            }
        ),
        encoding="utf-8",
    )

    selected = select_source_asset_revision(
        base,
        manifest_name="cold_memory_manifest.json",
        expected_source_sha256=expected,
        expected_source_model="qwen3-vl-plus",
        expected_schema_version="omniflow.mobilegpt-native-cold-memory.v1",
        expected_source_method="mobilegpt_native_source_cold",
    )

    assert selected == base / f"source_{expected[:12]}_r2"


def test_source_revision_skips_frozen_asset_rejected_by_validator(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt"
    expected = "2" * 64
    frozen = base / f"source_{expected[:12]}"
    frozen.mkdir(parents=True)
    manifest = {
        "schema_version": "omniflow.mobilegpt-native-cold-memory.v1",
        "source_method": "mobilegpt_native_source_cold",
        "source_model": "qwen3-vl-plus",
        "source_run_log": {"sha256": expected},
    }
    (frozen / "cold_memory_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    validated: list[tuple[Path, dict]] = []

    selected = select_source_asset_revision(
        base,
        manifest_name="cold_memory_manifest.json",
        expected_source_sha256=expected,
        expected_source_model="qwen3-vl-plus",
        expected_schema_version="omniflow.mobilegpt-native-cold-memory.v1",
        expected_source_method="mobilegpt_native_source_cold",
        candidate_validator=lambda candidate, payload: (
            validated.append((candidate, payload)) or False
        ),
    )

    assert selected == base / f"source_{expected[:12]}_r2"
    assert validated == [(frozen, manifest)]


def test_source_revision_ignores_terminal_failure_from_old_method(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt"
    expected = "2" * 64
    failed = base / f"source_{expected[:12]}"
    failed.mkdir(parents=True)
    (failed / "prep_failure.json").write_text(
        json.dumps(
            {
                "error": "mobilegpt_source_episode_failed:1",
                "retry_allowed": False,
            }
        ),
        encoding="utf-8",
    )
    (failed / "source_episode_command.json").write_text(
        json.dumps({"source_method": "fixed_replay"}),
        encoding="utf-8",
    )

    assert (
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=expected,
            expected_source_method="mobilegpt_native_source_cold",
        )
        == base / f"source_{expected[:12]}_r2"
    )


def test_source_revision_rejects_terminal_failure_for_exact_source_hash(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt"
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


def test_source_revision_advances_beyond_two_digit_failure_revision(
    tmp_path: Path,
) -> None:
    base = tmp_path / "mobilegpt"
    expected = "2" * 64
    prefix = f"source_{expected[:12]}"
    for revision in range(1, 11):
        suffix = "" if revision == 1 else f"_r{revision}"
        attempt = base / f"{prefix}{suffix}"
        attempt.mkdir(parents=True)
        (attempt / "prep_failure.json").write_text("{}", encoding="utf-8")

    assert (
        select_source_asset_revision(
            base,
            manifest_name="cold_memory_manifest.json",
            expected_source_sha256=expected,
        )
        == base / f"{prefix}_r11"
    )
