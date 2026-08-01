from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from PIL import Image
from runlog_fixtures import androidworld_run_log, androidworld_state

from src.experiment.androidworld import aggregate_task_results
from src.experiment.observation_evidence import (
    ObservationArchive,
    persist_target_run_evidence,
)


def test_archive_preserves_every_observation_and_deduplicates_images(
    tmp_path,
) -> None:
    states = [
        SimpleNamespace(
            pixels=Image.new("RGB", (4, 3), color="red"),
            package_name="com.android.settings",
            activity_name="com.android.settings/.Settings",
        ),
        SimpleNamespace(
            pixels=Image.new("RGB", (4, 3), color="red"),
            package_name="com.android.settings",
            activity_name="com.android.settings/.Settings",
        ),
    ]
    remaining = iter(states)
    archive = ObservationArchive(lambda: next(remaining))

    assert archive.get_state() is states[0]
    assert archive.get_state() is states[1]
    records = archive.persist(tmp_path)

    assert [record["observation_index"] for record in records] == [0, 1]
    assert records[0]["path"] == records[1]["path"]
    assert records[0]["sha256"] == records[1]["sha256"]
    assert records[0]["display"] == {"width": 4, "height": 3}
    assert records[0]["package_name"] == "com.android.settings"
    assert len(list((tmp_path / "observations" / "objects").glob("*.png"))) == 1
    assert json.loads((tmp_path / "observations" / "index.json").read_text()) == {
        "schema_version": "omniflow.androidworld-observations.v1",
        "observation_count": 2,
        "observations": records,
    }


def test_archive_reports_an_observation_without_pixels(tmp_path) -> None:
    state = SimpleNamespace(
        pixels=None,
        package_name="com.android.settings",
        activity_name="com.android.settings/.Settings",
    )
    archive = ObservationArchive(lambda: state)

    assert archive.get_state() is state

    assert archive.persist(tmp_path) == [
        {
            "observation_index": 0,
            "package_name": "com.android.settings",
            "activity_name": "com.android.settings/.Settings",
            "error": "observation_image_missing",
        }
    ]


def test_target_run_evidence_is_immutable_and_hash_addressable(tmp_path) -> None:
    run_log = androidworld_run_log(
        [{"action_type": "open_app", "app_name": "com.android.settings"}],
        observations=[androidworld_state("target-before")],
        task_name="OpenSettings",
        run_id="target-run",
        goal="Open Settings.",
    )
    run_log["steps"][0]["next_observation"] = androidworld_state("target-after")
    states = {
        "target-before": {
            "state_id": "target-before",
            "xml": "<hierarchy />",
        }
    }
    audit = {
        "referenced_state_ids": ["target-after", "target-before"],
        "captured_state_ids": ["target-before"],
        "missing_state_ids": ["target-after"],
        "referenced_state_count": 2,
        "captured_state_count": 1,
        "missing_state_count": 1,
        "complete": False,
    }

    first = persist_target_run_evidence(
        tmp_path,
        run_log=run_log,
        captured_transfer_states=states,
        transfer_state_audit=audit,
    )
    second = persist_target_run_evidence(
        tmp_path,
        run_log=run_log,
        captured_transfer_states=states,
        transfer_state_audit=audit,
    )

    assert first == second
    for path_key, sha_key in (
        ("target_run_log_path", "target_run_log_sha256"),
        ("target_transfer_states_path", "target_transfer_states_sha256"),
    ):
        path = Path(first[path_key])
        assert path.is_file()
        assert hashlib.sha256(path.read_bytes()).hexdigest() == first[sha_key]
    assert first["target_transfer_state_audit"]["missing_state_ids"] == [
        "target-after"
    ]


def test_target_evidence_provenance_survives_metrics_aggregation(tmp_path) -> None:
    result_path = tmp_path / "Task" / "ours" / "small5554" / "task_results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "Task",
                "official_validator_used": True,
                "success": True,
                "target_run_log_path": "/evidence/target.run_log.json",
                "target_run_log_sha256": "run-sha",
                "target_transfer_states_path": "/evidence/target.transfer_states.json",
                "target_transfer_states_sha256": "states-sha",
                "target_transfer_state_audit": {"complete": True},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    row = aggregate_task_results([result_path])["per_task"][0]

    assert row["target_run_log_sha256"] == "run-sha"
    assert row["target_transfer_states_sha256"] == "states-sha"
    assert row["target_transfer_state_audit"] == {"complete": True}


def test_metrics_preserve_missing_validator_as_unknown(tmp_path) -> None:
    result_path = tmp_path / "Task" / "fixed_replay" / "fold5564" / "task_results.jsonl"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(
        json.dumps(
            {
                "task_name": "Task",
                "official_validator_used": False,
                "success": False,
                "error": "FileNotFoundError: app database missing",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    summary = aggregate_task_results([result_path])

    assert summary["official_validator_task_count"] == 0
    assert summary["official_validator_coverage_rate"] == 0.0
    assert summary["per_task"][0]["official_validator_success"] is None
    assert summary["per_task"][0]["success"] is None
