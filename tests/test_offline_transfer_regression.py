from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from omniflow.core.model import Action, TransferResult
from src.experiment.offline_transfer_regression import (
    add_comparison_failures,
    add_embedding_negative,
    add_runlog_pair,
    annotate_case,
    run_regression_cycle,
    run_regression_dataset,
)


class _Encoder:
    name = "test-page-encoder"
    version = "test.v1"
    dimension = 2

    def embed(self, observation):
        vector = (
            np.asarray([0.0, 1.0], dtype=np.float32)
            if "different" in observation.xml
            else np.asarray([1.0, 0.0], dtype=np.float32)
        )
        return SimpleNamespace(vector=vector, elements=(object(),))


def _observation(label: str) -> dict[str, object]:
    return {
        "xml": (
            '<hierarchy bounds="[0,0][100,200]">'
            f'<node text="{label}" clickable="true" enabled="true" '
            'bounds="[40,40][80,80]" />'
            "</hierarchy>"
        ),
        "package_name": "example.app",
        "activity_name": "MainActivity",
        "display": {"width": 100, "height": 200},
    }


def test_dataset_checks_only_page_embedding_and_transfer(tmp_path: Path) -> None:
    dataset_path = tmp_path / "dataset.json"
    dataset_path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.offline-embedding-transfer-dataset.v1",
                "name": "smoke",
                "cases": [
                    {
                        "id": "positive-click",
                        "group": "smoke",
                        "source": _observation("same source"),
                        "target": _observation("same target"),
                        "source_action": {
                            "tool": "click",
                            "args": {"x": 500.0, "y": 250.0},
                        },
                        "expected": {
                            "page_match": True,
                            "minimum_page_similarity": 0.8,
                            "transfer_target_bounds": [40, 40, 80, 80],
                            "minimum_transfer_confidence": 0.8,
                        },
                    },
                    {
                        "id": "negative-page",
                        "group": "embedding-negatives",
                        "source": _observation("same source"),
                        "target": _observation("different target"),
                        "source_action": None,
                        "expected": {
                            "page_match": False,
                            "minimum_page_similarity": 0.8,
                        },
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    transfer_calls = []

    def transfer(action, target, source):
        transfer_calls.append((action, target, source))
        return TransferResult(
            Action("click", {"x": 600.0, "y": 300.0}),
            detail={"absolute_contextual_confidence": 0.92},
        )

    report = run_regression_dataset(
        dataset_path,
        page_encoder=_Encoder(),
        transfer=transfer,
    )

    assert report["status"] == "passed"
    assert report["case_count"] == 2
    assert report["passed_count"] == 2
    assert report["embedding_passed_count"] == 2
    assert report["transfer_case_count"] == 1
    assert report["transfer_passed_count"] == 1
    assert len(transfer_calls) == 1
    assert report["cases"][0]["page_similarity"] == 1.0
    assert report["cases"][0]["transfer_confidence"] == 0.92
    assert report["cases"][0]["mapped_target_point"] == [60.0, 60.0]
    assert report["cases"][1]["page_similarity"] == 0.0


def _run_log(path: Path, *, status: str, point: tuple[int, int]) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "omniflow.run_log.v1",
                "run_id": path.stem,
                "task_name": "ClickTask",
                "status": status,
                "success": status == "succeeded",
                "steps": [
                    {
                        "step_index": 0,
                        "observation": {
                            "xml": (
                                '<hierarchy bounds="[0,0][100,200]">'
                                '<node text="Click me" clickable="true" enabled="true" '
                                'bounds="[40,40][80,80]" />'
                                "</hierarchy>"
                            ),
                            "auxiliaries": {
                                "display": {"width": 100, "height": 200},
                                "package_name": "example.app",
                                "activity_name": "MainActivity",
                            },
                        },
                        "action": {
                            "action_type": "click",
                            "x": point[0],
                            "y": point[1],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_success_and_failure_pairs_share_one_deduplicated_dataset(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    source = _run_log(tmp_path / "source.json", status="succeeded", point=(50, 60))
    successful = _run_log(
        tmp_path / "successful.json", status="succeeded", point=(60, 60)
    )
    failed = _run_log(tmp_path / "failed.json", status="failed", point=(5, 5))

    added = add_runlog_pair(dataset, source, successful)
    assert added["added_count"] == 1
    assert added["pending_annotation_count"] == 0
    assert added["cases"][0]["group"] == "successful-pairs"
    assert added["cases"][0]["expected"]["transfer_target_bounds"] == [
        40.0,
        40.0,
        80.0,
        80.0,
    ]

    duplicate = add_runlog_pair(dataset, source, successful)
    assert duplicate["added_count"] == 0
    assert duplicate["updated_count"] == 0
    assert len(duplicate["cases"]) == 1

    with_failure = add_runlog_pair(dataset, source, failed)
    assert with_failure["added_count"] == 1
    assert with_failure["pending_annotation_count"] == 1
    failure = with_failure["cases"][1]
    assert failure["group"] == "runtime-failures"
    assert failure["annotation_status"] == "pending"
    assert failure["observed_target_action"]["x"] == 5

    annotated = add_runlog_pair(
        dataset,
        source,
        failed,
        expected_bounds={0: [40, 40, 80, 80]},
    )
    assert annotated["added_count"] == 0
    assert annotated["updated_count"] == 1
    assert len(annotated["cases"]) == 2
    assert annotated["cases"][1]["annotation_status"] == "ready"


def test_comparison_failures_are_promoted_into_runtime_error_group(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    source = _run_log(tmp_path / "source.json", status="succeeded", point=(50, 60))
    target = _run_log(tmp_path / "target.json", status="succeeded", point=(60, 60))
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps(
            {
                "schema_version": (
                    "omniflow.androidworld.offline-runlog-transfer-comparison.v1"
                ),
                "source_run_log": {"path": str(source)},
                "target_run_log": {"path": str(target)},
                "steps": [
                    {
                        "function_step_index": 0,
                        "top_candidate_contains_recorded_point": False,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = add_comparison_failures(dataset, comparison)

    assert result["added_count"] == 1
    assert result["cases"][0]["group"] == "runtime-failures"
    assert result["cases"][0]["annotation_status"] == "ready"
    assert result["cases"][0]["provenance"]["comparison_report"] == str(
        comparison.resolve()
    )


def test_cycle_keeps_pending_and_failed_cases_in_error_view(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    source = _run_log(tmp_path / "source.json", status="succeeded", point=(50, 60))
    failed = _run_log(tmp_path / "failed.json", status="failed", point=(5, 5))
    add_runlog_pair(dataset, source, failed)

    report_path = tmp_path / "report.json"
    errors_path = tmp_path / "errors.json"
    report = run_regression_cycle(
        dataset,
        report_path=report_path,
        errors_path=errors_path,
        page_encoder=_Encoder(),
        transfer=lambda _action, _target, _source: TransferResult(
            Action("click", {"x": 50.0, "y": 50.0}),
            detail={"absolute_contextual_confidence": 0.9},
        ),
    )

    assert report["status"] == "incomplete"
    assert report["pending_annotation_count"] == 1
    errors = json.loads(errors_path.read_text(encoding="utf-8"))
    assert errors["schema_version"] == ("omniflow.offline-embedding-transfer-errors.v1")
    assert errors["case_count"] == 1
    assert errors["cases"][0]["status"] == "pending_annotation"
    assert report_path.is_file()


def test_pair_snapshots_screenshots_beside_dataset(tmp_path: Path) -> None:
    source = _run_log(tmp_path / "source.json", status="succeeded", point=(50, 60))
    target = _run_log(tmp_path / "target.json", status="succeeded", point=(60, 60))
    for index, path in enumerate((source, target)):
        screenshot = tmp_path / f"screen-{index}.png"
        screenshot.write_bytes(b"saved-image")
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["steps"][0]["observation"]["screenshot"] = {
            "path": str(screenshot),
            "width": 100,
            "height": 200,
            "mime_type": "image/png",
        }
        path.write_text(json.dumps(payload), encoding="utf-8")

    dataset = tmp_path / "suite" / "dataset.json"
    result = add_runlog_pair(dataset, source, target)

    case = result["cases"][0]
    assert not Path(case["source"]["screenshot_path"]).is_absolute()
    assert (dataset.parent / case["source"]["screenshot_path"]).read_bytes() == (
        b"saved-image"
    )
    assert (dataset.parent / case["target"]["screenshot_path"]).is_file()


def test_manual_negative_pair_reuses_saved_observations(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    source = _run_log(tmp_path / "source.json", status="succeeded", point=(50, 60))
    first = _run_log(tmp_path / "first.json", status="succeeded", point=(60, 60))
    second = _run_log(tmp_path / "second.json", status="succeeded", point=(60, 60))
    add_runlog_pair(dataset, source, first)
    current = add_runlog_pair(dataset, source, second)
    first_id, second_id = [case["id"] for case in current["cases"]]

    result = add_embedding_negative(dataset, first_id, second_id)

    negative = next(
        case for case in result["cases"] if case["group"] == "embedding-negatives"
    )
    assert negative["source_action"] is None
    assert negative["expected"]["page_match"] is False
    assert negative["annotation_status"] == "ready"
    assert negative["provenance"] == {
        "source_case_id": first_id,
        "target_case_id": second_id,
        "kind": "manual_embedding_negative",
    }


def test_objective_package_mismatch_becomes_null_transfer_regression(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.json"
    source = _run_log(tmp_path / "source.json", status="succeeded", point=(50, 60))
    failed = _run_log(tmp_path / "failed.json", status="failed", point=(5, 5))
    payload = json.loads(failed.read_text(encoding="utf-8"))
    observation = payload["steps"][0]["observation"]
    observation["xml"] = observation["xml"].replace("Click me", "different page")
    observation["auxiliaries"]["package_name"] = "other.app"
    failed.write_text(json.dumps(payload), encoding="utf-8")

    result = add_runlog_pair(dataset, source, failed)
    case = result["cases"][0]

    assert case["annotation_status"] == "ready"
    assert case["expected"]["page_match"] is False
    assert case["expected"]["transfer_outcome"] == "null"
    report = run_regression_dataset(
        dataset,
        page_encoder=_Encoder(),
        transfer=lambda _action, _target, _source: TransferResult(
            None,
            reason="target_page_identity_mismatch",
        ),
    )
    assert report["status"] == "passed"
    assert report["transfer_case_count"] == 1
    assert report["transfer_passed_count"] == 1


def test_pending_case_can_be_human_annotated_as_expected_null(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.json"
    source = _run_log(tmp_path / "source.json", status="succeeded", point=(50, 60))
    failed = _run_log(tmp_path / "failed.json", status="failed", point=(5, 5))
    pending = add_runlog_pair(dataset, source, failed)["cases"][0]

    result = annotate_case(
        dataset,
        pending["id"],
        page_match=False,
        transfer_outcome="null",
    )

    case = result["cases"][0]
    assert case["annotation_status"] == "ready"
    assert case["expected"]["page_match"] is False
    assert case["expected"]["transfer_outcome"] == "null"
    assert case["expected"]["transfer_target_bounds"] is None
