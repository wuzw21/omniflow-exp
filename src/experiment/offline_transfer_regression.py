from __future__ import annotations

import argparse
from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from omniflow.core.model import Action, Observation, TransferResult
from omniflow.core.trajectory import (
    observation_display,
    observation_screenshot,
    observation_xml,
)
from omniflow.runlog import project_androidworld_step_actions
from omniflow.transfer.embedding import PageEncoder

DATASET_SCHEMA_VERSION = "omniflow.offline-embedding-transfer-dataset.v1"
REPORT_SCHEMA_VERSION = "omniflow.offline-embedding-transfer-report.v1"
ERRORS_SCHEMA_VERSION = "omniflow.offline-embedding-transfer-errors.v1"


def add_runlog_pair(
    dataset_path: str | Path,
    source_run_log: str | Path,
    target_run_log: str | Path,
    *,
    step_indices: set[int] | None = None,
    expected_bounds: dict[int, list[float]] | None = None,
    group: str | None = None,
    extra_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize one source/target RunLog pair into the persistent dataset."""

    dataset_path = Path(dataset_path).expanduser().resolve()
    source_path = Path(source_run_log).expanduser().resolve()
    target_path = Path(target_run_log).expanduser().resolve()
    source = _read_object(source_path)
    target = _read_object(target_path)
    source_sha = _sha256(source_path)
    target_sha = _sha256(target_path)
    target_succeeded = (
        target.get("status") == "succeeded" and target.get("success") is True
    )
    selected_group = group or (
        "successful-pairs" if target_succeeded else "runtime-failures"
    )
    source_steps = {_step_key(step): step for step in source.get("steps", [])}
    target_steps = {_step_key(step): step for step in target.get("steps", [])}
    annotations = expected_bounds or {}
    cases: list[dict[str, Any]] = []
    for step_index in sorted(set(source_steps) & set(target_steps)):
        if step_indices is not None and step_index not in step_indices:
            continue
        source_step = source_steps[step_index]
        target_step = target_steps[step_index]
        source_action = _projected_point_action(source_step)
        if source_action is None:
            continue
        case_hash = hashlib.sha256(
            f"{source_sha}:{target_sha}:{step_index}".encode()
        ).hexdigest()[:16]
        case_id = (
            f"{_safe_id(str(target.get('task_name') or 'task'))}-"
            f"{step_index}-{case_hash}"
        )
        saved_source = _saved_observation(
            source_step.get("observation"),
            dataset_root=dataset_path.parent,
            source_root=source_path.parent,
            asset_name=f"{case_id}-source",
        )
        saved_target = _saved_observation(
            target_step.get("observation"),
            dataset_root=dataset_path.parent,
            source_root=target_path.parent,
            asset_name=f"{case_id}-target",
        )
        target_point = _recorded_point(target_step)
        annotated_bounds = annotations.get(step_index)
        inferred_bounds = (
            _smallest_containing_bounds(
                observation_xml(target_step.get("observation") or {}),
                target_point,
            )
            if target_succeeded and target_point is not None
            else None
        )
        target_bounds = _bounds_list(annotated_bounds) or inferred_bounds
        source_package = str(saved_source.get("package_name") or "")
        target_package = str(saved_target.get("package_name") or "")
        objective_page_mismatch = bool(
            source_package and target_package and source_package != target_package
        )
        if target_bounds is not None:
            page_expectation: bool | None = True
            transfer_outcome: str | None = "mapped"
        elif objective_page_mismatch:
            page_expectation = False
            transfer_outcome = "null"
        else:
            page_expectation = None
            transfer_outcome = None
        annotation_status = "ready" if transfer_outcome is not None else "pending"
        provenance = {
            "source_run_log": str(source_path),
            "source_run_log_sha256": source_sha,
            "source_run_status": source.get("status"),
            "target_run_log": str(target_path),
            "target_run_log_sha256": target_sha,
            "target_run_status": target.get("status"),
            "target_run_success": target.get("success"),
            "function_step_index": step_index,
        }
        if extra_provenance:
            provenance.update(extra_provenance)
        cases.append(
            {
                "id": case_id,
                "group": selected_group,
                "annotation_status": annotation_status,
                "task_name": str(
                    target.get("task_name") or source.get("task_name") or ""
                ),
                "source": saved_source,
                "target": saved_target,
                "source_action": source_action,
                "observed_target_action": target_step.get("action"),
                "expected": {
                    "page_match": page_expectation,
                    "minimum_page_similarity": 0.8,
                    "transfer_outcome": transfer_outcome,
                    "transfer_target_bounds": target_bounds,
                    "minimum_transfer_confidence": 0.8,
                },
                "provenance": provenance,
            }
        )
    if not cases:
        raise ValueError("offline_transfer_pair_has_no_coordinate_steps")

    dataset = _load_or_create_dataset(dataset_path)
    existing = {
        str(case.get("id")): case for case in dataset["cases"] if isinstance(case, dict)
    }
    added_count = 0
    updated_count = 0
    for case in cases:
        previous = existing.get(case["id"])
        if previous is None:
            dataset["cases"].append(case)
            existing[case["id"]] = case
            added_count += 1
        elif previous != case:
            index = dataset["cases"].index(previous)
            dataset["cases"][index] = case
            existing[case["id"]] = case
            updated_count += 1
    dataset["cases"] = sorted(dataset["cases"], key=lambda case: case["id"])
    _write_json(dataset_path, dataset)
    return {
        **dataset,
        "added_count": added_count,
        "updated_count": updated_count,
        "pending_annotation_count": sum(
            case.get("annotation_status") == "pending" for case in dataset["cases"]
        ),
    }


def add_comparison_failures(
    dataset_path: str | Path,
    comparison_path: str | Path,
    *,
    expected_bounds: dict[int, list[float]] | None = None,
) -> dict[str, Any]:
    """Promote failed rows from the canonical offline comparison into regressions."""

    comparison_path = Path(comparison_path).expanduser().resolve()
    comparison = _read_object(comparison_path)
    if comparison.get("schema_version") != (
        "omniflow.androidworld.offline-runlog-transfer-comparison.v1"
    ):
        raise ValueError("offline_transfer_comparison_schema_invalid")
    failed_steps = {
        int(row["function_step_index"])
        for row in comparison.get("steps", [])
        if isinstance(row, dict)
        and row.get("function_step_index") is not None
        and row.get("top_candidate_contains_recorded_point") is not True
    }
    if not failed_steps:
        raise ValueError("offline_transfer_comparison_has_no_failures")
    source_path = _evidence_path(comparison.get("source_run_log"), "source")
    target_path = _evidence_path(comparison.get("target_run_log"), "target")
    return add_runlog_pair(
        dataset_path,
        source_path,
        target_path,
        step_indices=failed_steps,
        expected_bounds=expected_bounds,
        group="runtime-failures",
        extra_provenance={"comparison_report": str(comparison_path)},
    )


def add_embedding_negative(
    dataset_path: str | Path,
    source_case_id: str,
    target_case_id: str,
) -> dict[str, Any]:
    """Add one human-selected different-page pair to the same dataset."""

    path = Path(dataset_path).expanduser().resolve()
    dataset = _load_or_create_dataset(path)
    indexed = {
        str(case.get("id")): case for case in dataset["cases"] if isinstance(case, dict)
    }
    try:
        source_case = indexed[source_case_id]
        target_case = indexed[target_case_id]
    except KeyError as error:
        raise ValueError(f"offline_transfer_case_not_found:{error.args[0]}") from error
    digest = hashlib.sha256(
        f"negative:{source_case_id}:{target_case_id}".encode()
    ).hexdigest()[:16]
    negative = {
        "id": f"embedding-negative-{digest}",
        "group": "embedding-negatives",
        "annotation_status": "ready",
        "task_name": "",
        "source": dict(source_case["source"]),
        "target": dict(target_case["target"]),
        "source_action": None,
        "observed_target_action": None,
        "expected": {
            "page_match": False,
            "minimum_page_similarity": 0.8,
        },
        "provenance": {
            "source_case_id": source_case_id,
            "target_case_id": target_case_id,
            "kind": "manual_embedding_negative",
        },
    }
    previous = indexed.get(negative["id"])
    added_count = int(previous is None)
    updated_count = int(previous is not None and previous != negative)
    if previous is None:
        dataset["cases"].append(negative)
    elif previous != negative:
        dataset["cases"][dataset["cases"].index(previous)] = negative
    dataset["cases"] = sorted(dataset["cases"], key=lambda case: case["id"])
    _write_json(path, dataset)
    return {
        **dataset,
        "added_count": added_count,
        "updated_count": updated_count,
        "pending_annotation_count": sum(
            case.get("annotation_status") == "pending" for case in dataset["cases"]
        ),
    }


def annotate_case(
    dataset_path: str | Path,
    case_id: str,
    *,
    page_match: bool,
    transfer_outcome: str,
    target_bounds: list[float] | None = None,
) -> dict[str, Any]:
    """Attach human ground truth to one pending pair without replacing evidence."""

    if transfer_outcome not in {"mapped", "null"}:
        raise ValueError("offline_transfer_annotation_outcome_invalid")
    normalized_bounds = _bounds_list(target_bounds)
    if transfer_outcome == "mapped" and normalized_bounds is None:
        raise ValueError("offline_transfer_annotation_bounds_required")
    if transfer_outcome == "null" and target_bounds is not None:
        raise ValueError("offline_transfer_null_annotation_forbids_bounds")
    path = Path(dataset_path).expanduser().resolve()
    dataset = _load_or_create_dataset(path)
    selected = next(
        (
            case
            for case in dataset["cases"]
            if isinstance(case, dict) and case.get("id") == case_id
        ),
        None,
    )
    if selected is None:
        raise ValueError(f"offline_transfer_case_not_found:{case_id}")
    expected = dict(selected.get("expected") or {})
    expected.update(
        {
            "page_match": page_match,
            "minimum_page_similarity": float(
                expected.get("minimum_page_similarity", 0.8)
            ),
            "transfer_outcome": transfer_outcome,
            "transfer_target_bounds": normalized_bounds,
            "minimum_transfer_confidence": float(
                expected.get("minimum_transfer_confidence", 0.8)
            ),
        }
    )
    selected["expected"] = expected
    selected["annotation_status"] = "ready"
    _write_json(path, dataset)
    return {
        **dataset,
        "added_count": 0,
        "updated_count": 1,
        "pending_annotation_count": sum(
            case.get("annotation_status") == "pending" for case in dataset["cases"]
        ),
    }


def run_regression_dataset(
    dataset_path: str | Path,
    *,
    page_encoder: Any | None = None,
    transfer: Callable[[Action, Observation, Observation], TransferResult]
    | None = None,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved page pairs with production embedding and transfer seams."""

    path = Path(dataset_path).expanduser().resolve()
    dataset = _read_object(path)
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("offline_transfer_dataset_schema_invalid")
    cases = dataset.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("offline_transfer_dataset_cases_required")

    encoder = page_encoder or PageEncoder()
    if transfer is None:
        from omniflow.runtime.execution import default_transfer

        transfer = default_transfer

    rows = [
        _run_case(
            value,
            dataset_root=path.parent,
            page_encoder=encoder,
            transfer=transfer,
        )
        for value in cases
    ]
    pending_count = sum(row["status"] == "pending_annotation" for row in rows)
    failed_count = sum(row["status"] == "failed" for row in rows)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "dataset": {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "name": str(dataset.get("name") or path.stem),
        },
        "status": (
            "incomplete" if pending_count else ("failed" if failed_count else "passed")
        ),
        "case_count": len(rows),
        "passed_count": sum(row["status"] == "passed" for row in rows),
        "failed_count": failed_count,
        "pending_annotation_count": pending_count,
        "embedding_case_count": sum(row["embedding_tested"] for row in rows),
        "embedding_passed_count": sum(
            row["embedding_tested"] and row["embedding_passed"] for row in rows
        ),
        "transfer_probe_count": sum(row["transfer_probed"] for row in rows),
        "transfer_case_count": sum(row["transfer_tested"] for row in rows),
        "transfer_passed_count": sum(
            row["transfer_tested"] and row["transfer_passed"] for row in rows
        ),
        "model_calls": 0,
        "device_calls": 0,
        "cases": rows,
    }
    if output_path is not None:
        destination = Path(output_path).expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return report


def run_regression_cycle(
    dataset_path: str | Path,
    *,
    report_path: str | Path,
    errors_path: str | Path,
    page_encoder: Any | None = None,
    transfer: Callable[[Action, Observation, Observation], TransferResult]
    | None = None,
) -> dict[str, Any]:
    """Run every saved pair and refresh the generated error-set view."""

    report = run_regression_dataset(
        dataset_path,
        page_encoder=page_encoder,
        transfer=transfer,
        output_path=report_path,
    )
    errors = {
        "schema_version": ERRORS_SCHEMA_VERSION,
        "dataset": report["dataset"],
        "report_path": str(Path(report_path).expanduser().resolve()),
        "case_count": sum(case["status"] != "passed" for case in report["cases"]),
        "cases": [case for case in report["cases"] if case["status"] != "passed"],
    }
    _write_json(Path(errors_path).expanduser().resolve(), errors)
    return report


def _run_case(
    value: Any,
    *,
    dataset_root: Path,
    page_encoder: Any,
    transfer: Callable[[Action, Observation, Observation], TransferResult],
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("offline_transfer_case_must_be_object")
    case_id = str(value.get("id") or "").strip()
    if not case_id:
        raise ValueError("offline_transfer_case_id_required")
    source = _observation(value.get("source"), dataset_root=dataset_root)
    target = _observation(value.get("target"), dataset_root=dataset_root)
    expected = value.get("expected")
    if not isinstance(expected, dict) or expected.get("page_match") not in {
        True,
        False,
        None,
    }:
        raise ValueError(f"offline_transfer_case_expectation_invalid:{case_id}")

    source_page = page_encoder.embed(source)
    target_page = page_encoder.embed(target)
    similarity = _cosine(source_page.vector, target_page.vector)
    page_threshold = float(expected.get("minimum_page_similarity", 0.8))
    page_match = similarity >= page_threshold
    embedding_tested = isinstance(expected["page_match"], bool)
    embedding_passed = (
        page_match is expected["page_match"] if embedding_tested else False
    )

    action_value = value.get("source_action")
    transfer_probed = action_value is not None
    transfer_outcome = expected.get("transfer_outcome")
    if transfer_outcome is None and expected.get("transfer_target_bounds") is not None:
        transfer_outcome = "mapped"
    if transfer_outcome not in {None, "mapped", "null"}:
        raise ValueError(f"offline_transfer_outcome_invalid:{case_id}")
    transfer_tested = transfer_probed and transfer_outcome is not None
    transfer_passed = not transfer_probed
    confidence: float | None = None
    mapped_point: list[float] | None = None
    transfer_reason: str | None = None
    transfer_detail: dict[str, Any] | None = None
    if transfer_probed:
        action = Action.from_value(action_value)
        result = transfer(action, target, source)
        if not isinstance(result, TransferResult):
            raise TypeError(f"offline_transfer_result_invalid:{case_id}")
        confidence = _confidence(result.detail)
        transfer_reason = result.reason
        transfer_detail = dict(result.detail)
        mapped_point = _raw_mapped_point(result.action, target)
        if transfer_outcome == "null":
            transfer_passed = result.action is None
        elif transfer_outcome == "mapped":
            bounds = _bounds(expected.get("transfer_target_bounds"))
            minimum_confidence = float(expected.get("minimum_transfer_confidence", 0.8))
            transfer_passed = (
                result.action is not None
                and confidence is not None
                and confidence >= minimum_confidence
                and _contains(bounds, mapped_point)
            )
        else:
            transfer_passed = False

    annotation_status = str(value.get("annotation_status") or "ready")
    passed = embedding_tested and embedding_passed and transfer_passed
    return {
        "id": case_id,
        "group": str(value.get("group") or "default"),
        "status": (
            "pending_annotation"
            if annotation_status == "pending"
            else ("passed" if passed else "failed")
        ),
        "expected_page_match": expected["page_match"],
        "page_match": page_match,
        "page_similarity": similarity,
        "minimum_page_similarity": page_threshold,
        "embedding_tested": embedding_tested,
        "embedding_passed": embedding_passed,
        "transfer_probed": transfer_probed,
        "transfer_tested": transfer_tested,
        "expected_transfer_outcome": transfer_outcome,
        "transfer_passed": transfer_passed,
        "transfer_confidence": confidence,
        "mapped_target_point": mapped_point,
        "expected_target_bounds": expected.get("transfer_target_bounds"),
        "transfer_reason": transfer_reason,
        "transfer_detail": transfer_detail,
        "provenance": value.get("provenance"),
    }


def _observation(value: Any, *, dataset_root: Path) -> Observation:
    if not isinstance(value, dict):
        raise TypeError("offline_transfer_observation_must_be_object")
    xml = str(value.get("xml") or "")
    display = value.get("display")
    if not xml.strip() or not isinstance(display, dict):
        raise ValueError("offline_transfer_observation_xml_display_required")
    extra: dict[str, Any] = {"display": dict(display)}
    screenshot = str(value.get("screenshot_path") or "").strip()
    if screenshot:
        screenshot_path = Path(screenshot).expanduser()
        if not screenshot_path.is_absolute():
            screenshot_path = dataset_root / screenshot_path
        extra["screenshot_path"] = str(screenshot_path.resolve())
    return Observation(
        xml=xml,
        package_name=str(value.get("package_name") or ""),
        activity_name=str(value.get("activity_name") or ""),
        extra=extra,
    )


def _cosine(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(left_array, right_array) / denominator)


def _confidence(detail: dict[str, Any]) -> float | None:
    for key in ("absolute_contextual_confidence", "pair_confidence", "score"):
        try:
            value = float(detail[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return min(1.0, max(0.0, value))
    return None


def _raw_mapped_point(
    action: Action | None,
    observation: Observation,
) -> list[float] | None:
    if action is None:
        return None
    display = observation.extra.get("display")
    if not isinstance(display, dict):
        return None
    try:
        return [
            float(action.args["x"]) / 1000.0 * float(display["width"]),
            float(action.args["y"]) / 1000.0 * float(display["height"]),
        ]
    except (KeyError, TypeError, ValueError):
        return None


def _bounds(value: Any) -> tuple[float, float, float, float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        left, top, right, bottom = (float(item) for item in value)
    except (TypeError, ValueError):
        return None
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def _contains(
    bounds: tuple[float, float, float, float] | None,
    point: list[float] | None,
) -> bool:
    if bounds is None or point is None:
        return False
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"json_root_must_be_object:{path}")
    return value


def _load_or_create_dataset(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "name": path.stem,
            "cases": [],
        }
    dataset = _read_object(path)
    if dataset.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise ValueError("offline_transfer_dataset_schema_invalid")
    if not isinstance(dataset.get("cases"), list):
        raise TypeError("offline_transfer_dataset_cases_invalid")
    return dataset


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(rendered, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _step_key(value: Any) -> int:
    if not isinstance(value, dict):
        raise TypeError("offline_transfer_runlog_step_invalid")
    metadata = value.get("metadata")
    if isinstance(metadata, dict) and isinstance(
        metadata.get("function_step_index"), int
    ):
        return int(metadata["function_step_index"])
    return int(value.get("step_index") or 0)


def _projected_point_action(step: dict[str, Any]) -> dict[str, Any] | None:
    try:
        actions = project_androidworld_step_actions(step)
    except ValueError:
        return None
    for action in reversed(actions):
        if action.get("tool") in {"click", "input_text", "long_press"}:
            args = action.get("args")
            if isinstance(args, dict) and (
                all(args.get(key) is not None for key in ("x", "y"))
                or (
                    action.get("tool") == "input_text"
                    and str(args.get("target_description") or "").strip()
                )
            ):
                return action
    return None


def _recorded_point(step: dict[str, Any]) -> tuple[float, float] | None:
    action = step.get("action")
    action = action if isinstance(action, dict) else {}
    observation = step.get("observation")
    observation = observation if isinstance(observation, dict) else {}
    try:
        projected = project_androidworld_step_actions(step)
    except ValueError:
        return None
    for value in reversed(projected):
        args = value.get("args")
        if value.get("tool") not in {
            "click",
            "input_text",
            "long_press",
        } or not isinstance(args, dict):
            continue
        display = observation_display(observation)
        if display is None:
            return None
        try:
            return (
                float(args["x"]) / 1000.0 * display[0],
                float(args["y"]) / 1000.0 * display[1],
            )
        except (KeyError, TypeError, ValueError):
            return None
    try:
        return float(action["x"]), float(action["y"])
    except (KeyError, TypeError, ValueError):
        return None


def _saved_observation(
    value: Any,
    *,
    dataset_root: Path,
    source_root: Path,
    asset_name: str,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("offline_transfer_runlog_observation_required")
    xml = observation_xml(value)
    display = observation_display(value)
    if not xml or display is None:
        raise ValueError("offline_transfer_runlog_observation_incomplete")
    auxiliaries = value.get("auxiliaries")
    auxiliaries = auxiliaries if isinstance(auxiliaries, dict) else {}
    saved: dict[str, Any] = {
        "xml": xml,
        "package_name": str(auxiliaries.get("package_name") or _xml_package(xml)),
        "activity_name": str(auxiliaries.get("activity_name") or ""),
        "display": {"width": display[0], "height": display[1]},
    }
    screenshot = observation_screenshot(value)
    if isinstance(screenshot, dict):
        screenshot_path = str(screenshot.get("path") or "").strip()
        if screenshot_path:
            original = Path(screenshot_path).expanduser()
            if not original.is_absolute():
                original = source_root / original
            if not original.is_file():
                archived_sibling = source_root / "screenshots" / original.name
                if archived_sibling.is_file():
                    original = archived_sibling
            if original.is_file():
                digest = _sha256(original)
                suffix = original.suffix.lower() or ".png"
                relative = Path("assets") / f"{asset_name}-{digest[:12]}{suffix}"
                destination = dataset_root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.is_file():
                    shutil.copy2(original, destination)
                saved["screenshot_path"] = relative.as_posix()
    return saved


def _xml_package(xml: str) -> str:
    packages = [
        item
        for item in re.findall(r'\bpackage="([^"]+)"', xml)
        if item != "com.android.systemui"
    ]
    if not packages:
        return ""
    counts = {package: packages.count(package) for package in set(packages)}
    return min(counts, key=lambda package: (-counts[package], package))


def _smallest_containing_bounds(
    xml: str,
    point: tuple[float, float] | None,
) -> list[float] | None:
    if not xml or point is None:
        return None
    try:
        root = ET.fromstring(xml)
    except ET.ParseError:
        return None
    candidates: list[list[float]] = []
    for element in root.iter():
        bounds = _xml_bounds(element.attrib.get("bounds"))
        if bounds is None or not _contains(tuple(bounds), list(point)):
            continue
        candidates.append(bounds)
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda item: ((item[2] - item[0]) * (item[3] - item[1]), item),
    )


def _xml_bounds(value: Any) -> list[float] | None:
    numbers = re.findall(r"-?\d+(?:\.\d+)?", str(value or ""))
    return (
        _bounds_list([float(item) for item in numbers]) if len(numbers) == 4 else None
    )


def _bounds_list(value: Any) -> list[float] | None:
    bounds = _bounds(value)
    return list(bounds) if bounds is not None else None


def _safe_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")
    return normalized or "task"


def _evidence_path(value: Any, label: str) -> Path:
    if not isinstance(value, dict):
        raise TypeError(f"offline_transfer_comparison_{label}_missing")
    raw = str(value.get("path") or "").strip()
    if not raw:
        raise ValueError(f"offline_transfer_comparison_{label}_path_missing")
    path = Path(raw).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Maintain and run the device-free Page Embedding + OmniTransfer "
            "regression loop."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    add_pair = commands.add_parser(
        "add-pair", help="Add all coordinate steps from one RunLog pair."
    )
    add_pair.add_argument("--dataset", type=Path, required=True)
    add_pair.add_argument("--source-run-log", type=Path, required=True)
    add_pair.add_argument("--target-run-log", type=Path, required=True)
    add_pair.add_argument("--expected-bounds", type=Path)
    add_pair.add_argument("--group")

    add_errors = commands.add_parser(
        "add-errors",
        help="Promote failed rows from an offline comparison report.",
    )
    add_errors.add_argument("--dataset", type=Path, required=True)
    add_errors.add_argument("--comparison", type=Path, required=True)
    add_errors.add_argument("--expected-bounds", type=Path)

    add_negative = commands.add_parser(
        "add-negative",
        help="Pair two saved cases as a human-confirmed Embedding negative.",
    )
    add_negative.add_argument("--dataset", type=Path, required=True)
    add_negative.add_argument("--source-case", required=True)
    add_negative.add_argument("--target-case", required=True)

    annotate = commands.add_parser(
        "annotate", help="Attach human ground truth to one pending pair."
    )
    annotate.add_argument("--dataset", type=Path, required=True)
    annotate.add_argument("--case", required=True)
    annotate.add_argument("--page-match", choices=("true", "false"), required=True)
    annotate.add_argument(
        "--transfer-outcome", choices=("mapped", "null"), required=True
    )
    annotate.add_argument("--target-bounds", type=float, nargs=4)

    run = commands.add_parser(
        "run", help="Run the full saved dataset and refresh the error view."
    )
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--report", type=Path, required=True)
    run.add_argument("--errors", type=Path, required=True)
    return parser.parse_args()


def _load_expected_bounds(path: Path | None) -> dict[int, list[float]] | None:
    if path is None:
        return None
    value = _read_object(path.expanduser().resolve())
    bounds: dict[int, list[float]] = {}
    for key, item in value.items():
        normalized = _bounds_list(item)
        if normalized is None:
            raise ValueError(f"offline_transfer_expected_bounds_invalid:{key}")
        bounds[int(key)] = normalized
    return bounds


def main() -> int:
    args = _parse_args()
    if args.command == "add-pair":
        result = add_runlog_pair(
            args.dataset,
            args.source_run_log,
            args.target_run_log,
            expected_bounds=_load_expected_bounds(args.expected_bounds),
            group=args.group,
        )
    elif args.command == "add-errors":
        result = add_comparison_failures(
            args.dataset,
            args.comparison,
            expected_bounds=_load_expected_bounds(args.expected_bounds),
        )
    elif args.command == "add-negative":
        result = add_embedding_negative(
            args.dataset,
            args.source_case,
            args.target_case,
        )
    elif args.command == "annotate":
        result = annotate_case(
            args.dataset,
            args.case,
            page_match=args.page_match == "true",
            transfer_outcome=args.transfer_outcome,
            target_bounds=args.target_bounds,
        )
    else:
        result = run_regression_cycle(
            args.dataset,
            report_path=args.report,
            errors_path=args.errors,
        )
    if args.command in {"add-pair", "add-errors", "add-negative", "annotate"}:
        rendered = {
            "schema_version": result["schema_version"],
            "dataset": str(args.dataset.expanduser().resolve()),
            "case_count": len(result["cases"]),
            "added_count": result["added_count"],
            "updated_count": result["updated_count"],
            "pending_annotation_count": result["pending_annotation_count"],
        }
    else:
        rendered = {
            key: result[key]
            for key in (
                "schema_version",
                "status",
                "case_count",
                "passed_count",
                "failed_count",
                "pending_annotation_count",
                "embedding_case_count",
                "embedding_passed_count",
                "transfer_probe_count",
                "transfer_case_count",
                "transfer_passed_count",
                "model_calls",
                "device_calls",
            )
        }
    print(json.dumps(rendered, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result.get("status") in {None, "passed"} else 1


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "ERRORS_SCHEMA_VERSION",
    "REPORT_SCHEMA_VERSION",
    "add_comparison_failures",
    "add_embedding_negative",
    "add_runlog_pair",
    "annotate_case",
    "run_regression_cycle",
    "run_regression_dataset",
]


if __name__ == "__main__":
    raise SystemExit(main())
