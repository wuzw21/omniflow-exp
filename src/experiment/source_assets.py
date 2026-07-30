"""Deterministic source-only grounding for baseline asset preparation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
import xml.etree.ElementTree as ET

from omniflow.core.trajectory import state_id as observation_state_id
from omniflow.transfer.runtime import load_transfer_state_catalog
from src.integrations.runlog import (
    convert_legacy_run_log,
    import_run_log,
    import_run_log_evidence,
    project_androidworld_step_actions,
)


def select_source_asset_revision(
    base_root: str | Path,
    *,
    manifest_name: str,
    initial_revision: int = 3,
    expected_source_sha256: str = "",
    environment_repair_reason: str = "",
) -> Path:
    """Reuse the first frozen source asset or allocate a fresh revision path.

    Failed and incomplete revision directories are immutable evidence. When no
    frozen manifest exists, the returned path advances beyond every existing
    revision instead of overwriting one, unless an attempt explicitly forbids
    retry.
    """

    if initial_revision < 1:
        raise ValueError("initial_revision must be positive")
    manifest = str(manifest_name).strip()
    if not manifest or Path(manifest).name != manifest:
        raise ValueError("manifest_name must be one file name")
    source_sha256 = str(expected_source_sha256 or "").strip().lower()
    repair_reason = str(environment_repair_reason or "").strip()
    if source_sha256 and not re.fullmatch(r"[0-9a-f]{64}", source_sha256):
        raise ValueError("expected_source_sha256 must be one SHA-256 digest")
    base = Path(base_root).expanduser().resolve()
    if source_sha256:
        matches: list[Path] = []
        if base.is_dir():
            for candidate in base.iterdir():
                manifest_path = candidate / manifest
                if not candidate.is_dir() or not manifest_path.is_file():
                    continue
                try:
                    payload = json.loads(
                        manifest_path.read_text(encoding="utf-8")
                    )
                except (OSError, json.JSONDecodeError):
                    continue
                if _manifest_source_sha256(payload) == source_sha256:
                    matches.append(candidate.resolve())
        if len(matches) > 1:
            raise ValueError(
                "source_asset_revision_ambiguous:"
                + ",".join(str(path) for path in sorted(matches))
            )
        if matches:
            return matches[0]
        prefix = f"source_{source_sha256[:12]}"
        revisions: list[tuple[int, Path]] = []
        if base.is_dir():
            for candidate in base.iterdir():
                if not candidate.is_dir():
                    continue
                if candidate.name == prefix:
                    revisions.append((1, candidate.resolve()))
                    continue
                match = re.fullmatch(
                    rf"{re.escape(prefix)}_r([2-9]|[1-9][0-9]+)",
                    candidate.name,
                )
                if match:
                    revisions.append(
                        (int(match.group(1)), candidate.resolve())
                    )
        if not revisions:
            return base / prefix
        _reject_forbidden_source_retry(
            revisions,
            environment_repair_reason=repair_reason,
        )
        next_revision = max(revision for revision, _ in revisions) + 1
        return base / f"{prefix}_r{next_revision}"
    revisions: list[tuple[int, Path]] = []
    if base.is_dir():
        for candidate in base.iterdir():
            match = re.fullmatch(r"native_source_r([1-9][0-9]*)", candidate.name)
            if candidate.is_dir() and match:
                revision = int(match.group(1))
                if revision >= initial_revision:
                    revisions.append((revision, candidate.resolve()))
    frozen = sorted(
        (revision, candidate)
        for revision, candidate in revisions
        if (candidate / manifest).is_file()
    )
    if frozen:
        return frozen[0][1]
    _reject_forbidden_source_retry(
        revisions,
        environment_repair_reason=repair_reason,
    )
    next_revision = max(
        [initial_revision - 1, *(revision for revision, _ in revisions)]
    ) + 1
    return base / f"native_source_r{next_revision}"


def _reject_forbidden_source_retry(
    revisions: list[tuple[int, Path]],
    *,
    environment_repair_reason: str = "",
) -> None:
    for _, candidate in sorted(revisions):
        marker = candidate / "prep_failure.json"
        if not marker.is_file():
            continue
        try:
            failure = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(failure, dict) or failure.get("retry_allowed") is not False:
            continue
        if str(environment_repair_reason or "").strip():
            continue
        error = str(failure.get("error") or "terminal_source_failure").strip()
        raise ValueError(f"source_asset_retry_forbidden:{candidate}:{error}")


def _manifest_source_sha256(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    source_record = value.get("source_run_log")
    nested = (
        str(source_record.get("sha256") or "").strip()
        if isinstance(source_record, dict)
        else ""
    )
    return (
        nested
        or str(value.get("source_run_log_sha256") or "").strip()
    ).lower()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_frozen_file(
    value: str | Path,
    *,
    expected_sha256: str,
    label: str,
) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label}_missing:{path}")
    actual = _sha256(path)
    if not expected_sha256 or actual != str(expected_sha256):
        raise ValueError(
            f"{label}_hash_mismatch:"
            f"expected={expected_sha256 or 'missing'}:actual={actual}"
        )
    return path


def _index_reference(index_path: str | Path, value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"source_index_{label}_required")
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve()
    index = Path(index_path).expanduser().resolve()
    candidates = [index.parent / path]
    candidates.extend(parent / path for parent in index.parents)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _identity(value: dict[str, Any]) -> dict[str, str]:
    aliases = {
        "text": ("text", "label"),
        "content_desc": (
            "content_desc",
            "content-desc",
            "description",
        ),
        "resource_id": ("resource_id", "resource-id"),
    }
    result: dict[str, str] = {}
    for output_key, input_keys in aliases.items():
        for input_key in input_keys:
            text = str(value.get(input_key) or "").strip()
            if text:
                result[output_key] = text
                break
    return result


def _node_identity(node: ET.Element) -> dict[str, str]:
    return _identity(
        {
            "text": node.attrib.get("text"),
            "content_desc": node.attrib.get("content-desc"),
            "resource_id": node.attrib.get("resource-id"),
        }
    )


def _bounds(node: ET.Element) -> tuple[float, float, float, float] | None:
    value = str(node.attrib.get("bounds") or "")
    try:
        left_top, right_bottom = value.split("][")
        left, top = left_top.lstrip("[").split(",")
        right, bottom = right_bottom.rstrip("]").split(",")
        return float(left), float(top), float(right), float(bottom)
    except (TypeError, ValueError):
        return None


def _display_from_xml(xml_text: str) -> dict[str, int]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    candidates = []
    for node in root.iter():
        bounds = _bounds(node)
        if bounds is None:
            continue
        left, top, right, bottom = bounds
        if left <= 0 and top <= 0 and right > 0 and bottom > 0:
            candidates.append((right * bottom, right, bottom))
    if not candidates:
        return {}
    _, width, height = max(candidates)
    if not width.is_integer() or not height.is_integer():
        return {}
    return {"width": int(width), "height": int(height)}


def _source_display(value: Any) -> dict[str, int]:
    if not isinstance(value, dict):
        return {}
    width = value.get("width")
    height = value.get("height")
    if (
        not isinstance(width, int)
        or isinstance(width, bool)
        or width <= 0
        or not isinstance(height, int)
        or isinstance(height, bool)
        or height <= 0
    ):
        return {}
    return {"width": width, "height": height}


def _identity_at_action_point(
    xml_text: str,
    *,
    action_args: dict[str, Any],
    display: dict[str, Any],
) -> dict[str, str]:
    try:
        x = float(action_args["x"])
        y = float(action_args["y"])
        width = float(display["width"])
        height = float(display["height"])
        root = ET.fromstring(xml_text)
    except (KeyError, TypeError, ValueError, ET.ParseError):
        return {}
    if 0 <= x <= 1000 and 0 <= y <= 1000:
        x = x * width / 1000.0
        y = y * height / 1000.0
    candidates: list[tuple[float, int, dict[str, str]]] = []
    for depth, node in enumerate(root.iter()):
        bounds = _bounds(node)
        identity = _node_identity(node)
        if bounds is None or not identity:
            continue
        left, top, right, bottom = bounds
        if left <= x <= right and top <= y <= bottom:
            area = max(1.0, (right - left) * (bottom - top))
            candidates.append((area, -depth, identity))
    return min(candidates, default=(0.0, 0, {}))[2]


def _unique_editable_identity(xml_text: str) -> dict[str, str]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    nodes = [
        node
        for node in root.iter()
        if str(node.attrib.get("editable") or "").lower() == "true"
        or str(node.attrib.get("class") or "") == "android.widget.EditText"
    ]
    focused_nodes = [
        node
        for node in nodes
        if str(node.attrib.get("focused") or "").lower() == "true"
    ]
    if focused_nodes:
        nodes = focused_nodes if len(focused_nodes) == 1 else []
    else:
        page_nodes = [
            node
            for node in nodes
            if not re.search(
                r":id/(?:location_bar|omnibox|url_bar)$",
                str(node.attrib.get("resource-id") or "").lower(),
            )
        ]
        if page_nodes:
            nodes = page_nodes
    identities = [_node_identity(node) for node in nodes]
    identities = [identity for identity in identities if identity]
    return identities[0] if len(identities) == 1 else {}


def _source_target_audit(provenance: dict[str, Any]) -> dict[str, Any]:
    source_target_audit = provenance.get("source_target_audit")
    if not isinstance(source_target_audit, dict) or source_target_audit.get(
        "source_target_audit_complete"
    ) is not True:
        raise ValueError("source_target_audit_incomplete")
    return source_target_audit


def _normalized_semantic_text(value: Any) -> str:
    return " ".join(str(value or "").casefold().split())


def _unique_xml_identity_for_claim(
    xml_text: str,
    claim: dict[str, Any],
) -> dict[str, str]:
    expected = _identity(claim)
    if not expected:
        return {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    matches: list[dict[str, str]] = []
    for node in root.iter():
        identity = _node_identity(node)
        if identity and all(
            _normalized_semantic_text(identity.get(key))
            == _normalized_semantic_text(value)
            for key, value in expected.items()
        ):
            matches.append(identity)
    return matches[0] if len(matches) == 1 else {}


def _unique_xml_identity_for_description(
    xml_text: str,
    description: Any,
) -> dict[str, str]:
    expected = _normalized_semantic_text(description)
    if not expected:
        return {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return {}
    matches: list[dict[str, str]] = []
    for node in root.iter():
        identity = _node_identity(node)
        if identity and expected in {
            _normalized_semantic_text(value)
            for value in identity.values()
            if str(value).strip()
        }:
            matches.append(identity)
    return matches[0] if len(matches) == 1 else {}


def _verified_target_from_evidence(
    xml_text: str,
    evidence: Any,
) -> dict[str, str]:
    if not isinstance(evidence, dict) or not xml_text:
        return {}
    for key in ("element", "target"):
        claim = evidence.get(key)
        if isinstance(claim, dict):
            identity = _unique_xml_identity_for_claim(xml_text, claim)
            if identity:
                return identity
    return _unique_xml_identity_for_description(
        xml_text,
        evidence.get("target_description"),
    )


def _target_audit_from_embedded_evidence(
    canonical: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_targets: list[dict[str, Any]] = []
    evidence_count = 0
    for step_index, step in enumerate(canonical["steps"]):
        metadata = step.get("metadata")
        evidence = (
            metadata.get("source_target_evidence")
            if isinstance(metadata, dict)
            else None
        )
        if not isinstance(evidence, dict):
            continue
        evidence_count += 1
        identity = _verified_target_from_evidence(
            str(step["observation"].get("forest") or ""),
            evidence,
        )
        if identity:
            source_targets.append(
                {"step_index": step_index, "target": identity}
            )
    return {
        "source_targets": source_targets,
    }, {
        "source_target_evidence_source": (
            "embedded_source_run_log" if evidence_count else "none"
        ),
        "source_target_evidence_count": evidence_count,
        "verified_source_target_count": len(source_targets),
    }


def _target_audit_from_legacy_provenance(
    canonical: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    provenance = canonical.get("provenance")
    if not isinstance(provenance, dict) or provenance.get("kind") != "legacy_import":
        return {"source_targets": []}, {
            "source_target_evidence_source": "none",
            "source_target_evidence_count": 0,
            "verified_source_target_count": 0,
        }
    source_path = Path(str(provenance.get("source_path") or "")).expanduser()
    if not source_path.is_file():
        return {"source_targets": []}, {
            "source_target_evidence_source": "legacy_provenance_unavailable",
            "source_target_evidence_count": 0,
            "verified_source_target_count": 0,
        }
    expected_sha256 = str(provenance.get("source_sha256") or "").strip()
    actual_sha256 = _sha256(source_path)
    if not expected_sha256 or actual_sha256 != expected_sha256:
        raise ValueError(
            "source_legacy_provenance_hash_mismatch:"
            f"expected={expected_sha256 or 'missing'}:actual={actual_sha256}"
        )
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    reconverted = convert_legacy_run_log(
        raw,
        task_name=str(canonical["task_name"]),
        task_parameters=dict(canonical.get("task_parameters") or {}),
        seed=canonical.get("seed"),
        source_path=source_path,
        require_screenshots=False,
    )
    if len(reconverted["steps"]) != len(canonical["steps"]):
        raise ValueError("source_legacy_provenance_step_count_mismatch")
    for step_index, (source_step, canonical_step) in enumerate(
        zip(reconverted["steps"], canonical["steps"], strict=True)
    ):
        if source_step["action"] != canonical_step["action"]:
            raise ValueError(
                f"source_legacy_provenance_action_mismatch:{step_index}"
            )
        if str(source_step["observation"].get("forest") or "") != str(
            canonical_step["observation"].get("forest") or ""
        ):
            raise ValueError(
                f"source_legacy_provenance_observation_mismatch:{step_index}"
            )
    source_targets, audit = _target_audit_from_embedded_evidence(reconverted)
    audit["source_target_evidence_source"] = "verified_legacy_provenance"
    audit["source_target_evidence_sha256"] = actual_sha256
    return source_targets, audit


def _canonical_source_target_audit(
    canonical: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_targets, audit = _target_audit_from_embedded_evidence(canonical)
    if audit["source_target_evidence_count"]:
        return source_targets, audit
    return _target_audit_from_legacy_provenance(canonical)


def _ground_source_actions(
    canonical: dict[str, Any],
    states: dict[str, dict[str, Any]],
    source_target_audit: dict[str, Any],
) -> tuple[dict[str, Any], int]:
    targets_by_state: dict[str, dict[str, str]] = {}
    targets_by_step: dict[int, dict[str, str]] = {}
    for record in source_target_audit.get("source_targets") or []:
        if not isinstance(record, dict):
            continue
        target = _identity(
            record.get("target")
            if isinstance(record.get("target"), dict)
            else {}
        )
        if not target:
            continue
        state_id = str(record.get("source_state_id") or "").strip()
        if state_id:
            targets_by_state[state_id] = target
        try:
            targets_by_step[int(record["step_index"])] = target
        except (KeyError, TypeError, ValueError):
            pass

    grounded = json.loads(json.dumps(canonical, ensure_ascii=False))
    run_displays = {
        (display["width"], display["height"])
        for state in states.values()
        if isinstance(state, dict)
        for display in [
            _source_display(state.get("display"))
            or _display_from_xml(str(state.get("xml") or ""))
        ]
        if display
    }
    shared_display: dict[str, int] = {}
    if len(run_displays) == 1:
        width, height = next(iter(run_displays))
        shared_display = {"width": width, "height": height}
    semantic_action_count = 0
    for step_index, step in enumerate(grounded["steps"]):
        observation = step["observation"]
        state_identifier = observation_state_id(observation)
        state = states.get(state_identifier)
        if not isinstance(state, dict):
            raise ValueError(f"source_state_missing:{state_identifier}")
        action_type = str(step["action"].get("action_type") or "").strip()
        xml_text = str(state.get("xml") or observation.get("forest") or "").strip()
        auxiliaries = observation.get("auxiliaries")
        observation_display = _source_display(
            auxiliaries.get("display") if isinstance(auxiliaries, dict) else None
        )
        display = (
            _source_display(state.get("display"))
            or observation_display
            or _display_from_xml(xml_text)
            or shared_display
        )
        if display and not observation_display:
            projection_auxiliaries = dict(auxiliaries or {})
            projection_auxiliaries["display"] = display
            observation["auxiliaries"] = projection_auxiliaries
        projected_actions = (
            []
            if action_type in {"answer", "status", "unknown"}
            else project_androidworld_step_actions(step)
        )
        needs_element_grounding = any(
            action["tool"] in {"click", "long_press", "input_text"}
            for action in projected_actions
        )
        if not xml_text and needs_element_grounding:
            raise ValueError(f"source_state_xml_missing:{state_identifier}")
        target = (
            targets_by_state.get(state_identifier)
            or targets_by_step.get(step_index)
            or {}
        )
        if not target and action_type in {
            "click",
            "double_tap",
            "input_text",
            "long_press",
        }:
            point_action = next(
                (
                    action
                    for action in projected_actions
                    if action["tool"] in {"click", "long_press"}
                ),
                None,
            )
            if point_action is not None:
                target = _identity_at_action_point(
                    xml_text,
                    action_args=point_action["args"],
                    display=display,
                )
        if not target and action_type == "input_text":
            target = _unique_editable_identity(xml_text)
        source_context: dict[str, Any] = {}
        if xml_text:
            source_context["page"] = xml_text
        package_name = str(
            state.get("package_name")
            or (
                auxiliaries.get("package_name")
                if isinstance(auxiliaries, dict)
                else ""
            )
            or ""
        ).strip()
        if package_name:
            source_context["package_name"] = package_name
        if target:
            source_context["element"] = target
            semantic_action_count += 1
        metadata = dict(step.get("metadata") or {})
        metadata["source_context"] = source_context
        step["metadata"] = metadata
    return grounded, semantic_action_count


def build_grounded_teacher_run_log(
    *,
    source_run_log: str | Path,
    source_state_catalog: str | Path,
    provenance_manifest: str | Path,
    expected_source_run_log_sha256: str,
    expected_source_state_catalog_sha256: str,
    expected_provenance_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Join frozen source actions with frozen source UI identities.

    This reads no target task input, target observation, or validator state.
    It keeps the canonical source action sequence unchanged and adds only
    source-side semantic evidence consumed by baseline teacher adapters.
    """

    source_path = _require_frozen_file(
        source_run_log,
        expected_sha256=expected_source_run_log_sha256,
        label="source_run_log",
    )
    catalog_path = _require_frozen_file(
        source_state_catalog,
        expected_sha256=expected_source_state_catalog_sha256,
        label="source_state_catalog",
    )
    provenance_path = _require_frozen_file(
        provenance_manifest,
        expected_sha256=expected_provenance_sha256,
        label="source_provenance",
    )
    raw = json.loads(source_path.read_text(encoding="utf-8"))
    canonical = import_run_log(raw)
    states = load_transfer_state_catalog(catalog_path)
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    grounded, semantic_action_count = _ground_source_actions(
        canonical,
        states,
        _source_target_audit(provenance),
    )

    return grounded, {
        "schema_version": "omniflow.source-teacher-grounding.v1",
        "source_run_log": str(source_path),
        "source_run_log_sha256": _sha256(source_path),
        "source_state_catalog": str(catalog_path),
        "source_state_catalog_sha256": _sha256(catalog_path),
        "source_state_catalog_source": "frozen_catalog",
        "provenance_manifest": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "source_state_count": len(states),
        "semantic_action_count": semantic_action_count,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read": False,
    }


def _build_grounded_teacher_run_log_from_embedded_source(
    *,
    source_run_log: str | Path,
    provenance_manifest: str | Path,
    expected_source_run_log_sha256: str,
    expected_provenance_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = _require_frozen_file(
        source_run_log,
        expected_sha256=expected_source_run_log_sha256,
        label="source_run_log",
    )
    provenance_path = _require_frozen_file(
        provenance_manifest,
        expected_sha256=expected_provenance_sha256,
        label="source_provenance",
    )
    canonical, source_states = import_run_log_evidence(
        json.loads(source_path.read_text(encoding="utf-8")),
        evidence_root=source_path.parent,
    )
    states = source_states["states"]
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    provenance_source_sha256 = str(
        provenance.get("source_run_log_sha256") or ""
    ).strip()
    if (
        provenance_source_sha256
        and provenance_source_sha256 != expected_source_run_log_sha256
    ):
        replay_output_sha256 = str(
            provenance.get("output_source_run_log_sha256") or ""
        ).strip()
        if (
            provenance.get("schema_version")
            != "omniflow.source-replay-transfer-store.v1"
            or replay_output_sha256 != expected_source_run_log_sha256
        ):
            raise ValueError("source_provenance_run_log_mismatch")
    grounded, semantic_action_count = _ground_source_actions(
        canonical,
        states,
        _source_target_audit(provenance),
    )
    return grounded, {
        "schema_version": "omniflow.source-teacher-grounding.v1",
        "source_run_log": str(source_path),
        "source_run_log_sha256": _sha256(source_path),
        "source_state_catalog": str(source_path),
        "source_state_catalog_sha256": _sha256(source_path),
        "source_state_catalog_source": "embedded_source_run_log",
        "provenance_manifest": str(provenance_path),
        "provenance_sha256": _sha256(provenance_path),
        "source_state_count": len(states),
        "semantic_action_count": semantic_action_count,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read": False,
    }


def _build_grounded_teacher_run_log_from_canonical_source(
    *,
    source_run_log: str | Path,
    expected_source_run_log_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    source_path = _require_frozen_file(
        source_run_log,
        expected_sha256=expected_source_run_log_sha256,
        label="source_run_log",
    )
    canonical, source_states = import_run_log_evidence(
        json.loads(source_path.read_text(encoding="utf-8")),
        evidence_root=source_path.parent,
    )
    states = source_states["states"]
    source_target_audit, target_evidence_audit = (
        _canonical_source_target_audit(canonical)
    )
    grounded, semantic_action_count = _ground_source_actions(
        canonical,
        states,
        source_target_audit,
    )
    return grounded, {
        "schema_version": "omniflow.source-teacher-grounding.v1",
        "source_run_log": str(source_path),
        "source_run_log_sha256": _sha256(source_path),
        "source_state_catalog": str(source_path),
        "source_state_catalog_sha256": _sha256(source_path),
        "source_state_catalog_source": "embedded_source_run_log",
        "source_state_count": len(states),
        "semantic_action_count": semantic_action_count,
        "grounding_source": "canonical_androidworld_run_log",
        **target_evidence_audit,
        "target_inputs_read": False,
        "target_observations_read": False,
        "validator_state_read": False,
    }


def build_grounded_teacher_run_log_from_item(
    *,
    index_path: str | Path,
    item: Any,
    store_index_path: str | Path | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve one archive-index row and ground it from frozen source evidence."""

    meta = item.meta
    indexed_source_sha256 = str(
        meta.get("retained_source_run_log_sha256")
        or meta.get("source_run_log_sha256")
        or ""
    ).strip()
    if store_index_path is None and not meta.get("store_provenance"):
        return _build_grounded_teacher_run_log_from_canonical_source(
            source_run_log=item.source_run_log,
            expected_source_run_log_sha256=indexed_source_sha256,
        )
    store_row: dict[str, Any] = {}
    if store_index_path is not None:
        resolved_store_index = Path(store_index_path).expanduser().resolve()
        if not resolved_store_index.is_file():
            raise FileNotFoundError(
                f"store_index_missing:{resolved_store_index}"
            )
        store_payload = json.loads(
            resolved_store_index.read_text(encoding="utf-8")
        )
        candidate = (
            store_payload.get(str(item.task))
            if isinstance(store_payload, dict)
            else None
        )
        if not isinstance(candidate, dict):
            raise ValueError(f"store_index_task_missing:{item.task}")
        store_row = candidate

    source_provenance_value = meta.get("store_provenance")
    provenance_value = source_provenance_value or store_row.get(
        "provenance_path"
    )
    provenance_sha256 = (
        meta.get("store_provenance_sha256")
        or store_row.get("provenance_sha256")
    )
    source_run_log_value = (
        store_row.get("source_run_log_path")
        or item.source_run_log
    )
    source_run_log_sha256 = (
        store_row.get("source_run_log_sha256")
        or meta.get("retained_source_run_log_sha256")
        or meta.get("source_run_log_sha256")
        or ""
    )
    store_source_sha256 = str(
        store_row.get("source_run_log_sha256") or ""
    ).strip()
    if (
        indexed_source_sha256
        and store_source_sha256
        and indexed_source_sha256 != store_source_sha256
    ):
        raise ValueError("source_store_index_run_log_mismatch")
    provenance_path = _index_reference(
        index_path if source_provenance_value else store_index_path,
        provenance_value,
        label="store_provenance",
    )
    explicit_state_catalog = (
        meta.get("source_state_catalog")
        or meta.get("transfer_state_catalog")
    )
    if explicit_state_catalog:
        state_catalog_sha256 = str(
            meta.get("source_state_catalog_sha256")
            or meta.get("transfer_state_catalog_sha256")
            or ""
        ).strip()
        return build_grounded_teacher_run_log(
            source_run_log=source_run_log_value,
            source_state_catalog=_index_reference(
                index_path,
                explicit_state_catalog,
                label="source_state_catalog",
            ),
            provenance_manifest=provenance_path,
            expected_source_run_log_sha256=str(source_run_log_sha256),
            expected_source_state_catalog_sha256=state_catalog_sha256,
            expected_provenance_sha256=str(provenance_sha256 or ""),
        )

    try:
        return _build_grounded_teacher_run_log_from_embedded_source(
            source_run_log=source_run_log_value,
            provenance_manifest=provenance_path,
            expected_source_run_log_sha256=str(source_run_log_sha256),
            expected_provenance_sha256=str(provenance_sha256 or ""),
        )
    except ValueError as error:
        if not str(error).startswith(
            ("source_state_missing:", "source_state_xml_missing:")
        ) or not store_row:
            raise
    return build_grounded_teacher_run_log(
        source_run_log=source_run_log_value,
        source_state_catalog=_index_reference(
            store_index_path,
            store_row.get("transfer_states_path"),
            label="transfer_states_path",
        ),
        provenance_manifest=provenance_path,
        expected_source_run_log_sha256=str(source_run_log_sha256),
        expected_source_state_catalog_sha256=str(
            store_row.get("transfer_states_sha256") or ""
        ),
        expected_provenance_sha256=str(provenance_sha256 or ""),
    )


def resolve_store_source_run_log(
    store_index_path: str | Path,
    *,
    task_name: str,
) -> tuple[Path, str]:
    index_path = Path(store_index_path).expanduser().resolve()
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    row = payload.get(str(task_name)) if isinstance(payload, dict) else None
    if not isinstance(row, dict):
        raise ValueError(f"store_index_task_missing:{task_name}")
    source_path = _require_frozen_file(
        row.get("source_run_log_path"),
        expected_sha256=str(row.get("source_run_log_sha256") or ""),
        label=f"store_source_run_log:{task_name}",
    )
    return source_path, str(row["source_run_log_sha256"])
