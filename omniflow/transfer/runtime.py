from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

TRANSFER_STATE_CATALOG_FILENAME = "transfer_states.json"
TRANSFER_STATE_CATALOG_VERSION = "omniflow.transfer-state-catalog.v1"
_TRANSFER_STATE_FIELDS = {
    "state_id",
    "xml",
    "package_name",
    "activity_name",
    "display",
}


def capture_transfer_state(observation: Any) -> dict[str, Any]:
    identity = {
        key: value
        for key, value in observation.to_dict().items()
        if key in {"xml", "package_name", "activity_name"} and value not in {None, ""}
    }
    identity.update(
        {
            key: value
            for key, value in observation.extra.items()
            if key in {"display", "screenshot_path"}
            and value is not None
            and value != ""
        }
    )
    explicit_state_id = str(observation.extra.get("state_id") or "").strip()
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    state = {
        "state_id": explicit_state_id
        or "state_" + hashlib.sha256(encoded.encode()).hexdigest()[:20]
    }
    for key in ("xml", "package_name", "activity_name"):
        value = identity.get(key)
        if isinstance(value, str) and value:
            state[key] = value
    display = identity.get("display")
    if isinstance(display, dict) and set(display) == {"width", "height"}:
        state["display"] = dict(display)
    return state


def load_transfer_state_catalog(path: str | Path) -> dict[str, dict[str, Any]]:
    catalog_path = Path(path).expanduser().resolve()
    if not catalog_path.is_file():
        return {}
    payload = json.loads(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {
        "schema_version",
        "run_id",
        "states",
    }:
        raise ValueError("transfer_state_catalog_contract_invalid")
    if payload.get("schema_version") != TRANSFER_STATE_CATALOG_VERSION:
        raise ValueError("unsupported_transfer_state_catalog_version")
    if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
        raise ValueError("transfer_state_catalog_run_id_required")
    raw_states = payload.get("states")
    if not isinstance(raw_states, dict):
        raise ValueError("transfer_state_catalog_states_invalid")
    states: dict[str, dict[str, Any]] = {}
    for state_id, value in raw_states.items():
        state = _canonicalize_transfer_state(value)
        if str(state_id) != state["state_id"]:
            raise ValueError("transfer_state_catalog_key_mismatch")
        states[state["state_id"]] = state
    return states


def _canonicalize_transfer_state(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("transfer_state_must_be_object")
    unknown = sorted(set(value) - _TRANSFER_STATE_FIELDS)
    if unknown:
        raise ValueError(f"transfer_state_unknown_fields:{','.join(unknown)}")
    state_id = str(value.get("state_id") or "").strip()
    if not state_id:
        raise ValueError("transfer_state_id_required")
    state: dict[str, Any] = {"state_id": state_id}
    for key in ("xml", "package_name", "activity_name"):
        item = value.get(key)
        if item is None:
            continue
        if not isinstance(item, str):
            raise ValueError(f"transfer_state_{key}_must_be_string")
        state[key] = item
    display = value.get("display")
    if display is not None:
        if not isinstance(display, dict) or set(display) != {"width", "height"}:
            raise ValueError("transfer_state_display_invalid")
        width = display.get("width")
        height = display.get("height")
    else:
        width = height = None
    if width is not None or height is not None:
        if not all(
            isinstance(item, int) and not isinstance(item, bool) and item > 0
            for item in (width, height)
        ):
            raise ValueError("transfer_state_display_invalid")
        state["display"] = {"width": width, "height": height}
    return state


def load_omnitransfer() -> Any:
    configured_root = os.environ.get("OMNITRANSFER_ROOT")
    if configured_root:
        return _load_omnitransfer_from_root(Path(configured_root))
    try:
        return importlib.import_module("omnitransfer")
    except ModuleNotFoundError as error:
        if error.name != "omnitransfer":
            raise
    return _load_omnitransfer_from_root(
        Path.home() / "Projects" / "Omni" / "OmniTransfer"
    )


def _load_omnitransfer_from_root(root: Path) -> Any:
    root = root.expanduser().resolve()
    source_root = (root / "src").resolve()
    package_root = source_root / "omnitransfer"
    if not package_root.is_dir():
        raise RuntimeError(f"omnitransfer_root_missing:{root}")
    loaded = sys.modules.get("omnitransfer")
    if loaded is not None and _module_is_from(loaded, package_root):
        return loaded
    for name in tuple(sys.modules):
        if name == "omnitransfer" or name.startswith("omnitransfer."):
            del sys.modules[name]
    source_path = str(source_root)
    sys.path[:] = [
        item for item in sys.path if str(Path(item or ".").resolve()) != source_path
    ]
    sys.path.insert(0, source_path)
    importlib.invalidate_caches()
    module = importlib.import_module("omnitransfer")
    if not _module_is_from(module, package_root):
        raise RuntimeError(f"omnitransfer_import_outside_root:{module.__file__}")
    return module


def _module_is_from(module: Any, package_root: Path) -> bool:
    module_path = Path(str(getattr(module, "__file__", "") or "")).resolve()
    try:
        module_path.relative_to(package_root)
    except ValueError:
        return False
    return True


def required_transfer_state_ids(functions: Any) -> tuple[str, ...]:
    required: list[str] = []
    values = functions.values() if isinstance(functions, dict) else functions
    for function in values or ():
        for step in getattr(function, "steps", ()) or ():
            action = getattr(step, "action", None)
            if not _action_requires_transfer_state(action):
                continue
            state_id = str(getattr(step, "source_state_id", "") or "").strip()
            if state_id and state_id not in required:
                required.append(state_id)
    return tuple(required)


def transfer_state_coverage(
    functions: Any,
    states: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    required = required_transfer_state_ids(functions)
    missing = tuple(state_id for state_id in required if state_id not in states)
    return {
        "required_state_ids": required,
        "missing_state_ids": missing,
        "required_state_count": len(required),
        "available_state_count": len(states),
        "complete": not missing,
    }


def audit_transfer_action_sources(
    functions: Any,
    states: dict[str, dict[str, Any]],
    *,
    center_tolerance: float = 0.02,
) -> dict[str, Any]:
    tolerance = float(center_tolerance)
    if not math.isfinite(tolerance) or tolerance < 0.0 or tolerance > 0.5:
        raise ValueError("transfer_source_center_tolerance_invalid")
    audited: list[dict[str, Any]] = []
    values = functions.values() if isinstance(functions, dict) else functions
    for function in values or ():
        function_id = str(getattr(function, "id", "") or "")
        for step in getattr(function, "steps", ()) or ():
            action = getattr(step, "action", None)
            if not _action_requires_point_target(action):
                continue
            step_index = int(getattr(step, "step_index", 0))
            source_state_id = str(getattr(step, "source_state_id", "") or "")
            state = states.get(source_state_id)
            if not isinstance(state, dict):
                raise ValueError(
                    f"transfer_action_source_state_missing:{function_id}:"
                    f"{step_index}:{source_state_id}"
                )
            source_xml = str(state.get("xml") or "")
            source_size = _state_display_size(state, source_xml)
            if source_size is None:
                raise ValueError(
                    f"transfer_action_source_display_missing:{function_id}:"
                    f"{step_index}:{source_state_id}"
                )
            width, height = source_size
            source_point = (
                float(action.args["x"]) / 1000.0 * width,
                float(action.args["y"]) / 1000.0 * height,
            )
            _require_raw_source_target(
                source_xml,
                source_point,
                function_id=function_id,
                step_index=step_index,
                source_state_id=source_state_id,
            )
            result = transfer_action(
                source_xml=source_xml,
                target_xml=source_xml,
                source_point=source_point,
                source_package_name=str(state.get("package_name") or ""),
                target_package_name=str(state.get("package_name") or ""),
                source_activity_name=str(state.get("activity_name") or ""),
                target_activity_name=str(state.get("activity_name") or ""),
                action_type=str(getattr(action, "tool", "") or ""),
                top_k=3,
            )
            if result.get("mapped") is not True:
                reason = str(result.get("reason") or "failed")
                raise ValueError(
                    f"transfer_action_source_target_unresolved:{function_id}:"
                    f"{step_index}:{source_state_id}:{reason}"
                )
            source_element = result.get("src_element")
            if not isinstance(source_element, dict):
                raise ValueError(
                    f"transfer_action_source_target_unresolved:{function_id}:"
                    f"{step_index}:{source_state_id}"
                )
            offset_x, offset_y = _source_element_offset(
                source_element,
                source_point,
                function_id=function_id,
                step_index=step_index,
                source_state_id=source_state_id,
            )
            centered = max(abs(offset_x - 0.5), abs(offset_y - 0.5)) <= tolerance
            center_conflict = False
            if not centered:
                center_conflict = _source_center_conflicts(
                    source_xml=source_xml,
                    source_element=source_element,
                    state=state,
                    action_type=str(getattr(action, "tool", "") or ""),
                )
            if not centered and not center_conflict:
                raise ValueError(
                    f"transfer_action_source_point_not_centered:{function_id}:"
                    f"{step_index}:{source_state_id}:"
                    f"offset={offset_x:.6f},{offset_y:.6f}"
                )
            audited.append(
                {
                    "function_id": function_id,
                    "step_index": step_index,
                    "source_state_id": source_state_id,
                    "offset_x": offset_x,
                    "offset_y": offset_y,
                    "center_conflict": center_conflict,
                    "target": {
                        key: source_element[key]
                        for key in ("resource_id", "text", "content_desc", "class")
                        if source_element.get(key) not in (None, "")
                    },
                }
            )
    return {
        "source_target_audit_complete": True,
        "source_target_count": len(audited),
        "source_targets": audited,
    }


def _source_element_offset(
    source_element: dict[str, Any],
    point: tuple[float, float],
    *,
    function_id: str,
    step_index: int,
    source_state_id: str,
) -> tuple[float, float]:
    try:
        left, top, right, bottom = (float(item) for item in source_element["bounds"])
        offset_x = (point[0] - left) / (right - left)
        offset_y = (point[1] - top) / (bottom - top)
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as error:
        raise ValueError(
            f"transfer_action_source_target_offset_invalid:{function_id}:"
            f"{step_index}:{source_state_id}"
        ) from error
    if not all(math.isfinite(value) for value in (offset_x, offset_y)):
        raise ValueError(
            f"transfer_action_source_target_offset_invalid:{function_id}:"
            f"{step_index}:{source_state_id}"
        )
    return offset_x, offset_y


def _source_center_conflicts(
    *,
    source_xml: str,
    source_element: dict[str, Any],
    state: dict[str, Any],
    action_type: str,
) -> bool:
    try:
        left, top, right, bottom = (float(item) for item in source_element["bounds"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("transfer_action_source_target_offset_invalid") from error
    result = transfer_action(
        source_xml=source_xml,
        target_xml=source_xml,
        source_point=((left + right) / 2.0, (top + bottom) / 2.0),
        source_package_name=str(state.get("package_name") or ""),
        target_package_name=str(state.get("package_name") or ""),
        source_activity_name=str(state.get("activity_name") or ""),
        target_activity_name=str(state.get("activity_name") or ""),
        action_type=action_type,
        top_k=3,
    )
    centered_element = result.get("src_element")
    return (
        result.get("mapped") is not True
        or not isinstance(centered_element, dict)
        or _element_signature(centered_element) != _element_signature(source_element)
    )


def _element_signature(value: dict[str, Any]) -> tuple[Any, ...]:
    return tuple(
        json.dumps(value.get(key), ensure_ascii=False, sort_keys=True)
        for key in ("resource_id", "text", "content_desc", "class", "bounds")
    )


def _action_requires_transfer_state(action: Any) -> bool:
    tool = str(getattr(action, "tool", "") or "")
    args = getattr(action, "args", None)
    if not isinstance(args, dict):
        return False
    if tool in {"click", "input_text", "long_press"}:
        return all(args.get(key) is not None for key in ("x", "y"))
    if tool == "swipe":
        return all(args.get(key) is not None for key in ("x1", "y1", "x2", "y2"))
    return False


def _action_requires_point_target(action: Any) -> bool:
    tool = str(getattr(action, "tool", "") or "")
    args = getattr(action, "args", None)
    return (
        tool in {"click", "input_text", "long_press"}
        and isinstance(args, dict)
        and all(args.get(key) is not None for key in ("x", "y"))
    )


def _xml_display_size(xml_text: str) -> tuple[float, float] | None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return None
    bounds = [
        numbers
        for element in root.iter()
        if len(
            numbers := [
                int(item)
                for item in re.findall(
                    r"-?\d+",
                    str(element.attrib.get("bounds") or ""),
                )
            ]
        )
        == 4
        and numbers[2] > numbers[0]
        and numbers[3] > numbers[1]
    ]
    if not bounds:
        return None
    width = max(float(item[2]) for item in bounds)
    height = max(float(item[3]) for item in bounds)
    return (width, height) if width > 0.0 and height > 0.0 else None


def _state_display_size(
    state: dict[str, Any],
    xml_text: str,
) -> tuple[float, float] | None:
    xml_size = _xml_display_size(xml_text)
    display = state.get("display")
    action_size = None
    if isinstance(display, dict) and set(display) == {"width", "height"}:
        try:
            width = float(display.get("width") or 0)
            height = float(display.get("height") or 0)
        except (TypeError, ValueError):
            width = height = 0.0
        if width > 0 and height > 0:
            action_size = (width, height)
    if xml_size is None:
        return action_size
    if action_size is None:
        return xml_size
    return max(xml_size[0], action_size[0]), max(xml_size[1], action_size[1])


def _require_raw_source_target(
    xml_text: str,
    point: tuple[float, float],
    *,
    function_id: str,
    step_index: int,
    source_state_id: str,
) -> None:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as error:
        raise ValueError(
            f"transfer_action_source_state_xml_invalid:{function_id}:"
            f"{step_index}:{source_state_id}"
        ) from error
    x, y = point
    candidates: list[tuple[float, ET.Element]] = []
    for element in root.iter():
        numbers = [
            int(item)
            for item in re.findall(
                r"-?\d+",
                str(element.attrib.get("bounds") or ""),
            )
        ]
        if len(numbers) != 4:
            continue
        left, top, right, bottom = numbers
        if left <= x <= right and top <= y <= bottom and right > left and bottom > top:
            candidates.append((float((right - left) * (bottom - top)), element))
    if not candidates:
        raise ValueError(
            f"transfer_action_source_target_unresolved:{function_id}:"
            f"{step_index}:{source_state_id}"
        )
    target = min(
        candidates,
        key=lambda item: (
            item[0],
            0 if str(item[1].attrib.get("class") or "").strip() else 1,
        ),
    )[1]
    explicit_class = str(target.attrib.get("class") or "").strip()
    raw_attributes = {
        "package",
        "resource-id",
        "content-desc",
        "clickable",
        "enabled",
        "focusable",
        "scrollable",
    }
    if not explicit_class or not any(key in target.attrib for key in raw_attributes):
        raise ValueError(
            f"transfer_action_source_state_not_raw:{function_id}:"
            f"{step_index}:{source_state_id}"
        )


def transfer_action(**kwargs: Any) -> dict[str, Any]:
    action_transfer = getattr(load_omnitransfer(), "action_transfer", None)
    if not callable(action_transfer):
        raise RuntimeError("omnitransfer_action_transfer_unavailable")
    result = action_transfer(**kwargs)
    if not isinstance(result, dict):
        raise RuntimeError("omnitransfer_result_invalid")
    return result
