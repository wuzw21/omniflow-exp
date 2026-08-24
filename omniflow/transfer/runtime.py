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
import unicodedata
import xml.etree.ElementTree as ET

TRANSFER_STATE_CATALOG_FILENAME = "transfer_states.json"
TRANSFER_STATE_CATALOG_VERSION = "omniflow.transfer-state-catalog.v1"
_TRANSFER_STATE_FIELDS = {
    "state_id",
    "xml",
    "package_name",
    "activity_name",
    "display",
    "screenshot_path",
}
_OMNITRANSFER_CANDIDATE_FIELDS = {
    "target_xml",
    "source_xml",
    "source_point",
    "source_element_id",
    "source_offset",
    "source_screenshot_path",
    "target_screenshot_path",
    "source_visual_rgb",
    "target_visual_rgb",
    "action_type",
    "top_k",
}
_MIN_SOURCE_ANCHOR_OFFSET = -1.0
_MAX_SOURCE_ANCHOR_OFFSET = 2.0


def capture_transfer_state(observation: Any) -> dict[str, Any]:
    extra = observation.extra if isinstance(observation.extra, dict) else {}
    androidworld_state = (
        extra.get("androidworld_state")
        if isinstance(extra.get("androidworld_state"), dict)
        else {}
    )
    pixels = (
        androidworld_state.get("pixels")
        if isinstance(androidworld_state.get("pixels"), dict)
        else {}
    )
    identity = {
        key: value
        for key, value in observation.to_dict().items()
        if key in {"xml", "package_name", "activity_name"} and value not in {None, ""}
    }
    identity.update(
        {
            key: value
            for key, value in extra.items()
            if key in {"display", "screenshot_path"}
            and value is not None
            and value != ""
        }
    )
    screenshot_path = str(
        extra.get("screenshot_path") or pixels.get("path") or ""
    ).strip()
    if screenshot_path:
        identity["screenshot_path"] = screenshot_path
    explicit_state_id = str(extra.get("state_id") or "").strip()
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
    screenshot_path = identity.get("screenshot_path")
    if isinstance(screenshot_path, str) and screenshot_path:
        state["screenshot_path"] = screenshot_path
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
    screenshot_path = value.get("screenshot_path")
    if screenshot_path is not None:
        if not isinstance(screenshot_path, str) or not screenshot_path.strip():
            raise ValueError("transfer_state_screenshot_path_invalid")
        state["screenshot_path"] = screenshot_path.strip()
    return state


def load_omnitransfer() -> Any:
    configured_root = str(os.environ.get("OMNITRANSFER_ROOT") or "").strip()
    root = (
        Path(configured_root)
        if configured_root
        else Path.home() / "Projects" / "Omni" / "OmniTransfer"
    )
    return _load_omnitransfer_from_root(root)


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
) -> dict[str, Any]:
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
            source_point = _source_action_point(
                action,
                source_xml,
                width=width,
                height=height,
            )
            if source_point is None:
                raise ValueError(
                    f"transfer_action_source_target_unresolved:{function_id}:"
                    f"{step_index}:{source_state_id}"
                )
            source_grounding = _require_raw_source_target(
                source_xml,
                source_point,
                function_id=function_id,
                step_index=step_index,
                source_state_id=source_state_id,
            )
            source_element_id = source_semantic_anchor(
                source_xml,
                source_point,
            )
            transfer_request: dict[str, Any] = {
                "source_xml": source_xml,
                "target_xml": source_xml,
                "source_point": source_point,
                "source_package_name": str(state.get("package_name") or ""),
                "target_package_name": str(state.get("package_name") or ""),
                "source_activity_name": str(state.get("activity_name") or ""),
                "target_activity_name": str(state.get("activity_name") or ""),
                "action_type": str(getattr(action, "tool", "") or ""),
                "top_k": 3,
            }
            if source_element_id:
                transfer_request["source_element_id"] = source_element_id
                source_offset = source_semantic_offset(
                    source_xml,
                    source_point,
                    source_element_id,
                )
                if source_offset is not None:
                    transfer_request["source_offset"] = source_offset
            result = transfer_action(
                **transfer_request,
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
            if not all(
                _MIN_SOURCE_ANCHOR_OFFSET
                <= value
                <= _MAX_SOURCE_ANCHOR_OFFSET
                for value in (offset_x, offset_y)
            ):
                raise ValueError(
                    f"transfer_action_source_point_outside:{function_id}:"
                    f"{step_index}:{source_state_id}:"
                    f"offset={offset_x:.6f},{offset_y:.6f}"
                )
            audited.append(
                {
                    "function_id": function_id,
                    "step_index": step_index,
                    "source_state_id": source_state_id,
                    "source_grounding": source_grounding,
                    "offset_x": offset_x,
                    "offset_y": offset_y,
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


def _action_requires_transfer_state(action: Any) -> bool:
    tool = str(getattr(action, "tool", "") or "")
    args = getattr(action, "args", None)
    if not isinstance(args, dict):
        return False
    if tool == "input_text":
        return bool(str(args.get("target_description") or "").strip()) or all(
            args.get(key) is not None for key in ("x", "y")
        )
    if tool in {"click", "long_press"}:
        return all(args.get(key) is not None for key in ("x", "y"))
    if tool == "swipe":
        return all(args.get(key) is not None for key in ("x1", "y1", "x2", "y2"))
    return False


def _action_requires_point_target(action: Any) -> bool:
    tool = str(getattr(action, "tool", "") or "")
    args = getattr(action, "args", None)
    if not isinstance(args, dict):
        return False
    if tool == "input_text":
        return bool(str(args.get("target_description") or "").strip()) or all(
            args.get(key) is not None for key in ("x", "y")
        )
    return tool in {"click", "long_press"} and all(
        args.get(key) is not None for key in ("x", "y")
    )


def _source_action_point(
    action: Any,
    source_xml: str,
    *,
    width: float,
    height: float,
) -> tuple[float, float] | None:
    args = getattr(action, "args", None)
    if not isinstance(args, dict):
        return None
    if all(args.get(key) is not None for key in ("x", "y")):
        try:
            return (
                float(args["x"]) / 1000.0 * width,
                float(args["y"]) / 1000.0 * height,
            )
        except (TypeError, ValueError):
            return None
    if str(getattr(action, "tool", "") or "") != "input_text":
        return None
    return source_semantic_point(
        source_xml,
        str(args.get("target_description") or ""),
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
    display = state.get("display")
    if isinstance(display, dict) and set(display) == {"width", "height"}:
        try:
            width = float(display.get("width") or 0)
            height = float(display.get("height") or 0)
        except (TypeError, ValueError):
            width = height = 0.0
        if width > 0 and height > 0:
            return width, height
    return _xml_display_size(xml_text)


def _require_raw_source_target(
    xml_text: str,
    point: tuple[float, float],
    *,
    function_id: str,
    step_index: int,
    source_state_id: str,
) -> str:
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
    semantic_attributes = {
        "text",
        "content-desc",
        "resource-id",
        "class",
    }
    raw_attributes = {
        "id",
        "package",
        "resource-id",
        "content-desc",
        "clickable",
        "enabled",
        "focusable",
        "scrollable",
    }
    named = any(
        str(target.attrib.get(key) or "").strip()
        for key in semantic_attributes
    )
    actionable = bool(str(target.attrib.get("id") or "").strip()) and any(
        str(target.attrib.get(key) or "").strip().lower() == "true"
        for key in ("clickable", "scrollable")
    )
    if (
        not (named or actionable)
        or not any(key in target.attrib for key in raw_attributes)
    ):
        raise ValueError(
            f"transfer_action_source_state_not_raw:{function_id}:"
            f"{step_index}:{source_state_id}"
        )
    return "named_element" if named else "workflow_actionable_element"


def source_semantic_anchor(
    xml_text: str,
    point: tuple[float, float],
) -> str:
    """Resolve source-side row semantics for OmniTransfer candidate ranking.

    OOB observations may preserve accessibility nodes as an ordered, flat
    forest.  In that representation the clickable row and its visible title
    are siblings even though the title's bounds are inside the row.  Select
    that source title as the matcher anchor; this never identifies a target
    node and therefore cannot become a resource-id or coordinate replay path.
    """

    try:
        root = ET.fromstring(_normalize_legacy_flat_xml(xml_text))
    except ET.ParseError:
        return ""
    indexed: list[tuple[ET.Element, str, int]] = []
    parents: dict[ET.Element, ET.Element] = {}

    def visit(element: ET.Element, path: str, depth: int) -> None:
        indexed.append((element, path, depth))
        for index, child in enumerate(list(element)):
            parents[child] = element
            visit(child, f"{path}.{index}", depth + 1)

    visit(root, "0", 0)
    source = _actionable_source_element(indexed, point)
    if source is None:
        return ""
    source_element, source_path, _ = source
    promoted = False
    if _is_repeated_generic_affordance(source_element, indexed):
        ancestor = parents.get(source_element)
        while ancestor is not None:
            label = str(
                ancestor.attrib.get("text")
                or ancestor.attrib.get("content-desc")
                or ""
            ).strip()
            if (
                _semantic_label(label)
                and not _is_generic_affordance_label(label)
                and _element_bounds(ancestor) is not None
            ):
                source_element = ancestor
                source_path = next(
                    path for element, path, _ in indexed if element is ancestor
                )
                promoted = True
                break
            ancestor = parents.get(ancestor)
    source_bounds = _element_bounds(source_element)
    source_area = _bounds_area(source_bounds)
    descendants = set(source_element.iter())
    semantic: list[tuple[tuple[Any, ...], ET.Element, str]] = []
    for order, (element, path, depth) in enumerate(indexed):
        label = str(
            element.attrib.get("text")
            or element.attrib.get("content-desc")
            or ""
        ).strip()
        if not _semantic_label(label):
            continue
        bounds = _element_bounds(element)
        if element is not source_element and not (
            source_bounds is not None
            and bounds is not None
            and _bounds_contains(source_bounds, bounds)
            and _bounds_area(bounds) < source_area
        ):
            continue
        if bounds is not None and not _bounds_reaches_point(bounds, point):
            continue
        resource_tail = str(element.attrib.get("resource-id") or "").rsplit(
            "/", 1
        )[-1]
        semantic.append(
            (
                (
                    (
                        0
                        if (promoted and element is not source_element)
                        or (not promoted and element is source_element)
                        else 1
                    ),
                    0 if resource_tail == "title" else 1,
                    0 if element in descendants else 1,
                    -depth,
                    order,
                ),
                element,
                path,
            )
        )
    if not semantic:
        return ""
    _, anchor, anchor_path = min(semantic, key=lambda item: item[0])
    origin_id = str(anchor.attrib.get("id") or "").strip()
    if origin_id:
        return origin_id
    resource_id = str(anchor.attrib.get("resource-id") or "").strip()
    if resource_id and sum(
        str(element.attrib.get("resource-id") or "").strip() == resource_id
        for element, _, _ in indexed
    ) == 1:
        return resource_id
    return anchor_path


def source_semantic_point(
    xml_text: str,
    target_description: str,
) -> tuple[float, float] | None:
    """Resolve an input target only within its immutable source observation."""

    description = " ".join(str(target_description or "").split()).casefold()
    if not description:
        return None
    try:
        root = ET.fromstring(_normalize_legacy_flat_xml(xml_text))
    except ET.ParseError:
        return None
    matches: list[ET.Element] = []
    for element in root.iter():
        labels = {
            " ".join(str(element.attrib.get(attribute) or "").split()).casefold()
            for attribute in ("text", "content-desc")
        }
        resource_id = str(element.attrib.get("resource-id") or "").strip()
        if resource_id:
            labels.add(resource_id.rsplit("/", 1)[-1].casefold())
        if description in labels and _element_bounds(element) is not None:
            matches.append(element)
    if len(matches) != 1:
        return None
    left, top, right, bottom = _element_bounds(matches[0]) or (0.0, 0.0, 0.0, 0.0)
    return (left + right) / 2.0, (top + bottom) / 2.0


def source_semantic_offset(
    xml_text: str,
    point: tuple[float, float],
    element_id: str,
) -> tuple[float, float] | None:
    try:
        root = ET.fromstring(_normalize_legacy_flat_xml(xml_text))
    except ET.ParseError:
        return None
    indexed: list[tuple[ET.Element, str]] = []

    def visit(element: ET.Element, path: str) -> None:
        indexed.append((element, path))
        for index, child in enumerate(list(element)):
            visit(child, f"{path}.{index}")

    visit(root, "0")
    matching = [
        element
        for element, path in indexed
        if element_id
        in {
            str(element.attrib.get("id") or "").strip(),
            str(element.attrib.get("resource-id") or "").strip(),
            path,
        }
    ]
    if len(matching) != 1:
        return None
    bounds = _element_bounds(matching[0])
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    return (
        (point[0] - left) / (right - left),
        (point[1] - top) / (bottom - top),
    )


def _is_repeated_generic_affordance(
    element: ET.Element,
    indexed: list[tuple[ET.Element, str, int]],
) -> bool:
    label = str(
        element.attrib.get("text")
        or element.attrib.get("content-desc")
        or ""
    ).strip()
    resource_id = str(element.attrib.get("resource-id") or "").strip()
    resource_tail = resource_id.rsplit("/", 1)[-1].casefold()
    generic = _is_generic_affordance_label(label) or resource_tail in {
        "item_more",
        "more",
        "overflow",
        "overflow_menu",
    }
    if not generic:
        return False
    signature = (
        str(element.attrib.get("class") or "").strip(),
        resource_id,
        label.casefold(),
    )
    return sum(
        (
            str(candidate.attrib.get("class") or "").strip(),
            str(candidate.attrib.get("resource-id") or "").strip(),
            str(
                candidate.attrib.get("text")
                or candidate.attrib.get("content-desc")
                or ""
            ).strip().casefold(),
        )
        == signature
        for candidate, _, _ in indexed
    ) > 1


def _is_generic_affordance_label(value: str) -> bool:
    return value.strip().casefold() in {
        "menu",
        "more",
        "more actions",
        "more options",
        "options",
        "overflow",
    }


def _actionable_source_element(
    indexed: list[tuple[ET.Element, str, int]],
    point: tuple[float, float],
) -> tuple[ET.Element, str, int] | None:
    x, y = point
    containing = [
        item
        for item in indexed
        if (bounds := _element_bounds(item[0])) is not None
        and bounds[0] <= x <= bounds[2]
        and bounds[1] <= y <= bounds[3]
    ]
    actionable = [
        item
        for item in containing
        if str(item[0].attrib.get("enabled") or "true").lower() != "false"
        and any(
            str(item[0].attrib.get(attribute) or "").lower() == "true"
            for attribute in ("clickable", "editable", "scrollable")
        )
    ]
    candidates = actionable or containing
    return min(
        candidates,
        key=lambda item: (
            _bounds_area(_element_bounds(item[0])),
            -item[2],
            item[1],
        ),
        default=None,
    )


def _element_bounds(
    element: ET.Element,
) -> tuple[float, float, float, float] | None:
    numbers = [
        float(item)
        for item in re.findall(
            r"-?\d+(?:\.\d+)?",
            str(element.attrib.get("bounds") or ""),
        )
    ]
    if len(numbers) != 4 or numbers[2] <= numbers[0] or numbers[3] <= numbers[1]:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _bounds_area(bounds: tuple[float, float, float, float] | None) -> float:
    if bounds is None:
        return math.inf
    return (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _bounds_contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
) -> bool:
    return (
        outer[0] <= inner[0]
        and outer[1] <= inner[1]
        and outer[2] >= inner[2]
        and outer[3] >= inner[3]
    )


def _bounds_reaches_point(
    bounds: tuple[float, float, float, float],
    point: tuple[float, float],
) -> bool:
    # OmniTransfer accepts semantic anchor offsets in [-1, 2].  A farther label
    # cannot represent this click; let its containing actionable node anchor it.
    width = bounds[2] - bounds[0]
    height = bounds[3] - bounds[1]
    return (
        bounds[0] - width <= point[0] <= bounds[2] + width
        and bounds[1] - height <= point[1] <= bounds[3] + height
    )


def _semantic_label(value: str) -> bool:
    return bool(value) and any(
        not character.isspace() and unicodedata.category(character) != "Co"
        for character in value
    )


def _normalize_legacy_flat_xml(xml_text: str) -> str:
    """Restore hierarchy omitted by legacy OOB accessibility captures.

    Older collectors emitted accessibility nodes in native preorder but placed
    every node directly below one package wrapper.  OmniTransfer needs the
    original local graph relations, so rebuild only that recognisable flat
    representation from preorder and geometric containment.  Native nested
    AndroidWorld XML does not match this shape and passes through unchanged.
    """

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return xml_text
    flat_container = next(
        (
            element
            for element in root.iter()
            if len(children := list(element)) >= 3
            and all(not list(child) for child in children)
            and all(str(child.attrib.get("id") or "").strip() for child in children)
        ),
        None,
    )
    if flat_container is None:
        return xml_text
    flat_nodes = list(flat_container)
    bounds = [_element_bounds(node) for node in flat_nodes]
    depths: list[int] = []
    parents: list[int | None] = []
    for index, _ in enumerate(flat_nodes):
        node_bounds = bounds[index]
        possible = [
            parent_index
            for parent_index in range(index)
            if node_bounds is not None
            and bounds[parent_index] is not None
            and _bounds_contains(bounds[parent_index], node_bounds)
            and _legacy_node_can_contain(flat_nodes[parent_index])
        ]
        parent = min(
            possible,
            key=lambda parent_index: (
                _bounds_area(bounds[parent_index]),
                -depths[parent_index],
                -parent_index,
            ),
            default=None,
        )
        parents.append(parent)
        depths.append(0 if parent is None else depths[parent] + 1)

    for node in flat_nodes:
        flat_container.remove(node)
    top_level: list[ET.Element] = []
    for index, node in enumerate(flat_nodes):
        parent = parents[index]
        if parent is None:
            top_level.append(node)
        else:
            flat_nodes[parent].append(node)

    wrapper_is_capture_artifact = (
        flat_container is not root
        and len(list(root)) == 1
        and not str(flat_container.attrib.get("id") or "").strip()
        and not str(flat_container.attrib.get("class") or "").strip()
        and not str(flat_container.attrib.get("resource-id") or "").strip()
    )
    destination = root if wrapper_is_capture_artifact else flat_container
    if wrapper_is_capture_artifact:
        root.remove(flat_container)
    destination.extend(top_level)
    return ET.tostring(root, encoding="unicode")


def _legacy_node_can_contain(element: ET.Element) -> bool:
    class_name = str(element.attrib.get("class") or "").rsplit(".", 1)[-1]
    return class_name not in {
        "Button",
        "CheckBox",
        "EditText",
        "ImageButton",
        "ImageView",
        "RadioButton",
        "Switch",
        "SwitchCompat",
        "TextView",
    }


def transfer_action(**kwargs: Any) -> dict[str, Any]:
    kwargs = dict(kwargs)
    for field in ("source_xml", "target_xml"):
        value = kwargs.get(field)
        if isinstance(value, str) and value:
            kwargs[field] = _normalize_legacy_flat_xml(value)
    module = load_omnitransfer()
    rank_candidates = getattr(module, "rank_action_candidates", None)
    if callable(rank_candidates):
        ranking = rank_candidates(
            **{
                key: value
                for key, value in kwargs.items()
                if key in _OMNITRANSFER_CANDIDATE_FIELDS
            }
        )
        if not isinstance(ranking, dict):
            raise RuntimeError("omnitransfer_result_invalid")
        return _select_transfer_candidate(ranking, kwargs)
    action_transfer = getattr(module, "action_transfer", None)
    if not callable(action_transfer):
        raise RuntimeError("omnitransfer_action_transfer_unavailable")
    result = action_transfer(**kwargs)
    if not isinstance(result, dict):
        raise RuntimeError("omnitransfer_result_invalid")
    return result


def preflight_omnitransfer() -> dict[str, Any]:
    """Exercise the candidate API while keeping readiness policy in OmniFlow."""

    module = load_omnitransfer()
    rank_candidates = getattr(module, "rank_action_candidates", None)
    if not callable(rank_candidates):
        raise RuntimeError("omnitransfer_candidate_ranking_unavailable")
    source_xml = (
        '<hierarchy bounds="[0,0][100,100]">'
        '<node resource-id="preflight:id/search" text="Search" '
        'class="android.widget.Button" clickable="true" enabled="true" '
        'bounds="[10,20][50,60]" />'
        "</hierarchy>"
    )
    target_xml = (
        '<hierarchy bounds="[0,0][200,400]">'
        '<node resource-id="preflight:id/search" text="Search" '
        'class="android.widget.Button" clickable="true" enabled="true" '
        'bounds="[40,100][120,260]" />'
        "</hierarchy>"
    )
    result = rank_candidates(
        source_xml=source_xml,
        target_xml=target_xml,
        source_point=(30.0, 40.0),
    )
    if not isinstance(result, dict):
        raise RuntimeError("omnitransfer_result_invalid")
    candidates = result.get("candidates")
    if not isinstance(candidates, list) or not candidates:
        reason = str(result.get("reason") or "candidates_missing")
        error = str(result.get("error") or "")
        raise RuntimeError(
            f"omnitransfer_preflight_failed:{reason}:{error}".rstrip(":")
        )
    return {
        "ready": True,
        "backend": str(result.get("matcher_backend") or "unknown"),
        "mapping_mode": str(result.get("mapping_mode") or ""),
        "candidate_ranking_schema": str(result.get("schema_version") or ""),
        "matcher_release": str(result.get("matcher_release") or ""),
        "matcher_checkpoint_sha256": str(
            result.get("matcher_checkpoint_sha256") or ""
        ),
        "matcher_feature_schema": str(result.get("matcher_feature_schema") or ""),
        "matcher_feature_schema_sha256": str(
            result.get("matcher_feature_schema_sha256") or ""
        ),
    }


def _select_transfer_candidate(
    ranking: dict[str, Any],
    request: dict[str, Any],
) -> dict[str, Any]:
    """Apply OmniFlow policy to an OmniTransfer candidate ranking."""

    source_package = str(request.get("source_package_name") or "").strip()
    target_package = str(request.get("target_package_name") or "").strip()
    if source_package and target_package and source_package != target_package:
        return {
            **ranking,
            "mapped": False,
            "mapping_mode": "page_identity",
            "reason": "target_page_identity_mismatch",
        }
    candidates = ranking.get("candidates")
    if not isinstance(candidates, list):
        raise RuntimeError("omnitransfer_candidates_invalid")
    if not candidates:
        return {
            **ranking,
            "mapped": False,
            "reason": str(
                ranking.get("reason")
                or ranking.get("status")
                or "target_candidates_missing"
            ),
        }
    selected = candidates[0]
    if not isinstance(selected, dict):
        raise RuntimeError("omnitransfer_candidate_invalid")
    try:
        new_x = float(selected["new_x"])
        new_y = float(selected["new_y"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError("omnitransfer_candidate_coordinates_invalid") from error
    bounds = list(selected.get("bbox") or ())
    return {
        **ranking,
        "mapped": True,
        "selection_policy": "omniflow_top_candidate",
        "new_x": new_x,
        "new_y": new_y,
        "target_candidate_id": str(selected.get("candidate_id") or ""),
        "target_bbox": bounds,
        "target_center": (
            [(float(bounds[0]) + float(bounds[2])) / 2.0,
             (float(bounds[1]) + float(bounds[3])) / 2.0]
            if len(bounds) == 4
            else []
        ),
    }
