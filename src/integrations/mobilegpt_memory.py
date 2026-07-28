#!/usr/bin/env python3
"""Convert OmniFlow v2 Functions into frozen MobileGPT memory."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import importlib
import json
from pathlib import Path
import re
import shutil
import sys
from typing import Any, Iterable
import xml.etree.ElementTree as ET

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

Action = importlib.import_module("omniflow.core.model").Action
Function = importlib.import_module("omniflow.core.model").Function
FunctionStore = importlib.import_module("omniflow.functions.store").FunctionStore
transfer_module = importlib.import_module("omniflow.transfer.runtime")
load_transfer_state_catalog = transfer_module.load_transfer_state_catalog
transfer_action = transfer_module.transfer_action


ACTION_FUNCTION_ID = re.compile(r"^action_\d{3}_")
BINDING_TARGET = re.compile(
    r"^\$\.steps\[(?P<index>\d+)]\.action\.args"
    r"(?P<tail>(?:\.[A-Za-z_][A-Za-z0-9_-]*)*)$"
)
PLACEHOLDER = re.compile(r"<[^<>]+__-?\d+>")
COORDINATE_KEYS = {
    "bounds",
    "coordinates",
    "oob_bounds",
    "x",
    "x1",
    "x2",
    "y",
    "y1",
    "y2",
}
ACTION_HEADERS = ["subtask_name", "step", "action", "example"]
SUBTASK_HEADERS = ["name", "description", "parameters", "example"]
AVAILABLE_HEADERS = ["name", "description", "parameters"]


@dataclass(frozen=True)
class ColdActionSlot:
    page_index: int
    action: dict[str, Any]
    source_identity: frozenset[str]
    screen: str = ""


@dataclass(frozen=True)
class ColdAlignment:
    slots: tuple[ColdActionSlot, ...]
    matched_cold_action_count: int
    grounded_function_action_count: int


@dataclass(frozen=True)
class ElementTarget:
    element: ET.Element
    path: str
    bounds: tuple[float, float, float, float]
    identity: frozenset[str]
    resource_id: str
    offset: tuple[float, float] = (0.5, 0.5)


@dataclass(frozen=True)
class FunctionAction:
    source_index: int
    action: Action


@dataclass(frozen=True)
class Placement:
    function: Function
    start: int
    stop: int


@dataclass(frozen=True)
class NativeSegment:
    function: Function
    name: str
    complete_start: int
    complete_stop: int
    function_start: int
    function_stop: int
    page_index: int


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ValueError(f"mobilegpt_csv_missing:{path}")
    with path.open(encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _source_target_descriptions(store_path: Path) -> dict[tuple[str, int], str]:
    try:
        payload = json.loads(store_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"mobilegpt_store_json_invalid:{error}") from error
    functions = payload.get("functions") if isinstance(payload, dict) else None
    if not isinstance(functions, dict):
        raise ValueError("mobilegpt_store_functions_invalid")
    descriptions: dict[tuple[str, int], str] = {}
    for raw_function_id, raw_function in functions.items():
        if not isinstance(raw_function, dict):
            continue
        function_id = str(
            raw_function.get("function_id") or raw_function_id or ""
        ).strip()
        steps = raw_function.get("steps")
        if not function_id or not isinstance(steps, list):
            continue
        for fallback_index, raw_step in enumerate(steps):
            if not isinstance(raw_step, dict):
                continue
            try:
                step_index = int(raw_step.get("step_index", fallback_index))
            except (TypeError, ValueError):
                continue
            raw_action = raw_step.get("action")
            action_args = (
                raw_action.get("args") if isinstance(raw_action, dict) else None
            )
            target_description = str(
                (
                    action_args.get("target_description")
                    if isinstance(action_args, dict)
                    else ""
                )
                or ""
            ).strip()
            if target_description:
                descriptions[(function_id, step_index)] = target_description
    return descriptions


def _write_csv(
    path: Path,
    headers: list[str],
    rows: Iterable[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({header: row.get(header, "") for header in headers})


def _load_json_object(value: str, *, error: str) -> dict[str, Any]:
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(error) from exc
    if not isinstance(parsed, dict):
        raise ValueError(error)
    return parsed


def _integer(value: Any, *, error: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(error) from exc


def _mobile_name(action: Action) -> str:
    action_type = action.tool.strip()
    if action_type == "click":
        return "click"
    if action_type == "input_text":
        return "input"
    if action_type == "long_press":
        return "long-click"
    if action_type in {"scroll", "swipe"}:
        return "scroll"
    if action_type == "press_key":
        key = (
            str(
                action.args.get("key")
                or action.args.get("keycode")
                or action.args.get("name")
                or ""
            )
            .strip()
            .lower()
        )
        if key in {"4", "back", "keycode_back"}:
            return "back"
        raise ValueError(f"mobilegpt_unsupported_press_key:{key}")
    if action_type == "open_app":
        return "open_app"
    raise ValueError(f"mobilegpt_unsupported_function_action:{action_type}")


def _function_actions(function: Function) -> list[FunctionAction]:
    converted: list[FunctionAction] = []
    for source_index, action in enumerate(function.actions):
        mobile_name = _mobile_name(action)
        if mobile_name == "open_app":
            if source_index != 0:
                raise ValueError(
                    f"mobilegpt_mid_task_open_app_unsupported:{function.id}:{source_index}"
                )
            continue
        converted.append(FunctionAction(source_index, action))
    if not converted:
        raise ValueError(f"mobilegpt_function_has_no_primitive_actions:{function.id}")
    return converted


def _bound_tails(function: Function) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for binding in function.bindings:
        source = str(binding.get("source") or "")
        target = str(binding.get("target") or "")
        match = BINDING_TARGET.fullmatch(target)
        if match is None or not source.startswith("$.arguments."):
            raise ValueError(
                f"mobilegpt_binding_not_representable:{function.id}:{target}"
            )
        parameter = source[len("$.arguments.") :]
        if "." in parameter or "[" in parameter:
            raise ValueError(
                f"mobilegpt_nested_parameter_not_representable:{function.id}:{source}"
            )
        action_index = int(match.group("index"))
        result.setdefault(action_index, {})[match.group("tail")] = parameter
    return result


def _actions_compatible(
    complete_action: FunctionAction,
    subtask_action: FunctionAction,
) -> bool:
    return _mobile_name(complete_action.action) == _mobile_name(subtask_action.action)


def _matching_starts(complete: Function, subtask: Function) -> list[int]:
    complete_actions = _function_actions(complete)
    subtask_actions = _function_actions(subtask)
    if len(subtask_actions) > len(complete_actions):
        return []
    starts: list[int] = []
    for start in range(len(complete_actions) - len(subtask_actions) + 1):
        if all(
            _actions_compatible(
                complete_actions[start + offset],
                subtask_action,
            )
            for offset, subtask_action in enumerate(subtask_actions)
        ):
            starts.append(start)
    return starts


def _select_functions(
    store: FunctionStore,
    task_function_id: str,
) -> tuple[Function, list[Function]]:
    functions = [
        function
        for function in store.list_functions(limit=500)
        if ACTION_FUNCTION_ID.match(function.id) is None
    ]
    if not functions:
        raise ValueError("mobilegpt_semantic_functions_required")
    by_id = {function.id: function for function in functions}
    if task_function_id:
        task_function = by_id.get(task_function_id)
        if task_function is None:
            raise ValueError(f"mobilegpt_task_function_missing:{task_function_id}")
    elif len(functions) == 1:
        task_function = functions[0]
    else:
        candidates = [
            candidate
            for candidate in functions
            if all(
                other is candidate or bool(_matching_starts(candidate, other))
                for other in functions
            )
        ]
        maximum = max(
            (len(_function_actions(candidate)) for candidate in candidates),
            default=0,
        )
        candidates = [
            candidate
            for candidate in candidates
            if len(_function_actions(candidate)) == maximum
        ]
        if len(candidates) != 1:
            raise ValueError(
                "mobilegpt_complete_function_ambiguous:use_--task-function-id"
            )
        task_function = candidates[0]
    subtasks = [function for function in functions if function.id != task_function.id]
    return task_function, subtasks


def select_complete_function(
    store_path: str | Path,
    *,
    task_function_id: str = "",
) -> Function:
    store = FunctionStore(Path(store_path).expanduser().resolve())
    if store.load_errors:
        raise ValueError(
            "mobilegpt_function_store_invalid:"
            + ",".join(sorted(store.load_errors))
        )
    task_function, _ = _select_functions(store, task_function_id)
    return task_function


def _parse_explicit_placements(values: list[str]) -> dict[str, int]:
    placements: dict[str, int] = {}
    for value in values:
        function_id, separator, raw_start = value.rpartition(":")
        if not separator or not function_id:
            raise ValueError(f"mobilegpt_subtask_placement_invalid:{value}")
        start = _integer(
            raw_start,
            error=f"mobilegpt_subtask_placement_invalid:{value}",
        )
        if start < 0 or function_id in placements:
            raise ValueError(f"mobilegpt_subtask_placement_invalid:{value}")
        placements[function_id] = start
    return placements


def _place_subtasks(
    task_function: Function,
    subtask_functions: list[Function],
    explicit: dict[str, int],
) -> list[Placement]:
    unknown = sorted(set(explicit) - {function.id for function in subtask_functions})
    if unknown:
        raise ValueError(f"mobilegpt_subtask_placement_unknown:{','.join(unknown)}")
    task_action_count = len(_function_actions(task_function))
    placements: list[Placement] = []
    for function in subtask_functions:
        action_count = len(_function_actions(function))
        starts = _matching_starts(task_function, function)
        if function.id in explicit:
            start = explicit[function.id]
            if start not in starts:
                raise ValueError(
                    f"mobilegpt_subtask_placement_mismatch:{function.id}:{start}"
                )
        elif len(starts) == 1:
            start = starts[0]
        elif not starts:
            raise ValueError(
                f"mobilegpt_subtask_not_in_complete_function:{function.id}"
            )
        else:
            raise ValueError(
                f"mobilegpt_subtask_placement_ambiguous:{function.id}:{starts}"
            )
        stop = start + action_count
        if stop > task_action_count:
            raise ValueError(f"mobilegpt_subtask_placement_out_of_range:{function.id}")
        placements.append(Placement(function, start, stop))
    return sorted(
        placements, key=lambda item: (item.start, item.stop, item.function.id)
    )


def _cold_task_row(
    memory_root: Path,
    cold_task_name: str,
) -> dict[str, str]:
    rows = _read_csv(memory_root / "tasks.csv")
    if cold_task_name:
        matches = [row for row in rows if row.get("name") == cold_task_name]
    else:
        matches = rows
    if len(matches) != 1:
        raise ValueError("mobilegpt_cold_task_ambiguous:use_--cold-task-name")
    return matches[0]


def _package_name(function: Function) -> str:
    for action in function.actions:
        if action.tool == "open_app":
            package = str(
                action.args.get("package_name") or action.args.get("app_name") or ""
            ).strip()
            if package:
                return package
    raise ValueError(f"mobilegpt_task_package_missing:{function.id}")


def _cold_app_root(
    memory_root: Path,
    app_label: str,
    package_name: str,
) -> Path:
    for name in (app_label, package_name):
        candidate = memory_root / name
        if (candidate / "tasks.csv").is_file() and (candidate / "pages.csv").is_file():
            return candidate
    candidates = [
        path
        for path in memory_root.iterdir()
        if path.is_dir()
        and (path / "tasks.csv").is_file()
        and (path / "pages.csv").is_file()
        and (path / "hierarchy.csv").is_file()
    ]
    if len(candidates) != 1:
        raise ValueError("mobilegpt_cold_app_ambiguous")
    return candidates[0]


def _cold_path(app_root: Path, task_name: str) -> dict[int, list[str]]:
    rows = _read_csv(app_root / "tasks.csv")
    matches = [row for row in rows if row.get("name") == task_name]
    if len(matches) != 1 and len(rows) == 1:
        matches = rows
    if len(matches) != 1:
        raise ValueError("mobilegpt_cold_task_path_ambiguous")
    raw_path = _load_json_object(
        str(matches[0].get("path") or ""),
        error="mobilegpt_cold_task_path_invalid",
    )
    path: dict[int, list[str]] = {}
    for raw_index, names in raw_path.items():
        page_index = _integer(raw_index, error="mobilegpt_cold_page_index_invalid")
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError("mobilegpt_cold_task_path_invalid")
        path[page_index] = list(names)
    return dict(sorted(path.items()))


def _contains_coordinate(value: Any) -> str | None:
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key) in COORDINATE_KEYS:
                return str(key)
            found = _contains_coordinate(item)
            if found:
                return found
    if isinstance(value, list):
        for item in value:
            found = _contains_coordinate(item)
            if found:
                return found
    return None


def _normalized_index(value: Any) -> str:
    if isinstance(value, list):
        if len(value) != 1:
            return ""
        value = value[0]
    return str(value if value is not None else "").strip()


def _element_bounds(element: ET.Element) -> tuple[float, float, float, float] | None:
    values = [
        float(item)
        for item in re.findall(
            r"-?\d+(?:\.\d+)?",
            str(element.attrib.get("bounds") or ""),
        )
    ]
    if len(values) != 4 or values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values[0], values[1], values[2], values[3]


def _resource_identity(element: ET.Element) -> str:
    value = str(
        element.attrib.get("resource-id") or element.attrib.get("id") or ""
    ).strip()
    return value.rsplit("/", 1)[-1].casefold()


def _element_identity(element: ET.Element) -> set[str]:
    values: set[str] = set()
    for descendant in element.iter():
        candidates = [
            descendant.text,
            descendant.attrib.get("text"),
            descendant.attrib.get("description"),
            descendant.attrib.get("content-desc"),
        ]
        for candidate in candidates:
            normalized = str(candidate or "").strip().casefold()
            if normalized:
                values.add(normalized)
    return values


def _screen_targets(screen: str) -> list[ElementTarget]:
    try:
        root = ET.fromstring(str(screen or "").strip())
    except ET.ParseError:
        return []
    targets: list[ElementTarget] = []

    def visit(element: ET.Element, path: str) -> None:
        bounds = _element_bounds(element)
        if bounds is not None:
            targets.append(
                ElementTarget(
                    element=element,
                    path=path,
                    bounds=bounds,
                    identity=frozenset(_element_identity(element)),
                    resource_id=_resource_identity(element),
                )
            )
        for index, child in enumerate(list(element)):
            visit(child, f"{path}.{index}")

    visit(root, "0")
    return targets


def _screen_index_target(screen: str, index: Any) -> ElementTarget | None:
    normalized = _normalized_index(index)
    if not normalized:
        return None
    try:
        root = ET.fromstring(str(screen or "").strip())
    except ET.ParseError:
        return None
    matches: list[ElementTarget] = []

    def visit(element: ET.Element, path: str) -> None:
        if str(element.attrib.get("index") or "").strip() == normalized:
            matches.append(
                ElementTarget(
                    element=element,
                    path=path,
                    bounds=_element_bounds(element) or (0.0, 0.0, 1.0, 1.0),
                    identity=frozenset(_element_identity(element)),
                    resource_id=_resource_identity(element),
                )
            )
        for child_index, child in enumerate(list(element)):
            visit(child, f"{path}.{child_index}")

    visit(root, "0")
    return matches[0] if len(matches) == 1 else None


def _action_example_screen(row: dict[str, str], fallback: str) -> str:
    try:
        example = _load_json_object(
            str(row.get("example") or "{}"),
            error="mobilegpt_cold_action_example_invalid",
        )
    except ValueError:
        return fallback
    screen = str(example.get("screen") or "").strip()
    return screen or fallback


def _cold_slots(
    app_root: Path, task_path: dict[int, list[str]]
) -> list[ColdActionSlot]:
    page_screens = {
        _integer(row.get("index"), error="mobilegpt_page_index_invalid"): str(
            row.get("screen") or ""
        )
        for row in _read_csv(app_root / "pages.csv")
    }
    slots: list[ColdActionSlot] = []
    for page_index, subtask_names in task_path.items():
        action_rows = _read_csv(app_root / "pages" / str(page_index) / "actions.csv")
        for subtask_name in subtask_names:
            if subtask_name == "finish":
                continue
            matching_rows = [
                row for row in action_rows if row.get("subtask_name") == subtask_name
            ]
            matching_rows.sort(
                key=lambda row: _integer(
                    row.get("step"),
                    error="mobilegpt_cold_action_step_invalid",
                )
            )
            for row in matching_rows:
                action = _load_json_object(
                    str(row.get("action") or ""),
                    error="mobilegpt_cold_action_invalid",
                )
                if str(action.get("name") or "") == "finish":
                    continue
                coordinate = _contains_coordinate(action.get("parameters") or {})
                if coordinate:
                    raise ValueError(
                        f"mobilegpt_cold_coordinate_replay_forbidden:{coordinate}"
                    )
                screen = _action_example_screen(
                    row,
                    page_screens.get(page_index, ""),
                )
                parameters = action.get("parameters")
                target = (
                    _screen_index_target(screen, parameters.get("index"))
                    if isinstance(parameters, dict)
                    else None
                )
                if target is not None and target.resource_id:
                    action = {
                        **action,
                        "parameters": {
                            **dict(parameters or {}),
                            "id": target.resource_id,
                        },
                    }
                identity = set(_screen_action_identity(action, screen))
                if target is not None and target.resource_id:
                    identity.add(target.resource_id)
                slots.append(
                    ColdActionSlot(
                        page_index,
                        action,
                        frozenset(identity),
                        screen,
                    )
                )
    if not slots:
        raise ValueError("mobilegpt_cold_primitive_actions_required")
    return slots


def _screen_action_identity(action: dict[str, Any], screen: str) -> set[str]:
    parameters = action.get("parameters")
    if not isinstance(parameters, dict):
        return set()
    target_index = _normalized_index(parameters.get("index"))
    if not target_index:
        return set()
    try:
        root = ET.fromstring(str(screen or "").strip())
    except ET.ParseError:
        return set()
    target = next(
        (
            element
            for element in root.iter()
            if str(element.attrib.get("index") or "").strip() == target_index
        ),
        None,
    )
    if target is None:
        return set()
    values: set[str] = set()
    for element in target.iter():
        candidates = [
            element.text,
            element.attrib.get("text"),
            element.attrib.get("description"),
            element.attrib.get("content-desc"),
        ]
        for candidate in candidates:
            normalized = str(candidate or "").strip().casefold()
            if normalized:
                values.add(normalized)
    return values


def _cold_slot_matches(
    function_action: FunctionAction,
    slot: ColdActionSlot,
) -> bool:
    expected = _mobile_name(function_action.action)
    actual = str(slot.action.get("name") or "").strip()
    return expected == actual


def _cold_target_identity(
    slot: ColdActionSlot,
    source_target_description: str = "",
) -> dict[str, str]:
    parameters = slot.action.get("parameters")
    if not isinstance(parameters, dict):
        return {}
    preferred = str(source_target_description or "").strip()
    if preferred:
        normalized = preferred.casefold()
        if PLACEHOLDER.search(normalized) is None and normalized in slot.source_identity:
            return {"description": normalized}
        return {}
    for key in ("description", "text", "id"):
        value = str(parameters.get(key) or "").strip()
        normalized = value.rsplit("/", 1)[-1].casefold()
        if (
            normalized
            and PLACEHOLDER.search(normalized) is None
            and normalized in slot.source_identity
        ):
            return {key: value}
    if len(slot.source_identity) == 1:
        return {"description": next(iter(slot.source_identity))}
    return {}


def _align_complete_cold_slots(
    function_actions: list[FunctionAction],
    cold_slots: list[ColdActionSlot],
) -> list[ColdActionSlot]:
    @lru_cache(maxsize=None)
    def search(function_index: int, cold_start: int) -> tuple[tuple[int, ...], ...]:
        if function_index == len(function_actions):
            return ((),)
        matches: list[tuple[int, ...]] = []
        function_action = function_actions[function_index]
        for cold_index in range(cold_start, len(cold_slots)):
            if not _cold_slot_matches(
                function_action,
                cold_slots[cold_index],
            ):
                continue
            for suffix in search(function_index + 1, cold_index + 1):
                matches.append((cold_index, *suffix))
                if len(matches) == 2:
                    return tuple(matches)
        return tuple(matches)

    alignments = search(0, 0)
    if not alignments:
        raise ValueError(
            "mobilegpt_cold_primitive_alignment_missing:"
            f"function={len(function_actions)}:cold={len(cold_slots)}"
        )
    if len(alignments) != 1:
        raise ValueError(f"mobilegpt_cold_primitive_alignment_ambiguous:{alignments}")
    return [cold_slots[index] for index in alignments[0]]


def _source_action_target(
    function: Function,
    function_action: FunctionAction,
    states: dict[str, dict[str, Any]],
) -> tuple[ElementTarget, dict[str, Any]]:
    step = function.steps[function_action.source_index]
    state = states.get(step.source_state_id)
    if not isinstance(state, dict):
        raise ValueError(
            f"mobilegpt_source_state_missing:{function.id}:"
            f"{function_action.source_index}:{step.source_state_id}"
        )
    source_xml = str(state.get("xml") or "").strip()
    targets = _screen_targets(source_xml)
    if not targets:
        raise ValueError(
            f"mobilegpt_source_state_xml_invalid:{function.id}:"
            f"{function_action.source_index}:{step.source_state_id}"
        )
    action = function_action.action
    if all(action.args.get(key) is not None for key in ("x", "y")):
        display = state.get("display")
        if not isinstance(display, dict) or set(display) != {"width", "height"}:
            raise ValueError(
                f"mobilegpt_source_state_display_missing:{function.id}:"
                f"{function_action.source_index}:{step.source_state_id}"
            )
        try:
            point = (
                float(action.args["x"]) / 1000.0 * float(display["width"]),
                float(action.args["y"]) / 1000.0 * float(display["height"]),
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mobilegpt_source_action_point_invalid:{function.id}:"
                f"{function_action.source_index}"
            ) from exc
        candidates = [
            target
            for target in targets
            if target.bounds[0] <= point[0] <= target.bounds[2]
            and target.bounds[1] <= point[1] <= target.bounds[3]
        ]
        if not candidates:
            raise ValueError(
                f"mobilegpt_source_action_target_missing:{function.id}:"
                f"{function_action.source_index}"
            )
        target = min(
            candidates,
            key=lambda item: (
                (item.bounds[2] - item.bounds[0])
                * (item.bounds[3] - item.bounds[1]),
                not bool(item.resource_id or item.identity),
                -item.path.count("."),
                item.path,
            ),
        )
    elif _mobile_name(action) == "input":
        focused = [
            target
            for target in targets
            if str(target.element.attrib.get("focused") or "").casefold() == "true"
            and (
                target.element.tag.rsplit("}", 1)[-1] == "input"
                or str(target.element.attrib.get("class") or "").endswith(
                    "EditText"
                )
            )
        ]
        if len(focused) != 1:
            raise ValueError(
                f"mobilegpt_source_focused_input_ambiguous:{function.id}:"
                f"{function_action.source_index}:{len(focused)}"
            )
        target = focused[0]
        point = (
            (target.bounds[0] + target.bounds[2]) / 2.0,
            (target.bounds[1] + target.bounds[3]) / 2.0,
        )
    else:
        raise ValueError(
            f"mobilegpt_source_action_target_unsupported:{function.id}:"
            f"{function_action.source_index}"
        )
    width = target.bounds[2] - target.bounds[0]
    height = target.bounds[3] - target.bounds[1]
    offset = (
        (point[0] - target.bounds[0]) / width,
        (point[1] - target.bounds[1]) / height,
    )
    try:
        root = ET.fromstring(source_xml)
    except ET.ParseError as exc:
        raise ValueError("mobilegpt_source_state_xml_invalid") from exc
    elements_by_path: dict[str, ET.Element] = {}

    def collect(element: ET.Element, path: str) -> None:
        elements_by_path[path] = element
        for index, child in enumerate(list(element)):
            collect(child, f"{path}.{index}")

    collect(root, "0")
    semantic_element = elements_by_path[target.path]
    parent_path = target.path.rpartition(".")[0]
    while parent_path:
        parent = elements_by_path[parent_path]
        if str(parent.attrib.get("clickable") or "").casefold() == "true":
            semantic_element = parent
            break
        parent_path = parent_path.rpartition(".")[0]
    return (
        ElementTarget(
            element=target.element,
            path=target.path,
            bounds=target.bounds,
            identity=frozenset(_element_identity(semantic_element)),
            resource_id=target.resource_id,
            offset=offset,
        ),
        state,
    )


def _same_source_target(source: ElementTarget, target: ElementTarget) -> bool:
    return bool(
        source.resource_id
        and target.resource_id
        and source.resource_id == target.resource_id
    ) or bool(source.identity & target.identity)


def _grounded_slot_matches(
    function: Function,
    function_action: FunctionAction,
    slot: ColdActionSlot,
    states: dict[str, dict[str, Any]],
) -> bool:
    if not _cold_slot_matches(function_action, slot):
        return False
    parameters = slot.action.get("parameters")
    target = (
        _screen_index_target(slot.screen, parameters.get("index"))
        if isinstance(parameters, dict)
        else None
    )
    if target is None:
        return False
    source, _ = _source_action_target(function, function_action, states)
    return _same_source_target(source, target)


def _best_partial_alignment(
    function: Function,
    function_actions: list[FunctionAction],
    cold_slots: list[ColdActionSlot],
    states: dict[str, dict[str, Any]],
) -> tuple[tuple[int, int], ...]:
    @lru_cache(maxsize=None)
    def search(
        function_index: int,
        cold_index: int,
    ) -> tuple[int, tuple[tuple[tuple[int, int], ...], ...]]:
        if function_index == len(function_actions) or cold_index == len(cold_slots):
            return 0, ((),)
        options: list[
            tuple[int, tuple[tuple[tuple[int, int], ...], ...]]
        ] = [
            search(function_index + 1, cold_index),
            search(function_index, cold_index + 1),
        ]
        if _grounded_slot_matches(
            function,
            function_actions[function_index],
            cold_slots[cold_index],
            states,
        ):
            score, suffixes = search(function_index + 1, cold_index + 1)
            options.append(
                (
                    score + 1,
                    tuple(
                        ((function_index, cold_index), *suffix)
                        for suffix in suffixes
                    ),
                )
            )
        best_score = max(score for score, _ in options)
        alignments: list[tuple[tuple[int, int], ...]] = []
        for score, values in options:
            if score != best_score:
                continue
            for value in values:
                if value not in alignments:
                    alignments.append(value)
                if len(alignments) == 2:
                    return best_score, tuple(alignments)
        return best_score, tuple(alignments)

    score, alignments = search(0, 0)
    if score <= 0 or not alignments:
        raise ValueError("mobilegpt_cold_grounded_alignment_missing")
    if len(alignments) != 1:
        raise ValueError(
            f"mobilegpt_cold_grounded_alignment_ambiguous:{alignments}"
        )
    return alignments[0]


def _native_action_target(
    screen: str,
    mapped_bounds: Any,
    mobile_name: str,
) -> ElementTarget | None:
    if not isinstance(mapped_bounds, list) or len(mapped_bounds) != 4:
        return None
    try:
        expected = tuple(float(item) for item in mapped_bounds)
        root = ET.fromstring(str(screen or "").strip())
    except (TypeError, ValueError, ET.ParseError):
        return None
    parents: dict[ET.Element, ET.Element] = {
        child: parent for parent in root.iter() for child in parent
    }
    paths: dict[ET.Element, str] = {}

    def collect(element: ET.Element, path: str) -> None:
        paths[element] = path
        for index, child in enumerate(list(element)):
            collect(child, f"{path}.{index}")

    collect(root, "0")
    matches = [
        element
        for element in root.iter()
        if (bounds := _element_bounds(element)) is not None
        and all(abs(left - right) <= 1e-6 for left, right in zip(bounds, expected))
    ]
    if not matches:
        return None
    element = max(matches, key=lambda item: paths[item].count("."))

    def is_actionable(candidate: ET.Element) -> bool:
        tag = candidate.tag.rsplit("}", 1)[-1]
        class_name = str(candidate.attrib.get("class") or "")
        if mobile_name == "input":
            return tag == "input" or class_name.endswith("EditText")
        return (
            tag in {"button", "input"}
            or str(candidate.attrib.get("clickable") or "").casefold() == "true"
        )

    while element is not None and not is_actionable(element):
        element = parents.get(element)
    if element is None or not _normalized_index(element.attrib.get("index")):
        return None
    bounds = _element_bounds(element)
    if bounds is None:
        return None
    return ElementTarget(
        element=element,
        path=paths[element],
        bounds=bounds,
        identity=frozenset(_element_identity(element)),
        resource_id=_resource_identity(element),
    )


def _native_identity_parameters(target: ElementTarget) -> dict[str, str]:
    if target.resource_id:
        return {"id": target.resource_id}
    for key in ("description", "content-desc"):
        value = str(target.element.attrib.get(key) or "").strip()
        if value and PLACEHOLDER.search(value) is None:
            return {"description": value}
    identities = sorted(
        value for value in target.identity if PLACEHOLDER.search(value) is None
    )
    if len(identities) == 1:
        return {"description": identities[0]}
    return {}


def _ground_missing_action(
    function: Function,
    function_action: FunctionAction,
    states: dict[str, dict[str, Any]],
    page_screens: dict[int, str],
    page_indexes: list[int],
) -> ColdActionSlot:
    source, state = _source_action_target(function, function_action, states)
    mobile_name = _mobile_name(function_action.action)
    candidates: list[ColdActionSlot] = []
    for page_index in page_indexes:
        screen = page_screens[page_index]
        result = transfer_action(
            source_xml=str(state.get("xml") or ""),
            target_xml=screen,
            source_element_id=source.path,
            source_offset=source.offset,
            source_package_name=str(state.get("package_name") or ""),
            target_package_name=str(state.get("package_name") or ""),
            source_activity_name=str(state.get("activity_name") or ""),
            target_activity_name=str(state.get("activity_name") or ""),
            action_type=function_action.action.tool,
            top_k=3,
        )
        if result.get("mapped") is not True:
            continue
        target = _native_action_target(
            screen,
            result.get("target_bbox"),
            mobile_name,
        )
        if target is None or not _same_source_target(source, target):
            continue
        identity = _native_identity_parameters(target)
        if not identity:
            continue
        parameters: dict[str, Any] = {
            "index": _normalized_index(target.element.attrib.get("index")),
            **identity,
        }
        candidates.append(
            ColdActionSlot(
                page_index=page_index,
                action={"name": mobile_name, "parameters": parameters},
                source_identity=frozenset(
                    {*target.identity, *identity.values()}
                ),
                screen=screen,
            )
        )
    distinct = {
        (
            candidate.page_index,
            _normalized_index(candidate.action["parameters"].get("index")),
        ): candidate
        for candidate in candidates
    }
    if not distinct:
        raise ValueError(
            f"mobilegpt_missing_action_grounding_failed:{function.id}:"
            f"{function_action.source_index}"
        )
    if len(distinct) != 1:
        raise ValueError(
            f"mobilegpt_missing_action_grounding_ambiguous:{function.id}:"
            f"{function_action.source_index}:{sorted(distinct)}"
        )
    return next(iter(distinct.values()))


def _align_cold_slots(
    function: Function,
    function_actions: list[FunctionAction],
    cold_slots: list[ColdActionSlot],
    states: dict[str, dict[str, Any]],
    page_screens: dict[int, str],
) -> ColdAlignment:
    try:
        aligned = _align_complete_cold_slots(function_actions, cold_slots)
    except ValueError as exc:
        if not str(exc).startswith("mobilegpt_cold_primitive_alignment_missing:"):
            raise
    else:
        return ColdAlignment(
            slots=tuple(aligned),
            matched_cold_action_count=len(function_actions),
            grounded_function_action_count=0,
        )
    if not states:
        raise ValueError("mobilegpt_transfer_states_required_for_missing_actions")
    pairs = _best_partial_alignment(
        function,
        function_actions,
        cold_slots,
        states,
    )
    matched = {function_index: cold_index for function_index, cold_index in pairs}
    slots: list[ColdActionSlot] = []
    for function_index, function_action in enumerate(function_actions):
        cold_index = matched.get(function_index)
        if cold_index is not None:
            slots.append(cold_slots[cold_index])
            continue
        previous = [pair for pair in pairs if pair[0] < function_index]
        following = [pair for pair in pairs if pair[0] > function_index]
        if not previous or not following:
            raise ValueError(
                f"mobilegpt_missing_action_page_unbounded:{function.id}:"
                f"{function_action.source_index}"
            )
        previous_page = cold_slots[previous[-1][1]].page_index
        following_page = cold_slots[following[0][1]].page_index
        eligible_pages = [
            page_index
            for page_index in sorted(page_screens)
            if previous_page <= page_index <= following_page
        ]
        slots.append(
            _ground_missing_action(
                function,
                function_action,
                states,
                page_screens,
                eligible_pages,
            )
        )
    return ColdAlignment(
        slots=tuple(slots),
        matched_cold_action_count=len(pairs),
        grounded_function_action_count=len(function_actions) - len(pairs),
    )


def _parameter_questions(function: Function) -> dict[str, str]:
    properties = function.input_schema.get("properties") or {}
    questions: dict[str, str] = {}
    for name, raw_schema in properties.items():
        schema = raw_schema if isinstance(raw_schema, dict) else {}
        questions[str(name)] = str(
            schema.get("description") or schema.get("title") or name
        ).strip()
    return questions


def _scroll_direction(params: dict[str, Any]) -> str:
    coordinate_sets = (
        ("start_x", "start_y", "end_x", "end_y"),
        ("x1", "y1", "x2", "y2"),
        ("x", "y", "end_x", "end_y"),
    )
    for x1_key, y1_key, x2_key, y2_key in coordinate_sets:
        try:
            start_x = float(params[x1_key])
            start_y = float(params[y1_key])
            end_x = float(params[x2_key])
            end_y = float(params[y2_key])
        except (KeyError, TypeError, ValueError):
            continue
        delta_x = end_x - start_x
        delta_y = end_y - start_y
        if abs(delta_x) > abs(delta_y):
            return "left" if delta_x > 0 else "right"
        return "up" if delta_y > 0 else "down"
    direction = str(params.get("direction") or "").strip().lower()
    if direction in {"down", "left", "right", "up"}:
        return direction
    raise ValueError("mobilegpt_scroll_direction_missing")


def _converted_action(
    function: Function,
    function_action: FunctionAction,
    slot: ColdActionSlot,
    source_target_description: str = "",
) -> dict[str, Any]:
    mobile_name = _mobile_name(function_action.action)
    cold_parameters = slot.action.get("parameters")
    if not isinstance(cold_parameters, dict):
        raise ValueError("mobilegpt_cold_action_parameters_invalid")
    bound = _bound_tails(function).get(function_action.source_index, {})
    identity = _cold_target_identity(slot, source_target_description)
    parameters: dict[str, Any] = {}
    if mobile_name not in {"back"}:
        if "index" not in cold_parameters:
            raise ValueError("mobilegpt_cold_action_index_required")
        target_index = _normalized_index(cold_parameters["index"])
        if not target_index:
            raise ValueError("mobilegpt_cold_action_index_required")
        if not identity:
            raise ValueError(
                f"mobilegpt_cold_target_identity_required:{function.id}:"
                f"{function_action.source_index}:page={slot.page_index}"
            )
        parameters["index"] = target_index
    if mobile_name == "input":
        parameter = bound.get(".text")
        parameters["input_text"] = (
            f"<{parameter}__-1>"
            if parameter
            else str(function_action.action.args.get("text") or "")
        )
    elif mobile_name == "scroll":
        parameters["direction"] = _scroll_direction(function_action.action.args)
    elif mobile_name == "back":
        parameters = {}
    for key, value in identity.items():
        parameters[key] = value
    representable_tails = {
        ".text",
    }
    unsupported = sorted(set(bound) - representable_tails)
    if unsupported:
        raise ValueError(
            f"mobilegpt_binding_not_representable:{function.id}:{unsupported}"
        )
    return {"name": mobile_name, "parameters": parameters}


def _action_example(
    *,
    task_function: Function,
    function: Function,
    subtask_name: str,
    parameter_names: Iterable[str],
    action: dict[str, Any],
    screen: str,
    step: int,
    action_count: int,
) -> dict[str, Any]:
    example_parameters = {
        name: f"<{name}__-1>" for name in parameter_names
    }
    return {
        "instruction": task_function.description,
        "subtask": _json(
            {"name": subtask_name, "parameters": example_parameters}
        ),
        "screen": screen,
        "response": _json(
            {
                "reasoning": "Execute the next action defined by the Function.",
                "action": action,
                "completion_rate": min(
                    99,
                    round(100 * (step + 1) / max(action_count, 1)),
                ),
            }
        ),
    }


def _subtask_definition(function: Function) -> dict[str, Any]:
    return {
        "name": function.id,
        "description": function.description,
        "parameters": _parameter_questions(function),
    }


def _partitioned(
    placements: list[Placement],
    action_count: int,
) -> bool:
    if not placements:
        return False
    cursor = 0
    for placement in placements:
        if placement.start != cursor:
            return False
        cursor = placement.stop
    return cursor == action_count


def _native_segments(
    placement: Placement,
    slots: list[ColdActionSlot],
) -> list[NativeSegment]:
    function_actions = _function_actions(placement.function)
    if placement.stop - placement.start != len(function_actions):
        raise ValueError(
            f"mobilegpt_native_segment_length_mismatch:{placement.function.id}"
        )
    boundaries: list[tuple[int, int, int]] = []
    function_start = 0
    while function_start < len(function_actions):
        complete_start = placement.start + function_start
        page_index = slots[complete_start].page_index
        function_stop = function_start + 1
        while (
            function_stop < len(function_actions)
            and slots[placement.start + function_stop].page_index == page_index
        ):
            function_stop += 1
        boundaries.append((function_start, function_stop, page_index))
        function_start = function_stop

    segments: list[NativeSegment] = []
    for ordinal, (function_start, function_stop, page_index) in enumerate(boundaries):
        name = placement.function.id
        if ordinal:
            name = (
                f"{placement.function.id}"
                f"__omniflow_continuation_{ordinal:03d}"
            )
        segments.append(
            NativeSegment(
                function=placement.function,
                name=name,
                complete_start=placement.start + function_start,
                complete_stop=placement.start + function_stop,
                function_start=function_start,
                function_stop=function_stop,
                page_index=page_index,
            )
        )
    return segments


def _native_segment_definition(segment: NativeSegment) -> dict[str, Any]:
    questions = _parameter_questions(segment.function)
    bound = _bound_tails(segment.function)
    parameter_names = {
        parameter
        for function_action in _function_actions(segment.function)[
            segment.function_start : segment.function_stop
        ]
        for parameter in bound.get(function_action.source_index, {}).values()
    }
    return {
        "name": segment.name,
        "description": segment.function.description,
        "parameters": {
            name: questions[name]
            for name in questions
            if name in parameter_names
        },
    }


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "conversion_manifest.json":
            continue
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _make_tree_owner_writable(root: Path) -> None:
    for path in [root, *root.rglob("*")]:
        if path.is_symlink():
            continue
        path.chmod(path.stat().st_mode | 0o200)


def convert_memory(
    *,
    store_path: str | Path,
    cold_memory_root: str | Path,
    output_root: str | Path,
    task_name: str,
    source_seed: int,
    task_function_id: str = "",
    cold_task_name: str = "",
    subtask_placements: Iterable[str] = (),
) -> dict[str, Any]:
    store_file = Path(store_path).expanduser().resolve()
    cold_root = Path(cold_memory_root).expanduser().resolve()
    output = Path(output_root).expanduser().resolve()
    if not store_file.is_file():
        raise FileNotFoundError(store_file)
    if not cold_root.is_dir():
        raise FileNotFoundError(cold_root)
    if output.exists():
        raise FileExistsError(f"immutable output already exists:{output}")
    normalized_task_name = str(task_name or "").strip()
    if not normalized_task_name:
        raise ValueError("androidworld_task_name_required")
    if isinstance(source_seed, bool):
        raise ValueError("source_seed_invalid")
    normalized_source_seed = int(source_seed)

    store = FunctionStore(store_file)
    source_target_descriptions = _source_target_descriptions(store_file)
    task_function, subtask_functions = _select_functions(store, task_function_id)
    explicit = _parse_explicit_placements(list(subtask_placements))
    placements = _place_subtasks(task_function, subtask_functions, explicit)
    task_actions = _function_actions(task_function)

    cold_root_task = _cold_task_row(cold_root, cold_task_name)
    source_task_name = str(cold_root_task.get("name") or "").strip()
    app_label = str(cold_root_task.get("app") or "").strip()
    package_name = _package_name(task_function)
    cold_app = _cold_app_root(cold_root, app_label, package_name)
    cold_task_path = _cold_path(cold_app, source_task_name)
    cold_slots = _cold_slots(cold_app, cold_task_path)
    page_rows = _read_csv(cold_app / "pages.csv")
    page_screens = {
        _integer(row.get("index"), error="mobilegpt_page_index_invalid"): str(
            row.get("screen") or ""
        )
        for row in page_rows
    }
    page_indexes = {
        _integer(row.get("index"), error="mobilegpt_page_index_invalid")
        for row in page_rows
    }
    transfer_state_path = store_file.parent / "transfer_states.json"
    transfer_states = (
        load_transfer_state_catalog(transfer_state_path)
        if transfer_state_path.is_file()
        else {}
    )
    alignment = _align_cold_slots(
        task_function,
        task_actions,
        cold_slots,
        transfer_states,
        page_screens,
    )
    slots = list(alignment.slots)
    final_pages = [
        page_index for page_index, names in cold_task_path.items() if "finish" in names
    ]
    if len(final_pages) != 1:
        raise ValueError("mobilegpt_cold_finish_page_required")
    final_page = final_pages[0]

    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(cold_root, output)
    _make_tree_owner_writable(output)
    output_app = output / cold_app.name

    task_parameters = _parameter_questions(task_function)
    _write_csv(
        output / "tasks.csv",
        ["name", "description", "parameters", "app"],
        [
            {
                "name": task_function.id,
                "description": task_function.description,
                "parameters": _json(task_parameters),
                "app": app_label or cold_app.name,
            }
        ],
    )

    use_semantic_path = _partitioned(placements, len(task_actions))
    path_functions = (
        placements
        if use_semantic_path
        else [Placement(task_function, 0, len(task_actions))]
    )
    task_segments = [
        segment
        for placement in path_functions
        for segment in _native_segments(placement, slots)
    ]
    task_path: dict[int, list[str]] = {}
    for segment in task_segments:
        task_path.setdefault(segment.page_index, []).append(segment.name)
    task_path.setdefault(final_page, []).append("finish")
    _write_csv(
        output_app / "tasks.csv",
        ["name", "path"],
        [
            {
                "name": task_function.id,
                "path": _json(
                    {str(key): value for key, value in sorted(task_path.items())}
                ),
            }
        ],
    )

    if final_page not in page_indexes or any(
        slot.page_index not in page_indexes for slot in slots
    ):
        raise ValueError("mobilegpt_cold_page_missing")

    all_subtasks = list(placements)
    if not use_semantic_path:
        all_subtasks.append(Placement(task_function, 0, len(task_actions)))
    all_subtasks.sort(key=lambda item: (item.start, item.stop, item.function.id))
    all_segments = [
        segment
        for placement in all_subtasks
        for segment in _native_segments(placement, slots)
    ]
    segment_names = [segment.name for segment in all_segments]
    if len(segment_names) != len(set(segment_names)):
        raise ValueError("mobilegpt_native_subtask_name_collision")

    subtask_rows: dict[int, list[dict[str, Any]]] = {
        index: [] for index in page_indexes
    }
    available_rows: dict[int, list[dict[str, Any]]] = {
        index: [] for index in page_indexes
    }
    action_rows: dict[int, list[dict[str, Any]]] = {index: [] for index in page_indexes}
    available_definitions: dict[int, list[dict[str, Any]]] = {
        index: [] for index in page_indexes
    }

    for segment in all_segments:
        definition = _native_segment_definition(segment)
        start_page = segment.page_index
        available_definitions[start_page].append(definition)
        available_rows[start_page].append(
            {**definition, "parameters": _json(definition["parameters"])}
        )
        subtask_rows[start_page].append(
            {
                **definition,
                "parameters": _json(definition["parameters"]),
                "example": "{}",
            }
        )
        function_actions = _function_actions(segment.function)
        segment_actions = function_actions[
            segment.function_start : segment.function_stop
        ]
        for segment_step, function_action in enumerate(segment_actions):
            function_step = segment.function_start + segment_step
            complete_index = segment.complete_start + segment_step
            slot = slots[complete_index]
            converted_action = _converted_action(
                segment.function,
                function_action,
                slot,
                source_target_descriptions.get(
                    (segment.function.id, function_action.source_index),
                    "",
                ),
            )
            action_rows[slot.page_index].append(
                {
                    "subtask_name": segment.name,
                    "step": segment_step,
                    "action": _json(converted_action),
                    "example": _json(
                        _action_example(
                            task_function=task_function,
                            function=segment.function,
                            subtask_name=segment.name,
                            parameter_names=definition["parameters"],
                            action=converted_action,
                            screen=slot.screen or page_screens[slot.page_index],
                            step=function_step,
                            action_count=len(function_actions),
                        )
                    ),
                }
            )
        finish_page = (
            slots[segment.complete_stop].page_index
            if segment.complete_stop < len(slots)
            else final_page
        )
        action_rows[finish_page].append(
            {
                "subtask_name": segment.name,
                "step": len(segment_actions),
                "action": _json({"name": "finish", "parameters": {}}),
                "example": "{}",
            }
        )

    for page_index in sorted(page_indexes):
        page_root = output_app / "pages" / str(page_index)
        _write_csv(
            page_root / "subtasks.csv",
            SUBTASK_HEADERS,
            subtask_rows[page_index],
        )
        _write_csv(
            page_root / "available_subtasks.csv",
            AVAILABLE_HEADERS,
            available_rows[page_index],
        )
        _write_csv(
            page_root / "actions.csv",
            ACTION_HEADERS,
            action_rows[page_index],
        )

    updated_pages: list[dict[str, Any]] = []
    for row in page_rows:
        page_index = _integer(row.get("index"), error="mobilegpt_page_index_invalid")
        updated = dict(row)
        updated["available_subtasks"] = _json(available_definitions[page_index])
        updated_pages.append(updated)
    _write_csv(
        output_app / "pages.csv",
        ["index", "available_subtasks", "trigger_uis", "extra_uis", "screen"],
        updated_pages,
    )

    manifest = {
        "schema_version": "omniflow.mobilegpt-function-conversion.v2",
        "task_name": normalized_task_name,
        "source_seed": normalized_source_seed,
        "task_function_id": task_function.id,
        "semantic_subtask_ids": [placement.function.id for placement in placements],
        "task_path_uses_semantic_subtasks": use_semantic_path,
        "task_path_checkpointed": len(task_segments) > len(path_functions),
        "native_page_checkpoint_count": len(task_segments),
        "native_subtask_segments": [
            {
                "name": segment.name,
                "function_id": segment.function.id,
                "complete_start": segment.complete_start,
                "complete_stop": segment.complete_stop,
                "page_index": segment.page_index,
            }
            for segment in task_segments
        ],
        "source_store": str(store_file),
        "source_store_sha256": hashlib.sha256(store_file.read_bytes()).hexdigest(),
        "source_cold_memory": str(cold_root),
        "source_cold_memory_sha256": _tree_sha256(cold_root),
        "output_memory_sha256": _tree_sha256(output),
        "source_task_name": source_task_name,
        "mobilegpt_task_name": task_function.id,
        "app": app_label or cold_app.name,
        "package_name": package_name,
        "primitive_action_count": len(task_actions),
        "cold_primitive_action_count": len(cold_slots),
        "matched_cold_action_count": alignment.matched_cold_action_count,
        "discarded_cold_action_count": (
            len(cold_slots) - alignment.matched_cold_action_count
        ),
        "source_grounded_primitive_action_count": (
            alignment.grounded_function_action_count
        ),
        "canonical_omnitransfer_missing_action_grounding": (
            alignment.grounded_function_action_count > 0
        ),
        "source_transfer_states": (
            str(transfer_state_path) if transfer_state_path.is_file() else ""
        ),
        "source_transfer_states_sha256": (
            hashlib.sha256(transfer_state_path.read_bytes()).hexdigest()
            if transfer_state_path.is_file()
            else ""
        ),
        "page_count": len(page_indexes),
        "target_inputs_read": False,
        "coordinate_replay": False,
        "target_identity_grounding": (
            "function_source_description_verified_against_native_cold_screen"
        ),
    }
    (output / "conversion_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", required=True, help="OmniFlow v2 store.json")
    parser.add_argument(
        "--cold-memory",
        required=True,
        help="Native MobileGPT cold memory from the source demonstration",
    )
    parser.add_argument("--output", required=True, help="New immutable memory root")
    parser.add_argument("--task-name", required=True, help="AndroidWorld task name")
    parser.add_argument("--source-seed", required=True, type=int)
    parser.add_argument("--task-function-id", default="")
    parser.add_argument("--cold-task-name", default="")
    parser.add_argument(
        "--subtask-placement",
        action="append",
        default=[],
        metavar="FUNCTION_ID:START",
        help="Resolve an ambiguous semantic Function occurrence",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    manifest = convert_memory(
        store_path=args.store,
        cold_memory_root=args.cold_memory,
        output_root=args.output,
        task_name=args.task_name,
        source_seed=args.source_seed,
        task_function_id=args.task_function_id,
        cold_task_name=args.cold_task_name,
        subtask_placements=args.subtask_placement,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
