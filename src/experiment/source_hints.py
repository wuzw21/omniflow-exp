"""Source-action / T3A hint derivation from archived RunLogs (split out of run_task).

Moved verbatim from ``src.experiment.run_task``; that module re-exports these names.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Sequence
import xml.etree.ElementTree as ET
from src.experiment.paths import resolve_path
from src.experiment.source_records import CanonicalRunLog
from src.experiment.run_task_common import (
    REPO_ROOT,
    _safe_stem,
    canonicalize_source_run_log,
)


def _t3a_hint_forbidden_values(params: dict[str, Any]) -> tuple[str, ...]:
    values: set[str] = set()

    def collect(value: Any) -> None:
        if isinstance(value, dict):
            for item in value.values():
                collect(item)
            return
        if isinstance(value, (list, tuple, set)):
            for item in value:
                collect(item)
            return
        if value is None or isinstance(value, bool):
            return
        text = re.sub(r"\s+", " ", str(value)).strip().casefold()
        if text:
            values.add(text)
            suffix = Path(text).suffix
            if suffix:
                stem = Path(text).stem.strip().casefold()
                if len(stem) >= 3:
                    values.add(stem)

    collect(params)
    return tuple(sorted(values, key=len, reverse=True))


def _t3a_hint_redacted_text(
    value: Any,
    *,
    forbidden_values: Sequence[str],
    max_len: int = 320,
) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    lowered = text.casefold()
    if not text or any(
        marker in lowered
        for marker in (
            "<hierarchy",
            "<?xml",
            "screenshot",
            ".png",
            ".jpg",
            ".jpeg",
        )
    ):
        return ""
    for forbidden in forbidden_values:
        if len(forbidden) < 3:
            continue
        text = re.sub(
            re.escape(forbidden),
            "<current task value>",
            text,
            flags=re.IGNORECASE,
        )
    if len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip() + "..."
    return text


def _t3a_hint_text(
    value: Any,
    *,
    forbidden_values: Sequence[str],
    max_len: int = 100,
) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    lowered = text.casefold()
    if not text or any(
        marker in lowered
        for marker in (
            "<hierarchy",
            "<?xml",
            "screenshot",
            ".png",
            ".jpg",
            ".jpeg",
        )
    ):
        return ""
    for forbidden in forbidden_values:
        if lowered == forbidden:
            return ""
        if len(forbidden) >= 3 and re.search(
            rf"(?<!\w){re.escape(forbidden)}(?!\w)",
            lowered,
        ):
            return ""
    if len(text) > max_len:
        text = text[: max(0, max_len - 3)].rstrip() + "..."
    return text


def _t3a_hint_step_action(step: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(step, dict):
        return "", {}
    for key in ("tool_call", "action", "selected_action"):
        action = step.get(key)
        if not isinstance(action, dict):
            continue
        name = str(
            action.get("name")
            or action.get("type")
            or action.get("tool")
            or action.get("action_type")
            or ""
        ).strip()
        raw_params = action.get("params")
        if raw_params is None:
            raw_params = action.get("arguments")
        if raw_params is None:
            raw_params = action.get("args")
        if name:
            if isinstance(raw_params, dict):
                return name, dict(raw_params)
            return name, {
                str(param_key): param_value
                for param_key, param_value in action.items()
                if param_key
                not in {"name", "type", "tool", "action_type", "params", "arguments", "args"}
            }
    for key in ("executed_actions", "actions"):
        actions = step.get(key)
        if not isinstance(actions, list):
            continue
        for action in actions:
            if not isinstance(action, dict):
                continue
            name = str(
                action.get("name") or action.get("type") or action.get("tool") or ""
            ).strip()
            raw_params = action.get("params")
            if raw_params is None:
                raw_params = action.get("arguments")
            if raw_params is None:
                raw_params = action.get("args")
            if name:
                return name, dict(raw_params or {}) if isinstance(
                    raw_params, dict
                ) else {}
    return "", {}


def _t3a_hint_target(
    params: dict[str, Any],
    *,
    forbidden_values: Sequence[str],
) -> str:
    for key in (
        "target_description",
        "target",
        "label",
        "content_desc",
        "contentDescription",
    ):
        text = _t3a_hint_text(
            params.get(key),
            forbidden_values=forbidden_values,
        )
        if text:
            return text
    for container_key in ("target_evidence", "source_context"):
        container = params.get(container_key)
        if not isinstance(container, dict):
            continue
        for key in ("target_description", "label", "text", "content_desc"):
            text = _t3a_hint_text(
                container.get(key),
                forbidden_values=forbidden_values,
            )
            if text:
                return text
    return ""


def _t3a_hint_step_summary(
    step: Any,
    *,
    forbidden_values: Sequence[str],
) -> str:
    if not isinstance(step, dict):
        return ""
    candidates: list[Any] = []
    metadata = step.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("summary"))
    candidates.extend((step.get("summary"), step.get("title"), step.get("description")))
    for action_key in ("tool_call", "action", "selected_action"):
        action = step.get(action_key)
        if isinstance(action, dict):
            candidates.extend((action.get("reason"), action.get("description")))
    actions = step.get("actions")
    if isinstance(actions, list):
        candidates.extend(
            action.get("description")
            for action in actions
            if isinstance(action, dict)
        )
    for candidate in candidates:
        summary = re.sub(
            r"^Action selected:\s*\{.*?\}\.\s*",
            "",
            str(candidate or ""),
        )
        summary = _t3a_hint_redacted_text(
            summary,
            forbidden_values=forbidden_values,
        )
        if summary:
            return summary
    return ""


def _t3a_hint_summary_target(summary: str) -> str:
    patterns = (
        r"\bClicked\s+(?:the\s+)?(?P<target>.+?)(?:\s*\(index|\s+to\b|\s*;|\s*[—–-])",
        r"\bLong-pressed\s+(?:the\s+)?(?P<target>.+?)(?:\s*\(index|\s+to\b|\s*;|\s*[—–-])",
        r"\b(?:Entered|Typed|Inserted)\s+.+?\s+(?:into|in)\s+(?:the\s+)?(?P<target>.+?)(?:\s*\(index|\s*;|\s*[—–-])",
        r"\bRenamed\s+.+?\s+in\s+(?:the\s+)?(?P<target>.+?)(?:\s*\(index|\s*;|\s*[—–-])",
    )
    for pattern in patterns:
        match = re.search(pattern, summary, flags=re.IGNORECASE)
        if not match:
            continue
        target = re.sub(r'["“”]', "", match.group("target")).strip(" \t'.,")
        if target:
            return target
    return ""


def _t3a_hint_bounds(value: str) -> tuple[int, int, int, int] | None:
    match = re.fullmatch(
        r"\[(-?\d+),(-?\d+)\]\[(-?\d+),(-?\d+)\]",
        str(value or "").strip(),
    )
    if not match:
        return None
    return tuple(int(item) for item in match.groups())


def _t3a_hint_source_node(
    step: Any,
    *,
    forbidden_values: Sequence[str],
    editable_only: bool = False,
    preceding_step: Any = None,
) -> dict[str, str]:
    if not isinstance(step, dict):
        return {}
    observation = next(
        (
            value
            for value in (
                step.get("observation"),
                step.get("before"),
                step.get("observation_before_act"),
            )
            if isinstance(value, dict)
        ),
        {},
    )
    _, params = _t3a_hint_step_action(step)
    index = params.get("index")
    elements = observation.get("ui_elements")
    if (
        isinstance(index, int)
        and not isinstance(index, bool)
        and isinstance(elements, list)
        and 0 <= index < len(elements)
        and isinstance(elements[index], dict)
    ):
        element = elements[index]
        if editable_only and not bool(element.get("is_editable")):
            return {}
        evidence: dict[str, str] = {}
        for source_key, target_key in (
            ("class_name", "class_name"),
            ("text", "text"),
            ("content_description", "content_description"),
            ("resource_id", "resource_id"),
            ("resource_name", "resource_id"),
            ("package_name", "package_name"),
        ):
            if target_key == "text" and bool(element.get("is_editable")):
                continue
            value = _t3a_hint_redacted_text(
                element.get(source_key),
                forbidden_values=forbidden_values,
                max_len=120,
            )
            if value and target_key not in evidence:
                evidence[target_key] = value
        if evidence:
            return evidence
    if isinstance(elements, list):
        x = params.get("x")
        y = params.get("y")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            ui_candidates: list[tuple[tuple[bool, bool, float], dict[str, str]]] = []
            for element in elements:
                if not isinstance(element, dict):
                    continue
                bbox = element.get("bbox_pixels")
                if not isinstance(bbox, dict):
                    continue
                try:
                    left = float(bbox["x_min"])
                    top = float(bbox["y_min"])
                    right = float(bbox["x_max"])
                    bottom = float(bbox["y_max"])
                except (KeyError, TypeError, ValueError):
                    continue
                if not left <= x <= right or not top <= y <= bottom:
                    continue
                if editable_only and not bool(element.get("is_editable")):
                    continue
                evidence: dict[str, str] = {}
                for source_key, target_key in (
                    ("class_name", "class_name"),
                    ("text", "text"),
                    ("content_description", "content_description"),
                    ("resource_id", "resource_id"),
                    ("resource_name", "resource_id"),
                    ("package_name", "package_name"),
                ):
                    value = _t3a_hint_redacted_text(
                        element.get(source_key),
                        forbidden_values=forbidden_values,
                        max_len=120,
                    )
                    if value and target_key not in evidence:
                        evidence[target_key] = value
                actionable = any(
                    bool(element.get(key))
                    for key in ("is_clickable", "is_editable", "is_scrollable")
                )
                area = max(0.0, right - left) * max(0.0, bottom - top)
                ui_candidates.append(((not actionable, not bool(evidence), area), evidence))
            if ui_candidates:
                return min(ui_candidates, key=lambda candidate: candidate[0])[1]
    forest = next(
        (
            value
            for value in (
                observation.get("forest"),
                observation.get("xml"),
                observation.get("observation_xml"),
                observation.get("page"),
            )
            if isinstance(value, str) and "<node" in value
        ),
        "",
    )
    if not forest:
        source_context = params.get("source_context")
        if isinstance(source_context, dict):
            forest = next(
                (
                    value
                    for value in (
                        source_context.get("page"),
                        source_context.get("xml"),
                        source_context.get("observation_xml"),
                    )
                    if isinstance(value, str) and "<node" in value
                ),
                "",
            )
    if not isinstance(forest, str) or "<node" not in forest:
        return {}
    try:
        root = ET.fromstring(forest)
    except ET.ParseError:
        return {}
    action_name, _ = _t3a_hint_step_action(step)
    action_name = action_name.strip().lower()
    x = params.get("x")
    y = params.get("y")
    action_index = (
        index
        if isinstance(index, int) and not isinstance(index, bool)
        else None
    )
    metadata = step.get("metadata")
    raw_summary = (
        str(metadata.get("summary") or "") if isinstance(metadata, dict) else ""
    )
    index_match = re.search(r'"index"\s*:\s*(\d+)', raw_summary)
    indexed_id = index_match.group(1) if index_match else ""
    candidates: list[tuple[tuple[int, int, int], dict[str, str]]] = []
    for node in root.iter():
        if str(node.tag).rsplit("}", 1)[-1] != "node":
            continue
        attributes = {str(key): str(value) for key, value in node.attrib.items()}
        if editable_only and not (
            attributes.get("editable") == "true"
            or "edittext" in attributes.get("class", "").casefold()
        ):
            continue
        bounds = _t3a_hint_bounds(attributes.get("bounds", ""))
        coordinate_match = False
        if bounds is not None and isinstance(x, (int, float)) and isinstance(y, (int, float)):
            left, top, right, bottom = bounds
            coordinate_match = left <= x <= right and top <= y <= bottom
        index_match_node = bool(indexed_id and attributes.get("id") == indexed_id)
        if not coordinate_match and not index_match_node:
            continue
        actionable = any(
            attributes.get(key) == "true"
            for key in ("clickable", "editable", "scrollable")
        )
        semantic = any(
            attributes.get(key)
            for key in ("text", "content-desc", "resource-id")
        )
        area = (
            max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
            if bounds is not None
            else 10**12
        )
        candidates.append(((not actionable, not semantic, area), attributes))
    if (
        not candidates
        and action_name in {"click", "tap", "long_press"}
        and action_index is not None
    ):
        nodes = [
            node
            for node in root.iter()
            if str(node.tag).rsplit("}", 1)[-1] == "node"
        ]

        def index_matches(attributes: dict[str, str]) -> bool:
            node_id = attributes.get("id", "")
            return node_id == str(action_index) or node_id.endswith(
                f":{action_index}"
            )

        exact_matches = [
            {str(key): str(value) for key, value in node.attrib.items()}
            for node in nodes
            if index_matches({str(key): str(value) for key, value in node.attrib.items()})
        ]
        if len(exact_matches) == 1:
            attributes = exact_matches[0]
            candidates.append(((False, False, 0), attributes))
        else:
            # Manual source logs sometimes retain only an AndroidWorld action
            # index while the serialized observation retains XML. If the
            # clicked control disappears in the immediately following frame,
            # recover it from the before/after tree transition. Only accept a
            # unique clickable transition; ambiguous transitions remain a
            # conversion failure and must be recollected with source element
            # evidence.
            next_observation = (
                step.get("next_observation")
                if isinstance(step, dict)
                else None
            )
            next_forest = next_observation.get("xml") if isinstance(next_observation, dict) else ""
            if isinstance(next_forest, str) and "<node" in next_forest:
                try:
                    next_root = ET.fromstring(next_forest)
                except ET.ParseError:
                    next_root = None
                if next_root is not None:
                    next_ids = {
                        str(node.attrib.get("id"))
                        for node in next_root.iter()
                        if node.attrib.get("id")
                    }
                    transition_matches = []
                    for node in nodes:
                        attributes = {str(key): str(value) for key, value in node.attrib.items()}
                        node_id = attributes.get("id")
                        if not node_id or node_id in next_ids:
                            continue
                        if attributes.get("clickable") != "true":
                            continue
                        bounds = _t3a_hint_bounds(attributes.get("bounds", ""))
                        area = (
                            max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
                            if bounds is not None
                            else 10**12
                        )
                        transition_matches.append((area, attributes))
                    if len(transition_matches) == 1:
                        _, attributes = transition_matches[0]
                        candidates.append(((False, False, 0), attributes))
    if not candidates and action_name in {
        "input_text",
        "type_text",
        "set_text",
        "enter_text",
    } and not (
        (isinstance(x, (int, float)) and isinstance(y, (int, float)))
        or indexed_id
    ):
        # Some Android accessibility snapshots do not expose focus state. In
        # that case, use the preceding click and its post-action XML to recover
        # the selected editable element without carrying source coordinates
        # into the T3A hint.
        preceding_name, preceding_params = _t3a_hint_step_action(preceding_step)
        if preceding_name.strip().lower() in {"click", "tap"}:
            preceding_observation = (
                preceding_step.get("next_observation")
                if isinstance(preceding_step, dict)
                else None
            )
            preceding_forest = next(
                (
                    value
                    for value in (
                        preceding_observation.get("forest"),
                        preceding_observation.get("xml"),
                        preceding_observation.get("observation_xml"),
                        preceding_observation.get("page"),
                    )
                    if isinstance(value, str) and "<node" in value
                ),
                "",
            ) if isinstance(preceding_observation, dict) else ""
            preceding_x = preceding_params.get("x")
            preceding_y = preceding_params.get("y")
            if preceding_forest and isinstance(preceding_x, (int, float)) and isinstance(preceding_y, (int, float)):
                try:
                    preceding_root = ET.fromstring(preceding_forest)
                except ET.ParseError:
                    preceding_root = None
                if preceding_root is not None:
                    for node in preceding_root.iter():
                        if str(node.tag).rsplit("}", 1)[-1] != "node":
                            continue
                        attributes = {str(key): str(value) for key, value in node.attrib.items()}
                        if attributes.get("editable") != "true":
                            continue
                        bounds = _t3a_hint_bounds(attributes.get("bounds", ""))
                        if bounds is None:
                            continue
                        left, top, right, bottom = bounds
                        if left <= preceding_x <= right and top <= preceding_y <= bottom:
                            candidates.append(((False, False, 0), attributes))
                            break
        editable_nodes: list[tuple[bool, int, dict[str, str]]] = []
        for node in root.iter():
            if str(node.tag).rsplit("}", 1)[-1] != "node":
                continue
            attributes = {str(key): str(value) for key, value in node.attrib.items()}
            if attributes.get("editable") != "true":
                continue
            bounds = _t3a_hint_bounds(attributes.get("bounds", ""))
            area = (
                max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])
                if bounds is not None
                else 10**12
            )
            editable_nodes.append((attributes.get("focused") == "true", area, attributes))
        focused_nodes = [item for item in editable_nodes if item[0]]
        if focused_nodes:
            candidates.extend(
                ((False, False, area), attributes)
                for _, area, attributes in focused_nodes
            )
        elif len(editable_nodes) == 1:
            _, area, attributes = editable_nodes[0]
            candidates.append(((False, False, area), attributes))
        elif not candidates and editable_nodes:
            # AndroidWorld's file dialog opens with the first field selected,
            # even when the exported XML omits focused=true.
            _, area, attributes = editable_nodes[0]
            candidates.append(((False, False, area), attributes))
    if not candidates:
        return {}
    attributes = min(candidates, key=lambda item: item[0])[1]
    field_map = {
        "id": "node_id",
        "class": "class_name",
        "text": "text",
        "content-desc": "content_description",
        "resource-id": "resource_id",
        "package": "package_name",
    }
    evidence: dict[str, str] = {}
    for source_key, target_key in field_map.items():
        if target_key == "text" and attributes.get("editable") == "true":
            continue
        value = _t3a_hint_redacted_text(
            attributes.get(source_key),
            forbidden_values=forbidden_values,
            max_len=120,
        )
        if value:
            evidence[target_key] = value
    return evidence


def _t3a_semantic_hint_step(
    step: Any,
    *,
    forbidden_values: Sequence[str],
    preceding_step: Any = None,
) -> dict[str, Any] | None:
    name, params = _t3a_hint_step_action(step)
    action = name.strip().lower()
    if not action or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", action):
        return None
    if action in {"status", "finish", "done"}:
        return None
    semantic: dict[str, Any] = {"action": action}
    purpose = _t3a_hint_step_summary(
        step,
        forbidden_values=forbidden_values,
    )
    source_node = _t3a_hint_source_node(
        step,
        forbidden_values=forbidden_values,
        preceding_step=preceding_step,
    )
    target = _t3a_hint_target(params, forbidden_values=forbidden_values)
    if not target:
        target = _t3a_hint_summary_target(purpose)
    if not target:
        target = (
            source_node.get("content_description")
            or source_node.get("text")
            or source_node.get("resource_id")
        )
    if (
        action in {"input_text", "type_text", "set_text", "enter_text"}
        and not target
        and not source_node
    ):
        preceding_action, _ = _t3a_hint_step_action(preceding_step)
        if preceding_action.strip().lower() in {"click", "tap"}:
            source_node = _t3a_hint_source_node(
                preceding_step,
                forbidden_values=forbidden_values,
                editable_only=True,
                preceding_step=None,
            )
            if source_node:
                target = "editable text field selected by the preceding action"
    if action in {"click", "tap", "long_press", "input_text", "type_text", "set_text", "enter_text"} and not target and not source_node:
        raise ValueError(f"t3a_hint_unidentified_target:{action}")
    if target:
        semantic["target"] = target
    if purpose:
        semantic["purpose"] = purpose
    if source_node:
        semantic["source_node"] = source_node
    if action in {"open_app", "launch_app"}:
        app = _t3a_hint_text(
            params.get("app_name")
            or params.get("app")
            or params.get("package_name")
            or params.get("packageName"),
            forbidden_values=forbidden_values,
            max_len=80,
        )
        if app:
            semantic["app"] = app
    if action in {"swipe", "scroll"}:
        direction = _t3a_hint_text(
            params.get("direction"),
            forbidden_values=forbidden_values,
            max_len=24,
        )
        if direction:
            semantic["direction"] = direction
    if action in {"press_key", "key_event"}:
        key = _t3a_hint_text(
            params.get("key") or params.get("key_name"),
            forbidden_values=forbidden_values,
            max_len=24,
        )
        if key:
            semantic["key"] = key
    return semantic


def _source_action_hint_path_for_item(
    item: CanonicalRunLog,
    *,
    output_root: str | Path,
    repo_root: Path = REPO_ROOT,
) -> Path:
    payload, _, profile, _ = canonicalize_source_run_log(
        item,
    )
    forbidden_values = _t3a_hint_forbidden_values(item.params)
    source_steps = list(payload.get("steps") or [])
    semantic_source = "source_run_log"
    semantic_input_steps = source_steps
    semantic_preceding_steps = [None, *source_steps[:-1]]
    store_alignment_mode = "not_applicable"
    semantic_steps = []
    for step, preceding_step in zip(
        semantic_input_steps,
        semantic_preceding_steps,
        strict=True,
    ):
        semantic = _t3a_semantic_hint_step(
            step,
            forbidden_values=forbidden_values,
            preceding_step=preceding_step,
        )
        if semantic is not None:
            semantic_steps.append(semantic)
    if not semantic_steps:
        raise ValueError(f"source runlog produced no safe T3A hints: {item.task}")
    hint_payload = {
        "schema_version": "omniflow.t3a_semantic_hint.v2",
        "task": item.task,
        "semantic_source": semantic_source,
        "store_alignment_mode": store_alignment_mode,
        "source_step_count": len(source_steps),
        "semantic_step_count": len(semantic_steps),
        "steps": semantic_steps,
    }
    hint_dir = resolve_path(output_root, root=repo_root) / "_source_action_hints"
    hint_dir.mkdir(parents=True, exist_ok=True)
    hint_path = hint_dir / f"{_safe_stem(item.task)}.source_action_hints.json"
    hint_path.write_text(
        json.dumps(hint_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return hint_path.resolve()
