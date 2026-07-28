"""Deterministic monotonic alignment for raw OmniFlow RunLogs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any

import numpy as np

from omniflow.embedding import ElementEmbedding, PageEncoder, TreeEmbedding
from omniflow.model import Observation

_POINT_ACTIONS = frozenset({"click", "input_text", "long_press"})
_ACTION_TOOL_ALIASES = {
    "back": "press_key",
    "input": "input_text",
    "key_event": "press_key",
    "launch_app": "open_app",
    "navigate_back": "press_key",
    "openapp": "open_app",
    "press_back": "press_key",
    "presskey": "press_key",
    "scroll": "swipe",
    "set_text": "input_text",
    "tap": "click",
    "touch": "click",
    "type_text": "input_text",
}
_IGNORED_ACTION_ARGS = frozenset(
    {
        "duration_ms",
        "element_index",
        "node_id",
        "node_resource_id",
        "wait_after_s",
        "x",
        "x1",
        "x2",
        "y",
        "y1",
        "y2",
    }
)


@dataclass(frozen=True)
class RunlogEndpoint:
    run_id: str
    state_id: str
    step_index: int
    action_tool: str
    page_id: str
    package_name: str
    activity_name: str
    screenshot_path: str
    width: float
    height: float
    point: dict[str, Any] | None
    node: dict[str, Any] | None


@dataclass(frozen=True)
class RunlogStepPair:
    left_step_index: int
    right_step_index: int
    score: float
    page_similarity: float
    node_similarity: float
    left_node: dict[str, Any] | None
    right_node: dict[str, Any] | None
    left_endpoint: RunlogEndpoint
    right_endpoint: RunlogEndpoint


@dataclass(frozen=True)
class RunlogAlignment:
    left_run_id: str
    right_run_id: str
    pairs: tuple[RunlogStepPair, ...]
    score: float


@dataclass(frozen=True)
class _StepView:
    index: int
    action_tool: str
    action_args: dict[str, Any]
    package_name: str
    page: TreeEmbedding | None
    action_node: ElementEmbedding | None
    endpoint: RunlogEndpoint


@dataclass(frozen=True)
class _PairScore:
    total: float
    page: float
    node: float


def align_runlogs(
    left_runlog: dict[str, Any],
    right_runlog: dict[str, Any],
    *,
    min_score: float = 0.72,
    gap_penalty: float = -0.18,
    page_encoder: PageEncoder | None = None,
) -> RunlogAlignment:
    """Return high-confidence monotonic step and action-node pairs."""

    if not 0.0 <= min_score <= 1.0:
        raise ValueError("min_score_must_be_between_zero_and_one")
    if gap_penalty >= 0.0:
        raise ValueError("gap_penalty_must_be_negative")
    if not isinstance(left_runlog, dict) or not isinstance(right_runlog, dict):
        raise ValueError("runlogs_must_be_objects")
    encoder = page_encoder or PageEncoder()
    left = _runlog_steps(left_runlog, encoder)
    right = _runlog_steps(right_runlog, encoder)
    scores = [
        [_step_pair_score(left, right, left_index, right_index) for right_index in range(len(right))]
        for left_index in range(len(left))
    ]
    indices, alignment_score = _decode_monotonic(
        scores,
        min_score=min_score,
        gap_penalty=gap_penalty,
    )
    return RunlogAlignment(
        left_run_id=str(left_runlog.get("run_id") or "left"),
        right_run_id=str(right_runlog.get("run_id") or "right"),
        pairs=tuple(
            RunlogStepPair(
                left_step_index=left[left_index].index,
                right_step_index=right[right_index].index,
                score=round(scores[left_index][right_index].total, 6),
                page_similarity=round(scores[left_index][right_index].page, 6),
                node_similarity=round(scores[left_index][right_index].node, 6),
                left_node=_public_node(left[left_index].action_node),
                right_node=_public_node(right[right_index].action_node),
                left_endpoint=left[left_index].endpoint,
                right_endpoint=right[right_index].endpoint,
            )
            for left_index, right_index in indices
        ),
        score=round(alignment_score, 6),
    )


def _runlog_steps(runlog: dict[str, Any], encoder: PageEncoder) -> tuple[_StepView, ...]:
    raw_steps = runlog.get("steps")
    if not isinstance(raw_steps, list):
        raise ValueError("runlog_steps_must_be_array")
    run_id = str(runlog.get("run_id") or "run")
    coordinate_space = _runlog_coordinate_space(runlog)
    views = []
    for position, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError("runlog_step_must_be_object")
        action = _step_action(raw_step)
        observation = _step_observation(raw_step)
        page = _encode_page(observation, encoder)
        width, height = _page_dimensions(observation, page)
        point = _page_point(action, coordinate_space, width, height)
        action_node = _action_node(page, action, point)
        step_index = int(raw_step.get("step_index", position))
        state_id = _step_state_id(run_id, step_index, raw_step, observation)
        public_node = _public_node(action_node)
        views.append(
            _StepView(
                index=step_index,
                action_tool=action["tool"],
                action_args=action["args"],
                package_name=str(observation.get("package_name") or ""),
                page=page,
                action_node=action_node,
                endpoint=RunlogEndpoint(
                    run_id=run_id,
                    state_id=state_id,
                    step_index=step_index,
                    action_tool=action["tool"],
                    page_id=str(observation.get("page_id") or state_id),
                    package_name=str(observation.get("package_name") or ""),
                    activity_name=str(observation.get("activity_name") or ""),
                    screenshot_path=_screenshot_path(observation),
                    width=width,
                    height=height,
                    point=point,
                    node=public_node,
                ),
            )
        )
    return tuple(views)


def _step_action(step: dict[str, Any]) -> dict[str, Any]:
    candidates = step.get("executed_actions") or step.get("actions") or ()
    raw = candidates[0] if isinstance(candidates, list) and candidates else step.get("action")
    if not isinstance(raw, dict):
        return {"tool": "", "args": {}}
    raw_tool = str(raw.get("tool") or raw.get("type") or raw.get("name") or "")
    raw_tool = raw_tool.strip().lower().replace("-", "_")
    tool = _ACTION_TOOL_ALIASES.get(raw_tool, raw_tool)
    raw_args = raw.get("args") or raw.get("params") or raw.get("arguments") or {}
    args = dict(raw_args) if isinstance(raw_args, dict) else {}
    if tool == "press_key":
        key = args.get("key") or args.get("keycode")
        if raw_tool in {"back", "navigate_back", "press_back"}:
            key = "back"
        return {"tool": tool, "args": {"key": _canonical_key(key)}}
    if tool == "swipe":
        return {"tool": tool, "args": _canonical_swipe_args(args)}
    if tool in _POINT_ACTIONS:
        return {"tool": tool, "args": _canonical_point_args(tool, args)}
    if tool == "open_app":
        package_name = args.get("package_name") or args.get("package") or args.get("app_id")
        return {"tool": tool, "args": {"package_name": str(package_name or "")}}
    return {"tool": tool, "args": args}


def _canonical_key(value: Any) -> str:
    key = str(value or "").strip().lower()
    if key.startswith("keycode_"):
        key = key.removeprefix("keycode_")
    return key


def _canonical_point_args(tool: str, args: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for output, aliases in (("x", ("x", "px")), ("y", ("y", "py"))):
        value = _first_action_arg(args, aliases)
        if value is not None:
            canonical[output] = value
    if tool == "input_text":
        canonical["text"] = str(
            _first_action_arg(args, ("text", "input_text", "value")) or ""
        )
    if tool == "long_press" and args.get("duration_ms") is not None:
        canonical["duration_ms"] = args["duration_ms"]
    return canonical


def _canonical_swipe_args(args: dict[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    coordinate_aliases = {
        "x1": ("x1", "start_x", "from_x"),
        "y1": ("y1", "start_y", "from_y"),
        "x2": ("x2", "end_x", "to_x"),
        "y2": ("y2", "end_y", "to_y"),
    }
    for output, aliases in coordinate_aliases.items():
        value = _first_action_arg(args, aliases)
        if value is not None:
            canonical[output] = value
    direction = str(args.get("direction") or "").strip().lower()
    if not direction:
        direction = _direction_from_swipe(canonical)
    if direction:
        canonical["direction"] = direction
    if args.get("duration_ms") is not None:
        canonical["duration_ms"] = args["duration_ms"]
    return canonical


def _direction_from_swipe(args: dict[str, Any]) -> str:
    try:
        delta_x = float(args["x2"]) - float(args["x1"])
        delta_y = float(args["y2"]) - float(args["y1"])
    except (KeyError, TypeError, ValueError):
        return ""
    if abs(delta_x) > abs(delta_y):
        return "right" if delta_x > 0 else "left"
    if delta_y:
        return "down" if delta_y > 0 else "up"
    return ""


def _first_action_arg(args: dict[str, Any], names: tuple[str, ...]) -> Any:
    return next((args[name] for name in names if args.get(name) is not None), None)


def _step_observation(step: dict[str, Any]) -> dict[str, Any]:
    for value in (
        step.get("observation_before_act"),
        step.get("observation"),
        step.get("before"),
        (step.get("metadata") or {}).get("observation_before_act")
        if isinstance(step.get("metadata"), dict)
        else None,
    ):
        if isinstance(value, dict):
            return value
    return {}


def _encode_page(observation: dict[str, Any], encoder: PageEncoder) -> TreeEmbedding | None:
    xml = observation.get("xml")
    if not isinstance(xml, str) or not xml.strip():
        return None
    embedded = encoder.embed(
        Observation(
            xml=xml,
            package_name=str(observation.get("package_name") or ""),
            activity_name=str(observation.get("activity_name") or ""),
        )
    )
    return embedded if embedded.elements else None


def _action_node(
    page: TreeEmbedding | None,
    action: dict[str, Any],
    point: dict[str, Any] | None,
) -> ElementEmbedding | None:
    if page is None or action["tool"] not in _POINT_ACTIONS:
        return None
    actionable = [
        element
        for element in page.elements
        if element.attributes.get("enabled", True)
        and element.attributes.get("visible", True)
        and any(
            element.attributes.get(key)
            for key in ("clickable", "editable", "focusable", "long_clickable")
        )
    ]
    if action["tool"] == "input_text":
        focused_editable = [
            element
            for element in actionable
            if element.attributes.get("editable") and element.attributes.get("focused")
        ]
        if focused_editable and not {"x", "y"} <= action["args"].keys():
            return min(focused_editable, key=lambda element: _area(element.bounds))
    if point is None:
        return None
    coordinates = float(point["x"]), float(point["y"])
    containing = [
        element for element in actionable if _contains(element.bounds, coordinates)
    ]
    if containing:
        return min(containing, key=lambda element: _area(element.bounds))
    if not actionable:
        return None
    return min(
        actionable,
        key=lambda element: _center_distance(element.bounds, coordinates),
    )


def _runlog_coordinate_space(runlog: dict[str, Any]) -> str:
    explicit = str(runlog.get("action_coordinate_space") or "").strip()
    if explicit:
        if explicit not in {"page_pixels", "relative_0_1000"}:
            raise ValueError("runlog_action_coordinate_space_unsupported")
        return explicit
    if runlog.get("schema_version") == "omniflow.canonical_run_log.v1":
        return "relative_0_1000"
    return "page_pixels"


def _page_dimensions(
    observation: dict[str, Any],
    page: TreeEmbedding | None,
) -> tuple[float, float]:
    display = observation.get("display")
    candidates: list[tuple[float, float]] = []
    if isinstance(display, dict):
        candidates.append(
            (
                _positive_float(display.get("width")),
                _positive_float(display.get("height")),
            )
        )
    candidates.append(
        (
            _positive_float(
                observation.get("width") or observation.get("display_width")
            ),
            _positive_float(
                observation.get("height") or observation.get("display_height")
            ),
        )
    )
    screenshot = observation.get("screenshot")
    if isinstance(screenshot, dict):
        candidates.append(
            (
                _positive_float(screenshot.get("width")),
                _positive_float(screenshot.get("height")),
            )
        )
    if page is not None:
        left, top, right, bottom = page.root_bounds
        candidates.append((float(right - min(0, left)), float(bottom - min(0, top))))
    valid = [(width, height) for width, height in candidates if width > 0 and height > 0]
    if not valid:
        return 0.0, 0.0
    return max(width for width, _ in valid), max(height for _, height in valid)


def _page_point(
    action: dict[str, Any],
    coordinate_space: str,
    width: float,
    height: float,
) -> dict[str, Any] | None:
    if action["tool"] not in _POINT_ACTIONS or width <= 0 or height <= 0:
        return None
    try:
        x = float(action["args"]["x"])
        y = float(action["args"]["y"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(x) or not math.isfinite(y):
        return None
    if coordinate_space == "relative_0_1000":
        x = x / 1000.0 * width
        y = y / 1000.0 * height
    if x < 0 or y < 0 or x > width or y > height:
        return None
    return {"x": x, "y": y, "coordinate_space": "page_pixels"}


def _step_state_id(
    run_id: str,
    step_index: int,
    step: dict[str, Any],
    observation: dict[str, Any],
) -> str:
    explicit = str(
        step.get("before_state_id") or observation.get("state_id") or ""
    ).strip()
    if explicit:
        return explicit
    identity = json.dumps(
        {
            "run_id": run_id,
            "step_index": step_index,
            "xml": observation.get("xml"),
            "package_name": observation.get("package_name"),
            "activity_name": observation.get("activity_name"),
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return "state_" + hashlib.sha256(identity.encode()).hexdigest()[:20]


def _screenshot_path(observation: dict[str, Any]) -> str:
    direct = str(
        observation.get("screenshot_path") or observation.get("image_path") or ""
    ).strip()
    if direct:
        return direct
    screenshot = observation.get("screenshot")
    if isinstance(screenshot, dict):
        return str(
            screenshot.get("path") or screenshot.get("screenshot_path") or ""
        ).strip()
    return ""


def _positive_float(value: Any) -> float:
    try:
        number = float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
    return number if math.isfinite(number) and number > 0.0 else 0.0


def _step_pair_score(
    left: tuple[_StepView, ...],
    right: tuple[_StepView, ...],
    left_index: int,
    right_index: int,
) -> _PairScore:
    left_step = left[left_index]
    right_step = right[right_index]
    if not left_step.action_tool or left_step.action_tool != right_step.action_tool:
        return _PairScore(0.0, 0.0, 0.0)
    page_similarity = _embedding_similarity(left_step.page, right_step.page)
    node_similarity = _node_similarity(left_step.action_node, right_step.action_node)
    node_semantic_similarity = _node_semantic_similarity(
        left_step.action_node,
        right_step.action_node,
    )
    if node_semantic_similarity == 0.0:
        return _PairScore(0.0, page_similarity or 0.0, node_similarity or 0.0)
    total = 0.20
    if page_similarity is not None:
        total += 0.25 * page_similarity
    if node_similarity is not None:
        total += 0.22 * node_similarity
    if node_semantic_similarity is not None:
        total += 0.18 * node_semantic_similarity
    if left_step.package_name or right_step.package_name:
        total += 0.06 * float(left_step.package_name == right_step.package_name)
    args_similarity = _argument_similarity(left_step.action_args, right_step.action_args)
    if args_similarity is not None:
        total += 0.06 * args_similarity
    context_similarity = _context_similarity(left, right, left_index, right_index)
    if context_similarity is not None:
        total += 0.03 * context_similarity
    return _PairScore(
        max(0.0, min(1.0, total)),
        page_similarity or 0.0,
        node_similarity or 0.0,
    )


def _embedding_similarity(left: TreeEmbedding | None, right: TreeEmbedding | None) -> float | None:
    if left is None or right is None:
        return None
    return _cosine(left.vector, right.vector)


def _node_similarity(
    left: ElementEmbedding | None,
    right: ElementEmbedding | None,
) -> float | None:
    if left is None or right is None:
        return None
    return _cosine(left.vector, right.vector)


def _node_semantic_similarity(
    left: ElementEmbedding | None,
    right: ElementEmbedding | None,
) -> float | None:
    if left is None or right is None:
        return None
    left_tokens = _node_semantic_tokens(left)
    right_tokens = _node_semantic_tokens(right)
    if not left_tokens or not right_tokens:
        return None
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union)


def _node_semantic_tokens(node: ElementEmbedding) -> frozenset[str]:
    values = (
        node.attributes.get("text"),
        node.attributes.get("content_description"),
        node.attributes.get("resource_id"),
    )
    return frozenset(
        token
        for value in values
        for token in str(value or "").lower().replace("/", " ").replace("_", " ").split()
        if token
    )


def _argument_similarity(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    left_public = {key: value for key, value in left.items() if key not in _IGNORED_ACTION_ARGS}
    right_public = {key: value for key, value in right.items() if key not in _IGNORED_ACTION_ARGS}
    if not left_public and not right_public:
        return None
    left_tokens = _json_tokens(left_public)
    right_tokens = _json_tokens(right_public)
    union = left_tokens | right_tokens
    return len(left_tokens & right_tokens) / len(union) if union else 1.0


def _context_similarity(
    left: tuple[_StepView, ...],
    right: tuple[_StepView, ...],
    left_index: int,
    right_index: int,
) -> float | None:
    comparisons = []
    for offset in (-1, 1):
        left_neighbor = left_index + offset
        right_neighbor = right_index + offset
        if 0 <= left_neighbor < len(left) and 0 <= right_neighbor < len(right):
            comparisons.append(
                float(left[left_neighbor].action_tool == right[right_neighbor].action_tool)
            )
    return sum(comparisons) / len(comparisons) if comparisons else None


def _decode_monotonic(
    pair_scores: list[list[_PairScore]],
    *,
    min_score: float,
    gap_penalty: float,
) -> tuple[tuple[tuple[int, int], ...], float]:
    left_size = len(pair_scores)
    right_size = len(pair_scores[0]) if pair_scores else 0
    scores = np.zeros((left_size + 1, right_size + 1), dtype=np.float64)
    moves = np.full((left_size + 1, right_size + 1), "", dtype=object)
    for left_index in range(1, left_size + 1):
        scores[left_index, 0] = scores[left_index - 1, 0] + gap_penalty
        moves[left_index, 0] = "up"
    for right_index in range(1, right_size + 1):
        scores[0, right_index] = scores[0, right_index - 1] + gap_penalty
        moves[0, right_index] = "left"
    for left_index in range(1, left_size + 1):
        for right_index in range(1, right_size + 1):
            pair_score = pair_scores[left_index - 1][right_index - 1].total
            choices = (
                (scores[left_index - 1, right_index - 1] + pair_score - 0.5, "diag"),
                (scores[left_index - 1, right_index] + gap_penalty, "up"),
                (scores[left_index, right_index - 1] + gap_penalty, "left"),
            )
            scores[left_index, right_index], moves[left_index, right_index] = max(
                choices,
                key=lambda choice: (choice[0], choice[1] == "diag"),
            )
    matches = []
    left_index, right_index = left_size, right_size
    while left_index > 0 or right_index > 0:
        move = moves[left_index, right_index]
        if move == "diag":
            if pair_scores[left_index - 1][right_index - 1].total >= min_score:
                matches.append((left_index - 1, right_index - 1))
            left_index -= 1
            right_index -= 1
        elif move == "up":
            left_index -= 1
        else:
            right_index -= 1
    matches.reverse()
    return tuple(matches), float(scores[left_size, right_size])


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 0.0:
        return 0.0
    return max(0.0, min(1.0, float(np.dot(left, right) / denominator)))


def _contains(bounds: tuple[int, int, int, int], point: tuple[float, float]) -> bool:
    return bounds[0] <= point[0] <= bounds[2] and bounds[1] <= point[1] <= bounds[3]


def _area(bounds: tuple[int, int, int, int]) -> int:
    return max(0, bounds[2] - bounds[0]) * max(0, bounds[3] - bounds[1])


def _center_distance(bounds: tuple[int, int, int, int], point: tuple[float, float]) -> float:
    center = ((bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0)
    return math.dist(center, point)


def _json_tokens(value: dict[str, Any]) -> frozenset[str]:
    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return frozenset(token.lower() for token in serialized.replace('"', " ").replace(":", " ").split())


def _public_node(node: ElementEmbedding | None) -> dict[str, Any] | None:
    if node is None:
        return None
    return {
        "node_id": node.id,
        "bounds": list(node.bounds),
        "attributes": dict(node.attributes),
    }


__all__ = ["RunlogAlignment", "RunlogEndpoint", "RunlogStepPair", "align_runlogs"]
