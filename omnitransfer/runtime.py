"""Stable replay-time action transfer interface."""

from __future__ import annotations

from functools import lru_cache
import hashlib
import math
import os
from pathlib import Path
from typing import Any

from omnitransfer.mutual_matcher import MutualGraphMatcher
from omnitransfer.numpy_matcher import NumpyMutualGraphMatcher
from omnitransfer.ui_graph import UIGraph, UINode, graph_from_record


_MIN_ANCHOR_OFFSET = -1.0
_MAX_ANCHOR_OFFSET = 2.0
_MATCHER_MODE = "mutual_graph_matcher_no_null_v3"
_DEFAULT_MATCHER_MIN_PROBABILITY = 0.5
_DEFAULT_MATCHER_MIN_MARGIN = 0.0
_DEFAULT_MATCHER_CHECKPOINT = (
    Path(__file__).resolve().parent
    / "checkpoints"
    / "pair_evidence_mutual_no_null_v3_20260723"
    / "no_null_seed17.pt"
)
_DEFAULT_MATCHER_SHA256 = (
    "61beec6da26f7aab7c51fd778ea22b5cfc956ca0cb658f1e91f4e8debc6f95b8"
)
_DEFAULT_NUMPY_MATCHER_CHECKPOINT = _DEFAULT_MATCHER_CHECKPOINT.with_suffix(".npz")
_DEFAULT_NUMPY_MATCHER_SHA256 = (
    "6e5668343419da38776e1f32ad9da610abc323637d8f6c6df38fb72ddec062b8"
)


def action_transfer(
    *,
    target_xml: str,
    source_xml: str | None = None,
    source_point: tuple[float, float] | None = None,
    source_element_id: str | None = None,
    source_element: dict[str, Any] | None = None,
    source_offset: tuple[float, float] | None = None,
    source_coordinate_space: str | None = None,
    target_display_size: tuple[float, float] | None = None,
    source_package_name: str | None = None,
    target_package_name: str | None = None,
    source_activity_name: str | None = None,
    target_activity_name: str | None = None,
    action_type: str = "click",
    top_k: int = 1,
    history: Any = None,
) -> dict[str, Any]:
    """Relocate one recorded target and return coordinates in target XML pixels.

    Android display and screenshot scaling belong to the caller. Semantic
    source-to-target matching uses only the XML coordinate systems.
    """

    del source_element, source_coordinate_space, target_display_size, history
    if not source_xml:
        return {
            "mapped": False,
            "mapping_mode": _MATCHER_MODE,
            "reason": "source_graph_required",
        }
    source_package = str(source_package_name or "").strip()
    target_package = str(target_package_name or "").strip()
    if source_package and target_package and source_package != target_package:
        return {
            "mapped": False,
            "mapping_mode": "page_identity",
            "reason": "target_page_identity_mismatch",
            "source_package_name": source_package,
            "target_package_name": target_package,
        }
    source = graph_from_record({"xml": source_xml}, graph_id="source")
    target = graph_from_record({"xml": target_xml}, graph_id="target")
    source_node = _source_node(source, source_point, source_element_id)
    if source_node is None:
        return {
            "mapped": False,
            "mapping_mode": _MATCHER_MODE,
            "reason": "source_target_missing",
        }
    offset = _source_offset(source_node, source_point, source_offset)
    if offset is None:
        return {
            "mapped": False,
            "mapping_mode": _MATCHER_MODE,
            "reason": "source_point_or_offset_required",
        }
    equivalent_target = _equivalent_graph_target(source, target, source_node)
    if equivalent_target is not None:
        return _mapped_result(
            source=source,
            target=target,
            source_node=source_node,
            target_node=equivalent_target,
            offset=offset,
            mapping_mode="equivalent_ui_graph",
            score=1.0,
            margin=1.0,
            ranked=[(1.0, equivalent_target)],
            top_k=top_k,
            action_type=action_type,
            source_activity_name=source_activity_name,
            target_activity_name=target_activity_name,
        )
    candidates = tuple(
        node.node_id
        for node in target.nodes
        if node.bbox is not None and node.enabled
    )
    try:
        matcher = _get_matcher()
        match = matcher.predict(
            source,
            target,
            source_node_id=source_node.node_id,
            candidate_node_ids=candidates,
            min_probability=_matcher_threshold(
                "OMNITRANSFER_MATCHER_MIN_PROBABILITY",
                default=_DEFAULT_MATCHER_MIN_PROBABILITY,
            ),
            min_margin=_matcher_threshold(
                "OMNITRANSFER_MATCHER_MIN_MARGIN",
                default=_DEFAULT_MATCHER_MIN_MARGIN,
            ),
        )
    except Exception as error:
        return {
            "mapped": False,
            "mapping_mode": _MATCHER_MODE,
            "reason": "matcher_unavailable",
            "error": str(error) or type(error).__name__,
            "src_element": _node_dict(source_node),
            "source_size": _graph_size(source),
            "target_size": _graph_size(target),
        }
    ranked = _learned_candidates(target, match.scores)
    target_node = match.target_node
    if target_node is None or target_node.bbox is None:
        return {
            "mapped": False,
            "mapping_mode": _MATCHER_MODE,
            "reason": str(match.reason or "matcher_abstained"),
            "src_element": _node_dict(source_node),
            "source_size": _graph_size(source),
            "target_size": _graph_size(target),
            "score": float(match.probability),
            "margin": float(match.margin),
            "top_candidates": _candidate_dicts(ranked, top_k),
        }
    return _mapped_result(
        source=source,
        target=target,
        source_node=source_node,
        target_node=target_node,
        offset=offset,
        mapping_mode=_MATCHER_MODE,
        score=float(match.probability),
        margin=float(match.margin),
        ranked=ranked,
        top_k=top_k,
        action_type=action_type,
        source_activity_name=source_activity_name,
        target_activity_name=target_activity_name,
    )


def _mapped_result(
    *,
    source: UIGraph,
    target: UIGraph,
    source_node: UINode,
    target_node: UINode,
    offset: tuple[float, float],
    mapping_mode: str,
    score: float,
    margin: float,
    ranked: list[tuple[float, UINode]],
    top_k: int,
    action_type: str,
    source_activity_name: str | None,
    target_activity_name: str | None,
) -> dict[str, Any]:
    target_point = _project_offset(offset, target_node.bbox or (0.0, 0.0, 0.0, 0.0))
    return {
        "mapped": True,
        "mapping_mode": mapping_mode,
        "new_x": target_point[0],
        "new_y": target_point[1],
        "src_element": _node_dict(source_node),
        "source_size": _graph_size(source),
        "target_size": _graph_size(target),
        "target_candidate_id": _public_node_id(target_node),
        "target_bbox": list(target_node.bbox),
        "target_center": list(_center(target_node.bbox)),
        "score": score,
        "margin": margin,
        "top_candidates": _candidate_dicts(ranked, top_k),
        "action_type": action_type,
        "source_activity_name": str(source_activity_name or ""),
        "target_activity_name": str(target_activity_name or ""),
    }


def _get_matcher() -> MutualGraphMatcher:
    configured = str(os.environ.get("OMNITRANSFER_MATCHER_CHECKPOINT") or "").strip()
    checkpoint = Path(configured).expanduser() if configured else _DEFAULT_MATCHER_CHECKPOINT
    device = str(os.environ.get("OMNITRANSFER_MATCHER_DEVICE") or "cpu").strip()
    numpy_checkpoint = checkpoint if checkpoint.suffix == ".npz" else checkpoint.with_suffix(".npz")
    return _load_matcher(
        str(checkpoint.resolve()),
        str(numpy_checkpoint.resolve()),
        device,
    )


@lru_cache(maxsize=4)
def _load_matcher(
    checkpoint: str,
    numpy_checkpoint: str,
    device: str,
) -> MutualGraphMatcher | NumpyMutualGraphMatcher:
    path = Path(checkpoint)
    numpy_path = Path(numpy_checkpoint)
    if path.suffix == ".npz":
        return _load_numpy_matcher(path)
    if path.is_file() and path == _DEFAULT_MATCHER_CHECKPOINT.resolve():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != _DEFAULT_MATCHER_SHA256:
            raise ValueError("default mutual matcher checkpoint checksum mismatch")
    if path.is_file():
        try:
            return MutualGraphMatcher.from_checkpoint(path, device=device)
        except RuntimeError as error:
            if "PyTorch" not in str(error):
                raise
    if numpy_path.is_file():
        return _load_numpy_matcher(numpy_path)
    raise FileNotFoundError(
        f"mutual matcher checkpoints missing: {path}, {numpy_path}"
    )


def _load_numpy_matcher(path: Path) -> NumpyMutualGraphMatcher:
    if path == _DEFAULT_NUMPY_MATCHER_CHECKPOINT.resolve():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if not _DEFAULT_NUMPY_MATCHER_SHA256 or digest != _DEFAULT_NUMPY_MATCHER_SHA256:
            raise ValueError("default NumPy mutual matcher checkpoint checksum mismatch")
    return NumpyMutualGraphMatcher.from_checkpoint(path)


def matcher_backend() -> str:
    """Return the backend after loading the canonical matcher."""

    matcher = _get_matcher()
    return str(getattr(matcher, "backend", "pytorch"))


def runtime_preflight() -> dict[str, Any]:
    """Load the matcher and execute a deterministic end-to-end mapping."""

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
    result = action_transfer(
        source_xml=source_xml,
        target_xml=target_xml,
        source_point=(30.0, 40.0),
    )
    if not result.get("mapped"):
        reason = str(result.get("reason") or "mapping_failed")
        error = str(result.get("error") or "")
        raise RuntimeError(f"omnitransfer_preflight_failed:{reason}:{error}".rstrip(":"))
    return {
        "ready": True,
        "backend": matcher_backend(),
    }


def _matcher_threshold(name: str, *, default: float) -> float:
    configured = str(os.environ.get(name) or "").strip()
    value = float(configured) if configured else float(default)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _learned_candidates(
    graph: UIGraph,
    scores: tuple[tuple[str, float], ...],
) -> list[tuple[float, UINode]]:
    nodes = {node.node_id: node for node in graph.nodes}
    return sorted(
        (
            (float(score), nodes[node_id])
            for node_id, score in scores
            if node_id in nodes
        ),
        key=lambda item: (-item[0], item[1].node_id),
    )


def _equivalent_graph_target(
    source: UIGraph,
    target: UIGraph,
    source_node: UINode,
) -> UINode | None:
    if source.width != target.width or source.height != target.height:
        return None
    if len(source.nodes) != len(target.nodes):
        return None
    if any(
        _structural_identity(left) != _structural_identity(right)
        for left, right in zip(source.nodes, target.nodes, strict=True)
    ):
        return None
    target_by_id = {node.node_id: node for node in target.nodes}
    target_node = target_by_id.get(source_node.node_id)
    if target_node is None or target_node.bbox is None:
        return None
    source_subtree = _subtree(source, source_node.node_id)
    target_subtree = _subtree(target, target_node.node_id)
    if len(source_subtree) != len(target_subtree):
        return None
    if any(
        _identity_key(left) != _identity_key(right)
        for left, right in zip(source_subtree, target_subtree, strict=True)
    ):
        return None
    has_semantic_anchor = any(
        node.text or node.content_desc for node in source_subtree
    )
    has_unique_resource = bool(source_node.resource_id) and sum(
        node.resource_id == source_node.resource_id for node in source.nodes
    ) == 1
    return target_node if has_semantic_anchor or has_unique_resource else None


def _structural_identity(node: UINode) -> tuple[Any, ...]:
    return (
        node.node_id,
        node.parent_id,
        node.child_ids,
        node.origin_id,
        node.resource_id,
        node.class_name,
        node.bbox,
        node.clickable,
        node.editable,
        node.scrollable,
        node.enabled,
    )


def _subtree(graph: UIGraph, root_id: str) -> tuple[UINode, ...]:
    by_id = {node.node_id: node for node in graph.nodes}
    pending = [root_id]
    nodes: list[UINode] = []
    while pending:
        node = by_id.get(pending.pop(0))
        if node is None:
            return ()
        nodes.append(node)
        pending.extend(node.child_ids)
    return tuple(nodes)


def describe_action_target(
    *,
    source_xml: str,
    source_point: tuple[float, float],
    related_point: tuple[float, float] | None = None,
) -> dict[str, Any] | None:
    """Return the compact semantic target for one recorded source point."""

    source = graph_from_record({"xml": source_xml}, graph_id="source")
    source_node = _source_node(source, source_point, None)
    if source_node is None or source_node.bbox is None:
        return None
    descriptor = _node_dict(source_node)
    occurrence_index, occurrence_count = _occurrence(source, source_node)
    if not _has_stable_identity(source_node) and occurrence_count <= 1:
        return None
    offset = _offset(source_point, source_node.bbox)
    descriptor["offset_x"], descriptor["offset_y"] = offset
    if related_point is not None:
        raw_end_offset = _unclamped_offset(related_point, source_node.bbox)
        if not all(
            math.isfinite(value)
            and _MIN_ANCHOR_OFFSET <= value <= _MAX_ANCHOR_OFFSET
            for value in raw_end_offset
        ):
            return None
        end_offset = _offset(related_point, source_node.bbox)
        descriptor["end_offset_x"], descriptor["end_offset_y"] = end_offset
    if occurrence_count > 1:
        descriptor["occurrence_index"] = occurrence_index
        descriptor["occurrence_count"] = occurrence_count
    descriptor.pop("bounds", None)
    descriptor.pop("id", None)
    return descriptor


def _source_node(
    graph: UIGraph,
    point: tuple[float, float] | None,
    element_id: str | None,
) -> UINode | None:
    normalized = str(element_id or "").strip()
    if normalized:
        return next(
            (
                node
                for node in graph.nodes
                if normalized in {node.node_id, node.origin_id, node.resource_id}
            ),
            None,
        )
    if point is None:
        return None
    x, y = point
    containing = [
        node
        for node in graph.nodes
        if node.bbox is not None
        and node.bbox[0] <= x <= node.bbox[2]
        and node.bbox[1] <= y <= node.bbox[3]
    ]
    actionable = [
        node
        for node in containing
        if node.enabled and (node.clickable or node.editable or node.scrollable)
    ]
    candidates = actionable or containing
    return min(
        candidates,
        key=lambda node: (
            _area(node.bbox),
            not _has_stable_identity(node),
            -node.depth,
            node.node_id,
        ),
        default=None,
    )


def _has_stable_identity(node: UINode) -> bool:
    return bool(node.resource_id or node.text or node.content_desc)


def _occurrence(graph: UIGraph, source: UINode) -> tuple[int, int]:
    identity = _identity_key(source)
    matches = sorted(
        (
            node
            for node in graph.nodes
            if node.bbox is not None
            and node.enabled
            and _identity_key(node) == identity
        ),
        key=lambda node: (node.bbox[1], node.bbox[0], node.node_id),
    )
    for index, node in enumerate(matches):
        if node is source or node.node_id == source.node_id:
            return index, len(matches)
    return 0, len(matches)


def _identity_key(node: UINode) -> tuple[Any, ...]:
    return (
        _tail(node.resource_id),
        _text(node.text),
        _text(node.content_desc),
        _tail(node.class_name),
        node.clickable,
        node.editable,
        node.scrollable,
    )


def _source_offset(
    source: UINode,
    point: tuple[float, float] | None,
    explicit: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if explicit is not None:
        try:
            offset = float(explicit[0]), float(explicit[1])
        except (IndexError, TypeError, ValueError):
            return None
        if not all(
            math.isfinite(value)
            and _MIN_ANCHOR_OFFSET <= value <= _MAX_ANCHOR_OFFSET
            for value in offset
        ):
            return None
        return offset
    if point is not None and source.bbox is not None:
        return _offset(point, source.bbox)
    if source.bbox is not None:
        return 0.5, 0.5
    return None


def _offset(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    width = max(1.0, bounds[2] - bounds[0])
    height = max(1.0, bounds[3] - bounds[1])
    return (
        _clamp((float(point[0]) - bounds[0]) / width),
        _clamp((float(point[1]) - bounds[1]) / height),
    )


def _unclamped_offset(
    point: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> tuple[float, float]:
    width = max(1.0, bounds[2] - bounds[0])
    height = max(1.0, bounds[3] - bounds[1])
    return (
        (float(point[0]) - bounds[0]) / width,
        (float(point[1]) - bounds[1]) / height,
    )


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _project_offset(
    offset: tuple[float, float],
    target: tuple[float, float, float, float],
) -> tuple[float, float]:
    return (
        target[0] + offset[0] * (target[2] - target[0]),
        target[1] + offset[1] * (target[3] - target[1]),
    )


def _node_dict(node: UINode) -> dict[str, Any]:
    return {
        "id": _public_node_id(node),
        "resource_id": node.resource_id,
        "text": node.text,
        "content_desc": node.content_desc,
        "class": node.class_name,
        "bounds": list(node.bbox or ()),
        "clickable": node.clickable,
        "editable": node.editable,
        "scrollable": node.scrollable,
    }


def _candidate_dicts(
    ranked: list[tuple[float, UINode]],
    top_k: int,
) -> list[dict[str, Any]]:
    return [
        {
            "candidate_id": _public_node_id(candidate),
            "resource_id": candidate.resource_id,
            "text": candidate.text,
            "content_desc": candidate.content_desc,
            "class": candidate.class_name,
            "bbox": list(candidate.bbox or ()),
            "score": candidate_score,
        }
        for candidate_score, candidate in ranked[: max(1, int(top_k))]
    ]


def _public_node_id(node: UINode) -> str:
    return str(node.resource_id or node.origin_id or node.node_id)


def _graph_size(graph: UIGraph | None) -> list[float] | None:
    if graph is None or graph.width is None or graph.height is None:
        return None
    if graph.width <= 0 or graph.height <= 0:
        return None
    return [graph.width, graph.height]


def _center(bounds: tuple[float, float, float, float]) -> tuple[float, float]:
    return (bounds[0] + bounds[2]) / 2.0, (bounds[1] + bounds[3]) / 2.0


def _area(bounds: tuple[float, float, float, float] | None) -> float:
    return float("inf") if bounds is None else (bounds[2] - bounds[0]) * (bounds[3] - bounds[1])


def _tail(value: str) -> str:
    return _text(value).rsplit("/", 1)[-1].rsplit(".", 1)[-1]


def _text(value: str) -> str:
    return " ".join(str(value or "").strip().lower().split())
