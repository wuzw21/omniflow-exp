from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from omniflow.model import Observation

_GROUP_SIZES = (2, 5, 5, 7, 8)


@dataclass(frozen=True)
class EncoderWeights:
    view: tuple[float, ...]
    human: tuple[float, ...]
    programmer: tuple[float, ...]
    pooling: tuple[float, ...]
    slices: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "view", _normalized_group(self.view, 2))
        object.__setattr__(self, "human", _normalized_group(self.human, 5))
        object.__setattr__(
            self, "programmer", _normalized_group(self.programmer, 5)
        )
        object.__setattr__(self, "pooling", _positive_group(self.pooling, 7))
        object.__setattr__(self, "slices", _normalized_group(self.slices, 8))

    @classmethod
    def manual_default(cls) -> EncoderWeights:
        return cls(
            view=(0.55, 0.45),
            human=(0.50, 0.12, 0.15, 0.10, 0.13),
            programmer=(0.25, 0.30, 0.30, 0.08, 0.07),
            pooling=(0.45, 1.35, 1.30, 1.30, 1.80, 0.20, 0.40),
            slices=(0.03, 0.06, 0.07, 0.09, 0.03, 0.34, 0.30, 0.08),
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> EncoderWeights:
        return cls(
            view=tuple(value.get("view") or ()),
            human=tuple(value.get("human") or ()),
            programmer=tuple(value.get("programmer") or ()),
            pooling=tuple(value.get("pooling") or ()),
            slices=tuple(value.get("slices") or ()),
        )

    @classmethod
    def from_parameters(cls, parameters: np.ndarray) -> EncoderWeights:
        values = np.asarray(parameters, dtype=np.float64).reshape(-1)
        if values.shape != (sum(_GROUP_SIZES),) or not np.all(np.isfinite(values)):
            raise ValueError("encoder_parameters_must_be_27_finite_values")
        groups: list[np.ndarray] = []
        offset = 0
        for size in _GROUP_SIZES:
            groups.append(values[offset : offset + size])
            offset += size
        return cls(
            view=tuple(_softmax(groups[0])),
            human=tuple(_softmax(groups[1])),
            programmer=tuple(_softmax(groups[2])),
            pooling=tuple(np.logaddexp(0.0, groups[3])),
            slices=tuple(_softmax(groups[4])),
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "view": list(self.view),
            "human": list(self.human),
            "programmer": list(self.programmer),
            "pooling": list(self.pooling),
            "slices": list(self.slices),
        }

    def parameter_vector(self) -> np.ndarray:
        pooling = np.asarray(self.pooling, dtype=np.float64)
        inverse_softplus = np.where(
            pooling > 20.0,
            pooling,
            np.log(np.expm1(pooling)),
        )
        return np.concatenate(
            [
                np.log(np.asarray(self.view, dtype=np.float64)),
                np.log(np.asarray(self.human, dtype=np.float64)),
                np.log(np.asarray(self.programmer, dtype=np.float64)),
                inverse_softplus,
                np.log(np.asarray(self.slices, dtype=np.float64)),
            ]
        )

    @property
    def hash(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()[:16]


@dataclass(frozen=True)
class ElementEmbedding:
    id: str
    parent_id: str | None
    children_ids: tuple[str, ...]
    depth: int
    bounds: tuple[int, int, int, int]
    attributes: dict[str, Any]
    vector: np.ndarray


@dataclass(frozen=True)
class TreeEmbedding:
    elements: tuple[ElementEmbedding, ...]
    vector: np.ndarray
    root_bounds: tuple[int, int, int, int]
    encoder_version: str
    weights_hash: str


@dataclass(frozen=True)
class _Element:
    id: str
    parent_id: str | None
    children_ids: tuple[str, ...]
    depth: int
    bounds: tuple[int, int, int, int]
    attributes: dict[str, Any]


class PageEncoder:
    name = "page_vector"
    version = "page-vector.v1"
    element_dimension = 64
    dimension = 512

    def __init__(self, weights: EncoderWeights | None = None):
        self.weights = weights or EncoderWeights.manual_default()

    @classmethod
    def from_parameters(cls, parameters: np.ndarray) -> PageEncoder:
        return cls(EncoderWeights.from_parameters(parameters))

    def parameter_vector(self) -> np.ndarray:
        return self.weights.parameter_vector()

    def with_parameters(self, parameters: np.ndarray) -> PageEncoder:
        return type(self).from_parameters(parameters)

    def embed(self, value: Observation | dict[str, Any] | str) -> TreeEmbedding:
        observation = (
            Observation(xml=value)
            if isinstance(value, str)
            else Observation.from_value(value)
        )
        elements, root_bounds = _restore_tree(str(observation.xml or ""))
        if not elements:
            return TreeEmbedding(
                (),
                np.zeros(self.dimension, dtype=np.float32),
                root_bounds,
                self.version,
                self.weights.hash,
            )
        vectors = _embed_elements(elements, root_bounds, self.weights)
        masks = _slice_masks(elements, root_bounds)
        page_vector = _pool_tree(elements, vectors, masks, self.weights)
        embedded = tuple(
            ElementEmbedding(
                element.id,
                element.parent_id,
                element.children_ids,
                element.depth,
                element.bounds,
                dict(element.attributes),
                vectors[index],
            )
            for index, element in enumerate(elements)
        )
        return TreeEmbedding(
            embedded,
            page_vector,
            root_bounds,
            self.version,
            self.weights.hash,
        )


def _restore_tree(xml_text: str) -> tuple[tuple[_Element, ...], tuple[int, int, int, int]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return (), (0, 0, 1000, 1000)

    raw: list[dict[str, Any]] = []

    def visit(
        node: ET.Element,
        parent_id: str | None,
        depth: int,
        in_list: bool,
        has_siblings: bool,
    ) -> None:
        bounds = _bounds(node.attrib.get("bounds"))
        class_name = str(node.attrib.get("class") or node.tag).rsplit(".", 1)[-1]
        next_in_list = in_list or any(
            token in class_name.lower() for token in ("list", "recycler", "grid")
        )
        current_parent = parent_id
        current_depth = depth
        if bounds is not None:
            element_id = f"e{len(raw)}"
            attributes = {
                "class": class_name,
                "text": _text(node.attrib.get("text")),
                "content_description": _text(node.attrib.get("content-desc")),
                "resource_id": str(node.attrib.get("resource-id") or "").rsplit(
                    "/", 1
                )[-1],
                "clickable": _bool(node.attrib, "clickable"),
                "long_clickable": _bool(node.attrib, "long-clickable"),
                "focusable": _bool(node.attrib, "focusable"),
                "focused": _bool(node.attrib, "focused"),
                "editable": _bool(node.attrib, "editable")
                or "edittext" in class_name.lower(),
                "scrollable": _bool(node.attrib, "scrollable"),
                "checkable": _bool(node.attrib, "checkable"),
                "checked": _bool(node.attrib, "checked"),
                "enabled": _bool(node.attrib, "enabled", default=True),
                "selected": _bool(node.attrib, "selected"),
                "visible": _bool(node.attrib, "visible-to-user", default=True),
                "in_list": next_in_list,
                "has_siblings": has_siblings,
                "raw_child_count": len(node),
            }
            raw.append(
                {
                    "id": element_id,
                    "parent_id": parent_id,
                    "depth": depth,
                    "bounds": bounds,
                    "attributes": attributes,
                }
            )
            current_parent = element_id
            current_depth = depth + 1
        children = list(node)
        for child in children:
            visit(
                child,
                current_parent,
                current_depth,
                next_in_list,
                len(children) > 1,
            )

    visit(root, None, 0, False, False)
    if not raw:
        return (), (0, 0, 1000, 1000)
    child_ids: dict[str, list[str]] = {item["id"]: [] for item in raw}
    for item in raw:
        parent_id = item["parent_id"]
        if parent_id in child_ids:
            child_ids[parent_id].append(item["id"])
    elements = tuple(
        _Element(
            item["id"],
            item["parent_id"],
            tuple(child_ids[item["id"]]),
            item["depth"],
            item["bounds"],
            item["attributes"],
        )
        for item in raw
    )
    bounds = [item.bounds for item in elements]
    root_bounds = (
        min(item[0] for item in bounds),
        min(item[1] for item in bounds),
        max(item[2] for item in bounds),
        max(item[3] for item in bounds),
    )
    return elements, root_bounds


def _embed_elements(
    elements: tuple[_Element, ...],
    root_bounds: tuple[int, int, int, int],
    weights: EncoderWeights,
) -> np.ndarray:
    matrix = np.zeros((len(elements), 64), dtype=np.float32)
    by_id = {item.id: item for item in elements}
    root_area = max(
        1,
        (root_bounds[2] - root_bounds[0]) * (root_bounds[3] - root_bounds[1]),
    )
    for index, element in enumerate(elements):
        attrs = element.attributes
        content = " ".join(
            item
            for item in (attrs["text"], attrs["content_description"])
            if item
        )
        has_text = bool(attrs["text"])
        is_icon = bool(attrs["content_description"] and not has_text) or any(
            token in attrs["class"].lower() for token in ("image", "icon")
        )
        content_type = np.zeros(4, dtype=np.float32)
        content_type[(2 if has_text else 0) + (1 if is_icon else 0)] = 1.0
        affordance = np.asarray(
            [
                attrs["clickable"],
                attrs["editable"],
                attrs["scrollable"],
                attrs["checkable"],
            ],
            dtype=np.float32,
        )
        area = max(
            0,
            (element.bounds[2] - element.bounds[0])
            * (element.bounds[3] - element.bounds[1]),
        )
        prominence = np.zeros(4, dtype=np.float32)
        prominence[
            int(area / root_area > 0.005)
            + int(area / root_area > 0.02)
            + int(area / root_area > 0.08)
        ] = 1.0
        resource_id = attrs["resource_id"].lower()
        primary = attrs["selected"] or attrs["class"].lower().endswith("button")
        visual_state = np.asarray(
            [attrs["selected"], not attrs["enabled"], attrs["focused"], primary],
            dtype=np.float32,
        )
        child_classes = [
            by_id[child_id].attributes["class"]
            for child_id in element.children_ids
            if child_id in by_id
        ]
        structure = "|".join(
            [
                attrs["class"],
                str(len(element.children_ids)),
                *(sorted(child_classes)),
                "list" if attrs["in_list"] else "plain",
            ]
        )
        programmer_attributes = np.asarray(
            [
                attrs["clickable"],
                attrs["long_clickable"],
                attrs["focusable"],
                attrs["editable"],
                attrs["scrollable"],
                attrs["checkable"],
                attrs["enabled"],
                attrs["selected"],
            ],
            dtype=np.float32,
        )
        hierarchy = np.asarray(
            [not element.children_ids, attrs["has_siblings"]], dtype=np.float32
        )
        id_hint = np.asarray(
            [
                any(
                    token in resource_id
                    for token in ("btn", "button", "fab", "action", "submit", "save")
                ),
                any(
                    token in resource_id
                    for token in ("edit", "input", "search", "text")
                ),
            ],
            dtype=np.float32,
        )
        human = _weighted_blocks(
            (
                _hash_vector(content, 16),
                content_type,
                affordance,
                prominence,
                visual_state,
            ),
            weights.human,
        )
        programmer = _weighted_blocks(
            (
                _hash_vector(attrs["class"], 8),
                programmer_attributes,
                _hash_vector(structure, 12),
                hierarchy,
                id_hint,
            ),
            weights.programmer,
        )
        matrix[index] = np.concatenate(
            [human * weights.view[0], programmer * weights.view[1]]
        )
    return matrix


def _slice_masks(
    elements: tuple[_Element, ...],
    root_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    count = len(elements)
    masks = np.zeros((8, count), dtype=bool)
    root_height = max(1, root_bounds[3] - root_bounds[1])
    y_ratios = np.asarray(
        [
            ((item.bounds[1] + item.bounds[3]) / 2 - root_bounds[1]) / root_height
            for item in elements
        ],
        dtype=np.float32,
    )
    root_area = max(
        1,
        (root_bounds[2] - root_bounds[0]) * (root_bounds[3] - root_bounds[1]),
    )
    page_frame = np.asarray(
        [
            bool(item.children_ids)
            and (
                (item.bounds[2] - item.bounds[0])
                * (item.bounds[3] - item.bounds[1])
            )
            / root_area
            >= 0.80
            for item in elements
        ]
    )
    active = ~page_frame
    if not np.any(active):
        active = np.ones(count, dtype=bool)
    reference = y_ratios[active]
    top_cut = float(np.quantile(reference, 0.25))
    bottom_cut = float(np.quantile(reference, 0.75))
    has_text = np.asarray(
        [
            bool(item.attributes["text"] or item.attributes["content_description"])
            for item in elements
        ]
    )
    actionable = np.asarray(
        [
            bool(
                item.attributes["clickable"]
                or item.attributes["checkable"]
                or item.attributes["long_clickable"]
            )
            for item in elements
        ]
    )
    stateful = np.asarray(
        [
            bool(
                item.attributes["editable"]
                or item.attributes["focused"]
                or item.attributes["selected"]
                or item.attributes["checked"]
            )
            for item in elements
        ]
    )
    surface = np.asarray(
        [
            (
                (item.bounds[2] - item.bounds[0])
                * (item.bounds[3] - item.bounds[1])
            )
            / root_area
            >= 0.08
            for item in elements
        ]
    ) & active
    if not np.any(surface):
        surface = active.copy()
    masks[0] = active & has_text
    masks[1] = active & has_text & (y_ratios <= top_cut)
    masks[2] = active & has_text & (y_ratios >= bottom_cut)
    masks[3] = active & has_text & actionable
    masks[4] = active & has_text & stateful
    masks[5] = active
    masks[6] = active & (y_ratios > top_cut) & (y_ratios < bottom_cut)
    masks[7] = surface
    return masks


def _pool_tree(
    elements: tuple[_Element, ...],
    vectors: np.ndarray,
    masks: np.ndarray,
    weights: EncoderWeights,
) -> np.ndarray:
    attributes = [item.attributes for item in elements]
    text_leaf = np.asarray(
        [
            bool(item["text"] or item["content_description"])
            and not elements[index].children_ids
            for index, item in enumerate(attributes)
        ]
    )
    actionable = np.asarray(
        [
            bool(item["clickable"] or item["checkable"] or item["long_clickable"])
            for item in attributes
        ]
    )
    focus_target = np.asarray(
        [
            bool(item["focusable"] or item["editable"] or item["focused"])
            for item in attributes
        ]
    )
    selected = np.asarray(
        [bool(item["selected"] or item["checked"]) for item in attributes]
    )
    neutral = np.asarray(
        [
            bool(elements[index].children_ids)
            and not text_leaf[index]
            and not actionable[index]
            and not focus_target[index]
            for index in range(len(elements))
        ]
    )
    in_list = np.asarray([bool(item["in_list"]) for item in attributes])
    base = np.ones(len(elements), dtype=np.float32)
    base[neutral] *= weights.pooling[0]
    base[text_leaf] *= weights.pooling[1]
    base[actionable] *= weights.pooling[2]
    base[focus_target] *= weights.pooling[3]
    base[selected] *= weights.pooling[4]
    slices: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        current_weights = base.copy()
        current_weights[in_list] *= weights.pooling[5 if index < 5 else 6]
        matrix = vectors
        if index >= 5:
            matrix = vectors.copy()
            matrix[:, :16] = 0.0
        if not np.any(mask):
            pooled = np.zeros(64, dtype=np.float32)
        else:
            selected_weights = current_weights[mask]
            total = float(np.sum(selected_weights))
            pooled = (
                np.sum(matrix[mask] * selected_weights[:, None], axis=0) / total
                if total > 1e-9
                else np.zeros(64, dtype=np.float32)
            )
        slices.append(_normalize(pooled) * weights.slices[index])
    return _normalize(np.concatenate(slices)).astype(np.float32)


def _weighted_blocks(
    blocks: tuple[np.ndarray, ...], weights: tuple[float, ...]
) -> np.ndarray:
    weighted = [
        _normalize(np.asarray(block, dtype=np.float32)) * weights[index]
        for index, block in enumerate(blocks)
    ]
    return _normalize(np.concatenate(weighted)).astype(np.float32)


def _hash_vector(value: Any, dimension: int) -> np.ndarray:
    text = _text(value)
    vector = np.zeros(dimension, dtype=np.float32)
    if not text:
        return vector
    compact = re.sub(r"\s+", "_", text)
    tokens = set(re.findall(r"[\w.-]+", compact))
    if len(compact) < 3:
        tokens.add(compact)
    else:
        tokens.update(compact[index : index + 3] for index in range(len(compact) - 2))
    for token in sorted(tokens):
        digest = hashlib.blake2s(token.encode(), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "big") % dimension
        vector[bucket] += 1.0 if digest[4] & 1 else -1.0
    return _normalize(vector)


def _normalize(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(array))
    return (array / norm).astype(np.float32) if norm > 1e-9 else array


def _normalized_group(values: tuple[float, ...], size: int) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.shape != (size,) or not np.all(np.isfinite(array)) or np.any(array <= 0):
        raise ValueError(f"weight_group_must_have_{size}_positive_values")
    total = float(np.sum(array))
    if total <= 0:
        raise ValueError("weight_group_must_have_positive_mass")
    return tuple(float(round(item, 12)) for item in array / total)


def _positive_group(values: tuple[float, ...], size: int) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if (
        array.shape != (size,)
        or not np.all(np.isfinite(array))
        or np.any(array <= 0)
    ):
        raise ValueError(f"weight_group_must_have_{size}_positive_values")
    return tuple(float(round(item, 12)) for item in array)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - float(np.max(values))
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent)


def _bounds(value: Any) -> tuple[int, int, int, int] | None:
    numbers = [int(item) for item in re.findall(r"-?\d+", str(value or ""))]
    if len(numbers) != 4 or numbers[2] <= numbers[0] or numbers[3] <= numbers[1]:
        return None
    return numbers[0], numbers[1], numbers[2], numbers[3]


def _bool(attributes: dict[str, str], key: str, *, default: bool = False) -> bool:
    value = attributes.get(key)
    return default if value is None else str(value).lower() == "true"


def _text(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


__all__ = [
    "ElementEmbedding",
    "EncoderWeights",
    "PageEncoder",
    "TreeEmbedding",
]
