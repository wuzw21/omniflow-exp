from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib
import inspect
import json
import os
from pathlib import Path
import re
import sys
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from omniflow.core.model import Observation

_GROUP_SIZES = (2, 5, 5, 7, 8)
_UNIFIED_NODE_CHECKPOINT_SHA256 = (
    "c262f03c32c4b88d2933323fe2b33007281224ef1a8aae1418a9844d354de232"
)


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
class PageWordInputs:
    descriptors: np.ndarray
    evidence: np.ndarray
    priors: np.ndarray
    word_counts: tuple[int, ...]


@dataclass(frozen=True)
class SoftPageWordWeights:
    input_projection: np.ndarray
    input_bias: np.ndarray
    attention_output: np.ndarray
    attention_bias: np.ndarray
    prior_strength: np.ndarray
    presence_output: np.ndarray
    presence_bias: np.ndarray

    @property
    def descriptor_dimension(self) -> int:
        return int(self.input_projection.shape[0] - 16)

    @property
    def hidden_dimension(self) -> int:
        return int(self.input_projection.shape[1])

    @property
    def parameter_count(self) -> int:
        return sum(
            int(np.asarray(value).size)
            for value in (
                self.input_projection,
                self.input_bias,
                self.attention_output,
                self.attention_bias,
                self.prior_strength,
                self.presence_output,
                self.presence_bias,
            )
        )

    @classmethod
    def from_npz(cls, path: str) -> SoftPageWordWeights:
        with np.load(path, allow_pickle=False) as checkpoint:
            weights = cls(
                input_projection=checkpoint["input_projection"].astype(
                    np.float32, copy=True
                ),
                input_bias=checkpoint["input_bias"].astype(np.float32, copy=True),
                attention_output=checkpoint["attention_output"].astype(
                    np.float32, copy=True
                ),
                attention_bias=checkpoint["attention_bias"].astype(
                    np.float32, copy=True
                ),
                prior_strength=checkpoint["prior_strength"].astype(
                    np.float32, copy=True
                ),
                presence_output=checkpoint["presence_output"].astype(
                    np.float32, copy=True
                ),
                presence_bias=checkpoint["presence_bias"].astype(
                    np.float32, copy=True
                ),
            )
        weights.validate()
        return weights

    def validate(self) -> None:
        descriptor_dimension = self.descriptor_dimension
        hidden_dimension = self.hidden_dimension
        expected = {
            "input_projection": (descriptor_dimension + 16, hidden_dimension),
            "input_bias": (hidden_dimension,),
            "attention_output": (hidden_dimension, 8),
            "attention_bias": (8,),
            "prior_strength": (8,),
            "presence_output": (hidden_dimension, 8),
            "presence_bias": (8,),
        }
        for name, shape in expected.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"soft_page_word_weight_invalid:{name}")


@dataclass(frozen=True)
class SoftPageWordOutput:
    vector: np.ndarray
    words: np.ndarray
    presence: np.ndarray
    attention: np.ndarray
    word_counts: tuple[int, ...]


@dataclass(frozen=True)
class _Element:
    id: str
    parent_id: str | None
    children_ids: tuple[str, ...]
    depth: int
    bounds: tuple[int, int, int, int]
    attributes: dict[str, Any]


class _UnifiedNodeEncoder:
    """Produce learned 64D node descriptors without owning page pooling."""

    dimension = 64

    def __init__(self) -> None:
        root = _canonical_omnitransfer_root()
        checkpoint = (
            root
            / "src"
            / "omnitransfer"
            / "checkpoints"
            / "omnitransfer_unified_association_v1_20260819"
            / "relation_slots_l3_h64_seed17.npz"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"omnitransfer_node_checkpoint_missing:{checkpoint}"
            )
        self.checkpoint_sha256 = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if self.checkpoint_sha256 != _UNIFIED_NODE_CHECKPOINT_SHA256:
            raise ValueError("omnitransfer_node_checkpoint_checksum_mismatch")
        source_root = str(root / "src")
        package_root = root / "src" / "omnitransfer"
        sys.path[:] = [
            item
            for item in sys.path
            if str(Path(item or ".").resolve()) != source_root
        ]
        sys.path.insert(0, source_root)
        importlib.invalidate_caches()
        for module_name in tuple(sys.modules):
            if module_name == "omnitransfer" or module_name.startswith(
                "omnitransfer."
            ):
                module_path = Path(
                    str(getattr(sys.modules[module_name], "__file__", ""))
                ).resolve()
                if package_root not in module_path.parents:
                    del sys.modules[module_name]
        learned = importlib.import_module("omnitransfer.learned_matcher")
        numpy_matcher = importlib.import_module("omnitransfer.numpy_v9_matcher")
        ui_graph = importlib.import_module("omnitransfer.ui_graph")
        self._matcher = numpy_matcher.NumpyGeometricAlignmentMatcher.from_checkpoint(
            checkpoint
        )
        if int(self._matcher.config.hidden_dim) != self.dimension:
            raise ValueError("omnitransfer_node_embedding_dimension_mismatch")
        self._encode_graph = learned.encode_graph
        self._graph_from_record = ui_graph.graph_from_record
        self._visual_inputs = numpy_matcher._visual_inputs
        self.version = (
            f"{self._matcher.config.architecture}:"
            f"{self._matcher.config.text_encoder}:{self._matcher.backend}"
        )

    def encode(
        self,
        observation: Observation,
        elements: tuple[_Element, ...],
    ) -> np.ndarray:
        record = _omnitransfer_record(observation)
        graph_id = hashlib.sha256(
            str(observation.xml or "").encode("utf-8")
        ).hexdigest()[:20]
        graph = self._graph_from_record(record, graph_id=graph_id)
        encoded = self._encode_graph(
            graph,
            config=self._matcher.config,
            feature_schema_id=self._matcher.feature_schema_id,
        )
        visual_parameters = inspect.signature(self._visual_inputs).parameters
        visual_kwargs: dict[str, Any] = {
            "patch_size": self._matcher.config.visual_patch_size,
            "canvas_size": self._matcher.config.visual_canvas_size,
        }
        if "visual_encoder" in visual_parameters:
            visual_kwargs["visual_encoder"] = self._matcher.config.visual_encoder
        if "context_scale" in visual_parameters:
            visual_kwargs["context_scale"] = getattr(
                self._matcher.config, "visual_context_scale", 3.0
            )
        visual, visual_mask = self._visual_inputs(graph, **visual_kwargs)
        vectors, _modalities = self._matcher._encode_nodes(
            np.asarray(encoded.token_ids, dtype=np.int64),
            np.asarray(encoded.numeric_features, dtype=np.float32),
            visual,
            visual_mask,
        )
        valid_indices = [
            index
            for index, node in enumerate(graph.nodes)
            if node.bbox is not None
            and node.bbox[2] > node.bbox[0]
            and node.bbox[3] > node.bbox[1]
        ]
        aligned = np.asarray(vectors[valid_indices], dtype=np.float32)
        if aligned.shape != (len(elements), self.dimension):
            raise ValueError(
                "omnitransfer_node_embedding_alignment_failed:"
                f"elements={len(elements)}:vectors={aligned.shape}"
            )
        if not np.all(np.isfinite(aligned)):
            raise ValueError("omnitransfer_node_embedding_unusable")
        return aligned


class PageEncoder:
    name = "page_vector"
    element_dimension = 64
    dimension = 512

    def __init__(self, weights: EncoderWeights | None = None):
        self.weights = weights or EncoderWeights.manual_default()
        self._node_encoder = _UnifiedNodeEncoder()
        self.checkpoint_sha256 = self._node_encoder.checkpoint_sha256
        self.version = f"page-vector.v2:{self._node_encoder.version}"

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
        vectors = self._node_encoder.encode(observation, elements)
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


def _canonical_omnitransfer_root() -> Path:
    configured = str(os.environ.get("OMNITRANSFER_ROOT") or "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else Path.home() / "Projects" / "Omni" / "OmniTransfer"
    ).resolve()
    canonical = (Path.home() / "Projects" / "Omni" / "OmniTransfer").resolve()
    if root != canonical:
        raise ValueError(f"canonical_omnitransfer_root_required:{canonical}")
    if not (root / "src" / "omnitransfer").is_dir():
        raise RuntimeError(f"omnitransfer_root_missing:{root}")
    return root


def _omnitransfer_record(observation: Observation) -> dict[str, Any]:
    record: dict[str, Any] = {"xml": str(observation.xml or "")}
    display = observation.extra.get("display")
    if isinstance(display, dict):
        record["width"] = display.get("width")
        record["height"] = display.get("height")
    androidworld_state = observation.extra.get("androidworld_state")
    if isinstance(androidworld_state, dict):
        pixels = androidworld_state.get("pixels")
        if isinstance(pixels, dict):
            record["width"] = pixels.get("width") or record.get("width")
            record["height"] = pixels.get("height") or record.get("height")
            if isinstance(pixels.get("path"), str):
                record["screenshot_path"] = pixels["path"]
    screenshot_path = observation.extra.get("screenshot_path")
    if isinstance(screenshot_path, str) and screenshot_path.strip():
        record["screenshot_path"] = screenshot_path
    return record


def pool_dynamic_page_words(
    value: Observation | dict[str, Any] | str,
    node_descriptors: np.ndarray,
    *,
    weights: EncoderWeights | None = None,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Pool aligned node descriptors into OmniFlow's eight dynamic page words."""

    observation = (
        Observation(xml=value)
        if isinstance(value, str)
        else Observation.from_value(value)
    )
    elements, root_bounds = _restore_tree(str(observation.xml or ""))
    descriptors = np.asarray(node_descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] <= 0:
        raise ValueError("page_word_descriptors_must_be_n_by_d")
    if descriptors.shape[0] != len(elements):
        raise ValueError(
            "page_word_descriptor_count_mismatch:"
            f"elements={len(elements)}:descriptors={descriptors.shape[0]}"
        )
    if not len(elements) or not np.all(np.isfinite(descriptors)):
        raise ValueError("page_word_descriptors_unusable")
    selected_weights = weights or EncoderWeights.manual_default()
    masks = _slice_masks(elements, root_bounds)
    vector = _pool_descriptor_words(
        elements,
        descriptors,
        masks,
        selected_weights,
    )
    return vector, tuple(int(np.count_nonzero(mask)) for mask in masks)


def prepare_page_word_inputs(
    value: Observation | dict[str, Any] | str,
    node_descriptors: np.ndarray,
) -> PageWordInputs:
    observation = (
        Observation(xml=value)
        if isinstance(value, str)
        else Observation.from_value(value)
    )
    elements, root_bounds = _restore_tree(str(observation.xml or ""))
    descriptors = np.asarray(node_descriptors, dtype=np.float32)
    if descriptors.ndim != 2 or descriptors.shape[1] <= 0:
        raise ValueError("page_word_descriptors_must_be_n_by_d")
    if descriptors.shape[0] != len(elements):
        raise ValueError(
            "page_word_descriptor_count_mismatch:"
            f"elements={len(elements)}:descriptors={descriptors.shape[0]}"
        )
    if not len(elements) or not np.all(np.isfinite(descriptors)):
        raise ValueError("page_word_descriptors_unusable")
    priors = _slice_masks(elements, root_bounds)
    evidence = _page_word_evidence(elements, root_bounds)
    return PageWordInputs(
        descriptors=np.asarray(
            [_normalize(row) for row in descriptors], dtype=np.float32
        ),
        evidence=evidence,
        priors=priors.astype(np.float32),
        word_counts=tuple(int(np.count_nonzero(mask)) for mask in priors),
    )


def pool_soft_page_words(
    value: Observation | dict[str, Any] | str,
    node_descriptors: np.ndarray,
    weights: SoftPageWordWeights,
) -> SoftPageWordOutput:
    weights.validate()
    inputs = prepare_page_word_inputs(value, node_descriptors)
    if inputs.descriptors.shape[1] != weights.descriptor_dimension:
        raise ValueError("soft_page_word_descriptor_dimension_mismatch")
    features = np.concatenate((inputs.descriptors, inputs.evidence), axis=1)
    hidden = np.tanh(features @ weights.input_projection + weights.input_bias)
    logits = (
        hidden @ weights.attention_output
        + weights.attention_bias
        + inputs.priors.T * weights.prior_strength
    )
    attention = _column_softmax(logits).T
    words = np.asarray(
        [_normalize(row) for row in attention @ inputs.descriptors],
        dtype=np.float32,
    )
    contexts = attention @ hidden
    presence_logits = np.sum(
        contexts * weights.presence_output.T,
        axis=1,
    ) + weights.presence_bias
    presence = (1.0 / (1.0 + np.exp(-presence_logits))).astype(np.float32)
    vector = _normalize((words * presence[:, None]).reshape(-1))
    return SoftPageWordOutput(
        vector=vector.astype(np.float32),
        words=words,
        presence=presence,
        attention=attention.astype(np.float32),
        word_counts=inputs.word_counts,
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


def _page_word_evidence(
    elements: tuple[_Element, ...],
    root_bounds: tuple[int, int, int, int],
) -> np.ndarray:
    root_width = max(1, root_bounds[2] - root_bounds[0])
    root_height = max(1, root_bounds[3] - root_bounds[1])
    root_area = max(1, root_width * root_height)
    rows = []
    for element in elements:
        attrs = element.attributes
        width = max(0, element.bounds[2] - element.bounds[0])
        height = max(0, element.bounds[3] - element.bounds[1])
        center_x = (
            (element.bounds[0] + element.bounds[2]) / 2 - root_bounds[0]
        ) / root_width
        center_y = (
            (element.bounds[1] + element.bounds[3]) / 2 - root_bounds[1]
        ) / root_height
        area_ratio = min(1.0, width * height / root_area)
        has_text = bool(attrs["text"] or attrs["content_description"])
        rows.append(
            [
                center_x,
                center_y,
                width / root_width,
                height / root_height,
                area_ratio,
                min(1.0, element.depth / 16.0),
                float(has_text),
                float(attrs["clickable"] or attrs["long_clickable"]),
                float(attrs["editable"]),
                float(attrs["scrollable"]),
                float(attrs["checkable"]),
                float(attrs["selected"] or attrs["checked"]),
                float(attrs["focused"]),
                float(attrs["in_list"]),
                float(not element.children_ids),
                float(bool(element.children_ids)),
            ]
        )
    return np.asarray(rows, dtype=np.float32)


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


def _pool_descriptor_words(
    elements: tuple[_Element, ...],
    descriptors: np.ndarray,
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
    words: list[np.ndarray] = []
    for index, mask in enumerate(masks):
        current_weights = base.copy()
        current_weights[in_list] *= weights.pooling[5 if index < 5 else 6]
        if not np.any(mask):
            pooled = np.zeros(descriptors.shape[1], dtype=np.float32)
        else:
            word_weights = current_weights[mask]
            total = float(np.sum(word_weights))
            pooled = (
                np.sum(descriptors[mask] * word_weights[:, None], axis=0) / total
                if total > 1e-9
                else np.zeros(descriptors.shape[1], dtype=np.float32)
            )
        words.append(_normalize(pooled) * weights.slices[index])
    return _normalize(np.concatenate(words)).astype(np.float32)


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


def _column_softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=0, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / np.sum(exponent, axis=0, keepdims=True)


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
    "PageWordInputs",
    "SoftPageWordOutput",
    "SoftPageWordWeights",
    "TreeEmbedding",
    "pool_soft_page_words",
    "pool_dynamic_page_words",
    "prepare_page_word_inputs",
]
