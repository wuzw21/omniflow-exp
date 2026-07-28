"""NumPy inference for the canonical mutual assignment checkpoint."""

from __future__ import annotations

from dataclasses import asdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

from omnitransfer.learned_matcher import (
    LearnedMatch,
    MatcherConfig,
    cross_relation_features,
    encode_graph,
)
from omnitransfer.ui_graph import UIGraph, local_context_graph


_SCHEMA_VERSION = "omnitransfer_numpy_mutual_matcher_v2"
_COMPATIBLE_SCHEMA_VERSIONS = {
    "omnitransfer_numpy_mutual_matcher_v1",
    _SCHEMA_VERSION,
}


class NumpyMutualGraphMatcher:
    """Inference-only adapter for XML graph matching without PyTorch."""

    def __init__(
        self,
        weights: Mapping[str, Any],
        *,
        config: MatcherConfig,
    ) -> None:
        self.weights = dict(weights)
        self.config = config
        self.backend = "numpy"

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "NumpyMutualGraphMatcher":
        np = _require_numpy()
        with np.load(Path(path), allow_pickle=False) as checkpoint:
            schema_version = _decode(checkpoint["__schema_version__"])
            if schema_version not in _COMPATIBLE_SCHEMA_VERSIONS:
                raise ValueError("checkpoint is not a NumPy mutual assignment matcher")
            config = MatcherConfig(**json.loads(_decode(checkpoint["__config_json__"])))
            weights = {
                name: checkpoint[name].astype(np.float32, copy=True)
                for name in checkpoint.files
                if not name.startswith("__")
            }
        return cls(weights, config=config)

    def predict(
        self,
        source: UIGraph,
        target: UIGraph,
        *,
        source_node_id: str,
        candidate_node_ids: Iterable[str] | None = None,
        min_probability: float = 0.0,
        min_margin: float = 0.0,
    ) -> LearnedMatch:
        np = _require_numpy()
        if not any(node.node_id == source_node_id for node in source.nodes):
            return LearnedMatch(None, 0.0, 0.0, "source_node_missing", ())
        if source.metadata.get("screenshot_path") or target.metadata.get("screenshot_path"):
            raise RuntimeError("numpy_matcher_requires_xml_only_graphs")
        if len(source.nodes) > self.config.source_context_nodes:
            source = local_context_graph(
                source,
                anchor_node_id=source_node_id,
                max_nodes=self.config.source_context_nodes,
            )
        source_index = next(
            (
                index
                for index, node in enumerate(source.nodes)
                if node.node_id == source_node_id
            ),
            None,
        )
        if source_index is None:
            raise AssertionError("source context dropped its anchor node")
        allowed = set(candidate_node_ids or (node.node_id for node in target.nodes))
        candidate_indices = [
            index for index, node in enumerate(target.nodes) if node.node_id in allowed
        ]
        if not candidate_indices:
            return LearnedMatch(None, 0.0, 0.0, "target_candidates_missing", ())
        output = self._forward(source, target)
        selected_logits = output["logits_ab"][source_index][candidate_indices]
        selected_affinity = output["affinity"][source_index][candidate_indices]
        rank_probabilities = _softmax(selected_logits, axis=0)
        match_probabilities = _sigmoid(selected_affinity)
        ranked = sorted(
            (
                (target.nodes[index].node_id, float(rank_probabilities[position]))
                for position, index in enumerate(candidate_indices)
            ),
            key=lambda item: (-item[1], item[0]),
        )
        best_id, best_probability = ranked[0]
        best_position = next(
            position
            for position, index in enumerate(candidate_indices)
            if target.nodes[index].node_id == best_id
        )
        match_probability = float(match_probabilities[best_position])
        second_probability = max(
            (score for _, score in ranked[1:]),
            default=0.0,
        )
        margin = best_probability - second_probability
        scores = tuple(ranked)
        if match_probability < min_probability or margin < min_margin:
            return LearnedMatch(
                None,
                match_probability,
                margin,
                "learned_low_confidence",
                scores,
            )
        target_node = next(node for node in target.nodes if node.node_id == best_id)
        return LearnedMatch(target_node, match_probability, margin, "learned_match", scores)

    def _forward(self, source: UIGraph, target: UIGraph) -> dict[str, Any]:
        np = _require_numpy()
        source_graph = encode_graph(source, config=self.config)
        target_graph = encode_graph(target, config=self.config)
        source_token_ids = np.asarray(source_graph.token_ids, dtype=np.int64)
        target_token_ids = np.asarray(target_graph.token_ids, dtype=np.int64)
        source_numeric = np.asarray(source_graph.numeric_features, dtype=np.float32)
        target_numeric = np.asarray(target_graph.numeric_features, dtype=np.float32)
        source_relations = np.asarray(source_graph.relation_features, dtype=np.float32)
        target_relations = np.asarray(target_graph.relation_features, dtype=np.float32)
        pair_relations = np.asarray(
            cross_relation_features(source, target),
            dtype=np.float32,
        )
        source_states, source_semantic, source_attributes = self._encode_nodes(
            source_token_ids,
            source_numeric,
        )
        target_states, target_semantic, target_attributes = self._encode_nodes(
            target_token_ids,
            target_numeric,
        )
        for layer_index in range(self.config.num_layers):
            source_states = self._relation_layer(
                source_states,
                source_relations,
                layer_index,
            )
            target_states = self._relation_layer(
                target_states,
                target_relations,
                layer_index,
            )
        semantic = self._pair_head("semantic_score.network", source_semantic, target_semantic)
        attributes = self._pair_head(
            "attribute_score.network",
            source_attributes,
            target_attributes,
        )
        context = self._pair_head("context_score.network", source_states, target_states)
        geometry = self._mlp(
            pair_relations,
            "geometry_score",
            norm=True,
            final_index=3,
        ).squeeze(-1)
        visual = np.zeros_like(semantic)
        visual_available = np.zeros_like(semantic)
        anchor_seed = self._mlp(
            np.stack(
                (
                    semantic,
                    visual,
                    attributes,
                    context,
                    geometry,
                    visual_available,
                ),
                axis=-1,
            ),
            "anchor_seed",
            norm=True,
            final_index=3,
        ).squeeze(-1)
        anchor = self._anchor_support(anchor_seed, source_relations, target_relations)
        affinity = self._mlp(
            np.stack(
                (
                    semantic,
                    visual,
                    attributes,
                    context,
                    geometry,
                    anchor,
                    visual_available,
                ),
                axis=-1,
            ),
            "pair_fusion",
            norm=True,
            final_index=4,
        ).squeeze(-1)
        logits_ab, logits_ba = _mutual_assignment_logits(affinity)
        return {
            "logits_ab": logits_ab,
            "logits_ba": logits_ba,
            "affinity": affinity,
        }

    def _encode_nodes(self, token_ids: Any, numeric: Any) -> tuple[Any, Any, Any]:
        np = _require_numpy()
        mask = token_ids != 0
        embedded = self.weights["node_encoder.token_embedding.weight"][token_ids]
        token_sum = (embedded * mask[..., None]).sum(axis=1, dtype=np.float32)
        token_count = np.maximum(mask.sum(axis=1, keepdims=True), 1)
        token_states = self._linear(
            token_sum / token_count.astype(np.float32),
            "node_encoder.token_projection",
        )
        numeric_states = self._layer_norm(
            numeric,
            "node_encoder.numeric_projection.0",
        )
        numeric_states = _gelu(
            self._linear(numeric_states, "node_encoder.numeric_projection.1")
        )
        numeric_states = self._linear(
            numeric_states,
            "node_encoder.numeric_projection.3",
        )
        missing_visual = self.weights["node_encoder.missing_visual"][None, :]
        states = self._layer_norm(
            token_states + numeric_states + missing_visual,
            "node_encoder.output_norm",
        )
        return states, token_states, numeric_states

    def _relation_layer(
        self,
        states: Any,
        relations: Any,
        layer_index: int,
    ) -> Any:
        prefix = f"layers.{layer_index}"
        node_count = int(states.shape[0])
        head_dimension = self.config.hidden_dim // self.config.num_heads
        qkv = self._linear(states, f"{prefix}.qkv", bias=False).reshape(
            node_count,
            3,
            self.config.num_heads,
            head_dimension,
        )
        query, key, value = (qkv[:, index].transpose(1, 0, 2) for index in range(3))
        scores = (query @ key.transpose(0, 2, 1)) * np_float(head_dimension**-0.5)
        relation_bias = _gelu(
            self._linear(relations, f"{prefix}.relation_bias.0")
        )
        relation_bias = self._linear(
            relation_bias,
            f"{prefix}.relation_bias.2",
        ).transpose(2, 0, 1)
        attention = _softmax(scores + relation_bias, axis=-1)
        context = (attention @ value).transpose(1, 0, 2).reshape(
            node_count,
            self.config.hidden_dim,
        )
        states = self._layer_norm(
            states + self._linear(context, f"{prefix}.output", bias=False),
            f"{prefix}.attention_norm",
        )
        feed_forward = _gelu(
            self._linear(states, f"{prefix}.feed_forward.0")
        )
        feed_forward = self._linear(
            feed_forward,
            f"{prefix}.feed_forward.3",
        )
        return self._layer_norm(
            states + feed_forward,
            f"{prefix}.output_norm",
        )

    def _pair_head(self, prefix: str, source: Any, target: Any) -> Any:
        features = _require_numpy().concatenate(
            (
                abs(source[:, None, :] - target[None, :, :]),
                source[:, None, :] * target[None, :, :],
            ),
            axis=-1,
        )
        states = self._layer_norm(features, f"{prefix}.0")
        states = _gelu(self._linear(states, f"{prefix}.1"))
        return self._linear(states, f"{prefix}.4").squeeze(-1)

    def _anchor_support(
        self,
        seed: Any,
        source_relations: Any,
        target_relations: Any,
    ) -> Any:
        np = _require_numpy()
        alignment = np.exp(
            np_float(0.5)
            * (_log_softmax(seed, axis=1) + _log_softmax(seed, axis=0))
        )
        source_edges = self._local_edges(source_relations)
        target_edges = self._local_edges(target_relations)
        support = source_edges @ alignment @ target_edges.T
        source_degree = np.maximum(source_edges.sum(axis=1), np_float(1e-6))
        target_degree = np.maximum(target_edges.sum(axis=1), np_float(1e-6))
        normalizer = np.sqrt(source_degree[:, None] * target_degree[None, :])
        return support / normalizer

    def _local_edges(self, relations: Any) -> Any:
        learned = _gelu(self._linear(relations, "anchor_edge_score.0"))
        learned = _sigmoid(
            self._linear(learned, "anchor_edge_score.2").squeeze(-1)
        )
        return learned * relations[..., 17] * (np_float(1.0) - relations[..., 0])

    def _mlp(
        self,
        values: Any,
        prefix: str,
        *,
        norm: bool,
        final_index: int,
    ) -> Any:
        states = self._layer_norm(values, f"{prefix}.0") if norm else values
        states = _gelu(self._linear(states, f"{prefix}.1"))
        return self._linear(states, f"{prefix}.{final_index}")

    def _linear(self, values: Any, prefix: str, *, bias: bool = True) -> Any:
        result = values @ self.weights[f"{prefix}.weight"].T
        if bias:
            result = result + self.weights[f"{prefix}.bias"]
        return result

    def _layer_norm(self, values: Any, prefix: str) -> Any:
        np = _require_numpy()
        mean = values.mean(axis=-1, keepdims=True, dtype=np.float32)
        variance = ((values - mean) ** 2).mean(
            axis=-1,
            keepdims=True,
            dtype=np.float32,
        )
        normalized = (values - mean) / np.sqrt(variance + np_float(1e-5))
        return (
            normalized * self.weights[f"{prefix}.weight"]
            + self.weights[f"{prefix}.bias"]
        )


def save_numpy_mutual_matcher_checkpoint(
    path: str | Path,
    state_dict: Mapping[str, Any],
    *,
    config: MatcherConfig,
) -> None:
    """Export a PyTorch state dict as a safe, pickle-free NumPy archive."""

    np = _require_numpy()
    arrays: dict[str, Any] = {
        "__schema_version__": _encode(_SCHEMA_VERSION),
        "__config_json__": _encode(
            json.dumps(asdict(config), sort_keys=True, separators=(",", ":"))
        ),
    }
    for name, value in state_dict.items():
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        arrays[name] = np.asarray(value, dtype=np.float32)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)


def _mutual_assignment_logits(
    affinity: Any,
) -> tuple[Any, Any]:
    np = _require_numpy()
    row_log = _log_softmax(affinity, axis=1)
    column_log = _log_softmax(affinity, axis=0)
    mutual = np_float(0.5) * (
        row_log
        + column_log
    )
    return mutual, mutual.T


def _gelu(values: Any) -> Any:
    np = _require_numpy()
    scaled = values * np_float(1.0 / math.sqrt(2.0))
    sign = np.sign(scaled)
    absolute = np.abs(scaled)
    factor = np_float(1.0) / (np_float(1.0) + np_float(0.3275911) * absolute)
    polynomial = (
        (
            (
                (
                    np_float(1.061405429) * factor
                    - np_float(1.453152027)
                )
                * factor
                + np_float(1.421413741)
            )
            * factor
            - np_float(0.284496736)
        )
        * factor
        + np_float(0.254829592)
    ) * factor
    error_function = sign * (
        np_float(1.0) - polynomial * np.exp(-(absolute * absolute))
    )
    return np_float(0.5) * values * (np_float(1.0) + error_function)


def _softmax(values: Any, *, axis: int) -> Any:
    np = _require_numpy()
    shifted = values - np.max(values, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=axis, keepdims=True)


def _log_softmax(values: Any, *, axis: int) -> Any:
    np = _require_numpy()
    maximum = np.max(values, axis=axis, keepdims=True)
    shifted = values - maximum
    return shifted - np.log(np.exp(shifted).sum(axis=axis, keepdims=True))


def _sigmoid(values: Any) -> Any:
    np = _require_numpy()
    return np_float(1.0) / (np_float(1.0) + np.exp(-values))


def _encode(value: str) -> Any:
    return _require_numpy().frombuffer(value.encode("utf-8"), dtype=np_uint8()).copy()


def _decode(value: Any) -> str:
    return value.astype(np_uint8(), copy=False).tobytes().decode("utf-8")


def np_float(value: float) -> Any:
    return _require_numpy().float32(value)


def np_uint8() -> Any:
    return _require_numpy().uint8


def _require_numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise RuntimeError("NumPy is required for the mutual matcher") from exc
    return np
