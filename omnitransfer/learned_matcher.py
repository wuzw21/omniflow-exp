"""Lightweight relation-aware cross-attention for UI graph matching."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import re
from typing import Any, Iterable

from omnitransfer.ui_graph import BBox, UIGraph, UINode, local_context_graph


NUMERIC_FEATURE_DIM = 18
RELATION_FEATURE_DIM = 18


@dataclass(frozen=True)
class MatcherConfig:
    """Configuration for the compact learned matcher."""

    vocab_size: int = 8192
    max_tokens: int = 48
    token_dim: int = 48
    hidden_dim: int = 96
    relation_hidden_dim: int = 24
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.05
    visual_patch_size: int = 32
    visual_canvas_size: int = 384
    source_context_nodes: int = 48
    target_context_nodes: int = 64


@dataclass(frozen=True)
class EncodedGraph:
    """Torch-independent learned-matcher inputs for one UI graph."""

    graph_id: str
    node_ids: tuple[str, ...]
    origin_ids: tuple[str, ...]
    token_ids: tuple[tuple[int, ...], ...]
    numeric_features: tuple[tuple[float, ...], ...]
    relation_features: Any


@dataclass(frozen=True)
class _RelationContext:
    graph: UIGraph
    bboxes: tuple[BBox | None, ...]
    centers: tuple[tuple[float, float], ...]
    sizes: tuple[tuple[float, float], ...]
    node_indices: dict[str, int]
    ancestor_sets: dict[str, frozenset[str]]
    ancestor_paths: dict[str, tuple[str, ...]]
    path_positions: dict[str, dict[str, int]]


@dataclass(frozen=True)
class LearnedMatch:
    """One learned source-to-target match or a safe abstention."""

    target_node: UINode | None
    probability: float
    margin: float
    reason: str
    scores: tuple[tuple[str, float], ...]


def encode_graph(
    graph: UIGraph,
    *,
    config: MatcherConfig | None = None,
) -> EncodedGraph:
    """Encode UI attributes and pairwise relations without fixed match weights."""

    cfg = config or MatcherConfig()
    return EncodedGraph(
        graph_id=graph.graph_id,
        node_ids=tuple(node.node_id for node in graph.nodes),
        origin_ids=tuple(node.origin_id for node in graph.nodes),
        token_ids=tuple(_node_token_ids(node, config=cfg) for node in graph.nodes),
        numeric_features=tuple(_node_numeric_features(node, graph) for node in graph.nodes),
        relation_features=_relation_features(graph),
    )


def cross_relation_features(
    source: UIGraph,
    target: UIGraph,
) -> Any:
    """Return source-target geometric inputs for a learned pair head."""

    source_context = _relation_context(source)
    target_context = _relation_context(target)
    same_graph = source is target or source.graph_id == target.graph_id
    return _relation_matrix(
        source_context,
        target_context,
        same_graph=same_graph,
    )


def build_relation_aware_matcher(
    config: MatcherConfig | None = None,
) -> Any:
    """Build a compact relation-biased, bidirectional cross-attention matcher."""

    torch = _require_torch()
    nn = torch.nn
    cfg = config or MatcherConfig()
    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if cfg.source_context_nodes <= 0:
        raise ValueError("source_context_nodes must be positive")
    if cfg.target_context_nodes <= 0:
        raise ValueError("target_context_nodes must be positive")
    if cfg.visual_canvas_size < cfg.visual_patch_size:
        raise ValueError("visual_canvas_size must cover one visual patch")

    class RelationSelfAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 3, bias=False)
            self.relation_bias = nn.Sequential(
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, cfg.num_heads),
            )
            self.output = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.norm_attention = nn.LayerNorm(cfg.hidden_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.norm_output = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head_dim = cfg.hidden_dim // cfg.num_heads
            self.scale = self.head_dim**-0.5

        def forward(self, states: Any, relations: Any) -> Any:
            node_count = int(states.shape[0])
            qkv = self.qkv(states).reshape(
                node_count,
                3,
                cfg.num_heads,
                self.head_dim,
            )
            query, key, value = qkv.unbind(dim=1)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            scores = (query @ key.transpose(-1, -2)) * self.scale
            scores = scores + self.relation_bias(relations).permute(2, 0, 1)
            attention = torch.softmax(scores, dim=-1)
            context = attention @ value
            context = context.transpose(0, 1).reshape(node_count, cfg.hidden_dim)
            states = self.norm_attention(states + self.dropout(self.output(context)))
            return self.norm_output(states + self.dropout(self.feed_forward(states)))

    class MatcherLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attention = RelationSelfAttention()
            self.cross_attention = nn.MultiheadAttention(
                cfg.hidden_dim,
                cfg.num_heads,
                dropout=cfg.dropout,
                batch_first=True,
            )
            self.cross_norm = nn.LayerNorm(cfg.hidden_dim)
            self.cross_feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.output_norm = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)

        def forward(
            self,
            source_states: Any,
            target_states: Any,
            source_relations: Any,
            target_relations: Any,
        ) -> tuple[Any, Any]:
            source_states = self.self_attention(source_states, source_relations)
            target_states = self.self_attention(target_states, target_relations)
            source_context = self.cross_attention(
                source_states.unsqueeze(0),
                target_states.unsqueeze(0),
                target_states.unsqueeze(0),
                need_weights=False,
            )[0].squeeze(0)
            target_context = self.cross_attention(
                target_states.unsqueeze(0),
                source_states.unsqueeze(0),
                source_states.unsqueeze(0),
                need_weights=False,
            )[0].squeeze(0)
            source_states = self.cross_norm(source_states + self.dropout(source_context))
            target_states = self.cross_norm(target_states + self.dropout(target_context))
            source_states = self.output_norm(
                source_states + self.dropout(self.cross_feed_forward(source_states))
            )
            target_states = self.output_norm(
                target_states + self.dropout(self.cross_feed_forward(target_states))
            )
            return source_states, target_states

    class RelationAwareCrossAttentionMatcher(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(
                cfg.vocab_size,
                cfg.token_dim,
                padding_idx=0,
            )
            self.token_projection = nn.Linear(cfg.token_dim, cfg.hidden_dim)
            self.numeric_projection = nn.Sequential(
                nn.LayerNorm(NUMERIC_FEATURE_DIM),
                nn.Linear(NUMERIC_FEATURE_DIM, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            )
            self.visual_encoder = nn.Sequential(
                nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),
                nn.GELU(),
                nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.Conv2d(32, cfg.hidden_dim, kernel_size=3, stride=2, padding=1),
                nn.GELU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
            )
            self.missing_visual = nn.Parameter(torch.zeros(cfg.hidden_dim))
            self.input_norm = nn.LayerNorm(cfg.hidden_dim)
            self.layers = nn.ModuleList(MatcherLayer() for _ in range(cfg.num_layers))
            self.cross_relation_projection = nn.Sequential(
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
            )
            pair_dim = cfg.hidden_dim * 4 + cfg.relation_hidden_dim
            self.pair_head = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, 1),
            )
            self.matchability = nn.Linear(cfg.hidden_dim, 1)

        def _encode_nodes(
            self,
            token_ids: Any,
            numeric_features: Any,
            visual_patches: Any,
            visual_mask: Any,
        ) -> Any:
            mask = token_ids.ne(0).unsqueeze(-1)
            embedded = self.token_embedding(token_ids)
            token_sum = (embedded * mask).sum(dim=1)
            token_count = mask.sum(dim=1).clamp_min(1)
            token_states = self.token_projection(token_sum / token_count)
            numeric_states = self.numeric_projection(numeric_features)
            visual_states = self.visual_encoder(visual_patches)
            missing = self.missing_visual.unsqueeze(0).expand(visual_states.shape[0], -1)
            visual_states = visual_mask * visual_states + (1.0 - visual_mask) * missing
            return self.input_norm(token_states + numeric_states + visual_states)

        def _score_pairs(
            self,
            source_states: Any,
            target_states: Any,
            pair_relations: Any,
        ) -> Any:
            source = source_states[:, None, :].expand(-1, target_states.shape[0], -1)
            target = target_states[None, :, :].expand(source_states.shape[0], -1, -1)
            relation = self.cross_relation_projection(pair_relations)
            pair = torch.cat(
                [source, target, torch.abs(source - target), source * target, relation],
                dim=-1,
            )
            logits = self.pair_head(pair).squeeze(-1)
            return logits + self.matchability(source_states) + self.matchability(target_states).T

        def forward(
            self,
            source_token_ids: Any,
            source_numeric: Any,
            source_relations: Any,
            target_token_ids: Any,
            target_numeric: Any,
            target_relations: Any,
            source_target_relations: Any,
            source_visual: Any,
            source_visual_mask: Any,
            target_visual: Any,
            target_visual_mask: Any,
        ) -> dict[str, Any]:
            source_states = self._encode_nodes(
                source_token_ids,
                source_numeric,
                source_visual,
                source_visual_mask,
            )
            target_states = self._encode_nodes(
                target_token_ids,
                target_numeric,
                target_visual,
                target_visual_mask,
            )
            for layer in self.layers:
                source_states, target_states = layer(
                    source_states,
                    target_states,
                    source_relations,
                    target_relations,
                )
            pair_logits = self._score_pairs(
                source_states,
                target_states,
                source_target_relations,
            )
            reverse_relations = source_target_relations.transpose(0, 1).clone()
            reverse_relations[..., 9] = -reverse_relations[..., 9]
            reverse_relations[..., 10] = -reverse_relations[..., 10]
            reverse_relations[..., 13] = -reverse_relations[..., 13]
            reverse_relations[..., 14] = -reverse_relations[..., 14]
            reverse_logits = self._score_pairs(
                target_states,
                source_states,
                reverse_relations,
            )
            return {
                "logits_ab": pair_logits,
                "logits_ba": reverse_logits,
                "affinity": pair_logits,
                "source_states": source_states,
                "target_states": target_states,
            }

    return RelationAwareCrossAttentionMatcher()


def matcher_inputs(
    source: UIGraph,
    target: UIGraph,
    *,
    config: MatcherConfig | None = None,
    device: str | Any = "cpu",
) -> tuple[Any, ...]:
    """Convert two UI graphs to tensors accepted by the learned matcher."""

    torch = _require_torch()
    cfg = config or MatcherConfig()
    encoded_source = encode_graph(source, config=cfg)
    encoded_target = encode_graph(target, config=cfg)
    source_token_ids = torch.as_tensor(
        encoded_source.token_ids,
        dtype=torch.long,
        device=device,
    )
    target_token_ids = torch.as_tensor(
        encoded_target.token_ids,
        dtype=torch.long,
        device=device,
    )
    source_numeric = torch.as_tensor(
        encoded_source.numeric_features,
        dtype=torch.float32,
        device=device,
    )
    target_numeric = torch.as_tensor(
        encoded_target.numeric_features,
        dtype=torch.float32,
        device=device,
    )
    source_relations = torch.as_tensor(
        encoded_source.relation_features,
        dtype=torch.float32,
        device=device,
    )
    target_relations = torch.as_tensor(
        encoded_target.relation_features,
        dtype=torch.float32,
        device=device,
    )
    pair_relations = torch.as_tensor(
        cross_relation_features(source, target),
        dtype=torch.float32,
        device=device,
    )
    source_visual, source_visual_mask = _visual_inputs(
        source,
        patch_size=cfg.visual_patch_size,
        canvas_size=cfg.visual_canvas_size,
        torch=torch,
        device=device,
    )
    target_visual, target_visual_mask = _visual_inputs(
        target,
        patch_size=cfg.visual_patch_size,
        canvas_size=cfg.visual_canvas_size,
        torch=torch,
        device=device,
    )
    return (
        source_token_ids,
        source_numeric,
        source_relations,
        target_token_ids,
        target_numeric,
        target_relations,
        pair_relations,
        source_visual,
        source_visual_mask,
        target_visual,
        target_visual_mask,
    )


class LearnedGraphMatcher:
    """Inference adapter that abstains instead of replaying source coordinates."""

    def __init__(
        self,
        model: Any,
        *,
        config: MatcherConfig | None = None,
        device: str = "cpu",
    ) -> None:
        self.model = model
        self.config = config or MatcherConfig()
        self.device = device
        self.model.to(device)
        self.model.eval()

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "LearnedGraphMatcher":
        torch = _require_torch()
        payload = torch.load(Path(path), map_location=device)
        config = MatcherConfig(**dict(payload["matcher_config"]))
        model = build_relation_aware_matcher(config)
        model.load_state_dict(payload["state_dict"])
        return cls(model, config=config, device=device)

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
        torch = _require_torch()
        if not any(node.node_id == source_node_id for node in source.nodes):
            return LearnedMatch(None, 0.0, 0.0, "source_node_missing", ())
        if len(source.nodes) > self.config.source_context_nodes:
            source = local_context_graph(
                source,
                anchor_node_id=source_node_id,
                max_nodes=self.config.source_context_nodes,
            )
        source_index = next(
            (index for index, node in enumerate(source.nodes) if node.node_id == source_node_id),
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
        inputs = matcher_inputs(
            source,
            target,
            config=self.config,
            device=self.device,
        )
        with torch.no_grad():
            output = self.model(*inputs)
            selected_logits = output["logits_ab"][source_index][candidate_indices]
            selected_affinity = output["affinity"][source_index][candidate_indices]
            rank_probabilities = torch.softmax(selected_logits, dim=0)
            match_probabilities = torch.sigmoid(selected_affinity)
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
            return LearnedMatch(None, match_probability, margin, "learned_low_confidence", scores)
        target_node = next(node for node in target.nodes if node.node_id == best_id)
        return LearnedMatch(target_node, match_probability, margin, "learned_match", scores)


def save_matcher_checkpoint(
    path: str | Path,
    model: Any,
    *,
    config: MatcherConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a state-dict checkpoint with an explicit architecture contract."""

    torch = _require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "omnitransfer_relation_matcher_v2",
            "matcher_config": asdict(config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )


def parameter_count(model: Any) -> int:
    """Return trainable parameter count for experiment reporting."""

    return sum(int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad)


def prepare_visual_asset(
    screenshot_path: str | Path,
    *,
    canvas_size: int = 384,
) -> tuple[int, int, int]:
    """Decode and resize one screenshot before latency-critical matching."""

    path = Path(screenshot_path)
    if not path.is_file():
        raise FileNotFoundError(f"screenshot is missing: {path}")
    array = _load_rgb_array(str(path.resolve()), canvas_size)
    return tuple(int(value) for value in array.shape)


def _visual_inputs(
    graph: UIGraph,
    *,
    patch_size: int,
    canvas_size: int,
    torch: Any,
    device: str | Any,
) -> tuple[Any, Any]:
    screenshot_path = str(graph.metadata.get("screenshot_path") or "")
    empty = torch.zeros(
        (len(graph.nodes), 3, patch_size, patch_size),
        dtype=torch.float32,
        device=device,
    )
    mask = torch.zeros((len(graph.nodes), 1), dtype=torch.float32, device=device)
    if not screenshot_path:
        return empty, mask
    path = Path(screenshot_path)
    if not path.is_file():
        return empty, mask
    image_array = _load_rgb_array(str(path.resolve()), canvas_size)
    image = torch.as_tensor(image_array, dtype=torch.uint8, device=device)
    image = image.permute(2, 0, 1).unsqueeze(0).to(dtype=torch.float32).div_(255.0)
    graph_width = float(graph.width or image_array.shape[1])
    graph_height = float(graph.height or image_array.shape[0])
    normalized_boxes: list[tuple[float, float, float, float]] = []
    available: list[float] = []
    for node in graph.nodes:
        visual_bbox = _visual_bbox(node)
        if visual_bbox is None or graph_width <= 0.0 or graph_height <= 0.0:
            normalized_boxes.append((-1.0, -1.0, -1.0, -1.0))
            available.append(0.0)
            continue
        normalized_boxes.append(
            (
                _clip(2.0 * visual_bbox[0] / graph_width - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[1] / graph_height - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[2] / graph_width - 1.0, -1.0, 1.0),
                _clip(2.0 * visual_bbox[3] / graph_height - 1.0, -1.0, 1.0),
            )
        )
        available.append(1.0)
    boxes = torch.as_tensor(
        normalized_boxes,
        dtype=torch.float32,
        device=device,
    )
    offsets = torch.linspace(0.0, 1.0, patch_size, device=device)
    grid_x = boxes[:, 0, None, None] + (
        boxes[:, 2] - boxes[:, 0]
    )[:, None, None] * offsets[None, None, :]
    grid_y = boxes[:, 1, None, None] + (
        boxes[:, 3] - boxes[:, 1]
    )[:, None, None] * offsets[None, :, None]
    grid_x = grid_x.expand(-1, patch_size, -1)
    grid_y = grid_y.expand(-1, -1, patch_size)
    grid = torch.stack((grid_x, grid_y), dim=-1)
    patches = torch.nn.functional.grid_sample(
        image.expand(len(graph.nodes), -1, -1, -1),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )
    patches = _apply_visual_transform(patches, graph=graph, torch=torch, device=device)
    return (
        patches,
        torch.tensor(available, dtype=torch.float32, device=device).unsqueeze(1),
    )


def _visual_bbox(node: UINode) -> BBox | None:
    value = node.metadata.get("visual_bbox")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        try:
            x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
        except (TypeError, ValueError):
            return node.bbox
        if x2 > x1 and y2 > y1:
            return x1, y1, x2, y2
    return node.bbox


def _apply_visual_transform(
    patches: Any,
    *,
    graph: UIGraph,
    torch: Any,
    device: str | Any,
) -> Any:
    value = graph.metadata.get("visual_transform")
    if not isinstance(value, dict):
        return patches
    try:
        brightness = float(value.get("brightness") or 0.0)
        contrast = float(value.get("contrast") or 1.0)
        channel_scale = tuple(float(item) for item in value.get("channel_scale") or ())
    except (TypeError, ValueError):
        return patches
    if len(channel_scale) != 3 or contrast <= 0.0:
        return patches
    gains = torch.tensor(
        channel_scale,
        dtype=patches.dtype,
        device=device,
    ).view(1, 3, 1, 1)
    return (((patches - 0.5) * contrast + 0.5 + brightness) * gains).clamp_(0.0, 1.0)


@lru_cache(maxsize=512)
def _load_rgb_array(screenshot_path: str, canvas_size: int) -> Any:
    try:
        import numpy as np
        from PIL import Image
    except Exception as exc:
        raise RuntimeError(
            "Visual UI encoding requires NumPy and Pillow. Install omnitransfer[train]."
        ) from exc

    with Image.open(screenshot_path) as image:
        image = image.convert("RGB")
        if canvas_size > 0 and max(image.size) > canvas_size:
            scale = canvas_size / max(image.size)
            image = image.resize(
                (
                    max(1, round(image.width * scale)),
                    max(1, round(image.height * scale)),
                ),
                resample=Image.Resampling.BILINEAR,
            )
        return np.array(image, dtype=np.uint8, copy=True)


def _node_token_ids(node: UINode, *, config: MatcherConfig) -> tuple[int, ...]:
    pieces: list[str] = ["node"]
    fields: list[tuple[str, str]] = [
        ("text", node.text),
        ("desc", node.content_desc),
        ("resource", node.resource_id),
        ("class", node.class_name),
    ]
    metadata = node.metadata or {}
    for field_name in ("action_type", "parent_text", "screen_region"):
        fields.append((field_name, str(metadata.get(field_name) or "")))
    for field_name in ("sibling_texts", "nearby_texts"):
        values = metadata.get(field_name) or ()
        if isinstance(values, str):
            values = (values,)
        fields.append((field_name, " ".join(str(value) for value in values)))
    for field_name, value in fields:
        normalized = _normalize_text(value)
        if not normalized:
            continue
        pieces.append(f"field:{field_name}")
        words = re.findall(r"[a-z0-9]+", normalized)
        for word in words:
            pieces.append(f"{field_name}:word:{word}")
            if len(word) >= 3:
                padded = f"^{word}$"
                pieces.extend(
                    f"{field_name}:ngram:{padded[index:index + 3]}"
                    for index in range(len(padded) - 2)
                )
    token_ids: list[int] = []
    seen: set[int] = set()
    for piece in pieces:
        token_id = _token_bucket(piece, config.vocab_size)
        if token_id in seen:
            continue
        seen.add(token_id)
        token_ids.append(token_id)
        if len(token_ids) >= config.max_tokens:
            break
    return tuple(token_ids + [0] * (config.max_tokens - len(token_ids)))


def _node_numeric_features(node: UINode, graph: UIGraph) -> tuple[float, ...]:
    bbox = _normalized_bbox(node.bbox, graph)
    if bbox is None:
        x1 = y1 = x2 = y2 = center_x = center_y = width = height = area = aspect = 0.0
        has_bbox = 0.0
    else:
        x1, y1, x2, y2 = bbox
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)
        center_x = x1 + width / 2.0
        center_y = y1 + height / 2.0
        area = width * height
        aspect = _clip(math.log(max(width, 1e-6) / max(height, 1e-6)) / 4.0, -1.0, 1.0)
        has_bbox = 1.0
    values = (
        float(node.clickable),
        float(node.editable),
        float(node.scrollable),
        float(node.enabled),
        has_bbox,
        x1,
        y1,
        x2,
        y2,
        center_x,
        center_y,
        width,
        height,
        area,
        aspect,
        min(float(node.depth) / 32.0, 1.0),
        min(float(len(node.child_ids)) / 16.0, 1.0),
        float(node.parent_id is not None),
    )
    if len(values) != NUMERIC_FEATURE_DIM:
        raise AssertionError("unexpected numeric feature dimension")
    return values


def _relation_features(
    graph: UIGraph,
) -> Any:
    context = _relation_context(graph)
    return _relation_matrix(context, context, same_graph=True)


def _node_relation(
    source: UINode,
    target: UINode,
    source_graph: UIGraph,
    target_graph: UIGraph,
) -> tuple[float, ...]:
    same_graph = source_graph is target_graph or source_graph.graph_id == target_graph.graph_id
    source_context = _relation_context(source_graph)
    target_context = _relation_context(target_graph)
    source_index = next(
        index for index, node in enumerate(source_graph.nodes) if node.node_id == source.node_id
    )
    target_index = next(
        index for index, node in enumerate(target_graph.nodes) if node.node_id == target.node_id
    )
    return _contextual_node_relation(
        source_index,
        target_index,
        source_context,
        target_context,
        same_graph=same_graph,
    )


def _relation_context(graph: UIGraph) -> _RelationContext:
    nodes_by_id = {node.node_id: node for node in graph.nodes}
    bboxes = tuple(_normalized_bbox(node.bbox, graph) for node in graph.nodes)
    ancestor_paths: dict[str, tuple[str, ...]] = {}
    for node in graph.nodes:
        path = [node.node_id]
        current = node
        visited = {node.node_id}
        while current.parent_id and current.parent_id not in visited:
            path.append(current.parent_id)
            visited.add(current.parent_id)
            parent = nodes_by_id.get(current.parent_id)
            if parent is None:
                break
            current = parent
        ancestor_paths[node.node_id] = tuple(path)
    return _RelationContext(
        graph=graph,
        bboxes=bboxes,
        centers=tuple(_center(bbox) for bbox in bboxes),
        sizes=tuple(_size(bbox) for bbox in bboxes),
        node_indices={node.node_id: index for index, node in enumerate(graph.nodes)},
        ancestor_sets={
            node_id: frozenset(path[1:]) for node_id, path in ancestor_paths.items()
        },
        ancestor_paths=ancestor_paths,
        path_positions={
            node_id: {ancestor_id: index for index, ancestor_id in enumerate(path)}
            for node_id, path in ancestor_paths.items()
        },
    )


def _relation_matrix(
    source_context: _RelationContext,
    target_context: _RelationContext,
    *,
    same_graph: bool,
) -> Any:
    np = _require_numpy()
    source_count = len(source_context.graph.nodes)
    target_count = len(target_context.graph.nodes)
    values = np.zeros(
        (source_count, target_count, RELATION_FEATURE_DIM),
        dtype=np.float32,
    )
    source_centers = np.asarray(source_context.centers, dtype=np.float32)
    target_centers = np.asarray(target_context.centers, dtype=np.float32)
    source_sizes = np.asarray(source_context.sizes, dtype=np.float32)
    target_sizes = np.asarray(target_context.sizes, dtype=np.float32)
    delta_x = target_centers[None, :, 0] - source_centers[:, None, 0]
    delta_y = target_centers[None, :, 1] - source_centers[:, None, 1]
    source_width = source_sizes[:, None, 0]
    source_height = source_sizes[:, None, 1]
    target_width = target_sizes[None, :, 0]
    target_height = target_sizes[None, :, 1]
    same_row = np.abs(delta_y) <= np.maximum(
        np.maximum(source_height, target_height),
        0.02,
    ) * 0.5
    same_column = np.abs(delta_x) <= np.maximum(
        np.maximum(source_width, target_width),
        0.02,
    ) * 0.5
    overlap = _pairwise_iou(source_context.bboxes, target_context.bboxes, np=np)
    parent = np.zeros((source_count, target_count), dtype=bool)
    child = np.zeros_like(parent)
    sibling = np.zeros_like(parent)
    ancestor = np.zeros_like(parent)
    descendant = np.zeros_like(parent)
    identity = np.zeros_like(parent)
    tree_distance = np.full((source_count, target_count), 16.0, dtype=np.float32)
    if same_graph:
        source_ids = np.asarray(
            [node.node_id for node in source_context.graph.nodes],
            dtype=object,
        )
        target_ids = np.asarray(
            [node.node_id for node in target_context.graph.nodes],
            dtype=object,
        )
        source_parents = np.asarray(
            [node.parent_id or "" for node in source_context.graph.nodes],
            dtype=object,
        )
        target_parents = np.asarray(
            [node.parent_id or "" for node in target_context.graph.nodes],
            dtype=object,
        )
        identity = source_ids[:, None] == target_ids[None, :]
        parent = source_ids[:, None] == target_parents[None, :]
        child = source_parents[:, None] == target_ids[None, :]
        sibling = (
            (source_parents[:, None] == target_parents[None, :])
            & (source_parents[:, None] != "")
            & ~identity
        )
        for target_index, target_node in enumerate(target_context.graph.nodes):
            for ancestor_id in target_context.ancestor_sets.get(target_node.node_id, ()):
                source_index = source_context.node_indices.get(ancestor_id)
                if source_index is not None:
                    ancestor[source_index, target_index] = True
        for source_index, source_node in enumerate(source_context.graph.nodes):
            for ancestor_id in source_context.ancestor_sets.get(source_node.node_id, ()):
                target_index = target_context.node_indices.get(ancestor_id)
                if target_index is not None:
                    descendant[source_index, target_index] = True
            for target_index, target_node in enumerate(target_context.graph.nodes):
                tree_distance[source_index, target_index] = _context_tree_distance(
                    source_node.node_id,
                    target_node.node_id,
                    source_context,
                )
    values[..., 0] = identity
    values[..., 1] = parent
    values[..., 2] = child
    values[..., 3] = sibling
    values[..., 4] = ancestor
    values[..., 5] = descendant
    values[..., 6] = same_row
    values[..., 7] = same_column
    values[..., 8] = overlap > 0.0
    values[..., 9] = np.clip(delta_x, -1.0, 1.0)
    values[..., 10] = np.clip(delta_y, -1.0, 1.0)
    values[..., 11] = np.minimum(np.abs(delta_x), 1.0)
    values[..., 12] = np.minimum(np.abs(delta_y), 1.0)
    values[..., 13] = np.clip(
        np.log(np.maximum(target_width, 1e-6) / np.maximum(source_width, 1e-6)) / 4.0,
        -1.0,
        1.0,
    )
    values[..., 14] = np.clip(
        np.log(np.maximum(target_height, 1e-6) / np.maximum(source_height, 1e-6)) / 4.0,
        -1.0,
        1.0,
    )
    values[..., 15] = overlap
    values[..., 16] = np.minimum(tree_distance / 16.0, 1.0)
    values[..., 17] = (
        sibling
        | parent
        | child
        | (np.hypot(delta_x, delta_y) <= 0.25)
    )
    return values


def _pairwise_iou(
    source_bboxes: tuple[BBox | None, ...],
    target_bboxes: tuple[BBox | None, ...],
    *,
    np: Any,
) -> Any:
    source = np.asarray(
        [bbox or (0.0, 0.0, 0.0, 0.0) for bbox in source_bboxes],
        dtype=np.float32,
    )
    target = np.asarray(
        [bbox or (0.0, 0.0, 0.0, 0.0) for bbox in target_bboxes],
        dtype=np.float32,
    )
    left = np.maximum(source[:, None, 0], target[None, :, 0])
    top = np.maximum(source[:, None, 1], target[None, :, 1])
    right = np.minimum(source[:, None, 2], target[None, :, 2])
    bottom = np.minimum(source[:, None, 3], target[None, :, 3])
    intersection = np.maximum(0.0, right - left) * np.maximum(0.0, bottom - top)
    source_area = np.maximum(0.0, source[:, 2] - source[:, 0]) * np.maximum(
        0.0,
        source[:, 3] - source[:, 1],
    )
    target_area = np.maximum(0.0, target[:, 2] - target[:, 0]) * np.maximum(
        0.0,
        target[:, 3] - target[:, 1],
    )
    union = source_area[:, None] + target_area[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def _contextual_node_relation(
    source_index: int,
    target_index: int,
    source_context: _RelationContext,
    target_context: _RelationContext,
    *,
    same_graph: bool,
) -> tuple[float, ...]:
    source = source_context.graph.nodes[source_index]
    target = target_context.graph.nodes[target_index]
    source_bbox = source_context.bboxes[source_index]
    target_bbox = target_context.bboxes[target_index]
    source_center = source_context.centers[source_index]
    target_center = target_context.centers[target_index]
    delta_x = target_center[0] - source_center[0]
    delta_y = target_center[1] - source_center[1]
    source_width, source_height = source_context.sizes[source_index]
    target_width, target_height = target_context.sizes[target_index]
    parent = same_graph and target.parent_id == source.node_id
    child = same_graph and source.parent_id == target.node_id
    sibling = bool(
        same_graph
        and source.parent_id
        and source.parent_id == target.parent_id
        and source.node_id != target.node_id
    )
    ancestor = same_graph and source.node_id in target_context.ancestor_sets.get(
        target.node_id,
        (),
    )
    descendant = same_graph and target.node_id in source_context.ancestor_sets.get(
        source.node_id,
        (),
    )
    same_row = abs(delta_y) <= max(source_height, target_height, 0.02) * 0.5
    same_column = abs(delta_x) <= max(source_width, target_width, 0.02) * 0.5
    overlap = _iou(source_bbox, target_bbox)
    tree_distance = _context_tree_distance(
        source.node_id,
        target.node_id,
        source_context,
    ) if same_graph else 16
    values = (
        float(same_graph and source.node_id == target.node_id),
        float(parent),
        float(child),
        float(sibling),
        float(ancestor),
        float(descendant),
        float(same_row),
        float(same_column),
        float(overlap > 0.0),
        _clip(delta_x, -1.0, 1.0),
        _clip(delta_y, -1.0, 1.0),
        min(abs(delta_x), 1.0),
        min(abs(delta_y), 1.0),
        _clip(math.log(max(target_width, 1e-6) / max(source_width, 1e-6)) / 4.0, -1.0, 1.0),
        _clip(math.log(max(target_height, 1e-6) / max(source_height, 1e-6)) / 4.0, -1.0, 1.0),
        overlap,
        min(float(tree_distance) / 16.0, 1.0),
        float(sibling or parent or child or math.hypot(delta_x, delta_y) <= 0.25),
    )
    if len(values) != RELATION_FEATURE_DIM:
        raise AssertionError("unexpected relation feature dimension")
    return values


def _context_tree_distance(
    source_id: str,
    target_id: str,
    context: _RelationContext,
) -> int:
    if source_id == target_id:
        return 0
    source_positions = context.path_positions.get(source_id, {})
    target_path = context.ancestor_paths.get(target_id, (target_id,))
    distances = (
        source_positions[node_id] + target_index
        for target_index, node_id in enumerate(target_path)
        if node_id in source_positions
    )
    return min(distances, default=16)


def _normalized_bbox(bbox: BBox | None, graph: UIGraph) -> BBox | None:
    if bbox is None:
        return None
    width = float(graph.width or max((node.bbox or (0.0, 0.0, 1.0, 1.0))[2] for node in graph.nodes))
    height = float(graph.height or max((node.bbox or (0.0, 0.0, 1.0, 1.0))[3] for node in graph.nodes))
    if width <= 0.0 or height <= 0.0:
        return None
    return (
        _clip(bbox[0] / width, 0.0, 1.0),
        _clip(bbox[1] / height, 0.0, 1.0),
        _clip(bbox[2] / width, 0.0, 1.0),
        _clip(bbox[3] / height, 0.0, 1.0),
    )


def _center(bbox: BBox | None) -> tuple[float, float]:
    if bbox is None:
        return 0.0, 0.0
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


def _size(bbox: BBox | None) -> tuple[float, float]:
    if bbox is None:
        return 0.0, 0.0
    return max(0.0, bbox[2] - bbox[0]), max(0.0, bbox[3] - bbox[1])


def _iou(first: BBox | None, second: BBox | None) -> float:
    if first is None or second is None:
        return 0.0
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[2], second[2])
    bottom = min(first[3], second[3])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _is_ancestor(ancestor_id: str, node_id: str, graph: UIGraph) -> bool:
    nodes = {node.node_id: node for node in graph.nodes}
    current = nodes.get(node_id)
    visited: set[str] = set()
    while current is not None and current.parent_id and current.parent_id not in visited:
        if current.parent_id == ancestor_id:
            return True
        visited.add(current.parent_id)
        current = nodes.get(current.parent_id)
    return False


def _tree_distance(source_id: str, target_id: str, graph: UIGraph) -> int:
    if source_id == target_id:
        return 0
    source_path = _ancestor_path(source_id, graph)
    target_path = _ancestor_path(target_id, graph)
    source_positions = {node_id: index for index, node_id in enumerate(source_path)}
    distances = [
        source_positions[node_id] + target_index
        for target_index, node_id in enumerate(target_path)
        if node_id in source_positions
    ]
    return min(distances) if distances else 16


def _ancestor_path(node_id: str, graph: UIGraph) -> list[str]:
    nodes = {node.node_id: node for node in graph.nodes}
    path = [node_id]
    current = nodes.get(node_id)
    visited = {node_id}
    while current is not None and current.parent_id and current.parent_id not in visited:
        path.append(current.parent_id)
        visited.add(current.parent_id)
        current = nodes.get(current.parent_id)
    return path


def _normalize_text(value: str) -> str:
    value = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _token_bucket(piece: str, vocab_size: int) -> int:
    if vocab_size < 2:
        raise ValueError("vocab_size must be at least 2")
    digest = hashlib.blake2b(piece.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (vocab_size - 1) + 1


def _clip(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(float(value), maximum))


def _require_torch() -> Any:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError(
            "PyTorch is required for the learned matcher. Install omnitransfer[train]."
        ) from exc
    return torch


def _require_numpy() -> Any:
    try:
        import numpy as np
    except Exception as exc:
        raise RuntimeError(
            "NumPy is required for the learned matcher. Install omnitransfer[train]."
        ) from exc
    return np
