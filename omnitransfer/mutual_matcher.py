"""Mutual source-target assignment from explicit pair evidence matrices."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

from omnitransfer.learned_matcher import (
    LearnedGraphMatcher,
    MatcherConfig,
    NUMERIC_FEATURE_DIM,
    RELATION_FEATURE_DIM,
)


class MutualGraphMatcher(LearnedGraphMatcher):
    """Inference adapter for the single-matrix mutual matcher."""

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str = "cpu",
    ) -> "MutualGraphMatcher":
        torch = _require_torch()
        payload = torch.load(Path(path), map_location=device)
        if payload.get("schema_version") != "omnitransfer_mutual_matcher_v3":
            raise ValueError("checkpoint is not a mutual assignment matcher")
        config = MatcherConfig(**dict(payload["matcher_config"]))
        model = build_mutual_assignment_matcher(config)
        model.load_state_dict(payload["state_dict"])
        return cls(model, config=config, device=device)


def save_mutual_matcher_checkpoint(
    path: str | Path,
    model: Any,
    *,
    config: MatcherConfig,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save a versioned checkpoint that cannot be confused with the baseline."""

    torch = _require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": "omnitransfer_mutual_matcher_v3",
            "matcher_config": asdict(config),
            "state_dict": model.state_dict(),
            "metadata": dict(metadata or {}),
        },
        output,
    )


def build_mutual_assignment_matcher(
    config: MatcherConfig | None = None,
) -> Any:
    """Build one mutual matrix from explicit semantic and structural evidence."""

    torch = _require_torch()
    nn = torch.nn
    cfg = config or MatcherConfig()
    if cfg.hidden_dim % cfg.num_heads != 0:
        raise ValueError("hidden_dim must be divisible by num_heads")
    if cfg.visual_canvas_size < cfg.visual_patch_size:
        raise ValueError("visual_canvas_size must cover one visual patch")

    class NodeEncoder(nn.Module):
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
            self.output_norm = nn.LayerNorm(cfg.hidden_dim)

        def forward(
            self,
            token_ids: Any,
            numeric_features: Any,
            visual_patches: Any,
            visual_mask: Any,
        ) -> Any:
            mask = token_ids.ne(0).unsqueeze(-1)
            token_values = self.token_embedding(token_ids)
            token_sum = (token_values * mask).sum(dim=1)
            token_count = mask.sum(dim=1).clamp_min(1)
            token_states = self.token_projection(token_sum / token_count)
            numeric_states = self.numeric_projection(numeric_features)
            visual_states = self.visual_encoder(visual_patches)
            missing = self.missing_visual.unsqueeze(0).expand(visual_states.shape[0], -1)
            fused_visual = visual_mask * visual_states + (1.0 - visual_mask) * missing
            states = self.output_norm(token_states + numeric_states + fused_visual)
            return states, token_states, numeric_states, visual_states

    class PairEvidenceHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            pair_dim = cfg.hidden_dim * 2
            self.network = nn.Sequential(
                nn.LayerNorm(pair_dim),
                nn.Linear(pair_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim, 1),
            )

        def forward(self, source: Any, target: Any) -> Any:
            source_pairs = source[:, None, :]
            target_pairs = target[None, :, :]
            features = torch.cat(
                [
                    torch.abs(source_pairs - target_pairs),
                    source_pairs * target_pairs,
                ],
                dim=-1,
            )
            return self.network(features).squeeze(-1)

    class RelationEncoderLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.qkv = nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 3, bias=False)
            self.relation_bias = nn.Sequential(
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, cfg.num_heads),
            )
            self.output = nn.Linear(cfg.hidden_dim, cfg.hidden_dim, bias=False)
            self.attention_norm = nn.LayerNorm(cfg.hidden_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(cfg.hidden_dim, cfg.hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim),
            )
            self.output_norm = nn.LayerNorm(cfg.hidden_dim)
            self.dropout = nn.Dropout(cfg.dropout)
            self.head_dim = cfg.hidden_dim // cfg.num_heads

        def forward(self, states: Any, relations: Any) -> Any:
            node_count = int(states.shape[0])
            query, key, value = self.qkv(states).reshape(
                node_count,
                3,
                cfg.num_heads,
                self.head_dim,
            ).unbind(dim=1)
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            scores = (query @ key.transpose(-1, -2)) * self.head_dim**-0.5
            scores = scores + self.relation_bias(relations).permute(2, 0, 1)
            attention = torch.softmax(scores, dim=-1)
            context = (attention @ value).transpose(0, 1).reshape(
                node_count,
                cfg.hidden_dim,
            )
            states = self.attention_norm(
                states + self.dropout(self.output(context))
            )
            return self.output_norm(
                states + self.dropout(self.feed_forward(states))
            )

    class MutualAssignmentNetwork(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.node_encoder = NodeEncoder()
            self.layers = nn.ModuleList(
                RelationEncoderLayer() for _ in range(cfg.num_layers)
            )
            self.semantic_score = PairEvidenceHead()
            self.visual_score = PairEvidenceHead()
            self.attribute_score = PairEvidenceHead()
            self.context_score = PairEvidenceHead()
            self.geometry_score = nn.Sequential(
                nn.LayerNorm(RELATION_FEATURE_DIM),
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, 1),
            )
            self.anchor_seed = nn.Sequential(
                nn.LayerNorm(6),
                nn.Linear(6, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, 1),
            )
            self.anchor_edge_score = nn.Sequential(
                nn.Linear(RELATION_FEATURE_DIM, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.relation_hidden_dim, 1),
            )
            self.pair_fusion = nn.Sequential(
                nn.LayerNorm(7),
                nn.Linear(7, cfg.relation_hidden_dim),
                nn.GELU(),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.relation_hidden_dim, 1),
            )

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
            (
                source_states,
                source_semantic,
                source_attributes,
                source_visual_states,
            ) = self.node_encoder(
                source_token_ids,
                source_numeric,
                source_visual,
                source_visual_mask,
            )
            (
                target_states,
                target_semantic,
                target_attributes,
                target_visual_states,
            ) = self.node_encoder(
                target_token_ids,
                target_numeric,
                target_visual,
                target_visual_mask,
            )
            for layer in self.layers:
                source_states = layer(source_states, source_relations)
                target_states = layer(target_states, target_relations)
            visual_available = source_visual_mask * target_visual_mask.T
            semantic = self.semantic_score(source_semantic, target_semantic)
            visual = self.visual_score(
                source_visual_states,
                target_visual_states,
            ) * visual_available
            attributes = self.attribute_score(
                source_attributes,
                target_attributes,
            )
            context = self.context_score(source_states, target_states)
            geometry = self.geometry_score(source_target_relations).squeeze(-1)
            anchor_seed_features = torch.stack(
                [
                    semantic,
                    visual,
                    attributes,
                    context,
                    geometry,
                    visual_available,
                ],
                dim=-1,
            )
            anchor_seed = self.anchor_seed(anchor_seed_features).squeeze(-1)
            anchor = self._anchor_support(
                anchor_seed,
                source_relations,
                target_relations,
            )
            feature_matrices = {
                "semantic": semantic,
                "visual": visual,
                "attributes": attributes,
                "context": context,
                "geometry": geometry,
                "anchor": anchor,
                "visual_available": visual_available,
            }
            affinity = self.pair_fusion(
                torch.stack(tuple(feature_matrices.values()), dim=-1)
            ).squeeze(-1)
            logits_ab, logits_ba = mutual_assignment_logits(affinity)
            return {
                "logits_ab": logits_ab,
                "logits_ba": logits_ba,
                "affinity": affinity,
                "feature_matrices": feature_matrices,
                "source_states": source_states,
                "target_states": target_states,
            }

        def _anchor_support(
            self,
            seed: Any,
            source_relations: Any,
            target_relations: Any,
        ) -> Any:
            row_log_probabilities = torch.log_softmax(seed, dim=1)
            column_log_probabilities = torch.log_softmax(seed, dim=0)
            alignment = torch.exp(
                0.5 * (row_log_probabilities + column_log_probabilities)
            )
            source_edges = self._local_edges(source_relations)
            target_edges = self._local_edges(target_relations)
            support = source_edges @ alignment @ target_edges.T
            source_degree = source_edges.sum(dim=1).clamp_min(1e-6)
            target_degree = target_edges.sum(dim=1).clamp_min(1e-6)
            normalizer = torch.sqrt(source_degree[:, None] * target_degree[None, :])
            return support / normalizer

        def _local_edges(self, relations: Any) -> Any:
            learned_weight = torch.sigmoid(
                self.anchor_edge_score(relations).squeeze(-1)
            )
            local_mask = relations[..., 17]
            non_identity = 1.0 - relations[..., 0]
            return learned_weight * local_mask * non_identity

    return MutualAssignmentNetwork()


def mutual_assignment_logits(
    affinity: Any,
) -> tuple[Any, Any]:
    """Return bidirectional pair logits without a learned NULL class."""

    torch = _require_torch()
    if affinity.ndim != 2:
        raise ValueError("affinity must be a two-dimensional matrix")
    row_log_probabilities = torch.log_softmax(affinity, dim=1)
    column_log_probabilities = torch.log_softmax(affinity, dim=0)
    mutual = 0.5 * (
        row_log_probabilities
        + column_log_probabilities
    )
    return mutual, mutual.T


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("mutual matching requires PyTorch") from exc
    return torch
