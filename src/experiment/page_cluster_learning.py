"""Learn layout-invariant page clusters from unified UI correspondence pairs."""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable

import numpy as np
import torch

from omniflow.transfer.embedding import EncoderWeights, SoftPageWordWeights

UI_CORRESPONDENCE_PAIR_SCHEMA = "omnitransfer.ui_correspondence_pair.v1"


@dataclass(frozen=True)
class UnifiedPage:
    page_id: str
    graph: dict[str, Any]
    action_node_ids: tuple[str, ...]
    screenshot_path: str = ""


@dataclass(frozen=True)
class UnifiedPagePair:
    pair_id: str
    cluster_id: str
    app_id: str
    source: UnifiedPage
    target: UnifiedPage


@dataclass(frozen=True)
class ClusterPage:
    cluster_id: str
    app_id: str
    page: UnifiedPage


@dataclass(frozen=True)
class EncodedClusterPage:
    cluster_id: str
    app_id: str
    page_id: str
    descriptors: np.ndarray
    structure_features: np.ndarray
    parent_indices: np.ndarray
    page: UnifiedPage | None = None
    masked_descriptors: np.ndarray | None = None


@dataclass(frozen=True)
class FunctionalPageView:
    descriptors: torch.Tensor
    structure_features: torch.Tensor
    parent_indices: torch.Tensor


def make_self_supervised_view(
    page: EncodedClusterPage,
    *,
    rng: random.Random,
    node_drop_probability: float = 0.25,
    edge_drop_probability: float = 0.10,
    descriptor_drop_probability: float = 0.10,
    semantic_drop_probability: float = 0.30,
    minimum_nodes: int = 4,
) -> FunctionalPageView:
    """Create one topology view while retaining every actionable XML node."""

    for probability in (
        node_drop_probability,
        edge_drop_probability,
        descriptor_drop_probability,
        semantic_drop_probability,
    ):
        if not 0.0 <= probability <= 1.0:
            raise ValueError("functional_page_view_probability_invalid")
    node_count = int(page.descriptors.shape[0])
    if node_count == 0 or page.structure_features.shape != (node_count, 8):
        raise ValueError("functional_page_encoded_page_invalid")
    required = {
        index
        for index in range(node_count)
        if page.structure_features[index, -1] > 0.5
    }
    kept = set(required)
    optional = [index for index in range(node_count) if index not in required]
    for index in optional:
        if rng.random() >= node_drop_probability:
            kept.add(index)
    rng.shuffle(optional)
    for index in optional:
        if len(kept) >= min(max(minimum_nodes, 1), node_count):
            break
        kept.add(index)
    if not kept:
        kept.add(rng.randrange(node_count))
    kept_indices = tuple(sorted(kept))
    new_index = {old: current for current, old in enumerate(kept_indices)}
    parent_indices = []
    for old_index in kept_indices:
        old_parent = int(page.parent_indices[old_index])
        parent = new_index.get(old_parent, -1)
        if parent >= 0 and rng.random() < edge_drop_probability:
            parent = -1
        parent_indices.append(parent)
    descriptor_values = page.descriptors
    if (
        page.masked_descriptors is not None
        and rng.random() < semantic_drop_probability
    ):
        descriptor_values = page.masked_descriptors
    descriptors = torch.as_tensor(
        descriptor_values[list(kept_indices)], dtype=torch.float32
    ).clone()
    if descriptor_drop_probability > 0.0:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(rng.getrandbits(63))
        keep_mask = torch.rand(
            descriptors.shape,
            generator=generator,
            dtype=descriptors.dtype,
        ).ge(descriptor_drop_probability)
        descriptors *= keep_mask
    return FunctionalPageView(
        descriptors=descriptors,
        structure_features=torch.as_tensor(
            page.structure_features[list(kept_indices)], dtype=torch.float32
        ),
        parent_indices=torch.as_tensor(parent_indices, dtype=torch.long),
    )


class FrozenV9NodeBackbone:
    """Expose frozen v9.2 node descriptors with an explicit modality switch."""

    def __init__(
        self,
        *,
        omnitransfer_root: str | Path,
        checkpoint: str | Path,
        device: str = "cpu",
    ) -> None:
        root = Path(omnitransfer_root).expanduser().resolve()
        checkpoint_path = Path(checkpoint).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ValueError(f"functional_page_checkpoint_missing:{checkpoint_path}")
        for value in (root, root / "src"):
            if str(value) not in sys.path:
                sys.path.insert(0, str(value))
        from omnitransfer.learned_matcher import (
            ALL_NODE_CANDIDATE_POLICY,
            RelationAwareMatcher,
            matcher_inputs,
        )
        from omnitransfer.ui_graph import graph_from_record

        from omnitransfer import learned_matcher

        matcher = RelationAwareMatcher.from_checkpoint(
            checkpoint_path,
            device=device,
        )
        self.config = replace(
            matcher.config,
            candidate_policy=ALL_NODE_CANDIDATE_POLICY,
        )
        self.model = matcher.model.to(device).eval()
        self.device = device
        self._graph_from_record = graph_from_record
        self._matcher_inputs = matcher_inputs
        self._masked_token_encoder = learned_matcher._masked_context_token_ids
        feature_schema = learned_matcher._feature_schema_for_config(self.config)
        if feature_schema in {
            learned_matcher.OMNITRANSFER_V4_FEATURE_SCHEMA_ID,
            learned_matcher.OMNITRANSFER_V7_FEATURE_SCHEMA_ID,
        }:
            self._token_encoder = learned_matcher._multimodal_text_token_ids
            self._numeric_encoder = learned_matcher._multimodal_xml_features
        elif feature_schema == learned_matcher.PEMM_V3_FEATURE_SCHEMA_ID:
            self._token_encoder = learned_matcher._pemm_v3_node_token_ids
            self._numeric_encoder = learned_matcher._pemm_v3_node_numeric_features
        else:
            raise ValueError(f"functional_page_feature_schema_unsupported:{feature_schema}")

    def encode(
        self,
        page: UnifiedPage,
        *,
        structure_only: bool,
        mask_visible_text: bool = False,
    ) -> np.ndarray:
        record = dict(page.graph)
        if not structure_only and page.screenshot_path:
            record["screenshot_path"] = page.screenshot_path
        graph = self._graph_from_record(record, graph_id=page.page_id)
        if structure_only:
            token_ids = torch.as_tensor(
                [
                    (
                        self._masked_token_encoder(node, config=self.config)
                        if mask_visible_text
                        else self._token_encoder(node, config=self.config)
                    )
                    for node in graph.nodes
                ],
                dtype=torch.long,
                device=self.device,
            )
            numeric = torch.as_tensor(
                [self._numeric_encoder(node, graph) for node in graph.nodes],
                dtype=torch.float32,
                device=self.device,
            )
            numeric = structure_only_numeric_features(numeric)
            visual_patches = torch.zeros(
                (
                    len(graph.nodes),
                    3,
                    self.config.visual_patch_size,
                    self.config.visual_patch_size,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            visual_mask = torch.zeros(
                (len(graph.nodes), 1),
                dtype=torch.float32,
                device=self.device,
            )
            with torch.inference_mode():
                _, descriptors, _ = self.model._encode_nodes(
                    token_ids,
                    numeric,
                    visual_patches,
                    visual_mask,
                )
            values = descriptors.detach().cpu().numpy().astype(
                np.float32, copy=False
            )
            if values.shape != (len(graph.nodes), 128):
                raise ValueError("functional_page_v9_descriptors_must_be_n_by_128")
            return values
        inputs = self._matcher_inputs(
            graph,
            graph,
            config=self.config,
            device=self.device,
        )
        numeric = inputs[1]
        visual_patches = inputs[7]
        visual_mask = inputs[8]
        with torch.inference_mode():
            _, descriptors, _ = self.model._encode_nodes(
                inputs[0],
                numeric,
                visual_patches,
                visual_mask,
            )
        values = descriptors.detach().cpu().numpy().astype(np.float32, copy=False)
        if values.shape != (len(graph.nodes), 128):
            raise ValueError("functional_page_v9_descriptors_must_be_n_by_128")
        return values


def collect_cluster_pages(
    pairs: Iterable[UnifiedPagePair],
) -> tuple[ClusterPage, ...]:
    """Deduplicate pages while unioning every observed Action anchor."""

    pages: dict[str, ClusterPage] = {}
    for pair in pairs:
        for page in (pair.source, pair.target):
            previous = pages.get(page.page_id)
            if previous is None:
                pages[page.page_id] = ClusterPage(pair.cluster_id, pair.app_id, page)
                continue
            if previous.cluster_id != pair.cluster_id or previous.app_id != pair.app_id:
                raise ValueError(f"functional_page_label_conflict:{page.page_id}")
            action_node_ids = tuple(
                sorted(set(previous.page.action_node_ids).union(page.action_node_ids))
            )
            pages[page.page_id] = ClusterPage(
                pair.cluster_id,
                pair.app_id,
                UnifiedPage(
                    page_id=previous.page.page_id,
                    graph=previous.page.graph,
                    action_node_ids=action_node_ids,
                    screenshot_path=previous.page.screenshot_path or page.screenshot_path,
                ),
            )
    return tuple(pages[key] for key in sorted(pages))


def encode_cluster_pages(
    backbone: FrozenV9NodeBackbone,
    pages: Iterable[ClusterPage],
    *,
    structure_only: bool,
    include_masked_view: bool = False,
) -> tuple[EncodedClusterPage, ...]:
    """Freeze node evidence once before training the small page head."""

    encoded = []
    for sample in pages:
        structure = build_structure_action_inputs(sample.page)
        descriptors = backbone.encode(sample.page, structure_only=structure_only)
        masked_descriptors = (
            backbone.encode(
                sample.page,
                structure_only=True,
                mask_visible_text=True,
            )
            if structure_only and include_masked_view
            else None
        )
        if descriptors.shape[0] != structure.features.shape[0]:
            raise ValueError(f"functional_page_node_alignment_failed:{sample.page.page_id}")
        encoded.append(
            EncodedClusterPage(
                cluster_id=sample.cluster_id,
                app_id=sample.app_id,
                page_id=sample.page.page_id,
                descriptors=descriptors,
                structure_features=structure.features,
                parent_indices=structure.parent_indices,
                page=sample.page,
                masked_descriptors=masked_descriptors,
            )
        )
    return tuple(encoded)


def functional_head_embeddings(
    head: FunctionalSlotPageHead,
    pages: tuple[EncodedClusterPage, ...],
    *,
    device: str = "cpu",
) -> torch.Tensor:
    """Encode pages deterministically for retrieval evaluation."""

    head = head.to(device).eval()
    vectors = []
    with torch.inference_mode():
        for page in pages:
            output = head(
                torch.as_tensor(page.descriptors, dtype=torch.float32, device=device),
                torch.as_tensor(
                    page.structure_features, dtype=torch.float32, device=device
                ),
                torch.as_tensor(page.parent_indices, dtype=torch.long, device=device),
            )
            if page.masked_descriptors is None:
                vectors.append(output.embedding)
                continue
            masked_output = head(
                torch.as_tensor(
                    page.masked_descriptors,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(
                    page.structure_features,
                    dtype=torch.float32,
                    device=device,
                ),
                torch.as_tensor(page.parent_indices, dtype=torch.long, device=device),
            )
            vectors.append(
                torch.nn.functional.normalize(
                    0.75 * output.embedding + 0.25 * masked_output.embedding,
                    dim=0,
                )
            )
    if not vectors:
        raise ValueError("functional_page_embedding_pages_required")
    return torch.stack(vectors)


def strip_action_context(
    pages: tuple[EncodedClusterPage, ...],
) -> tuple[EncodedClusterPage, ...]:
    """Create an evaluation view with observed trajectory actions removed."""

    stripped = []
    for page in pages:
        features = page.structure_features.copy()
        features[:, -1] = 0.0
        stripped.append(replace(page, structure_features=features))
    return tuple(stripped)


def baseline_page_embeddings(
    pages: tuple[EncodedClusterPage, ...],
    *,
    soft_weights: SoftPageWordWeights,
) -> dict[str, torch.Tensor]:
    """Evaluate the frozen fixed-pooling and SoftGate baselines."""

    fixed = []
    soft = []
    for page in pages:
        if page.page is None:
            raise ValueError("functional_page_baseline_graph_required")
        fixed.append(pool_fixed_page_words(page.page, page.descriptors))
        soft.append(
            pool_soft_page_words_from_graph(
                page.page,
                page.descriptors,
                soft_weights,
            )
        )
    return {
        "fixed_pooling_1024d": torch.as_tensor(np.stack(fixed)),
        "softgate_1024d": torch.as_tensor(np.stack(soft)),
    }


def train_functional_slot_head(
    train_pages: tuple[EncodedClusterPage, ...],
    dev_pages: tuple[EncodedClusterPage, ...],
    *,
    epochs: int = 30,
    learning_rate: float = 2.0e-3,
    weight_decay: float = 1.0e-4,
    clusters_per_batch: int = 16,
    primary_cluster_ids: frozenset[str] | None = None,
    auxiliary_cluster_ratio: float = 0.5,
    pages_per_cluster: int = 3,
    seed: int = 41,
    device: str = "cpu",
) -> tuple[FunctionalSlotPageHead, dict[str, Any]]:
    """Train only the anonymous functional-slot output head."""

    if (
        epochs <= 0
        or clusters_per_batch <= 1
        or not 0.0 < auxiliary_cluster_ratio <= 1.0
        or pages_per_cluster < 2
    ):
        raise ValueError("functional_page_training_schedule_invalid")
    if len(train_pages) < 2 or len(dev_pages) < 2:
        raise ValueError("functional_page_training_pages_insufficient")
    torch.manual_seed(seed)
    rng = random.Random(seed)
    head = FunctionalSlotPageHead().to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=learning_rate * 0.1,
    )
    grouped: dict[str, list[EncodedClusterPage]] = {}
    for page in train_pages:
        grouped.setdefault(page.cluster_id, []).append(page)
    cluster_ids = sorted(grouped)
    primary_ids = frozenset(primary_cluster_ids or ()).intersection(cluster_ids)
    auxiliary_ids = frozenset(cluster_ids).difference(primary_ids)
    label_by_cluster = {
        cluster_id: index for index, cluster_id in enumerate(cluster_ids)
    }
    best_key: tuple[float, float, float] | None = None
    best_state: dict[str, torch.Tensor] | None = None
    history = []
    for epoch in range(1, epochs + 1):
        head.train()
        epoch_cluster_ids = _balanced_cluster_epoch(
            cluster_ids,
            primary_cluster_ids=primary_ids,
            auxiliary_cluster_ratio=auxiliary_cluster_ratio,
            rng=rng,
        )
        losses = []
        for offset in range(0, len(epoch_cluster_ids), clusters_per_batch):
            selected_ids = epoch_cluster_ids[offset : offset + clusters_per_batch]
            batch_pages = []
            for cluster_id in selected_ids:
                candidates = grouped[cluster_id]
                if len(candidates) <= pages_per_cluster:
                    batch_pages.extend(candidates)
                else:
                    batch_pages.extend(rng.sample(candidates, pages_per_cluster))
            if len(batch_pages) < 2:
                continue
            first_outputs = []
            second_outputs = []
            labels = []
            for page in batch_pages:
                first_view = make_self_supervised_view(page, rng=rng)
                second_view = make_self_supervised_view(page, rng=rng)
                first_outputs.append(_forward_view(head, first_view, device=device))
                second_outputs.append(_forward_view(head, second_view, device=device))
                labels.append(label_by_cluster[page.cluster_id])
            embeddings = torch.stack(
                [output.embedding for output in first_outputs + second_outputs]
            )
            label_tensor = torch.as_tensor(
                labels + labels,
                dtype=torch.long,
                device=device,
            )
            contrastive = supervised_contrastive_loss(embeddings, label_tensor)
            positive_alignment = _positive_alignment_loss(
                embeddings,
                label_tensor,
                minimum_cosine=0.95,
            )
            action_consistency = torch.stack(
                [
                    1.0
                    - torch.nn.functional.cosine_similarity(
                        first.action_mass.unsqueeze(0),
                        second.action_mass.unsqueeze(0),
                    ).squeeze(0)
                    for first, second in zip(
                        first_outputs, second_outputs, strict=True
                    )
                ]
            ).mean()
            gate_consistency = torch.stack(
                [
                    torch.nn.functional.mse_loss(
                        first.slot_gates,
                        second.slot_gates,
                    )
                    for first, second in zip(
                        first_outputs, second_outputs, strict=True
                    )
                ]
            ).mean()
            diversity = torch.stack(
                [
                    _slot_diversity(output.slots)
                    for output in first_outputs + second_outputs
                ]
            ).mean()
            loss = (
                contrastive
                + positive_alignment
                + 0.15 * action_consistency
                + 0.05 * gate_consistency
                + 0.01 * diversity
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        scheduler.step()
        dev_vectors = functional_head_embeddings(head, dev_pages, device=device)
        dev_metrics = cluster_retrieval_metrics(
            dev_vectors.cpu(),
            _cluster_pages_from_encoded(dev_pages),
        )
        selection_key = (
            float(dev_metrics["hit_at_1"]),
            float(dev_metrics["positive_margin_rate"]),
            float(dev_metrics["mean_margin"]),
        )
        if best_key is None or selection_key > best_key:
            best_key = selection_key
            best_state = copy.deepcopy(head.state_dict())
        history.append(
            {
                "epoch": epoch,
                "loss": float(np.mean(losses)) if losses else None,
                "dev": dev_metrics,
            }
        )
    if best_state is None or best_key is None:
        raise ValueError("functional_page_training_produced_no_checkpoint")
    head.load_state_dict(best_state)
    return head, {
        "schema_version": "omniflow.functional-page-head-training.v1",
        "method": {
            "output": "8_anonymous_slots_x_128d",
            "dimension": 1024,
            "visual_input": False,
            "geometry_input": False,
            "visible_text_input": "auxiliary_with_0.3_view_dropout",
            "xml_identity_input": "class_and_resource_id_tokens",
            "xml_local_relation_blocks": 1,
            "action_conditioned_gate": True,
            "parameter_count": sum(
                parameter.numel() for parameter in head.parameters()
            ),
        },
        "training": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "learning_rate_schedule": "cosine_to_0.1x",
            "weight_decay": weight_decay,
            "clusters_per_batch": clusters_per_batch,
            "pages_per_cluster": pages_per_cluster,
            "seed": seed,
            "train_pages": len(train_pages),
            "train_clusters": len(grouped),
            "primary_clusters": len(primary_ids),
            "auxiliary_clusters": len(auxiliary_ids),
            "auxiliary_cluster_ratio": auxiliary_cluster_ratio,
            "source_balancing": (
                "all_primary_plus_ratio_rotating_auxiliary"
                if primary_ids and auxiliary_ids
                else "single_pool"
            ),
        },
        "best_dev": {
            "hit_at_1": best_key[0],
            "positive_margin_rate": best_key[1],
            "mean_margin": best_key[2],
        },
        "history": history,
    }


def _balanced_cluster_epoch(
    cluster_ids: list[str],
    *,
    primary_cluster_ids: frozenset[str],
    auxiliary_cluster_ratio: float,
    rng: random.Random,
) -> list[str]:
    """Keep every primary component and rotate an equal auxiliary subset."""

    primary = [value for value in cluster_ids if value in primary_cluster_ids]
    auxiliary = [value for value in cluster_ids if value not in primary_cluster_ids]
    if not primary or not auxiliary:
        selected = list(cluster_ids)
        rng.shuffle(selected)
        return selected
    auxiliary_count = min(
        max(1, round(len(primary) * auxiliary_cluster_ratio)),
        len(auxiliary),
    )
    selected = primary + rng.sample(auxiliary, auxiliary_count)
    rng.shuffle(selected)
    return selected


def _forward_view(
    head: FunctionalSlotPageHead,
    view: FunctionalPageView,
    *,
    device: str,
) -> FunctionalPageOutput:
    return head(
        view.descriptors.to(device),
        view.structure_features.to(device),
        view.parent_indices.to(device),
    )


def _slot_diversity(slots: torch.Tensor) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(slots, dim=1)
    similarity = normalized @ normalized.T
    identity = torch.eye(
        similarity.shape[0],
        dtype=torch.bool,
        device=similarity.device,
    )
    return similarity.masked_select(~identity).square().mean()


def _positive_alignment_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    minimum_cosine: float,
) -> torch.Tensor:
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.T
    identity = torch.eye(
        similarities.shape[0],
        dtype=torch.bool,
        device=similarities.device,
    )
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    return torch.relu(minimum_cosine - similarities[positive]).mean()


def _cluster_pages_from_encoded(
    pages: tuple[EncodedClusterPage, ...],
) -> tuple[ClusterPage, ...]:
    return tuple(
        ClusterPage(
            page.cluster_id,
            page.app_id,
            page.page or UnifiedPage(page.page_id, {"nodes": []}, ()),
        )
        for page in pages
    )


@dataclass(frozen=True)
class StructureActionInputs:
    node_ids: tuple[str, ...]
    features: np.ndarray
    parent_indices: np.ndarray


@dataclass(frozen=True)
class FunctionalPageOutput:
    embedding: torch.Tensor
    slots: torch.Tensor
    slot_gates: torch.Tensor
    attention: torch.Tensor
    action_mass: torch.Tensor


class FunctionalSlotPageHead(torch.nn.Module):
    """Pool node evidence into eight anonymous structure-action slots."""

    descriptor_dimension = 128
    structure_dimension = 8
    slot_count = 8
    output_dimension = 1024

    def __init__(self) -> None:
        super().__init__()
        self.descriptor_projection = torch.nn.Linear(128, 128)
        self.structure_projection = torch.nn.Linear(8, 128)
        self.parent_projection = torch.nn.Linear(128, 128, bias=False)
        self.child_projection = torch.nn.Linear(128, 128, bias=False)
        self.node_norm = torch.nn.LayerNorm(128)
        self.slot_queries = torch.nn.Parameter(torch.empty(8, 128))
        self.slot_action_bias = torch.nn.Parameter(torch.linspace(1.5, -1.5, 8))
        self.slot_gate = torch.nn.Linear(129, 1)
        torch.nn.init.normal_(self.slot_queries, mean=0.0, std=0.02)

    def forward(
        self,
        descriptors: torch.Tensor,
        structure_features: torch.Tensor,
        parent_indices: torch.Tensor,
    ) -> FunctionalPageOutput:
        if descriptors.ndim != 2 or descriptors.shape[1] != 128:
            raise ValueError("functional_page_descriptors_must_be_n_by_128")
        if structure_features.shape != (descriptors.shape[0], 8):
            raise ValueError("functional_page_structure_must_be_n_by_8")
        if parent_indices.shape != (descriptors.shape[0],):
            raise ValueError("functional_page_parent_indices_must_be_n")
        if descriptors.shape[0] == 0:
            raise ValueError("functional_page_requires_nodes")
        node_states = torch.nn.functional.gelu(
            self.descriptor_projection(descriptors)
            + self.structure_projection(structure_features)
        )
        parent_states = torch.zeros_like(node_states)
        valid_parent = parent_indices.ge(0)
        if torch.any(valid_parent):
            parent_states[valid_parent] = node_states[parent_indices[valid_parent]]
        child_sums = torch.zeros_like(node_states)
        child_counts = node_states.new_zeros((node_states.shape[0], 1))
        if torch.any(valid_parent):
            child_sums.index_add_(
                0,
                parent_indices[valid_parent],
                node_states[valid_parent],
            )
            child_counts.index_add_(
                0,
                parent_indices[valid_parent],
                torch.ones_like(child_counts[valid_parent]),
            )
        child_states = child_sums / child_counts.clamp_min(1.0)
        node_states = self.node_norm(
            node_states
            + self.parent_projection(parent_states)
            + self.child_projection(child_states)
        )
        attention_logits = self.slot_queries @ node_states.T / (128.0**0.5)
        attention_logits = attention_logits + (
            self.slot_action_bias[:, None]
            * structure_features[:, -1][None, :]
        )
        attention = torch.softmax(attention_logits, dim=1)
        slots = attention @ node_states
        action_mass = attention @ structure_features[:, -1]
        slot_gates = torch.sigmoid(
            self.slot_gate(
                torch.cat((slots, action_mass.unsqueeze(1)), dim=1)
            ).squeeze(1)
        )
        embedding = torch.nn.functional.normalize(
            (slots * slot_gates.unsqueeze(1)).reshape(-1),
            dim=0,
        )
        return FunctionalPageOutput(
            embedding=embedding,
            slots=slots,
            slot_gates=slot_gates,
            attention=attention,
            action_mass=action_mass,
        )


def structure_only_numeric_features(numeric: torch.Tensor) -> torch.Tensor:
    """Remove every PEMM-v9 geometry channel from frozen node inputs."""

    if numeric.ndim != 2 or numeric.shape[1] not in {18, 32}:
        raise ValueError("functional_page_v9_numeric_features_must_be_n_by_18_or_32")
    structural = numeric.clone()
    if numeric.shape[1] == 18:
        structural[:, 4:15] = 0.0
    else:
        structural[:, 4:9] = 0.0
    return structural


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float = 0.08,
) -> torch.Tensor:
    """Pull every same-component view together against in-batch components."""

    if embeddings.ndim != 2 or labels.shape != (embeddings.shape[0],):
        raise ValueError("functional_page_contrastive_batch_invalid")
    if embeddings.shape[0] < 2 or temperature <= 0.0:
        raise ValueError("functional_page_contrastive_batch_too_small")
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    logits = normalized @ normalized.T / temperature
    identity = torch.eye(
        embeddings.shape[0], dtype=torch.bool, device=embeddings.device
    )
    positive = labels[:, None].eq(labels[None, :]) & ~identity
    valid = positive.any(dim=1)
    if not torch.any(valid):
        raise ValueError("functional_page_contrastive_positive_required")
    logits = logits.masked_fill(identity, float("-inf"))
    log_probability = logits - torch.logsumexp(logits, dim=1, keepdim=True)
    positive_count = positive.sum(dim=1).clamp_min(1)
    per_anchor = -(
        log_probability.masked_fill(~positive, 0.0).sum(dim=1) / positive_count
    )
    return per_anchor[valid].mean()


def cluster_retrieval_metrics(
    embeddings: torch.Tensor,
    pages: tuple[ClusterPage, ...],
) -> dict[str, float | int]:
    """Measure same-App retrieval against multi-positive component labels."""

    if embeddings.ndim != 2 or embeddings.shape[0] != len(pages):
        raise ValueError("functional_page_retrieval_inputs_invalid")
    normalized = torch.nn.functional.normalize(embeddings, dim=1)
    similarities = normalized @ normalized.T
    hits = {1: 0, 3: 0, 5: 0}
    margins = []
    queries = 0
    for query_index, query in enumerate(pages):
        candidates = [
            index
            for index, candidate in enumerate(pages)
            if index != query_index and candidate.app_id == query.app_id
        ]
        positives = [
            index
            for index in candidates
            if pages[index].cluster_id == query.cluster_id
        ]
        if not positives:
            continue
        queries += 1
        ranking = sorted(
            candidates,
            key=lambda index: float(similarities[query_index, index]),
            reverse=True,
        )
        positive_set = set(positives)
        for cutoff in hits:
            hits[cutoff] += int(bool(positive_set.intersection(ranking[:cutoff])))
        negatives = [index for index in candidates if index not in positive_set]
        if negatives:
            best_positive = max(
                float(similarities[query_index, index]) for index in positives
            )
            best_negative = max(
                float(similarities[query_index, index]) for index in negatives
            )
            margins.append(best_positive - best_negative)
    if not queries:
        raise ValueError("functional_page_retrieval_has_no_positive_queries")
    return {
        "queries": queries,
        "hit_at_1": hits[1] / queries,
        "hit_at_3": hits[3] / queries,
        "hit_at_5": hits[5] / queries,
        "mean_margin": float(np.mean(margins)) if margins else 0.0,
        "positive_margin_rate": (
            sum(value > 0.0 for value in margins) / len(margins)
            if margins
            else 0.0
        ),
    }


def pool_fixed_page_words(
    page: UnifiedPage,
    descriptors: np.ndarray,
) -> np.ndarray:
    """Apply the frozen eight-slice pooling baseline to one graph."""

    arrays = _graph_word_arrays(page, descriptors)
    weights = EncoderWeights.manual_default()
    base = np.ones(arrays.descriptors.shape[0], dtype=np.float32)
    base[arrays.neutral] *= weights.pooling[0]
    base[arrays.text_leaf] *= weights.pooling[1]
    base[arrays.actionable] *= weights.pooling[2]
    base[arrays.focus_target] *= weights.pooling[3]
    base[arrays.selected] *= weights.pooling[4]
    words = []
    for index, mask in enumerate(arrays.masks):
        current = base.copy()
        current[arrays.in_list] *= weights.pooling[5 if index < 5 else 6]
        if not np.any(mask):
            pooled = np.zeros(arrays.descriptors.shape[1], dtype=np.float32)
        else:
            selected = current[mask]
            pooled = np.sum(
                arrays.descriptors[mask] * selected[:, None], axis=0
            ) / max(float(np.sum(selected)), 1.0e-9)
        words.append(_normalize(pooled) * weights.slices[index])
    return _normalize(np.concatenate(words)).astype(np.float32)


def pool_soft_page_words_from_graph(
    page: UnifiedPage,
    descriptors: np.ndarray,
    weights: SoftPageWordWeights,
) -> np.ndarray:
    """Apply the frozen learned SoftGate baseline to one graph."""

    weights.validate()
    arrays = _graph_word_arrays(page, descriptors)
    features = np.concatenate((arrays.descriptors, arrays.evidence), axis=1)
    hidden = np.tanh(features @ weights.input_projection + weights.input_bias)
    logits = (
        hidden @ weights.attention_output
        + weights.attention_bias
        + arrays.masks.T * weights.prior_strength
    )
    shifted = logits - np.max(logits, axis=0, keepdims=True)
    attention = (np.exp(shifted) / np.sum(np.exp(shifted), axis=0, keepdims=True)).T
    words = np.asarray(
        [_normalize(row) for row in attention @ arrays.descriptors],
        dtype=np.float32,
    )
    contexts = attention @ hidden
    presence_logits = np.sum(
        contexts * weights.presence_output.T,
        axis=1,
    ) + weights.presence_bias
    presence = 1.0 / (1.0 + np.exp(-presence_logits))
    return _normalize((words * presence[:, None]).reshape(-1)).astype(np.float32)


@dataclass(frozen=True)
class _GraphWordArrays:
    descriptors: np.ndarray
    masks: np.ndarray
    evidence: np.ndarray
    text_leaf: np.ndarray
    actionable: np.ndarray
    focus_target: np.ndarray
    selected: np.ndarray
    neutral: np.ndarray
    in_list: np.ndarray


def _graph_word_arrays(
    page: UnifiedPage,
    descriptors: np.ndarray,
) -> _GraphWordArrays:
    raw_nodes = page.graph.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("page_cluster_graph_nodes_required")
    nodes = tuple(node for node in raw_nodes if isinstance(node, dict))
    values = np.asarray(descriptors, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] != len(nodes):
        raise ValueError("page_cluster_descriptors_do_not_align_with_graph")
    if not np.all(np.isfinite(values)):
        raise ValueError("page_cluster_descriptors_unusable")
    normalized = np.asarray([_normalize(row) for row in values], dtype=np.float32)
    width = float(page.graph.get("width") or 0.0)
    height = float(page.graph.get("height") or 0.0)
    bounds = tuple(_node_bounds(node) for node in nodes)
    if width <= 0.0:
        width = max((bound[2] for bound in bounds), default=1.0)
    if height <= 0.0:
        height = max((bound[3] for bound in bounds), default=1.0)
    width = max(width, 1.0)
    height = max(height, 1.0)
    root_area = width * height
    center_y = np.asarray(
        [(bound[1] + bound[3]) * 0.5 / height for bound in bounds],
        dtype=np.float32,
    )
    areas = np.asarray(
        [
            max(0.0, bound[2] - bound[0])
            * max(0.0, bound[3] - bound[1])
            / root_area
            for bound in bounds
        ],
        dtype=np.float32,
    )
    has_children = np.asarray(
        [bool(node.get("child_ids")) for node in nodes], dtype=bool
    )
    page_frame = has_children & areas.__ge__(0.80)
    active = ~page_frame
    if not np.any(active):
        active = np.ones(len(nodes), dtype=bool)
    top_cut = float(np.quantile(center_y[active], 0.25))
    bottom_cut = float(np.quantile(center_y[active], 0.75))
    has_text = np.asarray(
        [bool(node.get("text") or node.get("content_desc")) for node in nodes]
    )
    actionable = np.asarray(
        [
            bool(node.get("clickable"))
            or _node_flag(node, "checkable")
            or _node_flag(node, "long_clickable")
            for node in nodes
        ]
    )
    focus_target = np.asarray(
        [
            bool(node.get("editable"))
            or _node_flag(node, "focusable")
            or _node_flag(node, "focused")
            for node in nodes
        ]
    )
    selected = np.asarray(
        [_node_flag(node, "selected") or _node_flag(node, "checked") for node in nodes]
    )
    stateful = focus_target | selected
    surface = (areas >= 0.08) & active
    if not np.any(surface):
        surface = active.copy()
    masks = np.zeros((8, len(nodes)), dtype=bool)
    masks[0] = active & has_text
    masks[1] = active & has_text & (center_y <= top_cut)
    masks[2] = active & has_text & (center_y >= bottom_cut)
    masks[3] = active & has_text & actionable
    masks[4] = active & has_text & stateful
    masks[5] = active
    masks[6] = active & (center_y > top_cut) & (center_y < bottom_cut)
    masks[7] = surface
    in_list = np.asarray([_node_flag(node, "in_list") for node in nodes])
    evidence = []
    for node, bound, area, text in zip(nodes, bounds, areas, has_text, strict=True):
        node_width = max(0.0, bound[2] - bound[0])
        node_height = max(0.0, bound[3] - bound[1])
        evidence.append(
            (
                (bound[0] + bound[2]) * 0.5 / width,
                (bound[1] + bound[3]) * 0.5 / height,
                node_width / width,
                node_height / height,
                area,
                min(max(float(node.get("depth") or 0), 0.0) / 16.0, 1.0),
                float(text),
                float(actionable[len(evidence)]),
                float(bool(node.get("editable"))),
                float(bool(node.get("scrollable"))),
                float(_node_flag(node, "checkable")),
                float(selected[len(evidence)]),
                float(_node_flag(node, "focused")),
                float(in_list[len(evidence)]),
                float(not node.get("child_ids")),
                float(bool(node.get("child_ids"))),
            )
        )
    text_leaf = has_text & ~has_children
    neutral = has_children & ~text_leaf & ~actionable & ~focus_target
    return _GraphWordArrays(
        descriptors=normalized,
        masks=masks,
        evidence=np.asarray(evidence, dtype=np.float32),
        text_leaf=text_leaf,
        actionable=actionable,
        focus_target=focus_target,
        selected=selected,
        neutral=neutral,
        in_list=in_list,
    )


def _node_bounds(node: dict[str, Any]) -> tuple[float, float, float, float]:
    value = node.get("bbox")
    if isinstance(value, (list, tuple)) and len(value) == 4:
        return tuple(float(item) for item in value)
    return (0.0, 0.0, 0.0, 0.0)


def _node_flag(node: dict[str, Any], name: str) -> bool:
    if name in node:
        return bool(node[name])
    metadata = node.get("metadata")
    return bool(metadata.get(name)) if isinstance(metadata, dict) else False


def _normalize(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / norm if norm > 1.0e-9 else values


def build_structure_action_inputs(page: UnifiedPage) -> StructureActionInputs:
    """Encode XML topology and observed trajectory actions without geometry."""

    raw_nodes = page.graph.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("page_cluster_graph_nodes_required")
    nodes = []
    for value in raw_nodes:
        if not isinstance(value, dict):
            raise ValueError("page_cluster_graph_node_object_required")
        nodes.append(value)
    node_ids = tuple(str(node.get("node_id") or "") for node in nodes)
    if not all(node_ids) or len(set(node_ids)) != len(node_ids):
        raise ValueError("page_cluster_graph_node_ids_invalid")
    index_by_id = {node_id: index for index, node_id in enumerate(node_ids)}
    action_ids = set(page.action_node_ids)
    features = []
    parent_indices = []
    for node_id, node in zip(node_ids, nodes, strict=True):
        child_ids = tuple(str(value) for value in node.get("child_ids") or ())
        parent_id = str(node.get("parent_id") or "")
        clickable = bool(node.get("clickable"))
        editable = bool(node.get("editable"))
        scrollable = bool(node.get("scrollable"))
        parent_indices.append(index_by_id.get(parent_id, -1))
        features.append(
            (
                float(clickable),
                float(editable),
                float(scrollable),
                float(node.get("enabled", True) is not False),
                min(max(float(node.get("depth") or 0), 0.0) / 32.0, 1.0),
                min(len(child_ids) / 16.0, 1.0),
                float(bool(parent_id)),
                float(node_id in action_ids),
            )
        )
    return StructureActionInputs(
        node_ids=node_ids,
        features=np.asarray(features, dtype=np.float32),
        parent_indices=np.asarray(parent_indices, dtype=np.int64),
    )


def load_unified_page_pairs(
    paths: Iterable[str | Path],
    *,
    assignment: str,
) -> tuple[UnifiedPagePair, ...]:
    """Load one component-isolated split from the canonical pair schema."""

    if assignment not in {"train", "dev", "test", "diagnostic"}:
        raise ValueError("page_cluster_assignment_invalid")
    pairs = []
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").split("\n"), start=1
        ):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(
                    f"page_cluster_pair_object_required:{path}:{line_number}"
                )
            if value.get("schema_version") != UI_CORRESPONDENCE_PAIR_SCHEMA:
                raise ValueError(
                    f"page_cluster_pair_schema_invalid:{path}:{line_number}"
                )
            if _assigned_split(value) != assignment:
                continue
            pairs.append(_page_pair(value))
    return tuple(pairs)


def _assigned_split(value: dict[str, Any]) -> str:
    split = str(value.get("split") or "")
    if split != "diagnostic":
        return split
    provenance = value.get("provenance")
    if isinstance(provenance, dict) and provenance.get("reserved_split"):
        return str(provenance["reserved_split"])
    return split


def _page_pair(value: dict[str, Any]) -> UnifiedPagePair:
    partition_keys = tuple(str(item) for item in value.get("partition_keys") or ())
    cluster_ids = tuple(
        item for item in partition_keys if item.startswith("pool:component:")
    )
    if len(cluster_ids) != 1:
        raise ValueError("page_cluster_component_id_required")
    matches = value.get("matches") or ()
    source_action_ids = []
    target_action_ids = []
    slices = value.get("slices")
    has_action_supervision = bool(
        isinstance(slices, dict) and str(slices.get("action") or "").strip()
    )
    for match in matches:
        if not isinstance(match, dict):
            raise ValueError("page_cluster_match_object_required")
        if has_action_supervision:
            source_action_ids.append(str(match.get("source_node_id") or ""))
            target_action_ids.extend(
                str(node_id) for node_id in match.get("target_node_ids") or ()
            )
    app_id = str(slices.get("app") or "unknown") if isinstance(slices, dict) else "unknown"
    return UnifiedPagePair(
        pair_id=str(value.get("pair_id") or ""),
        cluster_id=cluster_ids[0],
        app_id=app_id,
        source=_page(value.get("source"), source_action_ids),
        target=_page(value.get("target"), target_action_ids),
    )


def _page(value: Any, action_node_ids: Iterable[str]) -> UnifiedPage:
    if not isinstance(value, dict) or not isinstance(value.get("graph"), dict):
        raise ValueError("page_cluster_page_graph_required")
    page_id = str(value.get("page_id") or "")
    if not page_id:
        raise ValueError("page_cluster_page_id_required")
    normalized_action_ids = tuple(
        sorted({node_id for node_id in action_node_ids if node_id})
    )
    return UnifiedPage(
        page_id=page_id,
        graph=dict(value["graph"]),
        action_node_ids=normalized_action_ids,
        screenshot_path=str(value.get("screenshot_path") or ""),
    )


__all__ = [
    "FunctionalPageOutput",
    "FunctionalSlotPageHead",
    "FrozenV9NodeBackbone",
    "ClusterPage",
    "EncodedClusterPage",
    "FunctionalPageView",
    "StructureActionInputs",
    "UnifiedPage",
    "UnifiedPagePair",
    "build_structure_action_inputs",
    "baseline_page_embeddings",
    "collect_cluster_pages",
    "cluster_retrieval_metrics",
    "encode_cluster_pages",
    "functional_head_embeddings",
    "load_unified_page_pairs",
    "make_self_supervised_view",
    "pool_fixed_page_words",
    "pool_soft_page_words_from_graph",
    "structure_only_numeric_features",
    "strip_action_context",
    "supervised_contrastive_loss",
    "train_functional_slot_head",
]
