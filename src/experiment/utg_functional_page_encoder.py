"""Encode UTG pages with the learned XML-structure and Action page head."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import torch

from omniflow.core.model import Observation
from omniflow.transfer.page_store import EmbeddingConfig
from src.experiment.page_cluster_learning import (
    ClusterPage,
    FrozenV9NodeBackbone,
    FunctionalSlotPageHead,
    UnifiedPage,
    encode_cluster_pages,
    functional_head_embeddings,
)


@dataclass(frozen=True)
class FunctionalPageEmbedding:
    vector: np.ndarray
    element_count: int
    encoder_version: str


class FunctionalPageEncoder:
    """Apply the v7 functional cluster head to one XML page and UTG actions."""

    dimension = 1024

    def __init__(
        self,
        *,
        omnitransfer_root: Path,
        checkpoint: Path,
        functional_head_checkpoint: Path,
        device: str = "cpu",
    ) -> None:
        root = omnitransfer_root.expanduser().resolve()
        node_checkpoint = checkpoint.expanduser().resolve()
        head_checkpoint = functional_head_checkpoint.expanduser().resolve()
        if not head_checkpoint.is_file():
            raise ValueError(f"functional_page_head_missing:{head_checkpoint}")
        self.backbone = FrozenV9NodeBackbone(
            omnitransfer_root=root,
            checkpoint=node_checkpoint,
            device=device,
        )
        payload = torch.load(head_checkpoint, map_location="cpu", weights_only=False)
        self.head = FunctionalSlotPageHead()
        self.head.load_state_dict(payload["state_dict"])
        self.head.to(device).eval()
        self.device = device
        self.embedding_config = EmbeddingConfig(
            name="omnitransfer_v9_2_functional_structure_action_1024",
            dimension=1024,
            source_dimension=128,
            pooling="eight_learned_function_slots_x_128d",
            provenance={
                "node_checkpoint_path": str(node_checkpoint),
                "functional_head_checkpoint_path": str(head_checkpoint),
                "functional_head_checkpoint_sha256": _sha256(head_checkpoint),
                "visual_input": False,
                "absolute_geometry_input": False,
                "action_context": "outgoing_utg_action_nodes",
                "parameter_count": sum(
                    parameter.numel() for parameter in self.head.parameters()
                ),
            },
        )

    def embed(self, value: dict[str, object] | Observation) -> FunctionalPageEmbedding:
        observation = Observation.from_value(value)
        xml = str(observation.xml or "")
        if not xml.strip():
            raise ValueError("functional_page_xml_required")
        page_id = str(observation.extra.get("state_id") or "") or hashlib.sha256(
            xml.encode("utf-8")
        ).hexdigest()[:20]
        graph_record, action_node_ids = _page_graph(
            xml,
            page_id=page_id,
            actions=observation.extra.get("utg_actions") or (),
        )
        page = ClusterPage(
            "inference",
            str(observation.package_name or "unknown"),
            UnifiedPage(page_id, graph_record, action_node_ids),
        )
        encoded = encode_cluster_pages(
            self.backbone,
            (page,),
            structure_only=True,
            include_masked_view=True,
        )
        vector = functional_head_embeddings(
            self.head,
            encoded,
            device=self.device,
        )[0].detach().cpu().numpy().astype(np.float32, copy=False)
        return FunctionalPageEmbedding(
            vector=vector,
            element_count=len(graph_record["nodes"]),
            encoder_version="functional-page-head.v7",
        )


def _page_graph(
    xml: str,
    *,
    page_id: str,
    actions: Any,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    from omnitransfer.ui_graph import graph_from_record, graph_to_record

    graph = graph_from_record({"xml": xml}, graph_id=page_id)
    action_node_ids = []
    for action in actions if isinstance(actions, (list, tuple)) else ():
        if not isinstance(action, dict):
            continue
        selected = _bind_action(graph.nodes, action)
        if selected is not None:
            action_node_ids.append(selected.node_id)
    return graph_to_record(graph), tuple(sorted(set(action_node_ids)))


def _bind_action(nodes: tuple[Any, ...], action: dict[str, Any]) -> Any | None:
    resource_id = str(action.get("resource_id") or "")
    text = str(action.get("text") or "")
    candidates = [node for node in nodes if resource_id and node.resource_id == resource_id]
    if not candidates:
        candidates = [
            node
            for node in nodes
            if text and text in {node.text, node.content_desc}
        ]
    bounds = action.get("bounds")
    if not candidates and isinstance(bounds, (list, tuple)) and len(bounds) == 4:
        candidates = sorted(
            (node for node in nodes if node.bbox is not None),
            key=lambda node: _intersection(node.bbox, bounds),
            reverse=True,
        )
    return candidates[0] if candidates else None


def _intersection(left: Any, right: Any) -> float:
    return max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0,
        min(left[3], right[3]) - max(left[1], right[1]),
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = ["FunctionalPageEmbedding", "FunctionalPageEncoder"]
