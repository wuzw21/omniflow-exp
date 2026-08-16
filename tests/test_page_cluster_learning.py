import json
from pathlib import Path
import random

import numpy as np
import torch

from omniflow.transfer.embedding import SoftPageWordWeights
from src.experiment.page_cluster_learning import (
    ClusterPage,
    EncodedClusterPage,
    FunctionalSlotPageHead,
    UnifiedPage,
    build_structure_action_inputs,
    cluster_retrieval_metrics,
    load_unified_page_pairs,
    make_self_supervised_view,
    pool_fixed_page_words,
    pool_soft_page_words_from_graph,
    structure_only_numeric_features,
    supervised_contrastive_loss,
)


def _graph(page_id: str, *, action_node_id: str) -> dict[str, object]:
    return {
        "graph_id": page_id,
        "width": 100,
        "height": 200,
        "nodes": [
            {
                "node_id": "root",
                "origin_id": "root",
                "parent_id": None,
                "class_name": "FrameLayout",
                "bbox": [0, 0, 100, 200],
                "child_ids": [action_node_id],
                "depth": 0,
            },
            {
                "node_id": action_node_id,
                "origin_id": action_node_id,
                "parent_id": "root",
                "text": "Open",
                "class_name": "Button",
                "bbox": [10, 150, 90, 190],
                "clickable": True,
                "enabled": True,
                "child_ids": [],
                "depth": 1,
            },
        ],
    }


def test_unified_loader_restores_reserved_split_and_action_anchors(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bmoca.jsonl"
    record = {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": "pair-home",
        "split": "diagnostic",
        "label_status": "unreviewed",
        "source": {
            "page_id": "env-100-home",
            "platform": "android",
            "screenshot_path": "",
            "graph": _graph("env-100-home", action_node_id="open-source"),
        },
        "target": {
            "page_id": "env-101-home",
            "platform": "android",
            "screenshot_path": "",
            "graph": _graph("env-101-home", action_node_id="open-target"),
        },
        "matches": [
            {
                "source_node_id": "open-source",
                "target_node_ids": ["open-target"],
                "label": "correspondence",
            }
        ],
        "partition_keys": ["pool:component:home"],
        "provenance": {"reserved_split": "test"},
        "slices": {"app": "com.example", "action": "click"},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    pairs = load_unified_page_pairs((path,), assignment="test")

    assert len(pairs) == 1
    assert pairs[0].cluster_id == "pool:component:home"
    assert pairs[0].app_id == "com.example"
    assert pairs[0].source.action_node_ids == ("open-source",)
    assert pairs[0].target.action_node_ids == ("open-target",)


def test_unified_loader_does_not_treat_structural_matches_as_actions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "ase.jsonl"
    record = {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": "pair-home",
        "split": "train",
        "source": {
            "page_id": "ios-home",
            "graph": _graph("ios-home", action_node_id="source-node"),
        },
        "target": {
            "page_id": "android-home",
            "graph": _graph("android-home", action_node_id="target-node"),
        },
        "matches": [
            {
                "source_node_id": "source-node",
                "target_node_ids": ["target-node"],
                "label": "correspondence",
            }
        ],
        "partition_keys": ["pool:component:home"],
        "slices": {"app": "example"},
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    pairs = load_unified_page_pairs((path,), assignment="train")

    assert pairs[0].source.action_node_ids == ()
    assert pairs[0].target.action_node_ids == ()


def test_structure_action_inputs_ignore_screen_geometry() -> None:
    first_graph = _graph("home", action_node_id="open")
    second_graph = _graph("home", action_node_id="open")
    second_graph["width"] = 2208
    second_graph["height"] = 1840
    second_graph["nodes"][0]["bbox"] = [0, 0, 2208, 1840]
    second_graph["nodes"][1]["bbox"] = [2000, 1500, 2180, 1800]
    first = UnifiedPage("first", first_graph, ("open",))
    second = UnifiedPage("second", second_graph, ("open",))

    first_inputs = build_structure_action_inputs(first)
    second_inputs = build_structure_action_inputs(second)

    np.testing.assert_array_equal(first_inputs.features, second_inputs.features)
    np.testing.assert_array_equal(
        first_inputs.parent_indices,
        second_inputs.parent_indices,
    )
    assert first_inputs.features[:, -1].tolist() == [0.0, 1.0]


def test_functional_slot_head_returns_order_invariant_1024d_embedding() -> None:
    torch.manual_seed(11)
    head = FunctionalSlotPageHead().eval()
    descriptors = torch.zeros((2, 128), dtype=torch.float32)
    descriptors[0, 0] = 1.0
    descriptors[1, 1] = 1.0
    features = torch.tensor(
        [
            [0, 0, 0, 1, 0, 1 / 16, 0, 0],
            [1, 0, 0, 1, 1 / 32, 0, 1, 1],
        ],
        dtype=torch.float32,
    )
    parent_indices = torch.tensor([-1, 0], dtype=torch.long)

    original = head(descriptors, features, parent_indices)
    reordered = head(
        descriptors[[1, 0]],
        features[[1, 0]],
        torch.tensor([1, -1], dtype=torch.long),
    )

    assert original.embedding.shape == (1024,)
    assert original.slot_gates.shape == (8,)
    assert original.attention.shape == (8, 2)
    torch.testing.assert_close(original.embedding.norm(), torch.tensor(1.0))
    torch.testing.assert_close(original.embedding, reordered.embedding, atol=1e-6, rtol=1e-6)


def test_fixed_and_softgate_baselines_share_1024d_page_contract() -> None:
    page = UnifiedPage("home", _graph("home", action_node_id="open"), ("open",))
    descriptors = np.zeros((2, 128), dtype=np.float32)
    descriptors[0, 0] = 1.0
    descriptors[1, 1] = 1.0
    generator = np.random.default_rng(19)
    weights = SoftPageWordWeights(
        input_projection=generator.normal(0.0, 0.02, (144, 32)).astype(np.float32),
        input_bias=np.zeros(32, dtype=np.float32),
        attention_output=generator.normal(0.0, 0.02, (32, 8)).astype(np.float32),
        attention_bias=np.zeros(8, dtype=np.float32),
        prior_strength=np.ones(8, dtype=np.float32),
        presence_output=generator.normal(0.0, 0.02, (32, 8)).astype(np.float32),
        presence_bias=np.zeros(8, dtype=np.float32),
    )

    fixed = pool_fixed_page_words(page, descriptors)
    soft = pool_soft_page_words_from_graph(page, descriptors, weights)

    assert fixed.shape == (1024,)
    assert soft.shape == (1024,)
    np.testing.assert_allclose(np.linalg.norm(fixed), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(soft), 1.0, atol=1e-6)


def test_structure_only_numeric_features_remove_every_geometry_channel() -> None:
    pemm_numeric = torch.arange(36, dtype=torch.float32).reshape(2, 18)
    v9_numeric = torch.arange(64, dtype=torch.float32).reshape(2, 32)

    pemm_structural = structure_only_numeric_features(pemm_numeric)
    v9_structural = structure_only_numeric_features(v9_numeric)

    torch.testing.assert_close(pemm_structural[:, :4], pemm_numeric[:, :4])
    torch.testing.assert_close(pemm_structural[:, 4:15], torch.zeros((2, 11)))
    torch.testing.assert_close(pemm_structural[:, 15:], pemm_numeric[:, 15:])
    torch.testing.assert_close(v9_structural[:, :4], v9_numeric[:, :4])
    torch.testing.assert_close(v9_structural[:, 4:9], torch.zeros((2, 5)))
    torch.testing.assert_close(v9_structural[:, 9:], v9_numeric[:, 9:])


def test_cluster_metrics_and_loss_use_multi_positive_component_labels() -> None:
    pages = (
        ClusterPage("cluster-a", "app", UnifiedPage("a1", _graph("a1", action_node_id="x"), ("x",))),
        ClusterPage("cluster-a", "app", UnifiedPage("a2", _graph("a2", action_node_id="x"), ("x",))),
        ClusterPage("cluster-b", "app", UnifiedPage("b1", _graph("b1", action_node_id="x"), ("x",))),
        ClusterPage("cluster-b", "app", UnifiedPage("b2", _graph("b2", action_node_id="x"), ("x",))),
    )
    aligned = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    crossed = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    metrics = cluster_retrieval_metrics(aligned, pages)

    assert metrics["hit_at_1"] == 1.0
    assert metrics["queries"] == 4
    assert supervised_contrastive_loss(aligned, labels) < supervised_contrastive_loss(crossed, labels)


def test_self_supervised_view_never_drops_action_anchor() -> None:
    descriptors = np.eye(4, 128, dtype=np.float32)
    features = np.zeros((4, 8), dtype=np.float32)
    features[-1, -1] = 1.0
    encoded = EncodedClusterPage(
        cluster_id="cluster",
        app_id="app",
        page_id="page",
        descriptors=descriptors,
        structure_features=features,
        parent_indices=np.asarray([-1, 0, 1, 2], dtype=np.int64),
    )

    view = make_self_supervised_view(
        encoded,
        rng=random.Random(3),
        node_drop_probability=1.0,
        edge_drop_probability=1.0,
        descriptor_drop_probability=0.0,
        minimum_nodes=2,
    )

    assert view.structure_features.shape[0] == 2
    assert torch.any(view.structure_features[:, -1].eq(1.0))
    assert torch.all(view.parent_indices.eq(-1))
