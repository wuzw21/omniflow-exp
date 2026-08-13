from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from src.experiment.mapping_datasets import (
    _baseline_page_mapping,
    _review_payload,
    evaluate_ase_records,
    evaluate_mobileviews_graph_records,
    evaluate_mobileviews_records,
    write_report,
)


def _page(page_id: str, node_id: str) -> dict:
    return {
        "page_id": page_id,
        "platform": "android",
        "screenshot_path": "/tmp/missing-in-unit-test.png",
        "graph": {
            "graph_id": page_id,
            "width": 100,
            "height": 200,
            "nodes": [
                {
                    "node_id": node_id,
                    "origin_id": node_id,
                    "class_name": "android.widget.Button",
                    "bbox": [0, 0, 50, 40],
                    "clickable": True,
                }
            ],
        },
    }


def _pair(pair_id: str, source: dict, target: dict) -> dict:
    return {
        "schema_version": "omnitransfer.ui_correspondence_pair.v1",
        "pair_id": pair_id,
        "split": "diagnostic",
        "label_status": "self_supervised",
        "source": source,
        "target": target,
        "matches": [
            {
                "source_node_id": source["graph"]["nodes"][0]["node_id"],
                "target_node_ids": [target["graph"]["nodes"][0]["node_id"]],
                "label": "correspondence",
            }
        ],
        "partition_keys": ["mobileviews:trace:test"],
        "provenance": {
            "dataset": "MobileViews_Apps_CompleteTraces",
            "annotation": "automatic_same_view_str_proposal",
        },
        "slices": {"app": "test.app"},
    }


def test_mobileviews_retrieves_with_native_page_encoder_before_mapping() -> None:
    records = [
        _pair("p1", _page("source-a", "sa"), _page("target-a", "ta")),
        _pair("p2", _page("source-b", "sb"), _page("target-b", "tb")),
    ]

    @dataclass
    class Embedded:
        vector: np.ndarray
        elements: tuple[int, ...] = (1,)

    class Encoder:
        dimension = 512

        def __init__(self):
            self.page_ids = []

        def embed(self, observation):
            page_id = observation.extra["page_id"]
            self.page_ids.append(page_id)
            group = 1.0 if page_id.endswith("a") else -1.0
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = group
            return Embedded(vector)

    class Config:
        candidate_policy = "actionable"

    class Matcher:
        config = Config()

        def _forward(self, source, target):
            return {
                "logits_ab": np.ones((len(source.nodes), len(target.nodes))),
                "affinity": np.ones((len(source.nodes), len(target.nodes))),
            }

    encoder = Encoder()
    result = evaluate_mobileviews_records(
        records,
        Matcher(),
        page_encoder=encoder,
        page_top_k=1,
        mapping_top_k=1,
    )

    assert set(encoder.page_ids) == {"source-a", "target-a", "source-b", "target-b"}
    assert result["page_retrieval"]["hit_at_1"] == 1.0
    assert result["page_retrieval"]["hit_at_k"] == 1.0
    assert result["element_mapping"]["end_to_end_top1"] == 1.0
    assert result["label_boundary"]["status"] == "self_supervised_diagnostic"
    assert result["configuration"]["page_embedding"] == "omniflow_native_512d_page_embedding"


def test_mobileviews_graph_screening_precedes_page_wide_mapping() -> None:
    def graph(page_id: str, *, y: int, text: str) -> dict:
        return {
            "graph_id": page_id,
            "width": 100,
            "height": 200,
            "nodes": [
                {
                    "node_id": f"{page_id}-save",
                    "origin_id": f"{page_id}-save",
                    "resource_id": "pkg:id/save",
                    "class_name": "android.widget.Button",
                    "text": text,
                    "bbox": [0, y, 50, y + 30],
                    "clickable": True,
                },
                {
                    "node_id": f"{page_id}-search",
                    "origin_id": f"{page_id}-search",
                    "resource_id": "pkg:id/search",
                    "class_name": "android.widget.EditText",
                    "bbox": [0, 100, 80, 140],
                    "editable": True,
                },
            ],
            "metadata": {
                "package": "pkg",
                "foreground_activity": "pkg/.MainActivity",
                "screenshot_path": "/tmp/not-needed.png",
                "split": "test",
            },
        }

    records = [
        graph("a", y=0, text="Save"),
        graph("b", y=20, text="Store"),
        {
            **graph("c", y=0, text="Other"),
            "metadata": {
                "package": "other.pkg",
                "foreground_activity": "other.pkg/.MainActivity",
                "screenshot_path": "/tmp/not-needed.png",
                "split": "test",
            },
        },
    ]

    @dataclass
    class Embedded:
        vector: np.ndarray
        elements: tuple[int, ...] = (1,)

    class Encoder:
        dimension = 512

        def embed(self, observation):
            vector = np.zeros(512, dtype=np.float32)
            vector[0] = 1.0
            return Embedded(vector)

    class Config:
        candidate_policy = "actionable"

    class Matcher:
        config = Config()
        calls = 0

        def _forward(self, source_graph, target_graph):
            self.calls += 1
            return {
                "logits_ab": np.asarray([[5.0, 1.0], [1.0, 5.0]]),
                "affinity": np.asarray([[5.0, 1.0], [1.0, 5.0]]),
            }

    matcher = Matcher()
    result = evaluate_mobileviews_graph_records(
        records,
        matcher,
        page_encoder=Encoder(),
        retrieval_top_k=1,
        pair_limit=1,
        mapping_top_k=2,
    )

    assert result["screening"]["input_pages"] == 3
    assert result["screening"]["eligible_pages"] == 3
    assert result["screening"]["selected_pairs"] == 1
    assert result["element_mapping"]["top1"] == 1.0
    assert result["methods"]["position_scaling"]["top1"] == 1.0
    assert result["methods"]["text_class_position"]["top1"] == 1.0
    assert result["methods"]["omnitransfer"]["top1"] == 1.0
    assert result["pairs"][0]["baselines"]["position_scaling"]["resource_id_input"] is False
    assert matcher.calls == 1
    assert result["label_boundary"]["node_label"] == "unique_resource_id_diagnostic"
    payload = _review_payload(result)
    assert payload["summary"]["task_count"] == 2
    assert {task["source"]["node"]["node_id"] for task in payload["pairs"]} == {
        "a-save",
        "a-search",
    }
    assert all(
        node["resource_id"] == ""
        for node in payload["pairs"][0]["source"]["candidates"]
    )


def test_mobileviews_baselines_share_candidates_without_resource_ids() -> None:
    def node(node_id: str, text: str, left: int) -> SimpleNamespace:
        return SimpleNamespace(
            node_id=node_id,
            text=text,
            content_desc="",
            class_name="android.widget.Button",
            bbox=(left, 0, left + 20, 20),
        )

    source = SimpleNamespace(
        graph_id="source",
        width=100,
        height=100,
        nodes=(node("s-save", "Save", 0), node("s-delete", "Delete", 80)),
    )
    target = SimpleNamespace(
        graph_id="target",
        width=100,
        height=100,
        nodes=(node("t-delete", "Delete", 0), node("t-save", "Save", 80)),
    )
    gold = {"s-save": {"t-save"}, "s-delete": {"t-delete"}}
    position = _baseline_page_mapping(
        source,
        target,
        source_node_ids=("s-save", "s-delete"),
        target_node_ids=("t-delete", "t-save"),
        gold=gold,
        top_k=2,
        method="position_scaling",
    )
    structured = _baseline_page_mapping(
        source,
        target,
        source_node_ids=("s-save", "s-delete"),
        target_node_ids=("t-delete", "t-save"),
        gold=gold,
        top_k=2,
        method="text_class_position",
    )

    assert position["gold_metrics"]["top1"] == 0.0
    assert structured["gold_metrics"]["top1"] == 1.0
    assert structured["resource_id_input"] is False


def test_ase_reports_public_gold_and_review_payload_labels_both_predictions() -> None:
    source = _page("source", "s1")
    target = _page("target", "t1")
    record = _pair("ase-pair", source, target)
    record["split"] = "test"
    record["label_status"] = "gold"
    record["provenance"] = {"dataset": "ase2023_vision_based_widget_mapping"}

    class Config:
        candidate_policy = "all_nodes"

    class Matcher:
        config = Config()

        def _forward(self, source_graph, target_graph):
            return {
                "logits_ab": np.ones((len(source_graph.nodes), len(target_graph.nodes))),
                "affinity": np.ones((len(source_graph.nodes), len(target_graph.nodes))),
            }

    result = evaluate_ase_records([record], Matcher(), mapping_top_k=1)
    payload = _review_payload(result)

    assert result["label_boundary"]["status"] == "public_gold"
    assert result["element_mapping"]["top1"] == 1.0
    assert payload["summary"]["review_ui"]["diagnostic_overlay"] == {
        "enabled": True,
        "methods": ["gold_proposal", "matcher_prediction"],
        "coordinate_space": "page_pixels",
    }
    assert payload["pairs"][0]["gold_proposal"]["node_id"] == "t1"
    assert payload["pairs"][0]["matcher_prediction"]["top1_node"]["node_id"] == "t1"


def test_ase_rejects_actionable_only_checkpoint() -> None:
    source = _page("source", "s1")
    target = _page("target", "t1")
    record = _pair("ase-pair", source, target)
    record["split"] = "test"
    record["label_status"] = "gold"

    class Config:
        candidate_policy = "actionable"

    class Matcher:
        config = Config()

    with pytest.raises(ValueError, match="requires an all-node OmniTransfer checkpoint"):
        evaluate_ase_records([record], Matcher())


def test_write_report_uses_canonical_review_with_real_screenshots(
    tmp_path: Path,
) -> None:
    source_image = tmp_path / "source.png"
    target_image = tmp_path / "target.png"
    Image.new("RGB", (100, 200), "white").save(source_image)
    Image.new("RGB", (100, 200), "black").save(target_image)
    source = _page("source", "s1")
    target = _page("target", "t1")
    source["screenshot_path"] = str(source_image)
    target["screenshot_path"] = str(target_image)
    record = _pair("ase-pair", source, target)
    record["split"] = "test"
    record["label_status"] = "gold"

    class Config:
        candidate_policy = "all_nodes"

    class Matcher:
        config = Config()

        def _forward(self, source_graph, target_graph):
            return {
                "logits_ab": np.ones((len(source_graph.nodes), len(target_graph.nodes))),
                "affinity": np.ones((len(source_graph.nodes), len(target_graph.nodes))),
            }

    report = evaluate_ase_records([record], Matcher(), mapping_top_k=1)
    manifest = write_report(report, tmp_path / "review")
    sidecar = json.loads(Path(manifest["review_sidecar"]).read_text())

    assert Path(manifest["review"]).read_text().count("__OMNITRANSFER_REVIEW_PAYLOAD__") == 0
    assert sidecar["pairs"][0]["source"]["screenshot_path"].startswith("screenshots/")
    assert sidecar["pairs"][0]["target"]["screenshot_path"].startswith("screenshots/")
    assert len(list((tmp_path / "review" / "screenshots").iterdir())) == 2
