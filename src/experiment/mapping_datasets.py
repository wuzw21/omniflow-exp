"""ASE and MobileViews diagnostics for page retrieval and page-wide mapping."""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import replace
from difflib import SequenceMatcher
import hashlib
import json
from pathlib import Path
import shutil
import time
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np

from omniflow.core.model import Observation
from omniflow.transfer.embedding import PageEncoder
from omniflow.transfer.review import canonical_review_template
from omniflow.transfer.runtime import load_omnitransfer

REPORT_SCHEMA = "omniflow.mapping-datasets-report.v1"
BASELINE_MAPPING_SCHEMA = "omniflow.mapping-dataset-baseline.v1"
MOBILEVIEWS_BASELINES = ("position_scaling", "text_class_position")
MOBILEVIEWS_LABEL_BOUNDARY = {
    "status": "self_supervised_diagnostic",
    "page_retrieval_label": "known_pair_membership",
    "element_label": "automatic_identity_proposal",
    "human_gold": False,
    "formal_accuracy_claim": False,
}


def evaluate_mobileviews_graph_records(
    records: Iterable[dict[str, Any]],
    matcher: Any,
    *,
    page_encoder: PageEncoder | None = None,
    retrieval_top_k: int = 5,
    pair_limit: int = 100,
    mapping_top_k: int = 3,
    minimum_shared_ids: int = 2,
    minimum_identity_overlap: float = 0.4,
    minimum_position_shift: float = 0.03,
    minimum_actionable_nodes: int = 2,
    maximum_actionable_nodes: int = 40,
    maximum_pairs_per_app: int = 3,
    require_same_activity: bool = True,
) -> dict[str, Any]:
    """Screen raw MobileViews pages, retrieve with 512D vectors, then map.

    Resource IDs are used only after retrieval to form diagnostic labels and
    pair-quality checks. They are never passed to PageEncoder or OmniTransfer.
    """

    if (
        retrieval_top_k <= 0
        or pair_limit <= 0
        or mapping_top_k <= 0
        or maximum_pairs_per_app <= 0
    ):
        raise ValueError("limits and top-k values must be positive")
    if minimum_shared_ids <= 0:
        raise ValueError("minimum_shared_ids must be positive")
    if not 0.0 <= minimum_identity_overlap <= 1.0:
        raise ValueError("minimum_identity_overlap must be in [0, 1]")
    load_omnitransfer()
    from omnitransfer.page_mapping import map_page
    from omnitransfer.ui_graph import graph_from_record

    encoder = page_encoder or PageEncoder()
    if int(getattr(encoder, "dimension", 0)) != 512:
        raise ValueError("MobileViews retrieval requires OmniFlow's native 512D encoder")
    raw_records = list(records)
    pages: list[dict[str, Any]] = []
    rejected = Counter()
    for raw in raw_records:
        graph_id = str(raw.get("graph_id") or raw.get("screen_id") or "").strip()
        metadata = raw.get("metadata") if isinstance(raw.get("metadata"), dict) else {}
        package = str(metadata.get("package") or metadata.get("app") or "").strip()
        if not graph_id or not package:
            rejected["missing_page_or_app"] += 1
            continue
        graph = graph_from_record(raw, graph_id=graph_id)
        actionable = tuple(node for node in graph.nodes if _actionable_node(node))
        if not minimum_actionable_nodes <= len(actionable) <= maximum_actionable_nodes:
            rejected["actionable_node_count"] += 1
            continue
        identities = _unique_resource_id_nodes(actionable)
        if len(identities) < minimum_shared_ids:
            rejected["too_few_unique_ids"] += 1
            continue
        vector = np.asarray(
            encoder.embed(
                Observation(
                    xml=_graph_xml_without_resource_ids(graph),
                    package_name=package,
                    activity_name=str(metadata.get("foreground_activity") or ""),
                    extra={"page_id": graph_id},
                )
            ).vector,
            dtype=np.float32,
        )
        if vector.shape != (512,) or not np.any(vector):
            rejected["unusable_page_embedding"] += 1
            continue
        pages.append(
            {
                "graph_id": graph_id,
                "package": package,
                "activity": str(metadata.get("foreground_activity") or ""),
                "record": raw,
                "graph": graph,
                "vector": vector,
                "identities": identities,
            }
        )

    by_app: dict[str, list[dict[str, Any]]] = {}
    for page in pages:
        by_app.setdefault(page["package"], []).append(page)
    retrieved_candidates = 0
    screened_candidates: list[dict[str, Any]] = []
    for source in sorted(pages, key=lambda item: item["graph_id"]):
        app_targets = [
            target
            for target in by_app[source["package"]]
            if target["graph_id"] != source["graph_id"]
            and (
                not require_same_activity
                or target["activity"] == source["activity"]
            )
        ]
        ranked = sorted(
            app_targets,
            key=lambda target: (
                -_cosine(source["vector"], target["vector"]),
                target["graph_id"],
            ),
        )[:retrieval_top_k]
        retrieved_candidates += len(ranked)
        for retrieval_rank, target in enumerate(ranked, start=1):
            quality = _mobileviews_pair_quality(
                source,
                target,
                minimum_shared_ids=minimum_shared_ids,
                minimum_identity_overlap=minimum_identity_overlap,
                minimum_position_shift=minimum_position_shift,
            )
            if quality is None:
                continue
            screened_candidates.append(
                {
                    "source": source,
                    "target": target,
                    "retrieval_rank": retrieval_rank,
                    "page_similarity": _cosine(source["vector"], target["vector"]),
                    **quality,
                }
            )

    screened_candidates.sort(
        key=lambda pair: (
            pair["identity_overlap"],
            -pair["changed_identity_fraction"],
            pair["page_similarity"],
            pair["source"]["graph_id"],
            pair["target"]["graph_id"],
        )
    )
    selected = []
    seen_pairs: set[tuple[str, str]] = set()
    app_counts: Counter[str] = Counter()
    for pair in screened_candidates:
        undirected = tuple(sorted((pair["source"]["graph_id"], pair["target"]["graph_id"])))
        if (
            undirected in seen_pairs
            or app_counts[pair["source"]["package"]] >= maximum_pairs_per_app
        ):
            continue
        selected.append(pair)
        seen_pairs.add(undirected)
        app_counts[pair["source"]["package"]] += 1
        if len(selected) >= pair_limit:
            break

    evaluated = []
    mapping_seconds = 0.0
    baseline_seconds: Counter[str] = Counter()
    for pair in selected:
        gold = pair["gold"]
        started = time.perf_counter()
        mapping = map_page(
            pair["source"]["graph"],
            pair["target"]["graph"],
            matcher,
            top_k=mapping_top_k,
            gold=gold,
            label_boundary="unique_resource_id_diagnostic",
        )
        mapping_seconds += time.perf_counter() - started
        source_node_ids = tuple(row["source_node_id"] for row in mapping["mappings"])
        target_node_ids = tuple(
            candidate["target_node_id"]
            for candidate in mapping["mappings"][0]["candidates"]
        )
        baselines = {}
        for method in MOBILEVIEWS_BASELINES:
            started = time.perf_counter()
            baselines[method] = _baseline_page_mapping(
                pair["source"]["graph"],
                pair["target"]["graph"],
                source_node_ids=source_node_ids,
                target_node_ids=target_node_ids,
                gold=gold,
                top_k=mapping_top_k,
                method=method,
            )
            baseline_seconds[method] += time.perf_counter() - started
        evaluated.append(
            {
                "pair_id": f"mobileviews:{pair['source']['graph_id']}->{pair['target']['graph_id']}",
                "dataset": "MobileViews-600K",
                "source": _graph_page_record(pair["source"]),
                "selected_target": _graph_page_record(pair["target"]),
                "retrieval_rank": pair["retrieval_rank"],
                "page_similarity": pair["page_similarity"],
                "identity_overlap": pair["identity_overlap"],
                "shared_identity_count": pair["shared_identity_count"],
                "changed_identity_count": pair["changed_identity_count"],
                "mapping": mapping,
                "baselines": baselines,
                "provenance": {
                    "dataset": "mllmTeam/MobileViews",
                    "annotation": "automatic_unique_resource_id_diagnostic",
                    "resource_ids_excluded_from_page_embedding": True,
                    "resource_ids_forbidden_from_omnitransfer": True,
                },
            }
        )

    gold_metrics = _aggregate_gold(evaluated)
    method_results = {
        "omnitransfer": {
            "description": "canonical learned OmniTransfer page-wide ranking",
            **gold_metrics,
            "latency_seconds": mapping_seconds,
            "average_latency_ms": (
                1000.0 * mapping_seconds / len(evaluated) if evaluated else 0.0
            ),
            **(
                _mapping_diagnostics(evaluated)
                if evaluated
                else {"mean_mutual_top1_rate": 0.0, "mean_collision_rate": 0.0}
            ),
        }
    }
    for method in MOBILEVIEWS_BASELINES:
        method_pairs = [{"mapping": pair["baselines"][method]} for pair in evaluated]
        method_results[method] = {
            "description": _baseline_description(method),
            **_aggregate_gold(method_pairs),
            "latency_seconds": baseline_seconds[method],
            "average_latency_ms": (
                1000.0 * baseline_seconds[method] / len(evaluated)
                if evaluated
                else 0.0
            ),
            **(
                _mapping_diagnostics(method_pairs)
                if method_pairs
                else {"mean_mutual_top1_rate": 0.0, "mean_collision_rate": 0.0}
            ),
        }
    return {
        "schema_version": REPORT_SCHEMA,
        "dataset": "MobileViews-600K",
        "configuration": {
            "page_embedding": "omniflow_native_512d_page_embedding",
            "page_embedding_dimension": 512,
            "retrieval_scope": "same_app",
            "retrieval_top_k": retrieval_top_k,
            "mapping": "canonical_omnitransfer_page_wide_single_forward",
            "mapping_top_k": mapping_top_k,
            "comparison_methods": {
                "position_scaling": _baseline_description("position_scaling"),
                "text_class_position": _baseline_description("text_class_position"),
                "omnitransfer": "canonical learned OmniTransfer page-wide ranking",
            },
            "maximum_pairs_per_app": maximum_pairs_per_app,
            "require_same_activity": require_same_activity,
        },
        "label_boundary": {
            "status": "self_supervised_diagnostic",
            "page_selection": "native_512d_embedding_only",
            "node_label": "unique_resource_id_diagnostic",
            "resource_id_page_embedding_input": False,
            "resource_id_matcher_input": False,
            "human_gold": False,
            "formal_accuracy_claim": False,
        },
        "screening": {
            "input_pages": len(raw_records),
            "eligible_pages": len(pages),
            "eligible_apps": len(by_app),
            "retrieved_candidates": retrieved_candidates,
            "quality_screened_candidates": len(screened_candidates),
            "selected_pairs": len(selected),
            "rejected_pages": dict(sorted(rejected.items())),
        },
        "element_mapping": {
            **method_results["omnitransfer"],
        },
        "methods": method_results,
        "pairs": evaluated,
    }


def evaluate_mobileviews_records(
    records: Iterable[dict[str, Any]],
    matcher: Any,
    *,
    page_encoder: PageEncoder | None = None,
    page_top_k: int = 3,
    mapping_top_k: int = 3,
) -> dict[str, Any]:
    """Retrieve target pages with native 512D vectors, then map each top page."""

    if page_top_k <= 0 or mapping_top_k <= 0:
        raise ValueError("top-k values must be positive")
    load_omnitransfer()
    from omnitransfer.mapping_dataset import validate_ui_correspondence_pair
    from omnitransfer.page_mapping import map_page

    normalized = [validate_ui_correspondence_pair(record) for record in records]
    if not normalized:
        raise ValueError("MobileViews diagnostic input is empty")
    encoder = page_encoder or PageEncoder()
    if int(getattr(encoder, "dimension", 0)) != 512:
        raise ValueError("MobileViews retrieval requires OmniFlow's native 512D encoder")

    pages: dict[str, dict[str, Any]] = {}
    target_groups: dict[str, list[str]] = {}
    vectors: dict[str, np.ndarray] = {}
    graphs: dict[str, Any] = {}
    for record in normalized:
        group = _app_group(record)
        target_id = record["target"]["page_id"]
        target_groups.setdefault(group, [])
        if target_id not in target_groups[group]:
            target_groups[group].append(target_id)
        for side in ("source", "target"):
            page = record[side]
            page_id = page["page_id"]
            pages.setdefault(page_id, page)
            if page_id in vectors:
                continue
            graph = _page_graph(page, record)
            graphs[page_id] = graph
            observation = Observation(
                xml=_graph_xml(graph),
                package_name=group,
                extra={"page_id": page_id},
            )
            embedded = encoder.embed(observation)
            vector = np.asarray(embedded.vector, dtype=np.float32)
            if vector.shape != (512,) or not getattr(embedded, "elements", ()):
                raise ValueError(f"page encoding is unusable: {page_id}")
            vectors[page_id] = vector

    retrieval_hits = Counter()
    mapping_hits = Counter()
    evaluated: list[dict[str, Any]] = []
    retrieval_seconds = 0.0
    mapping_seconds = 0.0
    for record in normalized:
        source_id = record["source"]["page_id"]
        gold_target_id = record["target"]["page_id"]
        group = _app_group(record)
        candidates = target_groups.get(group, [])
        if not candidates:
            raise ValueError(f"MobileViews target pool is empty for group: {group}")
        retrieval_started = time.perf_counter()
        ranked_pages = sorted(
            (
                {
                    "page_id": page_id,
                    "similarity": _cosine(vectors[source_id], vectors[page_id]),
                }
                for page_id in candidates
            ),
            key=lambda item: (-item["similarity"], item["page_id"]),
        )
        retrieval_seconds += time.perf_counter() - retrieval_started
        retrieved_ids = [item["page_id"] for item in ranked_pages[:page_top_k]]
        page_hit_at_1 = retrieved_ids[0] == gold_target_id
        page_hit_at_k = gold_target_id in retrieved_ids
        retrieval_hits["queries"] += 1
        retrieval_hits["hit_at_1"] += page_hit_at_1
        retrieval_hits["hit_at_k"] += page_hit_at_k

        selected_target_id = retrieved_ids[0]
        gold = _gold(record) if page_hit_at_1 else None
        mapping_started = time.perf_counter()
        mapping = map_page(
            graphs[source_id],
            graphs[selected_target_id],
            matcher,
            top_k=mapping_top_k,
            gold=gold,
            label_boundary="self_supervised_diagnostic" if gold else "unlabeled_retrieval_error",
        )
        mapping_seconds += time.perf_counter() - mapping_started
        element_top1 = (
            mapping["gold_metrics"]["top1"]
            if mapping.get("gold_metrics") is not None
            else 0.0
        )
        element_recall = (
            mapping["gold_metrics"]["recall_at_k"]
            if mapping.get("gold_metrics") is not None
            else 0.0
        )
        mapping_hits["queries"] += 1
        mapping_hits["page_hit_queries"] += page_hit_at_1
        mapping_hits["end_to_end_top1_sum"] += element_top1
        mapping_hits["end_to_end_recall_at_k_sum"] += element_recall
        evaluated.append(
            {
                "pair_id": record["pair_id"],
                "dataset": "MobileViews",
                "source": record["source"],
                "gold_target_page_id": gold_target_id,
                "retrieved_pages": ranked_pages[:page_top_k],
                "page_hit_at_1": page_hit_at_1,
                "page_hit_at_k": page_hit_at_k,
                "selected_target": pages[selected_target_id],
                "mapping": mapping,
                "provenance": record["provenance"],
            }
        )

    query_count = retrieval_hits["queries"]
    return {
        "schema_version": REPORT_SCHEMA,
        "dataset": "MobileViews",
        "configuration": {
            "page_embedding": "omniflow_native_512d_page_embedding",
            "page_embedding_dimension": 512,
            "page_top_k": page_top_k,
            "mapping": "canonical_omnitransfer_page_wide_single_forward",
            "mapping_top_k": mapping_top_k,
            "candidate_scope": "same_app_target_pages",
        },
        "label_boundary": dict(MOBILEVIEWS_LABEL_BOUNDARY),
        "page_retrieval": {
            "queries": query_count,
            "hit_at_1": retrieval_hits["hit_at_1"] / query_count,
            "hit_at_k": retrieval_hits["hit_at_k"] / query_count,
            "latency_seconds": retrieval_seconds,
            "average_latency_ms": 1000.0 * retrieval_seconds / query_count,
        },
        "element_mapping": {
            "queries": mapping_hits["queries"],
            "queries_with_correct_retrieved_page": mapping_hits["page_hit_queries"],
            "conditional_top1": (
                mapping_hits["end_to_end_top1_sum"] / mapping_hits["page_hit_queries"]
                if mapping_hits["page_hit_queries"]
                else 0.0
            ),
            "conditional_recall_at_k": (
                mapping_hits["end_to_end_recall_at_k_sum"]
                / mapping_hits["page_hit_queries"]
                if mapping_hits["page_hit_queries"]
                else 0.0
            ),
            "end_to_end_top1": mapping_hits["end_to_end_top1_sum"] / query_count,
            "end_to_end_recall_at_k": mapping_hits["end_to_end_recall_at_k_sum"] / query_count,
            "latency_seconds": mapping_seconds,
            "average_latency_ms": 1000.0 * mapping_seconds / query_count,
            **_mapping_diagnostics(evaluated),
        },
        "pairs": evaluated,
    }


def evaluate_ase_records(
    records: Iterable[dict[str, Any]],
    matcher: Any,
    *,
    mapping_top_k: int = 3,
) -> dict[str, Any]:
    """Map every actionable ASE source node and score public set-valued gold."""

    if mapping_top_k <= 0:
        raise ValueError("mapping_top_k must be positive")
    load_omnitransfer()
    from omnitransfer.learned_matcher import ALL_NODE_CANDIDATE_POLICY
    from omnitransfer.mapping_dataset import validate_ui_correspondence_pair
    from omnitransfer.page_mapping import map_page

    normalized = [validate_ui_correspondence_pair(record) for record in records]
    if not normalized:
        raise ValueError("ASE input is empty")
    if matcher.config.candidate_policy != ALL_NODE_CANDIDATE_POLICY:
        raise ValueError(
            "ASE all-node evaluation requires an all-node OmniTransfer checkpoint; "
            f"received candidate_policy={matcher.config.candidate_policy!r}"
        )
    evaluated = []
    mapping_seconds = 0.0
    for record in normalized:
        source = _page_graph(record["source"], record)
        target = _page_graph(record["target"], record)
        started = time.perf_counter()
        mapping = map_page(
            source,
            target,
            matcher,
            top_k=mapping_top_k,
            gold=_gold(record),
            label_boundary="ase_public_gold",
        )
        mapping_seconds += time.perf_counter() - started
        evaluated.append(
            {
                "pair_id": record["pair_id"],
                "dataset": "ASE",
                "source": record["source"],
                "selected_target": record["target"],
                "mapping": mapping,
                "provenance": record["provenance"],
            }
        )
    metrics = _aggregate_gold(evaluated)
    return {
        "schema_version": REPORT_SCHEMA,
        "dataset": "ASE",
        "configuration": {
            "mapping": "canonical_omnitransfer_page_wide_single_forward",
            "mapping_top_k": mapping_top_k,
            "source_scope": "all_actionable_nodes",
        },
        "label_boundary": {
            "status": "public_gold",
            "source": "ASE",
            "set_valued_gold": True,
        },
        "element_mapping": {
            **metrics,
            "latency_seconds": mapping_seconds,
            "average_latency_ms": 1000.0 * mapping_seconds / len(evaluated),
            **_mapping_diagnostics(evaluated),
        },
        "pairs": evaluated,
    }


def write_report(report: dict[str, Any], output_dir: str | Path) -> dict[str, Any]:
    """Write metrics, complete mapping evidence, and the canonical review UI."""

    output = Path(output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    payload = _review_payload(report)
    _materialize_review_screenshots(payload, output)
    sidecar = output / "review.html.payload.json"
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    template_path = canonical_review_template()
    template = template_path.read_text(encoding="utf-8")
    marker = "__OMNITRANSFER_REVIEW_PAYLOAD__"
    if template.count(marker) != 1:
        raise ValueError("canonical_review_template_payload_marker_invalid")
    review_path = output / "review.html"
    review_path.write_text(
        template.replace(
            marker,
            json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"),
        ),
        encoding="utf-8",
    )
    return {
        "report": str(report_path),
        "review": str(review_path),
        "review_sidecar": str(sidecar),
        "pairs": len(report["pairs"]),
    }


def _materialize_review_screenshots(
    payload: dict[str, Any], output: Path
) -> None:
    screenshot_dir = output / "screenshots"
    copied: dict[Path, str] = {}
    for task in payload["pairs"]:
        for side in ("source", "target"):
            value = str(task[side].get("screenshot_path") or "").strip()
            if not value:
                continue
            source = Path(value).expanduser().resolve()
            if not source.is_file():
                raise FileNotFoundError(f"review_screenshot_missing:{source}")
            relative = copied.get(source)
            if relative is None:
                screenshot_dir.mkdir(parents=True, exist_ok=True)
                digest = hashlib.sha256(str(source).encode()).hexdigest()[:16]
                destination = screenshot_dir / f"{digest}_{source.name}"
                shutil.copy2(source, destination)
                relative = f"screenshots/{destination.name}"
                copied[source] = relative
            task[side]["screenshot_path"] = relative


def _review_payload(report: dict[str, Any]) -> dict[str, Any]:
    tasks = []
    for pair in report["pairs"]:
        source_page = pair["source"]
        target_page = pair["selected_target"]
        source_nodes = {
            str(node.get("node_id")): node
            for node in source_page["graph"].get("nodes", ())
        }
        target_nodes = {
            str(node.get("node_id")): node
            for node in target_page["graph"].get("nodes", ())
        }
        for row in pair["mapping"]["mappings"]:
            source_node = source_nodes.get(row["source_node_id"])
            predicted = target_nodes.get(row["top1_node_id"])
            gold_nodes = [
                target_nodes[node_id]
                for node_id in row["gold_node_ids"]
                if node_id in target_nodes
            ]
            if not gold_nodes:
                continue
            reviewed_gold = [_review_node(node) for node in gold_nodes]
            reviewed_prediction = {
                    "node": _review_node(predicted),
                    "top1_node": _review_node(predicted),
                    "accepted": False,
                    "reason": "candidate_ranking_only",
                    "probability": row["pair_confidence"],
                    "margin": row["margin"],
                    "top_k": row["candidates"][: row["diagnostic_top_k"]],
            }
            tasks.append(
                {
                    "task_id": f"{pair['pair_id']}:{row['source_node_id']}",
                    "pair_id": pair["pair_id"],
                    "app": str(
                        pair["source"]["graph"].get("metadata", {}).get("app")
                        or pair["provenance"].get("dataset")
                        or report["dataset"]
                    ),
                    "label_status": report["label_boundary"]["status"],
                    "difficulty_score": 1.0 - float(row["pair_confidence"]),
                    "difficulty_reasons": ["page_wide_mapping"],
                    "source": _workbench_page(source_page, source_node),
                    "target": _workbench_page(target_page, reviewed_gold[0]),
                    "gold_proposal": reviewed_gold[0],
                    "gold_proposals": reviewed_gold,
                    "matcher_prediction": reviewed_prediction,
                    "selector_prediction": None,
                }
            )
    return {
        "summary": {
            "schema_version": "omniflow.mapping-datasets-review.v1",
            "task_count": len(tasks),
            "review_ui": {
                "protocol": "page_wide_mapping_diagnostic",
                "template_ids": [
                    "correct_correspondence",
                    "wrong_correspondence",
                    "ambiguous_or_absent",
                    "discard_bad_evidence",
                ],
                "template_overrides": {},
                "diagnostic_overlay": {
                    "enabled": True,
                    "methods": ["gold_proposal", "matcher_prediction"],
                    "coordinate_space": "page_pixels",
                },
            },
            "label_boundary": report["label_boundary"],
        },
        "pairs": tasks,
    }


def _workbench_page(page: dict[str, Any], node: dict[str, Any] | None) -> dict[str, Any]:
    graph = page["graph"]
    return {
        "page_id": page["page_id"],
        "width": graph.get("width"),
        "height": graph.get("height"),
        "screenshot_path": page.get("screenshot_path"),
        "node": _review_node(node),
        "candidates": [
            reviewed
            for candidate in graph.get("nodes", ())
            if (reviewed := _review_node(candidate)) is not None
        ],
    }


def _review_node(node: Any | None) -> dict[str, Any] | None:
    if node is None:
        return None
    value = node if isinstance(node, dict) else vars(node)
    bbox = value.get("bbox")
    return {
        "node_id": str(value.get("node_id") or ""),
        "origin_id": str(value.get("origin_id") or ""),
        "text": str(value.get("text") or ""),
        "content_desc": str(value.get("content_desc") or ""),
        "resource_id": "",
        "class_name": str(value.get("class_name") or ""),
        "bbox": list(bbox) if bbox else None,
        "clickable": bool(value.get("clickable")),
        "editable": bool(value.get("editable")),
        "scrollable": bool(value.get("scrollable")),
    }


def _graph_page_record(page: dict[str, Any]) -> dict[str, Any]:
    return {
        "page_id": page["graph_id"],
        "platform": "android",
        "screenshot_path": str(
            (page["record"].get("metadata") or {}).get("screenshot_path") or ""
        ),
        "graph": page["record"],
    }


def _baseline_page_mapping(
    source: Any,
    target: Any,
    *,
    source_node_ids: Sequence[str],
    target_node_ids: Sequence[str],
    gold: dict[str, set[str]],
    top_k: int,
    method: str,
) -> dict[str, Any]:
    if method not in MOBILEVIEWS_BASELINES:
        raise ValueError(f"unsupported MobileViews baseline: {method}")
    source_nodes_by_id = {node.node_id: node for node in source.nodes}
    target_nodes_by_id = {node.node_id: node for node in target.nodes}
    source_nodes = tuple(source_nodes_by_id[node_id] for node_id in source_node_ids)
    target_nodes = tuple(target_nodes_by_id[node_id] for node_id in target_node_ids)
    if not source_nodes or not target_nodes:
        raise ValueError("baseline mapping requires non-empty candidate sets")

    scores = np.asarray(
        [
            [
                _baseline_pair_score(
                    source_node,
                    target_node,
                    source,
                    target,
                    method=method,
                )
                for target_node in target_nodes
            ]
            for source_node in source_nodes
        ],
        dtype=np.float64,
    )
    reverse_best = {
        target_index: min(
            range(len(source_nodes)),
            key=lambda source_index: (
                -float(scores[source_index, target_index]),
                source_nodes[source_index].node_id,
            ),
        )
        for target_index in range(len(target_nodes))
    }
    mappings = []
    for source_index, source_node in enumerate(source_nodes):
        positions = sorted(
            range(len(target_nodes)),
            key=lambda target_index: (
                -float(scores[source_index, target_index]),
                target_nodes[target_index].node_id,
            ),
        )
        candidates = [
            {
                "rank": rank + 1,
                "target_node_id": target_nodes[target_index].node_id,
                "score": float(scores[source_index, target_index]),
                "bbox": (
                    list(target_nodes[target_index].bbox)
                    if target_nodes[target_index].bbox
                    else None
                ),
                "text": target_nodes[target_index].text,
                "content_desc": target_nodes[target_index].content_desc,
                "class_name": target_nodes[target_index].class_name,
            }
            for rank, target_index in enumerate(positions)
        ]
        best_target_index = positions[0]
        best_target_id = target_nodes[best_target_index].node_id
        second_score = (
            float(scores[source_index, positions[1]]) if len(positions) > 1 else 0.0
        )
        gold_node_ids = sorted(gold.get(source_node.node_id, ()))
        mappings.append(
            {
                "source_node_id": source_node.node_id,
                "source_bbox": list(source_node.bbox) if source_node.bbox else None,
                "top1_node_id": best_target_id,
                "score": float(scores[source_index, best_target_index]),
                "margin": float(scores[source_index, best_target_index]) - second_score,
                "mutual_top1": reverse_best[best_target_index] == source_index,
                "candidate_count": len(candidates),
                "candidates": candidates,
                "diagnostic_top_k": min(top_k, len(candidates)),
                "gold_node_ids": gold_node_ids,
                "top1_correct": (
                    best_target_id in gold_node_ids
                    if source_node.node_id in gold
                    else None
                ),
                "gold_in_top_k": (
                    any(
                        candidate["target_node_id"] in gold_node_ids
                        for candidate in candidates[:top_k]
                    )
                    if source_node.node_id in gold
                    else None
                ),
            }
        )
    predicted = [row["top1_node_id"] for row in mappings]
    collisions = sum(count - 1 for count in Counter(predicted).values() if count > 1)
    evaluated = [row for row in mappings if row["source_node_id"] in gold]
    top1_hits = sum(bool(row["top1_correct"]) for row in evaluated)
    recall_hits = sum(bool(row["gold_in_top_k"]) for row in evaluated)
    return {
        "schema_version": BASELINE_MAPPING_SCHEMA,
        "method": method,
        "description": _baseline_description(method),
        "resource_id_input": False,
        "source_graph_id": source.graph_id,
        "target_graph_id": target.graph_id,
        "source_node_count": len(source_nodes),
        "target_candidate_count": len(target_nodes),
        "mappings": mappings,
        "mutual_top1_rate": sum(bool(row["mutual_top1"]) for row in mappings)
        / len(mappings),
        "collision_rate": collisions / len(mappings),
        "gold_metrics": {
            "evaluated": len(evaluated),
            "top1_hits": top1_hits,
            "top1": top1_hits / len(evaluated) if evaluated else 0.0,
            "recall_at_k_hits": recall_hits,
            "recall_at_k": recall_hits / len(evaluated) if evaluated else 0.0,
            "coverage": len(evaluated) / len(gold) if gold else 0.0,
        },
    }


def _baseline_pair_score(
    source_node: Any,
    target_node: Any,
    source_graph: Any,
    target_graph: Any,
    *,
    method: str,
) -> float:
    position_similarity = _position_similarity(
        source_node,
        target_node,
        source_graph,
        target_graph,
    )
    if method == "position_scaling":
        return position_similarity
    if method == "text_class_position":
        semantic_similarity = _semantic_similarity(source_node, target_node)
        class_similarity = float(
            _class_tail(source_node.class_name) == _class_tail(target_node.class_name)
        )
        return (semantic_similarity + class_similarity + position_similarity) / 3.0
    raise ValueError(f"unsupported MobileViews baseline: {method}")


def _baseline_description(method: str) -> str:
    if method == "position_scaling":
        return "nearest normalized target center after page scaling"
    if method == "text_class_position":
        return "fixed equal-weight text, class, and normalized-position similarity"
    raise ValueError(f"unsupported MobileViews baseline: {method}")


def _position_similarity(
    source_node: Any,
    target_node: Any,
    source_graph: Any,
    target_graph: Any,
) -> float:
    distance = _position_shift(source_node, target_node, source_graph, target_graph)
    return max(0.0, 1.0 - distance / float(np.sqrt(2.0)))


def _semantic_similarity(source_node: Any, target_node: Any) -> float:
    source = " ".join(value for value in _node_semantics(source_node) if value).strip()
    target = " ".join(value for value in _node_semantics(target_node) if value).strip()
    if not source or not target:
        return 0.0
    return float(SequenceMatcher(None, source, target, autojunk=False).ratio())


def _class_tail(value: str) -> str:
    return str(value or "").rsplit(".", 1)[-1].casefold()


def _actionable_node(node: Any) -> bool:
    return bool(node.enabled and (node.clickable or node.editable or node.scrollable))


def _unique_resource_id_nodes(nodes: Iterable[Any]) -> dict[str, Any]:
    normalized = [
        (str(node.resource_id or "").strip(), node)
        for node in nodes
    ]
    counts = Counter(
        resource_id
        for resource_id, _ in normalized
        if resource_id and not _system_resource_id(resource_id)
    )
    return {
        resource_id: node
        for resource_id, node in normalized
        if resource_id and not _system_resource_id(resource_id) and counts[resource_id] == 1
    }


def _system_resource_id(value: str) -> bool:
    lowered = value.lower()
    return lowered.startswith("android:id/") or lowered.startswith("android.r.id/")


def _mobileviews_pair_quality(
    source: dict[str, Any],
    target: dict[str, Any],
    *,
    minimum_shared_ids: int,
    minimum_identity_overlap: float,
    minimum_position_shift: float,
) -> dict[str, Any] | None:
    shared = sorted(set(source["identities"]) & set(target["identities"]))
    if len(shared) < minimum_shared_ids:
        return None
    identity_overlap = len(shared) / min(
        len(source["identities"]), len(target["identities"])
    )
    if identity_overlap < minimum_identity_overlap:
        return None
    gold: dict[str, set[str]] = {}
    changed = 0
    for resource_id in shared:
        source_node = source["identities"][resource_id]
        target_node = target["identities"][resource_id]
        if not _compatible_action_role(source_node, target_node):
            continue
        gold[source_node.node_id] = {target_node.node_id}
        if _node_semantics(source_node) != _node_semantics(target_node) or _position_shift(
            source_node,
            target_node,
            source["graph"],
            target["graph"],
        ) >= minimum_position_shift:
            changed += 1
    if len(gold) < minimum_shared_ids or changed == 0:
        return None
    return {
        "gold": gold,
        "identity_overlap": identity_overlap,
        "shared_identity_count": len(gold),
        "changed_identity_count": changed,
        "changed_identity_fraction": changed / len(gold),
    }


def _compatible_action_role(source_node: Any, target_node: Any) -> bool:
    source_role = (
        bool(source_node.clickable),
        bool(source_node.editable),
        bool(source_node.scrollable),
        str(source_node.class_name or "").rsplit(".", 1)[-1].lower(),
    )
    target_role = (
        bool(target_node.clickable),
        bool(target_node.editable),
        bool(target_node.scrollable),
        str(target_node.class_name or "").rsplit(".", 1)[-1].lower(),
    )
    return source_role == target_role


def _node_semantics(node: Any) -> tuple[str, str]:
    return (
        " ".join(str(node.text or "").lower().split()),
        " ".join(str(node.content_desc or "").lower().split()),
    )


def _position_shift(source_node: Any, target_node: Any, source: Any, target: Any) -> float:
    source_center = _normalized_center(source_node, source)
    target_center = _normalized_center(target_node, target)
    if source_center is None or target_center is None:
        return 0.0
    return float(np.linalg.norm(np.asarray(source_center) - np.asarray(target_center)))


def _normalized_center(node: Any, graph: Any) -> tuple[float, float] | None:
    if node.bbox is None or not graph.width or not graph.height:
        return None
    left, top, right, bottom = node.bbox
    return ((left + right) / (2.0 * graph.width), (top + bottom) / (2.0 * graph.height))


def _page_graph(page: dict[str, Any], record: dict[str, Any]) -> Any:
    from omnitransfer.ui_graph import graph_from_record

    graph = graph_from_record(page["graph"], graph_id=page["page_id"])
    return replace(
        graph,
        metadata={
            **graph.metadata,
            "screenshot_path": page.get("screenshot_path"),
            "dataset": record["provenance"].get("dataset"),
        },
    )


def _graph_xml(graph: Any) -> str:
    nodes = {node.node_id: node for node in graph.nodes}
    roots = [node for node in graph.nodes if node.parent_id not in nodes]

    def element(node: Any, ancestry: frozenset[str]) -> ET.Element:
        if node.node_id in ancestry:
            raise ValueError(f"UI graph contains a cycle: {node.node_id}")
        bounds = node.bbox or (0.0, 0.0, 1.0, 1.0)
        attrs = {
            "node-id": node.node_id,
            "origin-id": node.origin_id,
            "class": node.class_name or "android.view.View",
            "text": node.text,
            "content-desc": node.content_desc,
            "resource-id": node.resource_id,
            "bounds": "[{:d},{:d}][{:d},{:d}]".format(*(round(value) for value in bounds)),
            "clickable": str(bool(node.clickable)).lower(),
            "editable": str(bool(node.editable)).lower(),
            "scrollable": str(bool(node.scrollable)).lower(),
            "enabled": str(bool(node.enabled)).lower(),
        }
        current = ET.Element("node", attrs)
        next_ancestry = ancestry | {node.node_id}
        for child_id in node.child_ids:
            if child_id in nodes:
                current.append(element(nodes[child_id], next_ancestry))
        return current

    hierarchy = ET.Element("hierarchy")
    for root in roots:
        hierarchy.append(element(root, frozenset()))
    return ET.tostring(hierarchy, encoding="unicode")


def _graph_xml_without_resource_ids(graph: Any) -> str:
    xml = _graph_xml(graph)
    root = ET.fromstring(xml)
    for node in root.iter():
        node.attrib["resource-id"] = ""
    return ET.tostring(root, encoding="unicode")


def _app_group(record: dict[str, Any]) -> str:
    return str(
        record.get("slices", {}).get("app")
        or record["provenance"].get("app_id")
        or record["provenance"].get("trace")
        or record["provenance"].get("dataset")
        or "unknown"
    )


def _gold(record: dict[str, Any]) -> dict[str, set[str]]:
    return {
        match["source_node_id"]: set(match["target_node_ids"])
        for match in record["matches"]
        if match["label"] == "correspondence"
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 0.0 else 0.0


def _aggregate_gold(evaluated: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = [
        pair["mapping"]["gold_metrics"]
        for pair in evaluated
        if pair["mapping"].get("gold_metrics") is not None
    ]
    evaluated_nodes = sum(item["evaluated"] for item in metrics)
    top1_hits = sum(item["top1_hits"] for item in metrics)
    recall_hits = sum(item["recall_at_k_hits"] for item in metrics)
    return {
        "evaluated_nodes": evaluated_nodes,
        "top1_hits": top1_hits,
        "top1": top1_hits / evaluated_nodes if evaluated_nodes else 0.0,
        "recall_at_k_hits": recall_hits,
        "recall_at_k": recall_hits / evaluated_nodes if evaluated_nodes else 0.0,
    }


def _mapping_diagnostics(evaluated: list[dict[str, Any]]) -> dict[str, float]:
    mappings = [pair["mapping"] for pair in evaluated]
    return {
        "mean_mutual_top1_rate": sum(item["mutual_top1_rate"] for item in mappings) / len(mappings),
        "mean_collision_rate": sum(item["collision_rate"] for item in mappings) / len(mappings),
    }


def _load_jsonl(paths: Sequence[str | Path], *, limit: int | None = None) -> list[dict[str, Any]]:
    records = []
    for path_value in paths:
        path = Path(path_value).expanduser().resolve()
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    records.append(json.loads(line))
                    if limit is not None and len(records) >= limit:
                        return records
    return records


def _matcher(checkpoint: str | Path) -> Any:
    load_omnitransfer()
    from omnitransfer.numpy_v9_matcher import NumpyGeometricAlignmentMatcher

    return NumpyGeometricAlignmentMatcher.from_checkpoint(
        Path(checkpoint).expanduser().resolve()
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="dataset", required=True)
    for dataset in ("ase", "mobileviews", "mobileviews-graphs"):
        command = subparsers.add_parser(dataset)
        command.add_argument("--input", action="append", required=True)
        command.add_argument("--checkpoint", required=True)
        command.add_argument("--output", required=True)
        command.add_argument("--mapping-top-k", type=int, default=3)
        command.add_argument("--limit", type=int)
        if dataset == "mobileviews":
            command.add_argument("--page-top-k", type=int, default=3)
        elif dataset == "mobileviews-graphs":
            command.add_argument("--retrieval-top-k", type=int, default=5)
            command.add_argument("--pair-limit", type=int, default=30)
            command.add_argument("--maximum-pairs-per-app", type=int, default=3)
            command.add_argument(
                "--allow-cross-activity",
                action="store_true",
                help="Allow retrieval candidates from another activity in the same App.",
            )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    records = _load_jsonl(args.input, limit=args.limit)
    matcher = _matcher(args.checkpoint)
    if args.dataset == "ase":
        report = evaluate_ase_records(
            records,
            matcher,
            mapping_top_k=args.mapping_top_k,
        )
    elif args.dataset == "mobileviews":
        report = evaluate_mobileviews_records(
            records,
            matcher,
            page_top_k=args.page_top_k,
            mapping_top_k=args.mapping_top_k,
        )
    else:
        report = evaluate_mobileviews_graph_records(
            records,
            matcher,
            retrieval_top_k=args.retrieval_top_k,
            pair_limit=args.pair_limit,
            mapping_top_k=args.mapping_top_k,
            maximum_pairs_per_app=args.maximum_pairs_per_app,
            require_same_activity=not args.allow_cross_activity,
        )
    print(json.dumps(write_report(report, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "evaluate_ase_records",
    "evaluate_mobileviews_graph_records",
    "evaluate_mobileviews_records",
    "main",
    "write_report",
]
