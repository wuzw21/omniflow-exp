"""Render transfer-pair memory with the canonical OmniTransfer workbench."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from omniflow.transfer_memory import TransferPair, TransferPairStore


REVIEW_TEMPLATE_RELATIVE_PATH = Path("tests/vector/review_annotation_template.html")
REVIEW_PAYLOAD_MARKER = "__OMNITRANSFER_REVIEW_PAYLOAD__"


def render_transfer_pair_review(
    memory_path: str | Path,
    output_html: str | Path,
    *,
    omnitransfer_root: str | Path | None = None,
) -> dict[str, Any]:
    store = TransferPairStore(memory_path)
    if not store.pairs:
        raise ValueError("transfer_pair_review_memory_empty")
    template_path = canonical_review_template(omnitransfer_root)
    template = template_path.read_text(encoding="utf-8")
    if template.count(REVIEW_PAYLOAD_MARKER) != 1:
        raise ValueError("canonical_review_template_payload_marker_invalid")
    memory_root = Path(memory_path).expanduser().resolve().parent
    tasks = [
        _review_task(pair, memory_root=memory_root)
        for pair in store.pairs.values()
    ]
    payload = {
        "summary": {
            "schema_version": "omniflow.transfer-pair-review.v1",
            "task_count": len(tasks),
            "review_ui": {
                "protocol": "bidirectional_pair_memory",
                "template_ids": [
                    "confirm_pair",
                    "reject_pair",
                    "ambiguous_pair",
                    "discard_bad_evidence",
                ],
                "template_overrides": {},
                "diagnostic_overlay": {
                    "enabled": True,
                    "methods": ["gold_proposal"],
                    "coordinate_space": "page_pixels",
                },
            },
        },
        "pairs": tasks,
    }
    output = Path(output_html).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    sidecar = output.with_name(output.name + ".payload.json")
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    rendered = template.replace(
        REVIEW_PAYLOAD_MARKER,
        json.dumps(payload, ensure_ascii=False).replace("</", "<\\/"),
    )
    output.write_text(rendered, encoding="utf-8")
    return {
        "schema_version": "omniflow.transfer-pair-review-manifest.v1",
        "pairs": len(tasks),
        "review_file": str(output),
        "sidecar": str(sidecar),
        "template": str(template_path),
    }


def canonical_review_template(
    omnitransfer_root: str | Path | None = None,
) -> Path:
    configured = str(omnitransfer_root or os.environ.get("OMNITRANSFER_ROOT") or "")
    root = (
        Path(configured).expanduser().resolve()
        if configured
        else (Path.home() / "Projects" / "Omni" / "OmniTransfer").resolve()
    )
    template = root / REVIEW_TEMPLATE_RELATIVE_PATH
    if not template.is_file():
        raise FileNotFoundError(f"canonical_review_template_missing:{template}")
    return template


def _review_task(pair: TransferPair, *, memory_root: Path) -> dict[str, Any]:
    source = _review_endpoint(pair.source, memory_root=memory_root)
    target = _review_endpoint(pair.target, memory_root=memory_root)
    app = source["package_name"] or target["package_name"] or "unknown"
    return {
        "task_id": pair.pair_id,
        "pair_id": pair.pair_id,
        "app": app,
        "label_status": "aligned_pair_evidence",
        "difficulty_score": 1.0 - float(pair.evidence.get("alignment_score") or 0.0),
        "difficulty_reasons": ["bidirectional_pair_memory"],
        "source": source,
        "target": target,
        "gold_proposal": dict(target["node"]),
        "gold_proposals": [dict(target["node"])],
        "matcher_prediction": {
            "node": None,
            "top1_node": None,
            "accepted": False,
            "reason": "not_evaluated",
            "probability": 0.0,
            "margin": 0.0,
        },
        "selector_prediction": {
            "node": None,
            "reason": "not_evaluated",
            "candidate_count": 0,
        },
        "evidence": dict(pair.evidence),
    }


def _review_endpoint(value: dict[str, Any], *, memory_root: Path) -> dict[str, Any]:
    screenshot = Path(value["screenshot_path"]).expanduser()
    if not screenshot.is_absolute():
        screenshot = memory_root / screenshot
    screenshot = screenshot.resolve()
    if not screenshot.is_file():
        raise FileNotFoundError(f"transfer_pair_screenshot_missing:{screenshot}")
    return {
        "page_id": value["page_id"],
        "package_name": value["package_name"],
        "width": value["width"],
        "height": value["height"],
        "screenshot_path": str(screenshot),
        "point": dict(value["point"]),
        "node": _review_node(value["node"]),
    }


def _review_node(value: dict[str, Any]) -> dict[str, Any]:
    attributes = value.get("attributes")
    attributes = attributes if isinstance(attributes, dict) else {}
    return {
        "node_id": str(value.get("node_id") or ""),
        "bbox": list(value.get("bounds") or ()),
        "text": str(attributes.get("text") or ""),
        "content_desc": str(attributes.get("content_description") or ""),
        "resource_id": str(attributes.get("resource_id") or ""),
        "class_name": str(attributes.get("class") or ""),
    }


__all__ = ["canonical_review_template", "render_transfer_pair_review"]
