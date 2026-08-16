"""Train and compare B-MOCA page-cluster output heads."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import torch

from omniflow.transfer.embedding import SoftPageWordWeights
from src.experiment.page_cluster_learning import (
    FrozenV9NodeBackbone,
    baseline_page_embeddings,
    cluster_retrieval_metrics,
    collect_cluster_pages,
    encode_cluster_pages,
    functional_head_embeddings,
    load_unified_page_pairs,
    strip_action_context,
    train_functional_slot_head,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument(
        "--additional-train-input",
        nargs="*",
        type=Path,
        default=(),
    )
    parser.add_argument("--omnitransfer-root", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--softgate-checkpoint", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--clusters-per-batch", type=int, default=16)
    parser.add_argument("--auxiliary-cluster-ratio", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2.0e-3)
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-eval-pairs", type=int, default=0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    if not output_dir.is_absolute():
        raise SystemExit("--output-dir must be absolute")
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = args.dataset_root.expanduser().resolve()
    paths = (dataset_root / "train.jsonl", dataset_root / "diagnostic.jsonl")
    primary_train_pairs = _limited(
        load_unified_page_pairs(paths, assignment="train"),
        args.max_train_pairs,
    )
    additional_train_pairs = load_unified_page_pairs(
        tuple(path.expanduser().resolve() for path in args.additional_train_input),
        assignment="train",
    )
    train_pairs = (*primary_train_pairs, *additional_train_pairs)
    dev_pairs = _limited(
        load_unified_page_pairs(paths, assignment="dev"),
        args.max_eval_pairs,
    )
    test_pairs = _limited(
        load_unified_page_pairs(paths, assignment="test"),
        args.max_eval_pairs,
    )
    primary_train_pages = collect_cluster_pages(primary_train_pairs)
    train_pages = collect_cluster_pages(train_pairs)
    dev_pages = collect_cluster_pages(dev_pairs)
    test_pages = collect_cluster_pages(test_pairs)
    backbone = FrozenV9NodeBackbone(
        omnitransfer_root=args.omnitransfer_root,
        checkpoint=args.checkpoint,
        device=args.device,
    )
    encoding_started = time.perf_counter()
    train_structure = encode_cluster_pages(
        backbone,
        train_pages,
        structure_only=True,
        include_masked_view=True,
    )
    dev_structure = encode_cluster_pages(
        backbone,
        dev_pages,
        structure_only=True,
        include_masked_view=True,
    )
    test_structure = encode_cluster_pages(
        backbone,
        test_pages,
        structure_only=True,
        include_masked_view=True,
    )
    dev_full = encode_cluster_pages(backbone, dev_pages, structure_only=False)
    test_full = encode_cluster_pages(backbone, test_pages, structure_only=False)
    encoding_seconds = time.perf_counter() - encoding_started
    soft_weights = SoftPageWordWeights.from_npz(
        str(args.softgate_checkpoint.expanduser().resolve())
    )
    baseline = {
        "dev": _baseline_metrics(dev_full, dev_pages, soft_weights),
        "test": _baseline_metrics(test_full, test_pages, soft_weights),
    }
    training_started = time.perf_counter()
    head, training = train_functional_slot_head(
        train_structure,
        dev_structure,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        clusters_per_batch=args.clusters_per_batch,
        primary_cluster_ids=frozenset(
            page.cluster_id for page in primary_train_pages
        ),
        auxiliary_cluster_ratio=args.auxiliary_cluster_ratio,
        seed=args.seed,
        device=args.device,
    )
    training_seconds = time.perf_counter() - training_started
    dev_vectors = functional_head_embeddings(
        head,
        dev_structure,
        device=args.device,
    ).cpu()
    test_vectors = functional_head_embeddings(
        head,
        test_structure,
        device=args.device,
    ).cpu()
    dev_structure_only_vectors = functional_head_embeddings(
        head,
        strip_action_context(dev_structure),
        device=args.device,
    ).cpu()
    test_structure_only_vectors = functional_head_embeddings(
        head,
        strip_action_context(test_structure),
        device=args.device,
    ).cpu()
    ours = {
        "dev": cluster_retrieval_metrics(dev_vectors, dev_pages),
        "test": cluster_retrieval_metrics(test_vectors, test_pages),
    }
    ours_structure_only = {
        "dev": cluster_retrieval_metrics(
            dev_structure_only_vectors,
            dev_pages,
        ),
        "test": cluster_retrieval_metrics(
            test_structure_only_vectors,
            test_pages,
        ),
    }
    checkpoint_path = output_dir / "functional_slot_head.pt"
    torch.save(
        {
            "schema_version": "omniflow.functional-page-head.v1",
            "state_dict": head.cpu().state_dict(),
            "method": training["method"],
        },
        checkpoint_path,
    )
    report = {
        "schema_version": "omniflow.bmoca-page-cluster-training.v1",
        "interpretation": (
            "B-MOCA alignment labels are automatic/unreviewed; metrics are "
            "diagnostic and not formal gold results."
        ),
        "data": {
            "dataset_root": str(dataset_root),
            "train_pairs": len(train_pairs),
            "additional_train_pairs": len(additional_train_pairs),
            "additional_train_inputs": [
                str(path.expanduser().resolve())
                for path in args.additional_train_input
            ],
            "dev_pairs": len(dev_pairs),
            "test_pairs": len(test_pairs),
            "train_pages": len(train_pages),
            "dev_pages": len(dev_pages),
            "test_pages": len(test_pages),
            "train_clusters": len({page.cluster_id for page in train_pages}),
            "dev_clusters": len({page.cluster_id for page in dev_pages}),
            "test_clusters": len({page.cluster_id for page in test_pages}),
        },
        "method": training["method"],
        "baselines": baseline,
        "ours": ours,
        "ours_structure_only": ours_structure_only,
        "training": training,
        "timing": {
            "encoding_seconds": encoding_seconds,
            "training_seconds": training_seconds,
        },
        "checkpoint": str(checkpoint_path),
    }
    report_path = output_dir / "training_report.json"
    temporary = report_path.with_suffix(".json.part")
    temporary.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(report_path)
    print(
        json.dumps(
            {
                "baselines": baseline["test"],
                "ours": ours["test"],
                "report": str(report_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _limited(values: tuple, limit: int) -> tuple:
    if limit < 0:
        raise ValueError("pair limit must be non-negative")
    return values[:limit] if limit else values


def _baseline_metrics(pages, cluster_pages, soft_weights):
    return {
        name: cluster_retrieval_metrics(vectors, cluster_pages)
        for name, vectors in baseline_page_embeddings(
            pages,
            soft_weights=soft_weights,
        ).items()
    }


if __name__ == "__main__":
    raise SystemExit(main())
