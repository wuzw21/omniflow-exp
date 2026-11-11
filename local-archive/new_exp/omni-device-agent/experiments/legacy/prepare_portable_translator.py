"""Lossless device remapping of an upstream checkpoint, not model conversion.

Uses torch's documented map_location and upstream's unchanged integrity loader.
Never modifies or overwrites the original file or its metadata.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import torch
from kvloom.training.checkpoint import CheckpointMetadata, load_translator


def prepare(source, target):
    source, target = Path(source), Path(target)
    sidecar = CheckpointMetadata.sidecar_path(source)
    target_sidecar = CheckpointMetadata.sidecar_path(target)
    if target.exists() or target_sidecar.exists():
        raise FileExistsError("Refusing to overwrite an existing checkpoint or sidecar")
    original = torch.load(source, weights_only=True, map_location="cpu")
    torch.save(original, target)
    shutil.copyfile(sidecar, target_sidecar)
    # Upstream checks the unchanged metadata hash and the complete state dict.
    restored = load_translator(target).state_dict()
    identical = original.keys() == restored.keys() and all(
        original[k].dtype == restored[k].dtype and torch.equal(original[k], restored[k])
        for k in original)
    if not identical:
        raise RuntimeError("CPU remapping changed tensor content")
    return {
        "status": "verified_lossless_cpu_device_remap",
        "source": str(source), "target": str(target),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "target_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "unchanged_upstream_state_hash": CheckpointMetadata.read(source).weights_hash,
        "all_tensor_keys_dtypes_values_equal": identical,
        "upstream_integrity_loader_passed": True,
        "tensor_count": len(original), "torch": torch.__version__,
        "limits": "Artifact portability check only; downstream real-model inference is separate.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(prepare(args.source, args.target), indent=2) + "\n")
