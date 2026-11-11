"""Actual operator measurements plus explicitly hypothetical WAN arithmetic."""
import argparse
import hashlib
import json
from pathlib import Path

from feasibility import serialization_seconds
from latent_transport_bounds import compute_budget_seconds


def analyze(result_path, manifest_path):
    result = json.loads(result_path.read_text())
    manifest = json.loads(manifest_path.read_text())
    if result["status"] != "completed" or result["trajectory"] != manifest["trajectory"]:
        raise ValueError("completed matching records required")
    for record in result["records"]:
        if not record["last_logit_finite"]:
            raise ValueError("non-finite operator check")
    raw_sizes = []
    for image in manifest["images"]:
        content = Path(image["path"]).read_bytes()
        if hashlib.sha256(content).hexdigest() != image["sha256"]:
            raise ValueError("image hash mismatch")
        raw_sizes.append(len(content))
    cases = {}
    for name, condition in result["conditions"].items():
        # trim3 removes original image 1, so its per-request image indices shift.
        offset = 1 if name.startswith("trim3/") else 0
        raw_bytes = sum(raw_sizes[i + offset] for i in condition["missing_encoder_images"])
        feature_bytes = condition["missing_encoder_output_bytes"]
        vision_time = condition["median_missing_encoder_seconds"]
        cases[name] = {"raw_webp_bytes": raw_bytes, "target_encoder_feature_bytes": feature_bytes,
            "features_to_raw_ratio": feature_bytes / raw_bytes if raw_bytes else None,
            "mps_encoder_seconds": vision_time,
            "network_sensitivity": [{"mbps": bw,
                "raw_seconds": serialization_seconds(raw_bytes, bw),
                "feature_seconds": serialization_seconds(feature_bytes, bw),
                "additional_compute_budget_seconds_vs_mps_encoder": compute_budget_seconds(vision_time, raw_bytes, feature_bytes, bw)}
                for bw in (20, 100, 1000)],
            "break_even_mbps_if_target_encoder_speed_relative_to_mps": {
                str(speed): max(0, feature_bytes - raw_bytes) * 8 / (vision_time / speed * 1_000_000)
                for speed in (1, 2, 5, 10)} if vision_time else {}}
    c = result["conditions"]
    return {"evidence": "real MPS operator timings plus analytic bandwidth sensitivity; not WAN measurements",
            "probe_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
            "image_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "cases": cases,
            "trim3_sparse_minus_continuous": {
                "text_seconds": c["trim3/sparse"]["median_text_seconds"] - c["trim3/continuous"]["median_text_seconds"],
                "encoder_seconds": c["trim3/sparse"]["median_missing_encoder_seconds"] - c["trim3/continuous"]["median_missing_encoder_seconds"],
                "sum_seconds": c["trim3/sparse"]["sum_of_operator_medians_seconds"] - c["trim3/continuous"]["sum_of_operator_medians_seconds"]},
            "limits": ["WebP sizes are these public files, not a measured live capture/upload protocol",
                       "Transferred features must be the target model's pooled AND DeepStack tensors",
                       "No proof that a different local model can produce these tensors",
                       "Local encoding/mapping, lookup/assembly, RTT, CPU serialization and energy are excluded",
                       "Target speed factors are sensitivity inputs, not CUDA measurements",
                       "Image privacy is not guaranteed by converting pixels to features"]}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--probe", type=Path, default=Path("results/mobile_operator_probe_mps.json"))
    p.add_argument("--manifest", type=Path, default=Path("results/mobile_prefill_image_manifest.json"))
    p.add_argument("--output", type=Path, default=Path("results/mobile_operator_transport.json"))
    args = p.parse_args()
    result = analyze(args.probe, args.manifest)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"trim3_sparse_minus_continuous": result["trim3_sparse_minus_continuous"],
                      "sparse_transfer": result["cases"]["trim3/sparse"]}, indent=2))
