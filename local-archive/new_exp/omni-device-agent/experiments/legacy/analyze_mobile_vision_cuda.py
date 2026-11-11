"""Use measured target CUDA encoding cost in existing transport arithmetic."""
import hashlib
import json
from pathlib import Path
import statistics

from feasibility import serialization_seconds
from latent_transport_bounds import compute_budget_seconds


def analyze(root):
    cuda_path = root / "results/mobile_vision_cuda.json"
    cuda = json.loads(cuda_path.read_text())
    mps_path = root / "results/mobile_operator_probe_mps.json"
    mps = json.loads(mps_path.read_text())
    prepared = json.loads((root / "results/mobile_vision_cuda_preparation.json").read_text())
    transport = json.loads((root / "results/mobile_operator_transport.json").read_text())
    if cuda["status"] != "completed" or not cuda["strict_weight_load_passed"]:
        raise ValueError("CUDA run incomplete or weight loading unverified")
    if cuda.get("conv_layout") != "channels_last_3d" or not cuda.get("layout_weight_values_equal"):
        raise ValueError("selected run must validate official convolution layout conversion")
    if any(cuda.get("conv_backend_selected", {}).get(str(n)) != "_ConvBackend.Cudnn" for n in (1, 3)):
        raise ValueError("do not select the known slow convolution fallback as target cost")
    if any(not cuda.get("layout_output_validation", {}).get(str(n), {}).get("finite") for n in (1, 3)):
        raise ValueError("layout output comparison missing or nonfinite")
    precision_path = root / "results/vision_precision.json"
    precision = json.loads(precision_path.read_text())
    if precision["status"] != "completed" or not precision["tf32_disabled"]:
        raise ValueError("FP32 numerical diagnostic incomplete")
    if cuda["source_sha256"] != hashlib.sha256((root / "benchmark_mobile_vision_cuda.py").read_bytes()).hexdigest():
        raise ValueError("benchmark source changed")
    expected_files = {x["name"]: x["sha256"] for x in prepared["local_assets"]}
    if cuda["weights_sha256"] != expected_files["vision.safetensors"] or cuda["pixels_sha256"] != expected_files["pixels.pt"]:
        raise ValueError("transferred assets differ")
    if ":".join(cuda["processed_image_hashes"]) != mps["conditions"]["trim3/sparse"]["encoder_batch_key"]:
        raise ValueError("CUDA pixels differ from MPS probe")
    cases = {}
    for n, source_case in [(1, "trim3/continuous"), (3, "trim3/sparse")]:
        records = [r for r in cuda["records"] if r["images"] == n]
        if len(records) != 12 or any(not r["finite"] for r in records):
            raise ValueError("invalid repetition coverage or output")
        measured = [r["seconds"] for r in records if not r["warmup"]]
        if len(measured) != 10:
            raise ValueError("expected ten post-warmup measurements")
        elapsed = statistics.median(measured)
        raw = transport["cases"][source_case]["raw_webp_bytes"]
        features = transport["cases"][source_case]["target_encoder_feature_bytes"]
        if any(r["output_bytes"] != features for r in records):
            raise ValueError("CUDA feature geometry differs")
        cases[str(n)] = {"cuda_encoder_median_seconds": elapsed,
            "cuda_encoder_min_seconds": min(measured), "cuda_encoder_max_seconds": max(measured),
            "raw_webp_bytes": raw, "feature_bytes": features,
            "break_even_mbps_excluding_extra_compute": 8 * (features - raw) / (elapsed * 1e6),
            "network_sensitivity": [{"mbps": bw,
                "raw_serialization_plus_cuda_encoder_seconds": serialization_seconds(raw, bw) + elapsed,
                "feature_serialization_seconds": serialization_seconds(features, bw),
                "maximum_feature_bytes_with_free_sender_compute": raw + elapsed * bw * 1e6 / 8,
                "additional_compute_budget_seconds": compute_budget_seconds(elapsed, raw, features, bw)}
                for bw in (20, 100, 1000)]}
    return {"evidence": "measured CUDA encoder, hypothetical network rates; no end-to-end edge/cloud run",
            "cuda_result_sha256": hashlib.sha256(cuda_path.read_bytes()).hexdigest(),
            "analyzer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "mps_result_sha256": hashlib.sha256(mps_path.read_bytes()).hexdigest(),
            "layout_output_validation": cuda["layout_output_validation"],
            "precision_result_sha256": hashlib.sha256(precision_path.read_bytes()).hexdigest(),
            "precision_reference": precision["records"],
            "cases": cases,
            "limits": ["2B model's target vision tower; not a larger model experiment",
                       "CUDA and MPS use different runtime versions; not a pure hardware comparison",
                       "Official convolution weight layout changes backend; output differences are recorded, not assumed bitwise identical",
                       "Both BF16 paths differ from FP32 features; neither decoder/task-quality equivalence nor higher accuracy of the fast path is established",
                       "Inputs are preprocessed and GPU resident; image decode/preprocessing and host-device input costs excluded",
                       "Sender computation and mapping are assumed free, a favorable bound for feature transfer",
                       "Same propagation/roundtrip assumed; no measured WAN rate, concurrency, queueing or energy",
                       "Known image cache residency, bulk batching and quality require separate serving validation"]}


if __name__ == "__main__":
    root = Path(__file__).parent
    data = analyze(root)
    (root / "results/mobile_vision_cuda_transport.json").write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data["cases"], indent=2))
