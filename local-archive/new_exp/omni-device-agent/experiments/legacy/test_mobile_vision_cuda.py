"""Replayable evidence regressions for the CUDA baseline failure.

These checks audit saved physical-GPU runs; they do not themselves execute CUDA.
Re-run benchmark_mobile_vision_cuda.py and validate_mobile_vision_precision.py
on the target GPU for deployment/runtime changes.
"""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_mobile_vision_cuda import analyze


class VisionCudaEvidenceTests(unittest.TestCase):
    root = Path(__file__).parent

    def read(self, filename):
        return json.loads((self.root / "results" / filename).read_text())

    def test_initial_and_control_runs_are_preserved_with_original_source(self):
        original = hashlib.sha256((self.root / "results/benchmark_mobile_vision_cuda_v1.py").read_bytes()).hexdigest()
        for filename in ("mobile_vision_cuda_initial.json", "vision_cuda_writable_cache.json"):
            data = self.read(filename)
            self.assertEqual(data["status"], "completed")
            self.assertEqual(data["source_sha256"], original)
            self.assertEqual(len(data["records"]), 24)

    def test_grid_placement_diagnostic_did_not_change_output(self):
        data = self.read("vision_grid_diagnostic.json")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["source_sha256"], hashlib.sha256((self.root / "diagnose_mobile_vision_cuda.py").read_bytes()).hexdigest())
        self.assertEqual([r["grid_device"] for r in data["records"]], ["cuda", "cpu", "cpu", "cuda"])
        for record in data["records"]:
            self.assertTrue(record["finite"])
            self.assertEqual(record["bit_equal"], [True] * 4)

    def test_precision_reference_covers_both_batches_and_both_backends(self):
        data = self.read("vision_precision.json")
        self.assertEqual(data["status"], "completed")
        self.assertTrue(data["tf32_disabled"])
        self.assertEqual(data["source_sha256"], hashlib.sha256((self.root / "validate_mobile_vision_precision.py").read_bytes()).hexdigest())
        records = {(r["images"], r["mode"]): r for r in data["records"]}
        self.assertEqual(set(records), {(n, m) for n in (1, 3) for m in ("bf16_contiguous", "bf16_channels_last")})
        for record in records.values():
            self.assertTrue(record["patch_vs_fp32"]["finite"])
            self.assertEqual(len(record["features_vs_fp32"]), 4)
            self.assertTrue(all(v["finite"] for v in record["features_vs_fp32"]))

    def test_selected_result_uses_cudnn_and_reproduces_byte_costs(self):
        actual = analyze(self.root)
        saved = self.read("mobile_vision_cuda_transport.json")
        self.assertEqual(actual, saved)
        self.assertEqual(actual["cases"]["3"]["feature_bytes"], 11304960)
        self.assertEqual(actual["cases"]["3"]["raw_webp_bytes"], 65758)

    def test_analyzer_rejects_slow_backend_even_with_completed_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "results").mkdir()
            for filename in ("mobile_vision_cuda.json", "mobile_operator_probe_mps.json",
                             "mobile_vision_cuda_preparation.json", "mobile_operator_transport.json"):
                (root / "results" / filename).write_bytes((self.root / "results" / filename).read_bytes())
            path = root / "results/mobile_vision_cuda.json"
            data = json.loads(path.read_text())
            data["conv_backend_selected"]["3"] = "_ConvBackend.SlowDilated3d"
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(ValueError, "slow convolution fallback"):
                analyze(root)


if __name__ == "__main__":
    unittest.main()
