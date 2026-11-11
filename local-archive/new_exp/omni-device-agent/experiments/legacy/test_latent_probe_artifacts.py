"""Audit actual-run evidence; these checks do not execute model inference.

Changing the probe invalidates this historical evidence until the real-model
runner is re-executed and the selected artifact updated. This prevents treating
a stale result or a synthetic score as a current deployment regression pass.
"""
import hashlib
import json
from pathlib import Path
import unittest

from analyze_latent_question_probe import analyze

ROOT = Path(__file__).parent
RESULT = ROOT / "results/latent_question_probe_mps_portable.json"


class ActualProbeEvidenceTests(unittest.TestCase):
    def test_real_run_matches_current_probe_source(self):
        data = json.loads(RESULT.read_text())
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["source_sha256"], hashlib.sha256(
            (ROOT / "run_latent_question_probe.py").read_bytes()).hexdigest())
        self.assertEqual(data["runtime"]["device"], "mps")

    def test_paired_coverage_and_split_identity_are_from_completed_run(self):
        summary = analyze(RESULT)
        self.assertEqual(summary["real_model_split_identity_checks_passed"], 8)
        self.assertEqual(summary["changed_question_pair_count"], 4)
        self.assertTrue(all(r["same_generated_token_ids"]
                            for r in summary["changed_question_comparisons"]))

    def test_portable_weights_match_original_published_tensor_hash(self):
        data = json.loads((ROOT / "results/latent_checkpoint_cpu_remap.json").read_text())
        probe = json.loads(RESULT.read_text())
        self.assertTrue(data["all_tensor_keys_dtypes_values_equal"])
        self.assertTrue(data["upstream_integrity_loader_passed"])
        self.assertEqual(data["target_sha256"], probe["adapter_sha256"])
        self.assertEqual(data["unchanged_upstream_state_hash"],
                         probe["adapter_metadata"]["weights_hash"])


if __name__ == "__main__":
    unittest.main()
