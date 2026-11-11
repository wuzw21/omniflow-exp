"""Audit completed MPS evidence, not a new inference or phone acceptance run."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parent


class CompletedReplayTests(unittest.TestCase):
    def setUp(self):
        self.data = json.loads((ROOT / "results/mobile_prefill_replay_mps_checked.json").read_text())

    def test_complete_result_matches_executed_source(self):
        self.assertEqual(self.data["status"], "completed")
        self.assertEqual(self.data["source_sha256"], hashlib.sha256(
            (ROOT / "run_mobile_prefill_replay.py").read_bytes()).hexdigest())
        self.assertEqual(len(self.data["records"]), 16)
        for name in ("keep4", "trim3"):
            for mode in ("full", "prefix"):
                records = [r for r in self.data["records"] if r["condition"] == name and r["mode"] == mode]
                self.assertEqual([r["repeat"] for r in records], [0, 1, 2, 3])

    def test_image_conditioning_check_is_present_and_finite(self):
        d = self.data["conditioning_diagnostics"]
        self.assertTrue(d["zero_images_finite"])
        self.assertGreater(d["zero_images_logit_max_abs_change"], 0)
        for value in self.data["conditions"].values():
            self.assertTrue(value["reference_diagnostics"]["finite"])

    def test_actual_cache_reuse_matches_one_position_logits(self):
        for r in self.data["records"]:
            self.assertEqual(r["logit_max_abs_error"], 0)
            self.assertTrue(r["next_token_agrees_with_full"])
        # This checks one prediction position only; it is not task accuracy.


if __name__ == "__main__":
    unittest.main()
