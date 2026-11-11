"""Operator boundary tests; no model/phone run is performed by these tests."""
import unittest
import hashlib
import json
from pathlib import Path
import torch
from run_mobile_operator_probe import select_missing_images, text_suffix


class OperatorBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.item = {"spans": [[2, 4], [6, 8], [10, 12]], "image_hashes": ["a", "b", "c"]}

    def test_decoder_prefix_already_covers_encoder_work(self):
        self.assertEqual(select_missing_images(self.item, 8, set()), [2])

    def test_encoder_cache_survives_prefix_invalidation(self):
        self.assertEqual(select_missing_images(self.item, 0, {"a", "b"}), [2])

    def test_sparse_target_must_compute_unseen_images(self):
        self.assertEqual(select_missing_images(self.item, 0, {"not-in-current-window"}), [0, 1, 2])

    def test_warm_encoder_and_cold_decoder_are_independent(self):
        self.assertEqual(select_missing_images(self.item, 0, {"a", "b", "c"}), [])

    def test_text_suffix_preserves_positions_mask_and_deepstack_alignment(self):
        cap = {"inputs_embeds": torch.arange(14).reshape(1, 7, 2),
               "position_ids": torch.arange(21).reshape(3, 1, 7),
               "attention_mask": torch.ones(1, 7),
               "visual_pos_masks": torch.tensor([[False, True, True, False, True, True, False]]),
               "deepstack_visual_embeds": [torch.arange(8).reshape(4, 2)]}
        sentinel = object()
        out = text_suffix(cap, 3, sentinel)
        self.assertIs(out["past_key_values"], sentinel)
        self.assertTrue(torch.equal(out["deepstack_visual_embeds"][0], cap["deepstack_visual_embeds"][0][2:]))
        self.assertTrue(torch.equal(out["position_ids"], cap["position_ids"][:, :, 3:]))
        self.assertEqual(out["attention_mask"].shape[-1], 7)
        self.assertEqual(out["inputs_embeds"].shape[1], 4)
        self.assertEqual(cap["inputs_embeds"].shape[1], 7)


class CompletedOperatorAuditTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).parent
        self.result = json.loads((self.root / "results/mobile_operator_probe_mps.json").read_text())

    def test_completed_native_operator_run_matches_sources(self):
        self.assertEqual(self.result["status"], "completed")
        for key, filename in [("source_sha256", "run_mobile_operator_probe.py"),
                              ("baseline_source_sha256", "run_mobile_prefill_replay.py")]:
            self.assertEqual(self.result[key], hashlib.sha256((self.root / filename).read_bytes()).hexdigest())
        self.assertEqual(len(self.result["records"]), 24)
        self.assertEqual(len(self.result["vision_records"]), 12)
        for r in self.result["records"]:
            self.assertTrue(r["last_logit_finite"])
            self.assertEqual(r["last_logit_max_abs_error"], 0)

    def test_sparse_schedule_does_not_receive_future_encoder_hits(self):
        c = self.result["conditions"]
        self.assertEqual(c["trim3/sparse"]["encoder_cache_hits_in_suffix"], 0)
        self.assertEqual(c["trim3/continuous"]["encoder_cache_hits_in_suffix"], 2)
        self.assertEqual(c["trim3/sparse"]["prefix_tokens"], c["trim3/continuous"]["prefix_tokens"])

    def test_transport_analysis_recomputes_from_actual_assets(self):
        from analyze_mobile_operator_probe import analyze
        actual = analyze(self.root / "results/mobile_operator_probe_mps.json",
                         self.root / "results/mobile_prefill_image_manifest.json")
        saved = json.loads((self.root / "results/mobile_operator_transport.json").read_text())
        self.assertEqual(actual, saved)
        sparse = actual["cases"]["trim3/sparse"]
        self.assertEqual(sparse["raw_webp_bytes"], 65758)
        self.assertEqual(sparse["target_encoder_feature_bytes"], 11304960)


if __name__ == "__main__":
    unittest.main()
