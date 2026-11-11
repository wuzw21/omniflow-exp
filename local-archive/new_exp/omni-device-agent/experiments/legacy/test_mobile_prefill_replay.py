"""Replay boundary regressions, separate from any device/agent acceptance."""
import copy
from pathlib import Path
import unittest

import torch

from run_mobile_prefill_replay import common_prefix, image_spans, upstream_history


def item(ids, hashes):
    return {"input_ids": torch.tensor([ids]),
            "position_ids": torch.arange(len(ids)).reshape(1, 1, -1).repeat(3, 1, 1),
            "spans": image_spans(ids, 99), "image_hashes": hashes}


class PrefixBoundaryTests(unittest.TestCase):
    def test_append_reuses_complete_old_input(self):
        self.assertEqual(common_prefix(item([1, 99, 99, 2], ["image-a"]),
                                       item([1, 99, 99, 2, 3], ["image-a"])), 4)

    def test_placeholder_tokens_do_not_prove_same_image(self):
        self.assertEqual(common_prefix(item([1, 99, 99, 2], ["image-a"]),
                                       item([1, 99, 99, 2], ["image-b"])), 1)

    def test_cannot_reuse_changed_rope_positions(self):
        a = item([1, 2, 3], [])
        b = copy.deepcopy(a)
        b["position_ids"][2, 0, 1] += 1
        self.assertEqual(common_prefix(a, b), 1)

    def test_no_partial_image_block_reuse(self):
        a = item([1, 99, 99, 99, 2], ["image-a"])
        b = copy.deepcopy(a)
        b["position_ids"][1, 0, 2] += 1
        self.assertEqual(common_prefix(a, b), 1)

    def test_prefix_stops_at_image_eviction(self):
        a = item([1, 99, 99, 2, 99, 99, 3], ["a", "b"])
        b = item([1, 5, 2, 99, 99, 3, 99, 99], ["b", "c"])
        self.assertEqual(common_prefix(a, b), 1)

    def test_original_policy_caps_three_images_at_fourth_turn(self):
        root = Path.home() / ".agent-reach/research/qwen-cua-8a295fd"
        if not (root / "runner.py").exists():
            self.skipTest("pinned upstream source unavailable")
        cls = upstream_history(root)
        h = cls(prompt="fixed fixture", history_n=50, image_max=3)
        h.screenshots = ["a", "b", "c", "d"]
        h.responses = ["one", "two", "three"]
        h.action_summaries = ["one", "two", "three"]
        messages = h.build_messages()
        self.assertEqual(sum(c["type"] == "image_url" for m in messages for c in m["content"]), 3)
        self.assertEqual(messages[1]["content"][-1]["text"], "This screenshot has been collapsed.")
        self.assertEqual(len(h.screenshots), 4)  # Rendering does not replace the history source.


if __name__ == "__main__":
    unittest.main()
