"""Synthetic counter-audit regressions; these do not run an agent or a model."""
import unittest
import json
from pathlib import Path
from analyze_mobile_cache import analyze, summarize, usage


def record(prompts, cached, identity="fixture"):
    steps = [{"usage": {"prompt_tokens": p, "cached_tokens": c,
                        "completion_tokens": 2, "reasoning_tokens": 1}} for p, c in zip(prompts, cached)]
    return {"id": identity, "steps": steps, "duration_s": 1,
            "usage": {k: sum(s["usage"][k] for s in steps) for k in steps[0]["usage"]}}


class CounterTests(unittest.TestCase):
    def test_missing_usage_is_not_zero_cache(self):
        with self.assertRaises(ValueError):
            usage({"prompt_tokens": 100})

    def test_invalid_cache_count_rejected(self):
        with self.assertRaises(ValueError):
            summarize([record([5], [6])])

    def test_fraction_is_token_weighted(self):
        result = summarize([record([10, 90], [10, 0])])
        self.assertEqual(result["token_weighted_cache_fraction"], 0.1)
        self.assertEqual(result["uncached_prompt_tokens"], 90)

    def test_first_step_may_be_warm(self):
        result = summarize([record([100, 110], [80, 90])])
        self.assertEqual(result["trajectories_with_first_step_cache_hits"], 1)
        self.assertEqual(result["later_steps_cache_fraction"], 90 / 110)

    def test_no_transition_across_trajectory_boundaries(self):
        result = summarize([record([100], [90]), record([100], [10], "second")])
        self.assertEqual(result["cached_token_drop_transitions"], 0)
        self.assertEqual(result["adjacent_transitions"], 0)
        self.assertIsNone(result["fourth_step_cache_change_median"])

    def test_fourth_step_drop_and_root_mismatch_reported(self):
        r = record([10, 20, 30, 32], [5, 10, 20, 5])
        r["usage"]["cached_tokens"] += 1
        result = summarize([r])
        self.assertEqual(result["fourth_step_cache_drops"], 1)
        self.assertEqual(result["fourth_step_cache_change_median"], -15)
        self.assertEqual(result["fourth_step_prompt_change_median"], 2)
        self.assertEqual(len(result["aggregate_counter_mismatches"]), 1)


class PublishedRecordAuditTests(unittest.TestCase):
    def test_saved_analysis_matches_complete_pinned_records(self):
        root = Path.home() / ".agent-reach/research/hcompany-trajectories"
        if not (root / "android_manifest.json").exists():
            self.skipTest("published record cache unavailable; no data audit performed")
        actual = analyze(root)
        saved = json.loads((Path(__file__).parent / "results/mobile_cache_audit.json").read_text())
        self.assertEqual(saved, actual)
        self.assertEqual(actual["all"]["trajectories"], 691)
        self.assertEqual(actual["all"]["aggregate_counter_mismatches"], [])


if __name__ == "__main__":
    unittest.main()
