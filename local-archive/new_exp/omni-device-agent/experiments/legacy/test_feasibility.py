"""Executable regressions for research reasoning; no model-quality claims."""
import unittest
from feasibility import evaluate_fixed_residency, iid_miss_estimate, kv_bytes, report, serialization_seconds


class FeasibilityTests(unittest.TestCase):
    def test_units_and_kv_shape(self):
        self.assertEqual(serialization_seconds(12500000, 100), 1)
        self.assertEqual(kv_bytes(8192, 48, 4, 128), 768 * 1024**2)

    def test_same_hit_rate_does_not_imply_same_dependency_latency(self):
        cases = report()["same_hit_rate_synthetic_counterexample"]
        a, b = cases.values()
        self.assertEqual(a["expert_hit_rate"], b["expert_hit_rate"])
        self.assertEqual(a["remote_expert_selections"], b["remote_expert_selections"])
        self.assertEqual(b["propagation_seconds_total"], 8 * a["propagation_seconds_total"])

    def test_full_residency_and_zero_residency_extremes(self):
        routes = [[[0, 1], [0, 1]], [[1, 2], [1, 2]]]
        full = evaluate_fixed_residency(routes, [{0, 1, 2}, {0, 1, 2}], 30)
        empty = evaluate_fixed_residency(routes, [set(), set()], 30)
        self.assertEqual(full["blocking_rpc_per_token"], [0, 0])
        self.assertEqual(empty["blocking_rpc_per_token"], [2, 2])
        self.assertEqual(empty["propagation_seconds_total"], .12)

    def test_adding_residency_cannot_increase_blocking(self):
        routes = [[[0, 1], [1, 2]], [[2, 3], [0, 3]]]
        a = evaluate_fixed_residency(routes, [{0}, {0}], 30)
        b = evaluate_fixed_residency(routes, [{0, 1, 2}, {0, 1, 2}], 30)
        self.assertLessEqual(b["propagation_seconds_total"], a["propagation_seconds_total"])

    def test_iid_extremes(self):
        self.assertEqual(iid_miss_estimate(1, 48, 8, 30)["propagation_seconds_per_token"], 0)
        self.assertEqual(iid_miss_estimate(0, 48, 8, 30)["propagation_seconds_per_token"], 1.44)

    def test_invalid_trace_rejected_not_silently_replayed(self):
        for trace in ([[[0], [1]]], [[[0, 0]]], [[[128]]]):
            with self.assertRaises(ValueError):
                evaluate_fixed_residency(trace, [{0}], 30)


if __name__ == "__main__":
    unittest.main()
