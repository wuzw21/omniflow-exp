"""Research regressions: bytes, wire cuts, and fair baseline accounting only."""
import unittest

from feasibility import kv_bytes
from latent_transport_bounds import compute_budget_seconds


class TransportBudgetTests(unittest.TestCase):
    def test_one_slot_and_three_slot_working_dtype_are_distinct(self):
        one = kv_bytes(1, 28, 8, 128, 2)
        three_fp32 = kv_bytes(3, 28, 8, 128, 4)
        self.assertEqual(one, 112 * 1024)
        self.assertEqual(three_fp32, 672 * 1024)

    def test_compact_latent_can_be_larger_than_the_source_token_request(self):
        payload = kv_bytes(3, 28, 8, 128, 4)
        self.assertGreater(payload, 2048 * 4)
        self.assertEqual(payload // 4, 172032)

    def test_zero_prefill_baseline_can_leave_no_compute_budget(self):
        # A hot target prefix is a different baseline from a cold target prefix.
        self.assertLess(compute_budget_seconds(0, 5000, 688128, 100), 0)
        self.assertGreater(compute_budget_seconds(.14, 5000, 688128, 100), 0)

    def test_extra_receiver_query_round_trip_is_not_free(self):
        direct = compute_budget_seconds(.14, 5000, 688128, 100)
        queried = compute_budget_seconds(.14, 5000, 688128, 100, 2, 50)
        self.assertAlmostEqual(direct - queried, .1)
        self.assertLess(queried, 0)

    def test_low_uplink_alone_can_eliminate_the_opportunity(self):
        self.assertLess(compute_budget_seconds(.14, 5000, 688128, 20), 0)


if __name__ == "__main__":
    unittest.main()
