import functools
import itertools
import unittest

import numpy as np

from streaming_expert_bounds import belady_misses, partition_bound
from window_oracle import contiguous_segments


def exhaustive_paging(sequence, capacity, initial=()):
    @functools.lru_cache(None)
    def visit(i, resident):
        if i == len(sequence):
            return 0
        page, pages = sequence[i], set(resident)
        if page in pages:
            return visit(i+1, resident)
        if len(pages) < capacity:
            return 1 + visit(i+1, tuple(sorted(pages | {page})))
        return 1 + min(visit(i+1, tuple(sorted(pages - {victim} | {page}))) for victim in pages)
    return visit(0, tuple(sorted(initial)))


class StreamingBoundsTests(unittest.TestCase):
    def test_MIN_matches_exhaustive_eviction_search(self):
        for seq in itertools.product(range(3), repeat=6):
            for capacity in (1, 2, 3):
                self.assertEqual(belady_misses(seq, capacity), exhaustive_paging(seq, capacity))

    def test_layer_ids_do_not_alias(self):
        routes = np.array([[[0], [0]]])
        self.assertEqual(partition_bound(routes, 1, 10, 10, 4), 10)

    def test_memory_can_hold_partial_expert_bytes(self):
        routes = np.array([[[0]], [[1]]])
        self.assertEqual(partition_bound(routes, 2, 15, 10, 4), 5)

    def test_initial_residency_is_free_each_window(self):
        routes = np.array([[[0]], [[1]], [[0]], [[1]]])
        self.assertEqual(partition_bound(routes, 2, 10, 10, 4), 20)
        self.assertEqual(partition_bound(routes, 4, 10, 10, 4), 10)
        # These alternative partitions must not be summed.
        self.assertEqual(max(partition_bound(routes, w, 10, 10, 4) for w in (2, 4)), 20)

    def test_tail_is_retained_without_crossing_reset(self):
        routes = np.array([[[0]], [[1]], [[2]], [[3]]])
        self.assertEqual(partition_bound(routes, 3, 10, 10, 4), 20)
        segments = contiguous_segments(np.array([0,0,1,1]), np.array([0,1,0,1]), np.zeros(4))
        self.assertEqual(sum(partition_bound(routes[a:b], 3, 20, 10, 4) for a,b in segments), 0)

    def test_zero_bound_is_not_zero_misses(self):
        routes = np.array([[[0]], [[1]], [[2]]])
        self.assertEqual(partition_bound(routes, 1, 10, 10, 4), 0)
        self.assertEqual(belady_misses([0,1,2], 1), 3)

    def test_window_bound_below_exhaustive_cold_cost(self):
        for seq in itertools.product(range(3), repeat=5):
            routes = np.array(seq).reshape(-1,1,1)
            for capacity in (1,2):
                bound = max(partition_bound(routes,w,capacity,1,3) for w in (1,2,3,5))
                self.assertLessEqual(bound, exhaustive_paging(seq, capacity))

    def test_bound_holds_with_best_free_initial_cache(self):
        for seq in itertools.product(range(3), repeat=5):
            routes = np.array(seq).reshape(-1,1,1)
            for capacity in (1,2):
                best_warm = min(exhaustive_paging(seq, capacity, initial)
                                for initial in itertools.combinations(range(3), capacity))
                bound = max(partition_bound(routes,w,capacity,1,3) for w in (1,2,3,5))
                self.assertLessEqual(bound, best_warm)

    def test_invalid_budget_rejected(self):
        for width, memory, size in ((0,10,10),(1,-1,10),(1,10,0)):
            with self.assertRaises(ValueError):
                partition_bound(np.array([[[0]]]), width, memory, size, 4)
        with self.assertRaises(ValueError):
            belady_misses([1], 0)


if __name__ == '__main__':
    unittest.main()
