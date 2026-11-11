"""Exact-splice semantics; tensor fixtures do not establish model quality."""
import unittest
import torch
from stage_a_core import Snapshot
from run_stage_a_splice import layer_mask, splice


class SpliceTests(unittest.TestCase):
    def test_lower_upper_interventions_are_independent(self):
        self.assertEqual(layer_mask(6, 2, 2, False, False), (False, False, True, True, False, False))
        self.assertEqual(layer_mask(6, 2, 2, True, False), (True, True, True, True, False, False))
        self.assertEqual(layer_mask(6, 2, 2, False, True), (False, False, True, True, True, True))
        self.assertEqual(layer_mask(6, 2, 2, True, True), (True,) * 6)

    def test_endpoint_equivalences_are_not_distinct_model_calls(self):
        self.assertEqual(layer_mask(6, 0, 2, False, False), layer_mask(6, 0, 2, True, False))
        self.assertEqual(layer_mask(6, 4, 2, False, False), layer_mask(6, 4, 2, False, True))
        self.assertEqual(layer_mask(6, 0, 2, False, True), (True,) * 6)

    def test_exact_values_and_target_prefix_are_preserved(self):
        source = Snapshot({}, tuple((torch.full((1, 1, 5, 2), i + 1.), torch.full((1, 1, 5, 2), i + 2.)) for i in range(4)), 2, 3)
        target = Snapshot({}, tuple((k + 100, v + 100) for k, v in source.kv), 2, 3)
        result = splice(source, target, (False, True, True, False))
        for i, pair in enumerate(result):
            for j, got in enumerate(pair):
                self.assertTrue(torch.equal(got[:, :, :2], target.kv[i][j][:, :, :2]))
                expected = target if i in (1, 2) else source
                self.assertTrue(torch.equal(got[:, :, 2:], expected.kv[i][j][:, :, 2:]))
        native = splice(source, target, (True,) * 4)
        self.assertTrue(all(torch.equal(u, v) for pair, orig in zip(native, target.kv) for u, v in zip(pair, orig)))
        with self.assertRaises(ValueError):
            splice(source, target, (1, 1, 1, 1))

    def test_invalid_interval_rejected(self):
        for start, width in [(-1, 2), (5, 2), (1, 0)]:
            with self.assertRaises(ValueError):
                layer_mask(6, start, width, False, False)


if __name__ == '__main__':
    unittest.main()
