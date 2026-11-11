import unittest
import json
from pathlib import Path
from minions_model_worker import prefill_ranges
from probe_minions_3b_suite import CASES, verify_tasks


class ChunkingContractTests(unittest.TestCase):
    def test_full_prefix_coverage_and_bounded_tail(self):
        for length in (1, 511, 512, 513, 883, 1024, 1025, 21841, 27748, 32000):
            for chunk in (512, 1024):
                ranges = prefill_ranges(length, chunk)
                cursor = 0
                for start, end in ranges:
                    self.assertEqual(start, cursor)
                    self.assertGreater(end, start)
                    self.assertLessEqual(end-start, chunk)
                    cursor = end
                self.assertGreater(length-cursor, 0)
                self.assertLessEqual(length-cursor, chunk)
        self.assertEqual(prefill_ranges(27748, 0), [])

    def test_task_selection_and_hashes(self):
        self.assertEqual(CASES, ('code', 'long_documents', 'finance'))
        self.assertEqual(len(verify_tasks()['files']), 6)

    def test_real_mps_smoke_and_prior_native_generation_prefix(self):
        root = Path(__file__).resolve().parent / 'results'
        smoke = json.loads((root / 'minions_chunk_smoke/evidence.json').read_text())
        self.assertEqual(smoke['ready']['model']['device'], 'mps')
        self.assertEqual(smoke['exit'], 0)
        self.assertEqual(smoke['True']['logits_relative_l2'], 0.0)
        self.assertEqual(smoke['True']['full_argmax'], smoke['True']['split_argmax'])
        out = smoke['False']
        self.assertEqual(out['first_forward_input_tokens'] + out['prefill_ranges'][-1][1], len(out['input_ids']))
        prior = [json.loads(s) for s in (root / 'minions_real_001/calls.jsonl').read_text().splitlines()][6]['result']
        self.assertEqual(out['input_ids'], prior['input_ids'])
        self.assertEqual(out['generated_ids'], prior['generated_ids'][:8])


if __name__ == '__main__':
    unittest.main()
