"""Real Mac + CUDA artifact regressions; do not infer quality or speed from them."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parent


class PapillonOverlapArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'results/papillon_overlap_002'
        cls.report = json.loads((cls.path / 'summary.json').read_text())

    def test_real_run_and_frozen_sources(self):
        r = self.report
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['process_exits'], {'local': 0, 'remote': 0})
        self.assertEqual(r['baseline_call_count'], 3)
        self.assertEqual(len(r['calls']), 5)
        for name, digest in r['scripts_sha256'].items():
            self.assertEqual(hashlib.sha256((self.path / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(r['selected']['tokens'], 87)
        self.assertEqual(r['selected']['row'], 409)

    def test_same_cloud_input_and_local_output_in_observed_pilot(self):
        for field in ['same_cloud_response', 'same_aggregation_input_ids',
                      'same_aggregation_generated_ids', 'same_parsed_final_output']:
            self.assertTrue(self.report[field])
        calls = self.report['calls']
        self.assertEqual(calls[1]['input_sha256'], calls[3]['input_sha256'])
        self.assertEqual(calls[2]['input_sha256'], calls[4]['input_sha256'])
        self.assertEqual(calls[2]['output_sha256'], calls[4]['output_sha256'])
        self.assertTrue(all(c['finish_reason'] == 'stop' for c in calls))

    def test_native_cache_offsets_and_full_token_prefix_validation(self):
        r = self.report
        self.assertEqual(r['prepared']['input_tokens'], 240)
        self.assertEqual(r['prepared']['all_layer_offsets'], [240] * 36)
        warm = r['calls'][-1]
        self.assertEqual(warm['reused_prefix_tokens'], 240)
        self.assertFalse(warm['prepared_prefix_mismatch'])
        self.assertLess(warm['reused_prefix_tokens'], warm['input_tokens'])

    def test_format_overhead_and_first_use_timing_are_preserved(self):
        a = json.loads((self.path / 'format_audit.json').read_text())
        self.assertEqual(a['parsed_final_tokens'], 67)
        self.assertEqual(a['local_calls'][1]['output_tokens'], 246)
        self.assertTrue(a['local_calls'][1]['begins_with_input_field'])
        # Keep the observed confound visible: these two cloud prefills are not
        # interchangeable controlled measurements of the scheduling mechanism.
        self.assertGreater(self.report['calls'][1]['prefill_seconds'],
                           self.report['calls'][3]['prefill_seconds'])

    def test_long_case_cannot_be_reported_as_output_preserving_speedup(self):
        path = ROOT / 'results/papillon_overlap_003'
        r = json.loads((path / 'summary.json').read_text())
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['process_exits'], {'local': 0, 'remote': 0})
        for name, digest in r['scripts_sha256'].items():
            self.assertEqual(hashlib.sha256((path / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(r['selected']['row'], 75)
        self.assertEqual(r['selected']['tokens'], 1613)
        self.assertTrue(r['same_cloud_response'])
        self.assertTrue(r['same_aggregation_input_ids'])
        self.assertFalse(r['same_aggregation_generated_ids'])
        self.assertFalse(r['same_parsed_final_output'])
        self.assertTrue(r['warm_serial_same_generated_ids'])
        local = [c for c in r['calls'] if c['endpoint'] == 'local']
        self.assertEqual([c['output_tokens'] for c in local], [353, 474, 288, 474])
        remote = [c for c in r['calls'] if c['endpoint'] == 'remote']
        self.assertEqual([c['finish_reason'] for c in remote], ['length'] * 3)
        self.assertEqual(r['prepared']['all_layer_offsets'], [1766] * 36)
        self.assertFalse(json.loads((path / 'trace_accounting.json').read_text())['speedup_claim_permitted'])


if __name__ == '__main__':
    unittest.main()
