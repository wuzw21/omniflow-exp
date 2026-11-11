"""Physical M4 replay regressions, not synthetic performance/quality claims."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).parent


class PapillonReplayArtifacts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.folder = ROOT / 'results/papillon_replay_001'
        cls.report = json.loads((cls.folder / 'summary.json').read_text())
        cls.rows = {(r['case'], r['mode']): r for r in cls.report['runs']}

    def test_completed_real_qwen_execution_and_frozen_sources(self):
        r = self.report
        self.assertEqual(r['status'], 'completed')
        self.assertEqual(r['worker_exit_code'], 0)
        self.assertEqual(r['hardware']['chip_type'], 'Apple M4')
        self.assertEqual(r['mlx_lm'], '0.32.2')
        self.assertEqual(len(r['runs']), 7)
        model = json.loads((self.folder / 'local.spec.json').read_text())
        self.assertEqual(model['repo'], 'Qwen/Qwen2.5-3B-Instruct')
        self.assertEqual(model['prefill_chunk_tokens'], 512)
        for name, expected in r['scripts_sha256'].items():
            self.assertEqual(hashlib.sha256((self.folder / name).read_bytes()).hexdigest(), expected)

    def test_same_input_controls_and_single_use_prefix_offsets(self):
        rows = [r for r in self.report['runs'] if r['case'] == 'p95' and r['mode'] != 'json']
        self.assertEqual(len({r['input_sha256'] for r in rows}), 1)
        self.assertEqual({r['input_tokens'] for r in rows}, {2596})
        aligned = self.rows['p95', 'aligned']
        self.assertEqual(aligned['prepared_tokens'], 1536)
        self.assertEqual(aligned['prepared_chunk_sizes'], [512, 512, 512])
        self.assertEqual(aligned['offsets'], [1536] * 36)
        self.assertEqual(self.rows['p95', 'unaligned']['offsets'], [1766] * 36)
        self.assertEqual(self.rows['median', 'full_before']['aligned_eligible_tokens'], 0)

    def test_stable_full_replay_matches_real_archived_generation(self):
        for key in [('p95', 'full_before'), ('p95', 'full_after'), ('median', 'full_before')]:
            self.assertTrue(self.rows[key]['matches_archived_generation'])
        self.assertEqual(self.rows['p95', 'full_before']['generated_sha256'],
                         self.rows['p95', 'full_after']['generated_sha256'])

    def test_observed_batch_boundary_failure_and_aligned_recovery(self):
        reference = self.rows['p95', 'full_before']
        aligned = self.rows['p95', 'aligned']
        unaligned = self.rows['p95', 'unaligned']
        self.assertEqual(aligned['output_tokens'], 474)
        self.assertEqual(aligned['generated_sha256'], reference['generated_sha256'])
        self.assertTrue(aligned['matches_archived_generation'])
        self.assertEqual(unaligned['output_tokens'], 288)
        self.assertNotEqual(unaligned['generated_sha256'], reference['generated_sha256'])
        self.assertFalse(unaligned['matches_archived_generation'])

    def test_boundaries_of_evidence_are_retained(self):
        self.assertFalse(self.report['speedup_claim_permitted'])
        self.assertFalse(self.report['quality_evaluated'])
        self.assertEqual(self.report['provenance'][0]['cloud_finish_reason'], 'length')
        self.assertEqual(self.report['provenance'][1]['cloud_finish_reason'], 'stop')
        for case in ['p95', 'median']:
            self.assertNotEqual(self.rows[case, 'json']['input_sha256'],
                                self.rows[case, 'full_before']['input_sha256'])
        for row in self.report['runs']:
            self.assertNotIn('text', row)
            self.assertNotIn('generated_ids', row)

    def test_native_json_format_control_is_not_a_quality_claim(self):
        audit = json.loads((self.folder / 'format_audit.json').read_text())
        source = self.folder / 'audit_papillon_replay.py'
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), audit['audit_source_sha256'])
        for row in audit['rows']:
            if row['mode'] == 'json':
                self.assertEqual(row['json_top_level_keys'], ['finalOutput'])
        for comparison in audit['comparisons']:
            self.assertFalse(comparison['same_parsed_final'])
            self.assertFalse(comparison['quality_evaluated'])
        self.assertEqual(self.rows['median', 'full_before']['output_tokens'], 246)
        self.assertEqual(self.rows['median', 'json']['output_tokens'], 76)
        self.assertTrue(self.rows['median', 'full_before']['begins_with_input_field'])
        self.assertFalse(self.rows['median', 'json']['raw_contains_input_field_markers'])


if __name__ == '__main__':
    unittest.main()
