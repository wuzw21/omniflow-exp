"""Real-result regression; requires the completed 9207 run, never fabricates it."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from analyze_stage_a_factorial import summarize

ROOT = Path(__file__).parent
ARTIFACT = ROOT / 'results/stage_a_factorial_001'


class FactorialArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ARTIFACT / 'results.json').read_text())

    def test_real_run_finished_and_archived_sources_match(self):
        self.assertEqual(self.report['status'], 'completed')
        self.assertEqual(json.loads((ARTIFACT / 'exit.json').read_text())['exit_code'], 0)
        for name, digest in self.report['source_sha256'].items():
            self.assertEqual(hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest(), digest)
        self.assertEqual(self.report['runtime']['host'], 'baldursgate')
        self.assertEqual(self.report['runtime']['gpu_names'], ['NVIDIA GeForce RTX 4090'] * 2)

    def test_all_frozen_inputs_and_seven_intervals_are_preserved(self):
        manifest = json.loads((ARTIFACT / 'stage_a_data_manifest.json').read_text())
        cases = {c['id']: c for line in Path(manifest['cases_file']).read_text().splitlines()
                 if (c := json.loads(line))['split'] == 'dev' and c['condition'] == 'P2'}
        for row in self.report['cases']:
            case = cases[row['id']]
            fields = [case[k] for k in ('prefix_source_ids', 'prefix_target_ids', 'shared_ids', 'suffix_ids')]
            self.assertEqual(hashlib.sha256(json.dumps(fields).encode()).hexdigest(), row['input_sha256'])
            self.assertEqual(row['gold_answer'], case['answer'])
        summary = summarize(self.report, draws=100)
        self.assertEqual(summary['question_count'], 8)
        self.assertEqual([i['start'] for i in summary['intervals']], list(range(0, 28, 4)))
        self.assertIsNone(summary['interval_candidate'])
        self.assertFalse(summary['held_out_gate_passed'])

    def test_rejects_partial_execution_and_missing_factorial_cell(self):
        r = copy.deepcopy(self.report)
        r['cases'][0]['status'] = 'running'
        with self.assertRaises(ValueError):
            summarize(r, draws=10)
        r = copy.deepcopy(self.report)
        del r['cases'][0]['cache_factorial']['At']
        with self.assertRaises(ValueError):
            summarize(r, draws=10)

    def test_rejects_skipped_questions_and_held_out_rows(self):
        r = copy.deepcopy(self.report)
        r['cases'].pop()
        with self.assertRaises(ValueError):
            summarize(r, draws=10)
        r = copy.deepcopy(self.report)
        r['cases'][0]['split'] = 'test'
        with self.assertRaises(ValueError):
            summarize(r, draws=10)

    def test_rejects_missing_intervals_and_invalid_token_counts(self):
        r = copy.deepcopy(self.report)
        r['cases'][0]['intervals'].pop()
        with self.assertRaises(ValueError):
            summarize(r, draws=10)
        r = copy.deepcopy(self.report)
        r['cases'][0]['cache_factorial']['As']['tokens'] += 1
        with self.assertRaises(ValueError):
            summarize(r, draws=10)

    def test_layer_zero_real_identities_and_native_outside_calibration(self):
        for row in self.report['cases']:
            scores = row['intervals'][0]['scores']
            self.assertEqual(scores['source'], scores['context_oracle'])
            self.assertEqual(scores['model_oracle'], scores['additive_oracle'])
            self.assertEqual(scores['model_oracle'], scores['joint_oracle'])
            for interval in row['intervals']:
                self.assertLessEqual(interval['scores']['joint_oracle']['interval_kv_error_max'], .03)
                # Record actual drift/sequence changes; never assume equality here.
                self.assertIn('generated_ids', interval['diagnostics']['joint_oracle_native_outside'])

    def test_preserves_harmful_oracle_and_native_failure_counterexamples(self):
        rows = {r['id']: r for r in self.report['cases']}
        boxing = rows['5ab915ad55429919ba4e239c']
        interval = next(i for i in boxing['intervals'] if i['start'] == 12)
        self.assertIn('Jimmy Ellis', interval['scores']['source']['prediction'])
        self.assertIn('Joe Frazier', interval['scores']['joint_oracle']['prediction'])
        self.assertLess(interval['scores']['joint_oracle']['f1'], interval['scores']['source']['f1'])
        self.assertEqual(rows['5ac1a6a85542994ab5c67db6']['cache_factorial']['Bt']['f1'], 0)
        summary = summarize(self.report, draws=100)
        late = next(i for i in summary['intervals'] if i['start'] == 16)
        self.assertLess(late['joint_minus_additive_f1']['mean'], 0)


if __name__ == '__main__':
    unittest.main()
