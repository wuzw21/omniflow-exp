"""Audit the completed 128-question physical-GPU experiment; never synthesize evidence."""
import hashlib
import json
from pathlib import Path
import unittest

from analyze_stage_a_factorial import summarize
from merge_stage_a_factorial import merge_reports

ROOT = Path(__file__).parent
ARTIFACT = ROOT / 'results/stage_a_full_p2_001'


class FullP2ArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.report = json.loads((ARTIFACT / 'merged.json').read_text())
        cls.shards = [json.loads((ARTIFACT / f'shard{i}.json').read_text()) for i in range(2)]

    def test_two_real_shards_completed_and_sources_are_preserved(self):
        for i, shard in enumerate(self.shards):
            self.assertEqual(shard['status'], 'completed')
            self.assertEqual(json.loads((ARTIFACT / f'shard{i}.exit.json').read_text())['exit_code'], 0)
            self.assertEqual(shard['runtime']['host'], 'baldursgate')
            self.assertEqual(shard['runtime']['gpu_names'], ['NVIDIA GeForce RTX 4090'] * 2)
            self.assertEqual(len(shard['cases']), 64)
            for name, digest in shard['source_sha256'].items():
                self.assertEqual(hashlib.sha256((ARTIFACT / name).read_bytes()).hexdigest(), digest)
        merged = merge_reports(self.shards)
        self.assertEqual(merged['cases'], self.report['cases'])
        for path, digest in self.report['shard_input_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / path).read_bytes()).hexdigest(), digest)

    def test_all_128_frozen_dev_inputs_match_the_original_manifest(self):
        manifest = json.loads((ARTIFACT / 'stage_a_data_manifest.json').read_text())
        raw = Path(manifest['cases_file']).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), manifest['cases_sha256'])
        cases = {c['id']: c for line in raw.decode().splitlines()
                 if (c := json.loads(line))['split'] == 'dev' and c['condition'] == 'P2'}
        self.assertEqual(len(cases), 128)
        self.assertEqual(set(cases), {r['id'] for r in self.report['cases']})
        for row in self.report['cases']:
            case = cases[row['id']]
            fields = [case[k] for k in ('prefix_source_ids', 'prefix_target_ids', 'shared_ids', 'suffix_ids')]
            self.assertEqual(hashlib.sha256(json.dumps(fields).encode()).hexdigest(), row['input_sha256'])
            self.assertEqual(row['gold_answer'], case['answer'])

    def test_all_interventions_present_and_no_heldout_claim(self):
        summary = summarize(self.report, draws=10)
        self.assertEqual(summary['question_count'], 128)
        self.assertEqual([i['start'] for i in summary['intervals']], list(range(0, 28, 4)))
        self.assertIsNone(summary['interval_candidate'])
        self.assertFalse(summary['held_out_gate_passed'])
        for row in self.report['cases']:
            for interval in row['intervals']:
                self.assertIn('exact_joint_source_outside', interval['diagnostics'])
                self.assertLessEqual(interval['scores']['joint_oracle']['interval_kv_error_max'], .03)

    def test_layer_zero_identity_and_exact_diagnostic_are_not_conflated(self):
        for row in self.report['cases']:
            scores = row['intervals'][0]['scores']
            self.assertEqual(scores['source'], scores['context_oracle'])
            self.assertEqual(scores['model_oracle'], scores['additive_oracle'])
            self.assertEqual(scores['model_oracle'], scores['joint_oracle'])
        # Preserve a historical sensitive case without requiring identical kernel outputs.
        row = next(r for r in self.report['cases'] if r['id'] == '5a89d36e554299515336132a')
        interval = next(i for i in row['intervals'] if i['start'] == 12)
        for score in (interval['scores']['joint_oracle'], interval['diagnostics']['exact_joint_source_outside']):
            self.assertIn('generated_ids', score)
            self.assertIn('prediction', score)
            self.assertIn('f1', score)

    def test_numerical_disagreements_and_truncation_remain_visible(self):
        # These are observed properties of this frozen artifact, not universal model guarantees.
        dense_disagreements = native_differences = exact_differences = capped = 0
        for row in self.report['cases']:
            dense_disagreements += row['calibration']['argmax_agreement'] < 1
            for interval in row['intervals']:
                diagnostics, scores = interval['diagnostics'], interval['scores']
                native_differences += (diagnostics['joint_oracle_native_outside']['generated_ids']
                                       != row['cache_factorial']['Bt']['generated_ids'])
                exact_differences += (diagnostics['exact_joint_source_outside']['generated_ids']
                                      != scores['joint_oracle']['generated_ids'])
                capped += sum(s['finish_reason'] == 'length' for s in scores.values())
        self.assertEqual((dense_disagreements, native_differences, exact_differences, capped),
                         (4, 13, 21, 3))


if __name__ == '__main__':
    unittest.main()
