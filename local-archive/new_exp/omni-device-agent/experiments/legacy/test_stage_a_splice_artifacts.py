"""Preserve actual negative results and numerical sensitivity, never mock success."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from analyze_stage_a_splice import summarize

ROOT = Path(__file__).parent / 'results'


class SpliceArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.path = ROOT / 'stage_a_splice_001'
        cls.report = json.loads((cls.path / 'results.json').read_text())

    def test_physical_runs_complete_with_matching_archived_code(self):
        for name in ('stage_a_splice_001', 'stage_a_splice_rounding_001'):
            folder = ROOT / name
            report = json.loads((folder / 'results.json').read_text())
            self.assertEqual(report['status'], 'completed')
            self.assertEqual(json.loads((folder / 'exit.json').read_text())['exit_code'], 0)
            self.assertEqual(report['runtime']['host'], 'baldursgate')
            for file, sha in report['source_sha256'].items():
                self.assertEqual(hashlib.sha256((folder / file).read_bytes()).hexdigest(), sha)

    def test_complete_mask_coverage_and_cross_run_baselines(self):
        summary = summarize(self.report)
        self.assertEqual(summary['question_count'], 8)
        self.assertEqual(summary['unique_cache_masks_scored'], 152)
        self.assertEqual(summary['exact_vs_recomputed_comparisons'], 56)
        self.assertEqual(summary['exact_vs_recomputed_same_sequences'], 55)

    def test_exact_splice_keeps_factual_failure_and_distinguishes_layer_location(self):
        row = next(r for r in self.report['cases'] if r['id'] == '5ab915ad55429919ba4e239c')
        interval = next(i for i in row['intervals'] if i['start'] == 12)
        scores = {k: row['scored_masks'][v] for k, v in interval['variants'].items()}
        self.assertIn('Joe Frazier', scores['source_both']['prediction'])
        self.assertIn('Joe Frazier', scores['target_lower']['prediction'])
        self.assertEqual(scores['target_upper']['prediction'], 'Jimmy Ellis')
        self.assertEqual(scores['target_lower']['target_layers'], 16)
        self.assertEqual(scores['target_upper']['target_layers'], 16)

    def test_same_process_repetition_preserves_both_numerical_cases(self):
        r = json.loads((ROOT / 'stage_a_splice_rounding_001/results.json').read_text())
        rows = {c['id']: c for c in r['cases']}
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(all(c['within_variant_identical'].values()) for c in rows.values()))
        self.assertTrue(rows['5ab915ad55429919ba4e239c']['same_generated_ids'])
        changed = rows['5a89d36e554299515336132a']
        self.assertFalse(changed['same_generated_ids'])
        self.assertEqual(changed['variants']['recomputed'][0]['prediction'], 'Franc Roddam')
        self.assertIn('directed', changed['variants']['exact'][0]['prediction'])

    def test_analyzer_rejects_missing_masks_and_misnamed_regions(self):
        r = copy.deepcopy(self.report)
        del r['cases'][0]['scored_masks'][r['cases'][0]['intervals'][3]['variants']['target_upper']]
        with self.assertRaises(ValueError):
            summarize(r)
        r = copy.deepcopy(self.report)
        r['cases'][0]['intervals'][3]['variants']['target_upper'] = r['cases'][0]['intervals'][3]['variants']['target_lower']
        with self.assertRaises(ValueError):
            summarize(r)

    def test_analyzer_rejects_partial_runs_and_unreproduced_baselines(self):
        r = copy.deepcopy(self.report)
        r['status'] = 'running'
        with self.assertRaises(ValueError):
            summarize(r)
        r = copy.deepcopy(self.report)
        r['cases'][0]['baseline_reproduction']['native_generated_ids_equal'] = False
        with self.assertRaises(ValueError):
            summarize(r)


if __name__ == '__main__':
    unittest.main()
