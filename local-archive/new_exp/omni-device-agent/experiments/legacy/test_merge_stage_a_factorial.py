"""Shard-merge regression using explicitly split copies of an existing pilot artifact."""
import copy
import json
from pathlib import Path
import unittest
from merge_stage_a_factorial import merge_reports
from analyze_stage_a_factorial import summarize


class MergeTests(unittest.TestCase):
    def setUp(self):
        path = Path(__file__).parent / 'results/stage_a_factorial_001/results.json'
        self.original = json.loads(path.read_text())
        self.shards = [copy.deepcopy(self.original) for _ in range(2)]
        for i, shard in enumerate(self.shards):
            shard['protocol']['shard'] = {'index': i, 'count': 2}
            shard['cases'] = shard['cases'][i::2]

    def test_merge_preserves_every_actual_row_in_frozen_order(self):
        result = merge_reports(self.shards[::-1])
        self.assertEqual(result['cases'], self.original['cases'])
        self.assertEqual(summarize(result, draws=10)['question_count'], 8)

    def test_running_failed_or_missing_shard_cannot_complete(self):
        for status in ('running', 'failed'):
            shards = copy.deepcopy(self.shards)
            shards[0]['status'] = status
            with self.assertRaises(ValueError): merge_reports(shards)
        with self.assertRaises(ValueError): merge_reports(self.shards[:1])

    def test_duplicate_index_and_wrong_question_order_rejected(self):
        with self.assertRaises(ValueError): merge_reports([self.shards[0], self.shards[0]])
        self.shards[1]['cases'].reverse()
        with self.assertRaises(ValueError): merge_reports(self.shards)

    def test_different_code_or_protocol_rejected(self):
        shards = copy.deepcopy(self.shards)
        shards[1]['source_sha256']['stage_a_core.py'] = 'changed'
        with self.assertRaises(ValueError): merge_reports(shards)
        self.shards[1]['protocol']['width'] = 2
        with self.assertRaises(ValueError): merge_reports(self.shards)

    def test_exact_diagnostic_exposes_the_real_fragile_joint_gain(self):
        report = copy.deepcopy(self.original)
        report['protocol']['exact_joint_diagnostic'] = True
        path = Path(__file__).parent / 'results/stage_a_splice_001/results.json'
        exact_rows = {r['id']: r for r in json.loads(path.read_text())['cases']}
        for row in report['cases']:
            exact = exact_rows[row['id']]
            for interval in row['intervals']:
                original = next(i for i in exact['intervals'] if i['start'] == interval['start'])
                interval['diagnostics']['exact_joint_source_outside'] = exact['scored_masks'][original['variants']['source_both']]
        analysis = summarize(report, draws=100)
        i = next(i for i in analysis['intervals'] if i['start'] == 12)
        self.assertGreater(i['joint_minus_additive_f1']['mean'], 0)
        self.assertEqual(i['exact_joint_diagnostic']['exact_minus_additive_f1']['mean'], 0)
        self.assertEqual(i['exact_joint_diagnostic']['same_generated_sequences'], 7)


if __name__ == '__main__':
    unittest.main()
