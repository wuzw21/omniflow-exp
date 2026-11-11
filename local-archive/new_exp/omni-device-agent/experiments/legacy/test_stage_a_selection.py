"""Selection bounds must not turn per-question gold choices into an actual policy."""
import unittest
from analyze_stage_a_selection import selection_headroom


def fixture(values):
    return {'status': 'completed', 'protocol': {'starts': [0, 4]}, 'cases': [
        {'id': str(n), 'status': 'completed', 'intervals': [
            {'start': start, 'scores': {'source': {'f1': value}}}
            for start, value in zip([0, 4], row)]} for n, row in enumerate(values)]}


class SelectionTests(unittest.TestCase):
    def test_opposite_optima_require_unavailable_gold_choices(self):
        r = selection_headroom(fixture([[1, 0], [0, 1]]), 'source')
        self.assertEqual(r['development_static_mean_f1'], .5)
        self.assertEqual(r['per_question_gold_oracle_mean_f1'], 1)
        self.assertEqual(r['oracle_minus_development_static'], .5)
        self.assertEqual(r['development_static_start'], 0)
        self.assertIn('unavailable answer labels', r['interpretation'])

    def test_dominant_static_interval_has_no_adaptive_headroom(self):
        r = selection_headroom(fixture([[.3, .5], [.8, 1]]), 'source')
        self.assertEqual(r['oracle_minus_development_static'], 0)
        self.assertEqual(r['questions_with_strict_improvement_over_static'], 0)

    def test_native_failures_are_not_filtered(self):
        r = selection_headroom(fixture([[0, 0], [1, 1]]), 'source')
        self.assertEqual(r['question_count'], 2)
        self.assertEqual(r['development_static_mean_f1'], .5)

    def test_incomplete_execution_and_intervals_rejected(self):
        r = fixture([[1, 0]])
        r['status'] = 'running'
        with self.assertRaises(ValueError): selection_headroom(r, 'source')
        r['status'] = 'completed'
        r['cases'][0]['intervals'].pop()
        with self.assertRaises(ValueError): selection_headroom(r, 'source')


if __name__ == '__main__':
    unittest.main()
