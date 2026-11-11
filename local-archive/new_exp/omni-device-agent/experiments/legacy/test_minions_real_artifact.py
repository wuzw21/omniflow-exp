"""Saved real-run regressions; passing does not mean the task succeeded."""
import json
from pathlib import Path
import unittest
from analyze_minions_collaboration import analyze
from probe_minions_collaboration import load_upstream, SOURCE

RUN = Path(__file__).resolve().parent / 'results/minions_real_001'


class RealTraceTests(unittest.TestCase):
    def test_failure_is_preserved_and_models_closed(self):
        e = json.loads((RUN / 'evidence.json').read_text())
        self.assertEqual(e['status'], 'protocol_failed')
        self.assertNotIn('minion_result', e)
        self.assertIn('JSONDecodeError', e['protocol_error'])
        self.assertEqual(e['closed_processes'], {'local': 0, 'remote': 0})
        self.assertEqual(e['endpoints']['local']['model']['device'], 'mps')
        self.assertEqual(e['endpoints']['remote']['model']['device'], 'cuda')

    def test_private_context_boundary_and_real_token_accounting(self):
        a = analyze(RUN)
        self.assertEqual(len(a['calls']), 8)
        for c in a['calls'][:6]:
            self.assertEqual(c['contains_entire_file'], c['endpoint'] == 'local')
        self.assertEqual(a['totals']['minion/remote']['eligible_own_prefix_tokens'], 2567)
        self.assertEqual(a['totals']['minion/local']['eligible_own_prefix_tokens'], 998)
        self.assertEqual(a['calls'][6]['finish_reason'], 'length')
        self.assertEqual(a['previous_output_copied_into_last_input_message'][0]['source_call'], 2)

    def test_original_parser_reproduces_observed_failure(self):
        Minion, _, _ = load_upstream()
        rows = [json.loads(s) for s in (RUN / 'calls.jsonl').read_text().splitlines()]
        with self.assertRaises(json.JSONDecodeError):
            Minion.__call__.__globals__['_extract_json'](rows[5]['result']['text'])

    def test_reference_answer_never_supplied_as_a_label(self):
        fixture = json.loads((SOURCE / 'minions/examples/code/task.json').read_text())
        rows = [json.loads(s) for s in (RUN / 'calls.jsonl').read_text().splitlines()]
        self.assertIn(fixture['question'], rows[0]['messages'][0]['content'])
        self.assertNotIn('filter_data', rows[0]['messages'][0]['content'])
        # Names occur naturally in code; check the complete independent reference prompt.
        expected = fixture['question'] + '\n\n' + (SOURCE / 'minions/examples/code/sample.txt').read_text()
        for row in rows[6:]:
            self.assertEqual(row['messages'], [{'role': 'user', 'content': expected}])


if __name__ == '__main__':
    unittest.main()
