"""Regressions over actual completed/aborted evidence, never a success metric."""
import json
from pathlib import Path
import unittest
from analyze_minions_3b_suite import analyze
from probe_minions_collaboration import SOURCE

RESULTS = Path(__file__).resolve().parent / 'results'


class ActualSuiteTests(unittest.TestCase):
    def test_completed_suite_and_preserved_aborted_attempt(self):
        e = json.loads((RESULTS / 'minions_3b_mlx_001/evidence.json').read_text())
        self.assertEqual(e['status'], 'execution_completed')
        self.assertEqual(e['closed_processes'], {'local': 0, 'remote': 0})
        self.assertEqual(e['endpoints']['local']['weight_dtypes'], ['mlx.core.bfloat16'])
        self.assertEqual(e['endpoints']['local']['mlx_lm'], '0.32.2')
        self.assertEqual(e['endpoints']['local']['tokenizer_sha256'], e['endpoints']['remote']['tokenizer_sha256'])
        old = json.loads((RESULTS / 'minions_3b_001/evidence.json').read_text())
        self.assertEqual(old['status'], 'execution_failed')
        self.assertEqual(old['closed_processes']['local'], -15)
        self.assertTrue((RESULTS / 'minions_3b_001/intentional_stop.json').exists())
        self.assertNotIn('minion_result', old['cases']['long_documents'])
        self.assertNotIn('finance', old['cases'])

    def test_full_documents_and_uncontaminated_no_document_controls(self):
        rows = [json.loads(s) for s in (RESULTS / 'minions_3b_mlx_001/calls.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 22)
        for r in rows:
            case, phase = r['phase'].split('/')
            root = SOURCE / f'minions/examples/{case}'
            question = json.loads((root / 'task.json').read_text())['question']
            doc = (root / 'sample.txt').read_text()
            if phase == 'no_document':
                self.assertEqual(r['messages'], [{'role': 'user', 'content': question}])
            if phase == 'direct':
                self.assertEqual(r['messages'], [{'role': 'user', 'content': question+'\n\n'+doc}])
            if phase == 'minion' and r['endpoint'] == 'local':
                self.assertIn(doc, r['messages'][0]['content'])

    def test_actual_prefill_boundaries_and_self_cache_counts(self):
        a = analyze(RESULTS / 'minions_3b_mlx_001')
        self.assertEqual(len(a['calls']), 22)
        for case in ['code', 'long_documents', 'finance']:
            self.assertEqual(a['totals_by_case'][case]['minion/local']['calls'], 1)
            self.assertEqual(a['totals_by_case'][case]['minion/local']['eligible_own_prefix_tokens'], 0)
        self.assertEqual(a['totals_by_case']['long_documents']['minion/local']['input_tokens'], 27832)
        self.assertEqual(a['totals_by_case']['finance']['minion/local']['input_tokens'], 21917)

    def test_copying_and_failed_quality_controls_are_retained(self):
        spans = json.loads((RESULTS / 'minions_3b_mlx_001/copy_spans.json').read_text())
        self.assertEqual(spans[0]['matched_tokens'], 482)
        self.assertEqual(spans[0]['generated_tokens'], 512)
        full = json.loads((RESULTS / 'minions_file_read_001/evidence.json').read_text())
        bounded = json.loads((RESULTS / 'minions_file_read_512/evidence.json').read_text())
        self.assertIn('sort_data', full['result']['final_answer'])
        self.assertNotIn('filter_data', json.dumps(bounded['result']['final_answer']))
        for e in [full, bounded]:
            self.assertEqual(e['process_exit'], 0)
            self.assertEqual(sum(c['local_model_calls'] for c in e['file_operations']), 0)

    def test_policy_and_numerical_differences_are_explicit(self):
        policy = json.loads((RESULTS / 'minions_3b_mlx_001/generation_policy_audit.json').read_text())
        self.assertNotEqual(policy['local_mlx']['repetition_penalty'], policy['torch_workers']['repetition_penalty'])
        e = json.loads((RESULTS / 'minions_file_read_001/evidence.json').read_text())
        check = e['remote_chunk_check']
        self.assertTrue(check['prefix_ranges'])
        self.assertGreater(check['logits_relative_l2'], 0)
        self.assertEqual(check['full_argmax'], check['split_argmax'])


if __name__ == '__main__':
    unittest.main()
