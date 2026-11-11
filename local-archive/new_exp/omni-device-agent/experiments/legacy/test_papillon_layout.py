"""Regressions for the real offline dataset/formatter audit, not inference tests."""
import hashlib
import csv
import json
from pathlib import Path
import unittest

from audit_papillon_prompt_layout import query_span

ROOT = Path(__file__).parent


class PapillonLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = json.loads((ROOT / 'results/papillon_layout_002.json').read_text())

    def test_real_artifact_covers_both_complete_public_files(self):
        d = self.result['datasets']
        self.assertEqual(set(d), {'PUPA_New.csv', 'PUPA_TNB.csv'})
        self.assertEqual([d[k]['stats']['query_tokens']['count'] for k in d], [664, 237])
        self.assertEqual(sum(x['query_length_counts']['2048'] for x in d.values()), 24)
        for name, digest in self.result['scripts_sha256'].items():
            self.assertEqual(hashlib.sha256((ROOT / name).read_bytes()).hexdigest(), digest)

    def test_shared_text_is_outside_exact_prefix_for_all_observed_rows(self):
        for dataset in self.result['datasets'].values():
            s = dataset['stats']
            self.assertEqual(s['cross_operator_prefix_tokens']['min'], 25)
            self.assertEqual(s['cross_operator_prefix_tokens']['max'], 25)
            self.assertEqual(s['creation_query_start']['min'], 203)
            self.assertEqual(s['aggregation_query_start']['min'], 169)
            self.assertGreater(s['repeated_query_interior_tokens']['min'], 0)
        # Only aggregate statistics are preserved here, never raw requests or model outputs.
        self.assertNotIn('predictions', self.result)
        self.assertNotIn('rows', self.result)

    def test_transformers_5_batchencoding_default_does_not_break_explicit_token_path(self):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.result['tokenizer'], local_files_only=True)
        query = 'Synthetic regression request: explain sorting.'
        messages = [{'role': 'system', 'content': 'Read the user request.'},
                    {'role': 'user', 'content': query}]
        ids, start, interior = query_span(tokenizer, messages, query)
        expected = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                 return_dict=False)
        self.assertEqual(ids, expected)
        self.assertEqual(ids[start:start + len(interior)], interior)
        self.assertGreater(len(interior), 0)

    def test_public_rows_are_not_misreported_as_independent_requests(self):
        data = Path.home() / '.agent-reach/research/papillon-594e7eb/pupa'
        files = {x['name']: x for x in self.result['dataset']['files']}
        groups = []
        for name in ('PUPA_New.csv', 'PUPA_TNB.csv'):
            path = data / name
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), files[name]['sha256'])
            with path.open() as stream:
                groups.append(list(csv.DictReader(stream)))
        a, b = groups
        self.assertEqual((len({r['user_query'] for r in a}), len({r['user_query'] for r in b})),
                         (462, 222))
        self.assertEqual(len({r['user_query'] for r in a} & {r['user_query'] for r in b}), 7)


if __name__ == '__main__':
    unittest.main()
