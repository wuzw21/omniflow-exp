"""Regression for fixed-target factorial contrasts; random weights are semantic checks only."""
import json
from pathlib import Path
import unittest

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from stage_a_core import assemble, capture
from run_stage_a_factorial import entry_variants, factorial_interaction, select_shard


class FactorialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(87)
        c = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                        num_hidden_layers=4, num_attention_heads=4,
                        num_key_value_heads=2, max_position_embeddings=128)
        c._attn_implementation = 'eager'
        a, b = Qwen2ForCausalLM(c).eval(), Qwen2ForCausalLM(c).eval()
        cls.states = {key: capture(model, prefix, [8, 9, 10], [0, 2])
                      for key, model, prefix in [('As', a, [1, 2]), ('At', a, [3, 4]),
                                                  ('Bs', b, [1, 2]), ('Bt', b, [3, 4])]}

    def test_all_four_grafts_have_target_prefix_and_intended_shared_cache(self):
        target = self.states['Bt']
        for state in self.states.values():
            for mixed, src, tgt in zip(assemble(state, target), state.kv, target.kv):
                for got, source, native in zip(mixed, src, tgt):
                    self.assertTrue(torch.equal(got[:, :, :2], native[:, :, :2]))
                    self.assertTrue(torch.equal(got[:, :, 2:], source[:, :, 2:]))

    def test_entry_oracles_and_additive_interaction_use_correct_cells(self):
        e = entry_variants(self.states, 2)
        for key, state in [('source', 'As'), ('context_oracle', 'At'), ('model_oracle', 'Bs'), ('joint_oracle', 'Bt')]:
            self.assertTrue(torch.equal(e[key], self.states[state].entries[2]))
        expected = e['model_oracle'] + (e['context_oracle'] - e['source'])
        torch.testing.assert_close(e['additive_oracle'], expected)

    def test_embedding_entry_collapses_context_and_additive_conditions(self):
        e = entry_variants(self.states, 0)
        self.assertTrue(torch.equal(e['source'], e['context_oracle']))
        self.assertTrue(torch.equal(e['model_oracle'], e['joint_oracle']))
        self.assertTrue(torch.equal(e['additive_oracle'], e['joint_oracle']))

    def test_interaction_is_signed_and_does_not_drop_native_failures(self):
        scores = {k: {'f1': v} for k, v in zip(('As', 'At', 'Bs', 'Bt'), (1, .5, .5, 0))}
        self.assertEqual(factorial_interaction(scores, 'f1'), 0)
        scores['Bt']['f1'] = .25
        self.assertEqual(factorial_interaction(scores, 'f1'), .25)

    def test_frozen_ids_keep_existing_order_and_no_held_out(self):
        root = Path(__file__).parent
        p = json.loads((root / 'stage_a_factorial_protocol.json').read_text())
        m = json.loads((root / 'results/stage_a_data_manifest.json').read_text())
        self.assertEqual(p['question_ids'], [q['id'] for q in m['questions'] if q['split'] == 'dev'][:8])
        self.assertEqual(p['starts'], list(range(0, 28, 4)))
        self.assertEqual(p['width'], 4)

    def test_full_development_shards_are_disjoint_complete_and_ordered(self):
        root = Path(__file__).parent
        manifest = json.loads((root / 'results/stage_a_data_manifest.json').read_text())
        cases = [{'id': q['id']} for q in manifest['questions'] if q['split'] == 'dev']
        shards = []
        for index in range(2):
            protocol = json.loads((root / f'stage_a_full_p2_shard{index}.json').read_text())
            selected = select_shard(cases, protocol)
            self.assertEqual(selected, cases[index::2])
            self.assertEqual(len(selected), 64)
            shards.append({c['id'] for c in selected})
        self.assertFalse(shards[0] & shards[1])
        self.assertEqual(shards[0] | shards[1], {c['id'] for c in cases})

    def test_shard_selection_rejects_changed_order_and_invalid_index(self):
        cases = [{'id': 'a'}, {'id': 'b'}]
        with self.assertRaises(ValueError):
            select_shard(cases, {'question_ids': ['b', 'a']})
        with self.assertRaises(ValueError):
            select_shard(cases, {'question_ids': ['a', 'b'], 'shard': {'index': 2, 'count': 2}})


if __name__ == '__main__':
    unittest.main()
