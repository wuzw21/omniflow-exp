"""Small random Qwen2 verifies intervention semantics, never model quality."""
import unittest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM
from stage_a_core import (additive_entry, assemble, capture, continuation_logits, distribution_metrics,
                          rebuild_interval, relative_error)


class StageACoreTests(unittest.TestCase):
    def test_additive_oracle_same_context_collapses_exactly_in_bfloat16(self):
        source=torch.tensor([1000.0,100.0,1.0],dtype=torch.bfloat16)
        target=torch.tensor([1.0,0.125,0.001],dtype=torch.bfloat16)
        self.assertFalse(torch.equal(target+source-source,target))
        self.assertTrue(torch.equal(additive_entry(source,source.clone(),target),target))

    @classmethod
    def setUpClass(cls):
        torch.manual_seed(31)
        config = Qwen2Config(vocab_size=97, hidden_size=32, intermediate_size=64,
                            num_hidden_layers=4, num_attention_heads=4,
                            num_key_value_heads=2, max_position_embeddings=256,
                            attention_dropout=0.0, use_sliding_window=False)
        config._attn_implementation = "eager"
        cls.model = Qwen2ForCausalLM(config).eval()
        cls.hs, cls.ht, cls.x, cls.q, cls.y = [2,3,4], [5,6,7], [8,9,10,11], [12,13], [14,15]
        cls.s = capture(cls.model, cls.hs, cls.x, range(4))
        cls.t = capture(cls.model, cls.ht, cls.x, range(4))

    def test_entry_zero_has_no_prefix_dependency(self):
        self.assertTrue(torch.equal(self.s.entries[0], self.t.entries[0]))
        with torch.inference_mode():
            embedding=self.model.get_input_embeddings()(torch.tensor([self.x])).cpu()
        self.assertTrue(torch.equal(embedding,self.t.entries[0]))
        self.assertGreater(relative_error(self.s.entries[1], self.t.entries[1]), 0)

    def test_oracle_rebuild_matches_native_for_all_valid_intervals(self):
        for start in range(4):
            for width in range(1,5-start):
                rebuilt = rebuild_interval(self.model, self.t, self.t.entries[start], start, width)
                for i, values in rebuilt.items():
                    for got, full in zip(values, self.t.kv[i]):
                        self.assertLess(relative_error(got, full[:,:,3:]), 2e-6)

    def test_cache_continuation_matches_unsplit_prefill(self):
        split = continuation_logits(self.model, self.t.kv, self.q, self.y)
        ids = torch.tensor([self.ht + self.x + self.q + self.y[:-1]])
        with torch.inference_mode():
            dense = self.model(ids, logits_to_keep=len(self.y)).logits[0].float()
        torch.testing.assert_close(split, dense, rtol=1e-5, atol=1e-7)

    def test_continuation_does_not_mutate_snapshot_on_repeat(self):
        before = tuple((k.clone(),v.clone()) for k,v in self.t.kv)
        a = continuation_logits(self.model, self.t.kv, self.q, self.y)
        b = continuation_logits(self.model, self.t.kv, self.q, self.y)
        self.assertTrue(torch.equal(a,b))
        for old,new in zip(before,self.t.kv):
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(old,new)))

    def test_source_history_control_is_native_source_not_target_prefix_graft(self):
        source_logits=continuation_logits(self.model,self.s.kv,self.q,self.y)
        with torch.inference_mode():
            expected=self.model(torch.tensor([self.hs+self.x+self.q+self.y[:-1]]),
                                logits_to_keep=len(self.y)).logits[0].float()
        torch.testing.assert_close(source_logits,expected,rtol=1e-5,atol=1e-7)
        graft=assemble(self.s,self.t)
        graft_logits=continuation_logits(self.model,graft,self.q,self.y)
        self.assertGreater(relative_error(source_logits,graft_logits),1e-5)
        for source,(k,v) in zip(self.s.kv,graft):
            self.assertFalse(torch.equal(source[0][:,:,:3],k[:,:,:3]))
            self.assertTrue(torch.equal(source[0][:,:,3:],k[:,:,3:]))

    def test_replacement_changes_only_selected_shared_region(self):
        rebuilt = rebuild_interval(self.model, self.t, self.t.entries[2], 2, 1)
        mixed = assemble(self.s,self.t,rebuilt)
        for i, (k,v) in enumerate(mixed):
            for j, tensor in enumerate((k,v)):
                self.assertTrue(torch.equal(tensor[:,:,:3],self.t.kv[i][j][:,:,:3]))
                expected = rebuilt[i][j] if i == 2 else self.s.kv[i][j][:,:,3:]
                self.assertTrue(torch.equal(tensor[:,:,3:],expected))

    def test_all_native_replacements_recover_native_cache(self):
        rep = {i: (k[:,:,3:],v[:,:,3:]) for i,(k,v) in enumerate(self.t.kv)}
        mixed = assemble(self.s,self.t,rep)
        for actual,expected in zip(mixed,self.t.kv):
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(actual,expected)))

    def test_invalid_shapes_and_intervals_rejected(self):
        for start,width in [(-1,1),(3,2),(0,0)]:
            with self.assertRaises(ValueError):
                rebuild_interval(self.model,self.t,self.t.entries[1],start,width)
        with self.assertRaises(ValueError):
            rebuild_interval(self.model,self.t,self.t.entries[1][:,:-1],1,1)
        with self.assertRaises(ValueError):
            distribution_metrics(torch.zeros(2,3),torch.zeros(3,3),[1,2])


if __name__ == "__main__":
    unittest.main()
