"""Executable regressions for the data and split contract, no quality claims."""
import copy
import unittest
from transformers import AutoTokenizer
from fetch_stage_a_models import CACHE,MODELS
from stage_a_data import build_case,select_split


class StageADataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer=AutoTokenizer.from_pretrained(CACHE/'models'/MODELS['instruct'][1],local_files_only=True)
        cls.row={'id':'fixture-001','type':'bridge','question':'Which city is linked through the two observations?',
                 'answer':'Example City',
                 'supporting_facts':{'title':['Observation A','Observation B'],'sent_id':[0,0]},
                 'context':{'title':['Distractor','Observation B','Observation A'],
                            'sentences':[['This is unrelated background.'],
                                         ['Observation B names Example City.'],
                                         ['Observation A points to Observation B.']]}}

    def test_split_is_disjoint_stratified_and_input_order_invariant(self):
        rows=[{'id':str(i),'type':'bridge' if i<80 else 'comparison'} for i in range(100)]
        a=select_split(rows,total=40,dev=10)
        b=select_split(list(reversed(rows)),total=40,dev=10)
        self.assertEqual(a,b)
        dev={r['id'] for split,r in a if split=='dev'}
        test={r['id'] for split,r in a if split=='test'}
        self.assertEqual((len(dev),len(test)),(10,30))
        self.assertFalse(dev & test)
        self.assertEqual(sum(r['type']=='comparison' for _,r in a),8)

    def test_duplicate_ids_are_rejected(self):
        with self.assertRaises(ValueError):
            select_split([{'id':'same','type':'bridge'}]*4,total=3,dev=1)

    def test_shared_tokens_and_positions_are_controlled(self):
        cases=[build_case(self.row,self.tokenizer,p) for p in ['P0','P1','P2']]
        for c in cases:
            self.assertEqual(len(c['prefix_source_ids']),len(c['prefix_target_ids']))
        self.assertEqual(cases[0]['prefix_source_ids'],cases[0]['prefix_target_ids'])
        self.assertNotEqual(cases[1]['prefix_source_ids'],cases[1]['prefix_target_ids'])
        self.assertEqual(cases[0]['shared_ids'],cases[1]['shared_ids'])

    def test_p2_moves_one_support_document_to_target_history(self):
        c=build_case(self.row,self.tokenizer,'P2')
        title=c['history_support_title']
        self.assertNotIn(title,c['shared_titles'])
        self.assertIn(title,self.tokenizer.decode(c['prefix_target_ids']))
        self.assertNotIn('Document: '+title,self.tokenizer.decode(c['prefix_source_ids']))
        self.assertEqual(len(set(c['support_titles']) & set(c['shared_titles'])),1)

    def test_gold_answer_does_not_change_model_inputs(self):
        changed=copy.deepcopy(self.row)
        changed['answer']='SENTINEL_DO_NOT_LEAK'
        for p in ['P0','P1','P2']:
            a,b=build_case(self.row,self.tokenizer,p),build_case(changed,self.tokenizer,p)
            for k in ['prefix_source_ids','prefix_target_ids','shared_ids','suffix_ids']:
                self.assertEqual(a[k],b[k])

    def test_budget_never_silently_discards_support_evidence(self):
        with self.assertRaises(ValueError):
            build_case(self.row,self.tokenizer,'P0',shared_budget=1)


if __name__=='__main__':
    unittest.main()
