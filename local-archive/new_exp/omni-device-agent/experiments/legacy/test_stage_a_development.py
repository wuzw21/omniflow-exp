"""Regression: development screening never includes held-out question IDs."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from run_stage_a_development import load_cases


class DevelopmentDataGateTests(unittest.TestCase):
    def fixture(self,folder):
        rows=[{'id':qid,'split':split,'condition':p} for qid,split in [('d0','dev'),('d1','dev'),('t0','test')]
              for p in ['P0','P1','P2']]
        cases=folder/'cases.jsonl'
        cases.write_text('\n'.join(json.dumps(r) for r in rows)+'\n')
        manifest={'split_counts':{'dev':2,'test':1},'questions':[{'id':qid,'split':split} for qid,split in [('d0','dev'),('d1','dev'),('t0','test')]],
                  'cases_sha256':hashlib.sha256(cases.read_bytes()).hexdigest()}
        path=folder/'manifest.json'
        path.write_text(json.dumps(manifest))
        return path,cases

    def test_only_requested_development_questions_and_conditions(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,c=self.fixture(Path(tmp))
            _,cases=load_cases(m,c,1,['P2'])
            self.assertEqual(cases,[{'id':'d0','split':'dev','condition':'P2'}])

    def test_cannot_expand_limit_into_test_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,c=self.fixture(Path(tmp))
            for limit in [0,3]:
                with self.assertRaises(ValueError):load_cases(m,c,limit,['P2'])

    def test_mutated_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,c=self.fixture(Path(tmp))
            c.write_text(c.read_text()+'\n')
            with self.assertRaises(ValueError):load_cases(m,c,1,['P2'])

    def test_missing_condition_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            m,c=self.fixture(Path(tmp))
            with self.assertRaises(ValueError):load_cases(m,c,1,['P3'])


if __name__=='__main__':unittest.main()
