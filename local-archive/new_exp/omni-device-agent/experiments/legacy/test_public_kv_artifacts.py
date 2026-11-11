"""Regression checks of saved real-model evidence; does not rerun inference."""
import hashlib
import json
from pathlib import Path
import unittest


class PublicKVEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).parent
        cls.path=cls.root/'results/public_kv_dependency_mps.json'
        cls.result=json.loads(cls.path.read_text())

    def test_completed_run_and_source_provenance(self):
        r=self.result
        self.assertEqual(r['status'],'completed_synthetic_state_dependency_probe')
        self.assertEqual(r['completed_forwards'],64)
        for key,path in [('source_sha256',self.root/'probe_public_kv_dependency.py'),
                         ('capture_sha256',self.root/'stage_a_core.py'),
                         ('protocol_sha256',self.path.with_suffix('.protocol.json'))]:
            self.assertEqual(r[key],hashlib.sha256(path.read_bytes()).hexdigest())

    def test_causal_controls_and_layer_zero_invariance(self):
        for document in self.result['documents'].values():
            self.assertEqual(len(document['public_ids']),128)
            for condition,record in document['conditions'].items():
                self.assertEqual(len(record['records']),8)
                for layer,state in record['distinct_states'].items():
                    expected=8 if condition=='private_before_public' and layer!='0' else 1
                    self.assertEqual(set(state.values()),{expected})

    def test_independent_candidate_challenges_and_payload_scope(self):
        for document in self.result['documents'].values():
            challenges=document['challenge']
            self.assertEqual([x['true_synthetic_label'] for x in challenges],list(reversed([str(i) for i in range(8)])))
            for record in challenges:
                self.assertEqual(len(record['matches_by_layer']['0']),8)
                for layer in range(1,24):
                    self.assertEqual(record['matches_by_layer'][str(layer)],[record['true_synthetic_label']])
            for record in document['conditions']['private_before_public']['records']:
                self.assertEqual(record['layers']['23']['kv_last_public_bytes'],512)


if __name__=='__main__':
    unittest.main()
