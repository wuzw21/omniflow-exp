import unittest
import torch
from stage_a_core import Snapshot
from probe_public_kv_dependency import public_slice, fingerprint, candidate_matches


class PublicExportTests(unittest.TestCase):
    def snapshot(self):
        k=torch.arange(16,dtype=torch.float32).reshape(1,1,8,2)
        return Snapshot({},((k,k+100),),3,5)

    def test_export_excludes_prefix_and_later_private_rows(self):
        snap=self.snapshot()
        k,v=public_slice(snap,2,0)
        self.assertEqual(k.flatten().tolist(),[6,7,8,9])
        self.assertEqual(v.flatten().tolist(),[106,107,108,109])

    def test_last_public_token_not_last_input_token(self):
        k,v=public_slice(self.snapshot(),2,0,last_only=True)
        self.assertEqual(k.flatten().tolist(),[8,9])
        self.assertEqual(v.flatten().tolist(),[108,109])

    def test_export_mutation_does_not_change_source(self):
        snap=self.snapshot(); before=fingerprint(snap.kv[0])
        k,v=public_slice(snap,2,0);k.zero_();v.zero_()
        self.assertEqual(fingerprint(snap.kv[0]),before)

    def test_changed_unexported_rows_do_not_change_packet(self):
        snap=self.snapshot();before=fingerprint(public_slice(snap,2,0))
        for tensor in snap.kv[0]:
            tensor[:,:,:3].fill_(-999)
            tensor[:,:,5:].fill_(-777)
        self.assertEqual(fingerprint(public_slice(snap,2,0)),before)

    def test_ambiguous_candidates_are_not_counted_as_recovery(self):
        self.assertEqual(candidate_matches({'a':'same','b':'same'},'same'),['a','b'])
        self.assertEqual(candidate_matches({'a':'different'},'same'),[])

    def test_invalid_public_span_fails(self):
        for count in (0,-1,6):
            with self.assertRaises(ValueError):public_slice(self.snapshot(),count,0)


if __name__=='__main__':
    unittest.main()
