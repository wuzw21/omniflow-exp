"""Conversation regression: causal decisions, bounded memory, no free evidence."""
import unittest
import json
import hashlib
from pathlib import Path
import numpy as np
from stream_gate.policies import Gate,replay,evidence_events,score

class StreamGateTests(unittest.TestCase):
    def test_prefix_invariant(self):
        a=[np.full((2,2),i/10) for i in range(8)]
        for k,p in [('all',0),('uniform',2),('last',.15),('memory',.15)]:
            self.assertEqual(replay(a,k,p)[1][:4],replay(a[:4],k,p)[1])
    def test_bounded_and_no_reuse_after_eviction(self):
        gate=Gate('memory',.01,2)
        for i,v in enumerate((0,.5,1)): gate.push(i,np.array([v]))
        self.assertTrue(gate.push(3,np.array([0]))[0])
        self.assertEqual(list(gate.memory),[2,3])
    def test_repeated_frame_does_not_encode(self):
        x=np.zeros((2,2)); gate=Gate('memory',.01)
        self.assertTrue(gate.push(0,x)[0]); self.assertFalse(gate.push(1,x)[0])
    def test_missing_frame_cannot_create_evidence(self):
        events=evidence_events([{'firstshade'},{'other'},{'other'}])
        self.assertEqual(score(events,[[0],[1],[1,2]])['hits'],0)
        self.assertEqual(score(events,[[0],[0,1],[0,1,2]])['hits'],1)
    def test_no_query_after_stream_end(self):
        self.assertEqual(evidence_events([{'persistent'},{'persistent','lastonly'}]),[])

class RecordedStreamGateTests(unittest.TestCase):
    root=Path(__file__).parent/'stream_gate'
    def setUp(self):
        if not (self.root/'analysis.json').exists():
            self.skipTest('public replay artifacts not generated')
        self.analysis=json.loads((self.root/'analysis.json').read_text())
    def test_public_frames_integrity(self):
        manifest=json.loads((self.root/'manifest.json').read_text())
        self.assertEqual(manifest['revision'],'ede2a362c99ddba0d03f439c47a0b8fea6abae22')
        for r in manifest['records']:
            for f in r['frames']:
                self.assertEqual(hashlib.sha256(Path(f['path']).read_bytes()).hexdigest(),f['sha256'])
    def test_reference_all_preserves_next_step_evidence(self):
        for r in self.analysis['records']:
            if r['kind']=='all': self.assertEqual(r['hits'],r['events'])
    def test_real_timer_change_is_not_in_skipped_memory(self):
        r=next(r for r in self.analysis['records'] if r['id']=='clocktimerentry-0-0a8e4277' and r['kind']=='memory' and r['param']==.01)
        self.assertNotIn(6,r['states'][7])
        self.assertTrue(any(e['support']==[6] for e in r['missed']))
    def test_live_timing_uses_identical_causal_selection(self):
        p=self.root/'timing.json'
        if not p.exists(): self.skipTest('native encoder timing not run')
        result=json.loads(p.read_text())
        self.assertEqual(result['status'],'completed')
        self.assertEqual(len(result['records']),72)
        for row in result['records']:
            a=next(r for r in self.analysis['records'] if r['id']==row['id'] and r['kind']==row['kind'] and r['param']==row['param'])
            self.assertEqual(row['selected'],a['selected'])
            self.assertGreater(row['encode_seconds'],0)
            self.assertGreaterEqual(row['total_seconds'],row['encode_seconds'])
    def test_recorded_qa_exposes_frozen_baseline_evidence_loss(self):
        p=self.root/'case_qa.json'
        if not p.exists(): self.skipTest('full Qwen failure check not run')
        result=json.loads(p.read_text())
        self.assertEqual(result['status'],'completed')
        records={r['kind']:r for r in result['records']}
        self.assertEqual(records['all']['answer'],result['human_verified_answer'])
        self.assertNotEqual(records['memory']['answer'],result['human_verified_answer'])
        for kind,row in records.items():
            a=next(r for r in self.analysis['records'] if r['id']==result['id'] and r['kind']==kind and r['param']==row['param'])
            self.assertEqual(row['retained_steps'],a['states'][result['query_step']])

if __name__=='__main__': unittest.main()
