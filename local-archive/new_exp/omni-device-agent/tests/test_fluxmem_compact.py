"""Executable regression entrypoints plus contracts over physical M4 evidence."""
import hashlib,json,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'experiments/fluxmem_incremental_compact'
E=BASE/'evidence'
def read(name):return json.loads((E/name).read_text())
class CompactTests(unittest.TestCase):
 def test_append_repro_red_on_legacy_green_on_compact(self):
  py=ROOT/'.venv-fluxmem/bin/python'
  if not py.exists() or not (ROOT/'runs/fluxmem_incremental_audit_001/question_41_A.pt').exists():
   self.skipTest('Real-device runtime/features unavailable; evidence is not a substitute for this execution')
  cmd=[str(py),str(BASE/'check_append.py')]
  old=subprocess.run(cmd+['--legacy'],capture_output=True,text=True)
  new=subprocess.run(cmd,capture_output=True,text=True)
  self.assertNotEqual(old.returncode,0);self.assertIn('append concatenates historical state',old.stderr)
  self.assertEqual(new.returncode,0,new.stderr);self.assertIn("'append_cat_calls': 0",new.stdout)
 def test_every_prefix_equal_and_lifecycle(self):
  r=read('validate_results.json');self.assertEqual(r['status'],'completed')
  self.assertEqual(r['chip'],'Apple M4');self.assertEqual(r['upstream_diff'],'')
  self.assertEqual([v['blocks'] for v in r['records']],list(range(1,42)))
  for v in r['records']:
   for k in ['vs_original','vs_C1','repeat_read']:
    self.assertTrue(v[k]['features_bitwise_equal']);self.assertEqual(v[k]['max_abs_error'],0)
  for k in ['reset_replay','snapshot_mutation_isolated']:self.assertTrue(r[k]['features_bitwise_equal'])
 def test_append_has_no_history_cat(self):
  copies=read('validate_results.json')['append_copies']
  self.assertEqual(copies['C1']['calls']['aten.cat.default'],280)
  self.assertEqual(copies['C2']['calls'].get('aten.cat.default',0),0)
  # Incoming normalization/feature copies remain; never claim all copying is zero.
  self.assertGreater(copies['C2']['logical_bytes']['aten.copy_.default'],0)
 def test_compact_state_below_visual_cache_without_short_window_growth(self):
  rows=read('validate_results.json')['records']
  for r in rows:
   self.assertEqual(r['C2_state']['components']['norm'],5*120*2048*2)
  self.assertLess(rows[-1]['C2_state']['tensor_storage_bytes'],41*120*2048*2)
  self.assertLess(rows[-1]['C2_state']['tensor_storage_bytes'],rows[-1]['C1_state']['tensor_storage_bytes']/3)
 def test_original_policy_counts_and_merge_boundary(self):
  rows=read('validate_results.json')['records']
  self.assertEqual(rows[35]['counts']['merge_frames'],0)
  self.assertEqual(rows[36]['counts']['merge_frames'],1)
  self.assertEqual(rows[-1]['counts']['match_calls'],77)
  self.assertEqual(rows[-1]['counts']['prune_frames'],37)
  self.assertEqual(rows[-1]['counts']['merge_frames'],5)
 def test_paired_timings_are_real_fresh_runs_with_balanced_order(self):
  for name,pairs,vision in [('memory_results.json',12,0),('frontend_results.json',6,41)]:
   r=read(name);self.assertEqual(r['status'],'completed');self.assertEqual(r['chip'],'Apple M4')
   self.assertEqual(len([v for v in r['runs'] if v['warmup']]),2)
   runs=[v for v in r['runs'] if not v['warmup']];self.assertEqual(len(runs),2*pairs)
   for i in range(pairs):
    self.assertEqual([v['condition'] for v in runs[2*i:2*i+2]],['B','C2'] if i%2==0 else ['C2','B'])
   for v in runs:
    self.assertEqual(v['vision_calls'],vision)
    self.assertEqual([e['blocks'] for e in v['events']],[2,26,41])
    self.assertTrue(all(e['equivalence']['features_bitwise_equal'] for e in v['events']))
    self.assertEqual(v['counts']['match_calls'],125 if v['condition']=='B' else 77)
    self.assertEqual(v['counts']['prune_frames'],59 if v['condition']=='B' else 37)
    self.assertEqual(v['counts']['merge_frames'],5)
    self.assertGreater(v['total_seconds'],0)
 def test_full_qa_equal_with_compact_snapshot(self):
  r=read('qa_results.json');self.assertEqual(r['status'],'completed');self.assertEqual(len(r['records']),9)
  self.assertEqual(r['audit_sha256'],hashlib.sha256((E/'validate_results.json').read_bytes()).hexdigest())
  for v in r['records']:
   self.assertTrue(v['first_logits_bitwise_equal']);self.assertTrue(v['generated_ids_equal_to_A'])
   self.assertTrue(all(e['bitwise_equal'] for e in v['memory_outputs_vs_A'].values()))
 def test_frozen_code_and_inputs(self):
  for name,folder in [('validate_results.json','validate_code'),('memory_results.json','memory_code'),('frontend_results.json','frontend_code'),('qa_results.json','qa_code')]:
   r=read(name)
   self.assertEqual(r['input_manifest_sha256'],read('validate_results.json')['input_manifest_sha256'])
   for f,sha in r['source_hashes'].items():self.assertEqual(hashlib.sha256((E/folder/f).read_bytes()).hexdigest(),sha)
if __name__=='__main__':unittest.main()
