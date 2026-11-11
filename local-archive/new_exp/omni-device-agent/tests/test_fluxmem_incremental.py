"""Contracts over real-device execution evidence; model execution scripts remain runnable."""
import hashlib,json,subprocess,sys,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'experiments/fluxmem_incremental';E=BASE/'evidence'
def read(name):return json.loads((E/name).read_text())
class IncrementalEvidenceTests(unittest.TestCase):
 def test_immutable_causal_sampling(self):
  x=read('input_manifest.json')
  self.assertEqual(x['pts_seconds'],list(range(82)))
  self.assertEqual(x['block_pts'],[[i,i+1] for i in range(0,82,2)])
  self.assertEqual([r['frames'] for r in x['queries']],[4,52,82])
  for q in x['queries']:
   t=sum(int(v)*mul for v,mul in zip(q['query']['time_stamp'].split(':'),[3600,60,1]))
   self.assertLess(x['pts_seconds'][q['frames']-1],t)
  self.assertEqual(len(x['block_pixel_sha256']),41)
 def test_all_41_prefixes_and_original_question_schedule(self):
  x=read('audit_results.json');self.assertEqual(x['status'],'completed');self.assertEqual(x['hardware']['chip'],'Apple M4')
  self.assertEqual(x['upstream_diff'],'')
  self.assertEqual([r['blocks'] for r in x['records'] if r['schedule']=='dense'],list(range(1,42)))
  self.assertEqual([r['blocks'] for r in x['records'] if r['schedule']=='questions'],[2,26,41])
  for r in x['records']:
   self.assertTrue(all(r['visual_bitwise_equal'].values()))
   self.assertTrue(r['read_did_not_mutate_execution_state'])
   for v in r['memory_equality'].values():
    self.assertTrue(v['indices_equal']);self.assertTrue(v['features_bitwise_equal']);self.assertEqual(v['max_abs_error'],0)
  # Long-term boundary is exercised, not inferred from short success.
  for n,merges in [(36,0),(37,1),(41,5)]:
   r=next(r for r in x['records'] if r['schedule']=='dense' and r['blocks']==n)
   self.assertEqual(r['counts']['C']['merge_frames'],merges)
 def test_compute_saved_on_real_three_query_schedule(self):
  r=next(r for r in read('audit_results.json')['records'] if r['schedule']=='questions' and r['blocks']==41)
  self.assertEqual(r['vision_blocks'],{'A':69,'B':41,'C':41})
  self.assertEqual(r['counts']['A'],r['counts']['B'])
  for c,matches,prunes,merges in [('A',125,59,5),('B',125,59,5),('C',77,37,5)]:
   self.assertEqual(r['counts'][c]['match_calls'],matches)
   self.assertEqual(r['counts'][c]['prune_frames'],prunes)
   self.assertEqual(r['counts'][c]['merge_frames'],merges)
  self.assertEqual(r['state']['B']['tensor_storage_bytes'],41*120*2048*2)
  self.assertGreater(r['state']['C']['tensor_storage_bytes'],3*r['state']['B']['tensor_storage_bytes'])
 def test_full_model_inputs_logits_and_outputs_match(self):
  x=read('qa_results.json');self.assertEqual(x['status'],'completed');self.assertEqual(len(x['records']),9)
  self.assertEqual(x['parameter_devices'],['mps:0'])
  self.assertTrue(all(not v for v in x['loading_info'].values()))
  for r in x['records']:
   self.assertTrue(r['generated_ids_equal_to_A']);self.assertTrue(r['first_logits_bitwise_equal'])
   self.assertEqual(r['first_logits_max_abs_error'],0)
   for v in r['memory_outputs_vs_A'].values():self.assertTrue(v['bitwise_equal'])
   self.assertEqual(r['actual_vision_calls'],r['blocks'] if r['condition']=='A' else 0)
   if r['condition']=='C':self.assertEqual(sum(r['memory_counts'].values()),0)
 def test_lifecycle_copy_cost_and_fault_sensitivity(self):
  r=read('replay_results.json');self.assertEqual(r['status'],'passed');self.assertEqual(r['device'],'mps')
  self.assertTrue(r['reset_and_replay_passed']);self.assertFalse(r['injected_read_mask_fault_detected']['indices_equal'])
  self.assertTrue(r['first_prune_boundary_vs_original']['features_bitwise_equal'])
  copies=r['checks'][-1]['cat_clone_logical_output_bytes']
  self.assertGreater(copies['aten.cat.default'],r['checks'][-1]['state']['tensor_storage_bytes'])
 def test_failed_meta_loading_is_not_hidden(self):
  r=read('failed_meta_load.json');self.assertEqual(r['status'],'failed');self.assertIn('Cannot copy out of meta tensor',r['error'])
 def test_qa_loader_memory_regression_was_verified_on_device(self):
  old=read('interrupted_qa_memory.json');new=read('memory_cleanup_smoke.json')
  self.assertEqual(old['status'],'interrupted');self.assertEqual(new['status'],'completed')
  self.assertEqual(old['input_manifest_sha256'],new['input_manifest_sha256'])
  self.assertGreater(old['records'][0]['measurement']['before']['physical_footprint_bytes'],14*2**30)
  first=next(e for e in new['allocation_trace'] if e['step']=='before_generation')
  self.assertLess(first['physical_footprint_bytes'],9*2**30)
  self.assertTrue(all(r['first_logits_bitwise_equal'] for r in new['records']))
 def test_evidence_code_snapshots_match_hashes(self):
  for result,folder in [('audit_results.json','audit_code'),('qa_results.json','qa_code')]:
   r=read(result)
   for name,sha in r['source_hashes'].items():self.assertEqual(hashlib.sha256((E/folder/name).read_bytes()).hexdigest(),sha)
 def test_run_entry_refuses_overwrite_before_loading_models(self):
  target=ROOT/'runs/fluxmem_incremental_audit_001'
  if not target.exists():self.skipTest('Raw local run not present')
  before=(target/'results.json').read_bytes()
  p=subprocess.run([sys.executable,str(BASE/'audit.py'),'--inputs','unused','--output',str(target)],capture_output=True,text=True)
  self.assertNotEqual(p.returncode,0);self.assertIn('FileExistsError',p.stderr)
  self.assertEqual(before,(target/'results.json').read_bytes())
if __name__=='__main__':unittest.main()
