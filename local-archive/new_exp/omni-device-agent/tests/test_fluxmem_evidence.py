"""Regression contracts for the REAL M4 run; no mocked model acceptance."""
import hashlib,json,subprocess,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
E=ROOT/'experiments/fluxmem/evidence'
def read(p):return json.loads((E/p).read_text())
class FluxMemEvidenceTests(unittest.TestCase):
 def test_complete_public_weights_verified(self):
  x=read('weights_verified.json')
  self.assertEqual(x['revision'],'66285546d2b821cf421d4f5eb2576359d3770cd3')
  shards=[r for r in x['files'] if r['path'].endswith('.safetensors')]
  self.assertEqual(len(shards),2)
  self.assertTrue(all(r['lfs_match'] and len(r['sha256'])==64 for r in shards))
 def test_actual_failed_backends_remain_failures(self):
  for name,version,error in [('fluxmem_smoke_001','2.6.0','Dimension out of range'),('fluxmem_smoke_002','2.9.1','not supported for torch.int64')]:
   x=read(name+'/results.json')
   self.assertEqual(x['status'],'failed');self.assertEqual(x['torch'],version)
   self.assertIn(error,x['error'])
  # Short success cannot hide failure in the long-term path.
  x=read('fluxmem_smoke_002/results.json')
  self.assertEqual(len(x['records']),1)
  self.assertEqual(x['records'][0]['query_index'],0)
 def test_full_model_and_long_term_path_ran_on_physical_m4(self):
  x=read('fluxmem_smoke_003/results.json')
  self.assertEqual(x['status'],'completed');self.assertEqual(x['hardware']['chip'],'Apple M4')
  self.assertEqual(x['parameter_devices'],['mps:0'])
  self.assertEqual(x['parameter_dtypes'],['torch.bfloat16'])
  self.assertTrue(all(not v for v in x['loading_info'].values()))
  self.assertEqual(x['upstream_diff'],'')
  self.assertEqual([r['query_index'] for r in x['records']],[0,2])
  r=x['records'][-1];event=next(e for e in r['events'] if e['stage']=='memory_compression')
  self.assertLess(event['output_tokens'],event['input_tokens'])
  self.assertEqual(event['input_tokens'],5056)
  self.assertGreater(r['generated_tokens'],0)
  self.assertFalse(r['hit_length_limit'])
  # 82 sampled frames / temporal stride 2 exceed short 4 + mid 32 grids.
  inp=read('prepared.json')['records'][2]
  self.assertGreater(inp['video_grid_thw'][0][0],(8+64)//2)
 def test_reader_pixels_and_fps_match_upstream_on_real_video(self):
  r=read('reader_equivalence.json')
  self.assertTrue(r['equal_pixels']);self.assertEqual(r['original_fps'],r['bounded_fps'])
  self.assertEqual(r['original_shape'],[4,3,1080,1920])
 def test_inputs_are_first_three_rows_and_no_future_frames(self):
  x=read('prepared.json');self.assertEqual(x['status'],'completed')
  self.assertEqual([r['cutoff_seconds'] for r in x['records']],[4,53,82])
  for r in x['records']:
   self.assertLessEqual(max(r['sampling_trace']['selected_pts_seconds']),r['cutoff_seconds'])
   self.assertEqual(r['query'],x['dataset']['queries'][r['index']])
 def test_retained_successful_smoke_scripts_are_exact(self):
  x=read('fluxmem_smoke_003/results.json')
  for name,sha in x['source_hashes'].items():
   self.assertEqual(hashlib.sha256((E/'fluxmem_smoke_003'/name).read_bytes()).hexdigest(),sha)
 def test_repeated_shell_entry_cannot_overwrite_evidence(self):
  # run_local checks mkdir before opening logs, using the actual existing run.
  target=ROOT/'runs/fluxmem_smoke_003/results.json'
  if not target.exists():self.skipTest('Local raw run not restored')
  before=target.read_bytes()
  r=subprocess.run(['bash','experiments/fluxmem/run_local.sh','profile','fluxmem_smoke_003'],cwd=ROOT,capture_output=True,text=True)
  self.assertNotEqual(r.returncode,0);self.assertEqual(before,target.read_bytes())
 def test_complete_interleaved_profile_and_accounting(self):
  x=read('fluxmem_profile_002/results.json')
  self.assertEqual(x['status'],'completed')
  self.assertEqual(x['remaining_active_stages'],[])
  self.assertEqual(len(x['records']),20)
  measured=[r for r in x['records'] if not r['warmup']]
  self.assertEqual(len(measured),18)
  self.assertEqual({(r['repeat'],r['query_index'],r['fluxmem']) for r in measured},
                   {(i,j,f) for i in range(3) for j in range(3) for f in [False,True]})
  self.assertEqual(x['prepared_manifest_sha256'],hashlib.sha256((E/'prepared.json').read_bytes()).hexdigest())
  for r in measured:
   self.assertEqual(r['generated_tokens'],2)
   self.assertEqual(r['answer'],r['gold'])
   self.assertFalse(r['hit_length_limit'])
   self.assertAlmostEqual(r['generation_seconds'],r['vision_seconds']+r['memory_seconds']+r['qa_after_visual_memory_seconds'],places=6)
   self.assertGreaterEqual(r['language_prefill_and_other_seconds'],0)
   events=r['events'];mem=[e for e in events if e['stage']=='memory_compression']
   self.assertEqual(len(mem),int(r['fluxmem']))
   pre=next(e for e in events if e['stage']=='prefill_forward')
   self.assertEqual(pre['kv_seq_length'],mem[0]['output_tokens'] if mem else pre['input_tokens'])
   for e in events:
    self.assertGreater(e['seconds'],0)
    self.assertGreaterEqual(e['sampled_peak']['physical_footprint_bytes'],e['before']['physical_footprint_bytes'])
  for name,sha in x['source_hashes'].items():
   self.assertEqual(hashlib.sha256((E/'fluxmem_profile_002'/name).read_bytes()).hexdigest(),sha)
 def test_actual_fixed_backend_ops_and_preprocessing_equivalence(self):
  x=read('backend_ops.json');self.assertEqual(x['status'],'passed');self.assertEqual(x['device'],'mps')
  self.assertEqual(x['scatter_reduce_int64_output'],[0,0,2,3])
  x=read('preprocessing_runtime_equivalence.json');self.assertEqual(x['status'],'passed')
  self.assertEqual(len(x['checks']),3)
  for r in x['checks']:
   self.assertTrue(r['sampling_trace_equal']);self.assertTrue(all(r['fields_equal'].values()))
 def test_summary_keeps_slow_measurements(self):
  x=read('fluxmem_profile_002/summary.json')
  self.assertEqual(x['source_status'],'completed')
  self.assertEqual(len(x['rows']),6)
  for r in x['rows']:
   self.assertEqual(r['n'],3)
   self.assertEqual(r['generation_seconds_max'],max(r['generation_seconds_each']))
  short=next(r for r in x['rows'] if r['query_index']==0 and not r['fluxmem'])
  self.assertGreater(short['generation_seconds_max'],40)
if __name__=='__main__':unittest.main()
