"""Summarize actual completed measurements; never synthesize absent results."""
import argparse,json,statistics
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('results',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
x=json.loads(a.results.read_text())
if x['status']!='completed':raise RuntimeError('Only completed runs can become performance tables')
rows=[]
for idx in sorted({r['query_index'] for r in x['records'] if not r['warmup']}):
 for enabled in [False,True]:
  group=[r for r in x['records'] if not r['warmup'] and r['query_index']==idx and r['fluxmem']==enabled]
  if not group:continue
  def med(k):return statistics.median(r[k] for r in group)
  def peak(r,stage,k):return max(e['sampled_peak'][k] for e in r['events'] if e['stage']==stage)
  stages={s:{k:statistics.median(peak(r,s,k) for r in group) for k in ['rss_bytes','physical_footprint_bytes','mps_allocated_bytes','mps_driver_bytes']}
          for s in ['generation','prefill_forward','vision']+(['memory_compression'] if enabled else [])}
  tokens=[next(e['kv_seq_length'] for e in r['events'] if e['stage']=='prefill_forward') for r in group]
  rows.append({'query_index':idx,'timestamp':group[0]['timestamp'],'fluxmem':enabled,'n':len(group),
               **{k:med(k) for k in ['vision_seconds','memory_seconds','language_prefill_and_other_seconds','qa_after_visual_memory_seconds','generation_seconds','decode_forward_seconds','first_forward_kv_bytes']},
               'generation_seconds_min':min(r['generation_seconds'] for r in group),'generation_seconds_max':max(r['generation_seconds'] for r in group),
               'generation_seconds_each':[r['generation_seconds'] for r in group],
               'prefill_kv_tokens_each':tokens,'answers':[r['answer'] for r in group],
               'gold':group[0]['gold'],'stage_peak_medians_bytes':stages})
summary={'source':str(a.results.resolve()),'source_status':x['status'],'statistic':'median of three interleaved measured repeats; first shape use retained',
         'hardware':x['hardware'],'config':x['config'],'rows':rows}
if a.output.exists():raise FileExistsError(a.output)
a.output.write_text(json.dumps(summary,indent=2)+'\n')
for r in rows:
 print(r['timestamp'],'on' if r['fluxmem'] else 'off','vision',round(r['vision_seconds'],3),'memory',round(r['memory_seconds'],3),'QA remainder',round(r['qa_after_visual_memory_seconds'],3),'total',round(r['generation_seconds'],3),'MPS GiB',round(r['stage_peak_medians_bytes']['generation']['mps_driver_bytes']/2**30,3),'answers',r['answers'])
