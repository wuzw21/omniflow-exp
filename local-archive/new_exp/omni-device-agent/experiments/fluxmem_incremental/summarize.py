"""Derive tables from completed real execution; no speedup claim from counts."""
import argparse,json
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--qa',type=Path,required=True);p.add_argument('--replay',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
if a.output.exists():raise FileExistsError(a.output)
x=json.loads(a.audit.read_text());q=json.loads(a.qa.read_text());rep=json.loads(a.replay.read_text())
assert x['status']==q['status']=='completed' and rep['status']=='passed'
s={'status':'completed','schedules':{},'qa_calls':len(q['records']),'qa_exact':all(r['generated_ids_equal_to_A'] and r['first_logits_bitwise_equal'] and all(e['bitwise_equal'] for e in r['memory_outputs_vs_A'].values()) for r in q['records']),'copy_replay':rep['checks'][-1]}
for schedule in ['dense','questions']:
 rows=[r for r in x['records'] if r['schedule']==schedule];last=rows[-1]
 s['schedules'][schedule]={'checkpoints':len(rows),'all_visual_exact':all(all(r['visual_bitwise_equal'].values()) for r in rows),'all_memory_exact':all(all(e['features_bitwise_equal'] for e in r['memory_equality'].values()) for r in rows),'conditions':{}}
 for cond in 'ABC':
  s['schedules'][schedule]['conditions'][cond]={'vision_blocks':last['vision_blocks'][cond],**last['counts'][cond],'persistent_tensor_bytes':last['state'][cond]['tensor_storage_bytes'],'python_object_bytes':last['state'][cond]['python_object_bytes'],'visual_seconds_total':sum(r['elapsed'][cond]['visual_seconds'] for r in rows),'memory_seconds_total':sum(r['elapsed'][cond]['memory_seconds'] for r in rows)}
a.output.write_text(json.dumps(s,indent=2)+'\n');print(json.dumps(s,indent=2))
