from pathlib import Path
from urllib.request import urlopen
from urllib.parse import quote
import json, hashlib, csv, io
from remotezip import RemoteZip
root=Path.home()/'.agent-reach/research/fluxmem-streamingbench-sample1'
root.mkdir(parents=True,exist_ok=True)
rev='48872fa707124474ce4c5172ddc58efb8bc88058'
base='https://huggingface.co/datasets/mjuicem/StreamingBench/resolve/'+rev+'/'
raw=urlopen(base+'StreamingBench/Real_Time_Visual_Understanding.csv',timeout=30).read()
assert hashlib.sha256(raw).hexdigest()=='0121417a71e1dcf367222ae97e9512e3e99725d72e60d50760a050816ccd58da', 'Annotation checksum mismatch'
(root/'Real_Time_Visual_Understanding.csv').write_bytes(raw)
rows=list(csv.DictReader(io.StringIO(raw.decode())))[:3]
print('selected',[(x['question_id'],x['time_stamp']) for x in rows],flush=True)
u=base+quote('Real-Time Visual Understanding_1-50.zip')
with RemoteZip(u, timeout=45) as z:
 names=z.namelist(); matches=[x for x in names if x.endswith('sample_1/video.mp4')]
 print('members',len(names),'matches',matches,flush=True)
 if len(matches)!=1: raise RuntimeError('Expected unique first sample video')
 member=matches[0]; info=z.getinfo(member)
 print('video bytes',info.file_size,'compressed',info.compress_size,flush=True)
 target=root/'sample_1.mp4'
 if not target.exists():
  temp=target.with_suffix('.partial')
  with z.open(member) as src, temp.open('wb') as dst:
   while block:=src.read(1024*1024): dst.write(block)
  temp.replace(target)
 sha=hashlib.sha256(target.read_bytes()).hexdigest()
 if sha!='ba1aa1242318beef20e56dd3b19e5cf15503a121fe1a348a6366926fb86a4146':raise RuntimeError('Video checksum mismatch')
 manifest={'dataset':'mjuicem/StreamingBench','revision':rev,'archive':'Real-Time Visual Understanding_1-50.zip','member':member,'video':str(target),'video_sha256':sha,'video_bytes':info.file_size,'annotation_sha256':hashlib.sha256(raw).hexdigest(),'selection':'first three rows in original Real_Time_Visual_Understanding.csv; before model outputs','queries':rows}
 (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 print('READY',target,sha,flush=True)
