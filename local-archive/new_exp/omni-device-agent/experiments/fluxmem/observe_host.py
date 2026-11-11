"""Read-only host observations; started late, not a causal diagnosis."""
import json,subprocess,time,argparse
from datetime import datetime,timezone
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);a=p.parse_args()
f=(a.run/'host_observations.jsonl').open('x')
for i in range(360):
 r=json.loads((a.run/'results.json').read_text())
 row={'utc':datetime.now(timezone.utc).isoformat(),'status':r['status'],'completed_calls':len(r['records']),
      'vm_stat':subprocess.check_output(['vm_stat'],text=True),
      'swap':subprocess.check_output(['sysctl','vm.swapusage'],text=True),
      'thermal':subprocess.check_output(['pmset','-g','therm'],text=True)}
 f.write(json.dumps(row)+'\n');f.flush()
 if r['status']!='running':break
 time.sleep(5)
f.close()
