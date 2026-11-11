"""One-shot dependency-gated experiment job for the authorized 9207 pilot.

No installation/restarts, no recurring scheduler, no full development/test run.
Wait for the already-running preparation, check the runtime and idle GPUs, then
run the existing two-question pilot once. Any failure terminates this job.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


def main(args):
    root=Path(args.root).resolve()
    results=root/'results'
    state_path=results/'queued_pilot_state.json'
    if state_path.exists():
        raise FileExistsError('A queued pilot record already exists; inspect it before any new job')
    expected_hashes=json.loads(Path(args.source_hashes).read_text())
    model_manifest=json.loads((root/'stage_a_7b_models.json').read_text())['models']
    output=results/'stage_a_7b_pilot.json'
    state={'pid':os.getpid(),'root':str(root),'started_unix':time.time(),
           'purpose':'One-shot 2-question development pilot; no held-out execution',
           'source_hashes':expected_hashes,'output':str(output),'children':[]}
    def save(phase,**extra):
        state.update(phase=phase,observed_unix=time.time(),**extra)
        tmp=state_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(state,indent=2)+'\n')
        tmp.replace(state_path)
        print(json.dumps({'phase':phase,**extra}),flush=True)
    env=os.environ.copy()
    env.update(PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH='',
               TMPDIR=str(root/'tmp'),XDG_CACHE_HOME=str(root/'cache'),
               CUDA_CACHE_PATH=str(root/'cache/cuda'),TORCH_HOME=str(root/'cache/torch'),
               TRITON_CACHE_DIR=str(root/'cache/triton'),HF_HUB_OFFLINE='1')
    python=str(root/'runtime/bin/python3.13')
    def run(command,log_name,timeout):
        log_path=results/log_name
        with log_path.open('x') as stream:
            child=subprocess.Popen(command,cwd=root,env=env,stdout=stream,stderr=subprocess.STDOUT)
            state['children'].append({'pid':child.pid,'command':command,'log':str(log_path)})
            save(state['phase'],active_child_pid=child.pid)
            try:
                code=child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                child.terminate()
                state['children'][-1]['timeout']=True
                raise RuntimeError(f'Child exceeded {timeout}s; termination requested; inspect PID {child.pid} before retry')
            state['children'][-1]['exit_code']=code
            if code:
                raise RuntimeError(f'Child failed with exit {code}; see {log_path}')
    try:
        save('waiting_for_existing_preparation')
        deadline=time.monotonic()+args.wait_seconds
        while True:
            installer=Path('/proc')/str(args.installer_pid)
            try:
                installing=bool((installer/'cmdline').read_bytes())
            except FileNotFoundError:
                installing=False
            verified=[]
            for role,model in model_manifest.items():
                path=root/'models'/model['revision']/'verified_manifest.json'
                if path.exists() and json.loads(path.read_text())==model:
                    verified.append(role)
            save('waiting_for_existing_preparation',installer_present=installing,verified_models=verified)
            if not installing and len(verified)==len(model_manifest):
                break
            if time.monotonic()>deadline:
                raise TimeoutError('Preparation wait expired; original preparation jobs were not stopped or restarted')
            time.sleep(60)
        if output.exists():
            raise FileExistsError('Pilot output already exists; refusing duplicate execution')
        for name,expected in expected_hashes.items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Queued source changed: {name}; refusing a different experiment')
        gpu=subprocess.run(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu',
                            '--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=30)
        availability={int(fields[0]):(int(fields[1]),int(fields[2]))
                      for line in gpu.stdout.strip().splitlines() if (fields:=line.split(','))}
        if any(i not in availability or availability[i][0]<20000 or availability[i][1]>=10 for i in (0,1)):
            raise RuntimeError(f'GPUs 0/1 are not sufficiently idle/free: {availability}; no pilot launched')
        save('checking_cuda',gpu_snapshot=availability)
        check="""import json,sys
from pathlib import Path
import torch,transformers
root=Path.cwd()
assert Path(sys.prefix).resolve()==root/'runtime'
assert torch.__version__.split('+')[0]=='2.9.1'
assert transformers.__version__=='4.57.6'
assert torch.cuda.is_available() and torch.cuda.device_count()>=2
torch.manual_seed(20261024)
gpus=[]
for i in (0,1):
    with torch.cuda.device(i):
        assert torch.cuda.is_bf16_supported()
    a=torch.randn(256,256,device=f'cuda:{i}',dtype=torch.bfloat16)
    b=a@a.T
    torch.cuda.synchronize(i)
    assert torch.isfinite(b).all().item()
    gpus.append({'index':i,'name':torch.cuda.get_device_name(i),'finite_bf16_matmul':True})
report={'status':'passed','scope':'CUDA initialization/BF16 matmul only, not model calibration',
        'python':sys.version,'torch':torch.__version__,'transformers':transformers.__version__,
        'cuda':torch.version.cuda,'torch_file':torch.__file__,'gpus':gpus}
path=root/'results/stage_a_cuda_runtime_check.json'
with path.open('x') as f: json.dump(report,f,indent=2)
print(json.dumps(report),flush=True)
"""
        run([python,'-B','-u','-c',check],'queued_cuda_check.log',180)
        save('checking_core_semantics')
        run([python,'-B','-u','-m','unittest','-v','test_stage_a_core.py'],'queued_core_checks.log',180)
        save('recording_installed_packages')
        run([python,'-B','-m','pip','freeze'],'stage_a_cuda_requirements.lock',120)
        save('running_two_question_pilot')
        command=[python,'-B','-u','run_stage_a_development.py','--cases','data/stage_a_cases.jsonl',
                 '--model-root',str(root/'models'),'--evaluator','data/hotpot_evaluate_v1.py',
                 '--source-device','cuda:0','--target-device','cuda:1',
                 '--limit','2','--conditions','P0','P2','--starts','0','4','8','12','16','20','24',
                 '--width','4','--max-new-tokens','64','--output',str(output)]
        run(command,'stage_a_7b_pilot.log',3600)
        report=json.loads(output.read_text())
        if report['status']!='completed' or len(report['cases'])!=4:
            raise RuntimeError('Pilot process ended without four completed development cases')
        save('pilot_completed_pending_scientific_review',active_child_pid=None)
    except Exception as error:
        save('failed',error=f'{type(error).__name__}: {error}')
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',required=True)
    p.add_argument('--installer-pid',type=int,required=True)
    p.add_argument('--source-hashes',required=True)
    p.add_argument('--wait-seconds',type=int,default=7200)
    main(p.parse_args())
