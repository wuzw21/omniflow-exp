"""Check real remote files and durable writes before starting any model process."""
import json
import subprocess


def remote_preflight(ssh, root, python, model_spec):
    config = {'root': root, 'python': python, 'model': model_spec['path'],
              'weights': list(model_spec['weights'])}
    script = 'CONFIG = ' + repr(config) + '\n' + r'''
import json, os, pathlib, subprocess, tempfile
r = {'config': CONFIG, 'checks': {}, 'errors': []}
root = pathlib.Path(CONFIG['root'])
r['checks']['root_exists'] = root.is_dir()
r['checks']['python_exists'] = pathlib.Path(CONFIG['python']).is_file()
r['missing_model_files'] = [name for name in CONFIG['weights'] + ['config.json', 'tokenizer_config.json', 'tokenizer.json'] if not (pathlib.Path(CONFIG['model']) / name).is_file()]
r['checks']['model_files_exist'] = not r['missing_model_files']
try:
    with tempfile.TemporaryDirectory(prefix='tau2-preflight-', dir=root) as d:
        p = pathlib.Path(d) / 'write'
        with p.open('wb') as f:
            f.write(b'\0' * 1024 * 1024); f.flush(); os.fsync(f.fileno())
        q = p.with_name('renamed'); p.rename(q)
        assert q.stat().st_size == 1024 * 1024
    r['checks']['write_fsync_rename_delete'] = True
except OSError as e:
    r['checks']['write_fsync_rename_delete'] = False
    r['errors'].append({'check': 'storage', 'errno': e.errno, 'type': type(e).__name__})
if r['checks']['python_exists']:
    try:
        p = subprocess.run([CONFIG['python'], '-c', 'import json,torch,transformers; print(json.dumps({"torch":torch.__version__,"transformers":transformers.__version__,"cuda":torch.cuda.is_available()}))'], capture_output=True, text=True, timeout=20)
        r['runtime_returncode'] = p.returncode
        if p.returncode == 0:
            r['runtime'] = json.loads(p.stdout.strip())
            r['checks']['runtime_ready'] = r['runtime']['cuda'] and r['runtime']['torch'] == '2.9.1+cu128' and r['runtime']['transformers'] == '4.57.6'
        else:
            r['checks']['runtime_ready'] = False
            r['runtime_error_tail'] = p.stderr.splitlines()[-3:]
    except (subprocess.TimeoutExpired, OSError, ValueError) as e:
        r['checks']['runtime_ready'] = False
        r['errors'].append({'check': 'runtime', 'type': type(e).__name__})
else:
    r['checks']['runtime_ready'] = False
r['status'] = 'ready' if all(r['checks'].values()) else 'unavailable'
print(json.dumps(r))
'''
    result = subprocess.run(ssh + ['/usr/bin/python3 -'], input=script,
                            capture_output=True, text=True, timeout=60)
    if result.returncode != 0:
        return {'status': 'unavailable', 'ssh_returncode': result.returncode,
                'error': 'Remote preflight command failed'}
    return json.loads(result.stdout)
