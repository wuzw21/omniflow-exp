"""Fixed upstream examples, stronger real local worker, unchanged Minion protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import traceback
from probe_minions_collaboration import ROOT, SOURCE, REMOTE, SSH, Endpoint, load_upstream

CASES = ('code', 'long_documents', 'finance')
REVISION = 'aa8e72537993ba99e69dfaafa59ed015b17504d1'


class LongContextEndpoint(Endpoint):
    def receive(self, timeout):
        return super().receive(max(timeout, 900))


def save(directory, evidence):
    tmp = directory / 'evidence.tmp'
    tmp.write_text(json.dumps(evidence, ensure_ascii=False, indent=2,
                  default=lambda x: x.to_dict() if hasattr(x, 'to_dict') else str(x)))
    tmp.replace(directory / 'evidence.json')


def verify_tasks():
    extra = json.loads((SOURCE / 'additional_tasks_manifest.json').read_text())
    for f in extra['files']:
        raw = (SOURCE / f['path']).read_bytes()
        assert hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest() == f['git_blob_sha1']
        assert hashlib.sha256(raw).hexdigest() == f['sha256']
    return extra


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--local-backend', choices=['torch', 'mlx'], default='torch')
    args = parser.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    Minion, Usage, upstream = load_upstream()
    tasks = verify_tasks()
    modelroot = Path.home() / '.agent-reach/edgecloud-stage-a/models' / REVISION
    manifest = json.loads((modelroot / 'download_manifest.json').read_text())
    local_spec = {'repo': manifest['repo'], 'revision': REVISION, 'path': str(modelroot),
                  'device': 'mps', 'prefill_chunk_tokens': 512, 'license': manifest['license'],
                  'weights': {f['name']: f['sha256'] for f in manifest['files'] if f['name'].endswith('.safetensors')}}
    if args.local_backend == 'mlx':
        local_spec['device'] = 'mlx_metal'
        local_spec['backend'] = 'mlx-lm-0.32.2'
    remote_model = json.loads((ROOT / 'stage_a_7b_models.json').read_text())['models']['instruct']
    remote_spec = {'repo': remote_model['repo'], 'revision': remote_model['revision'],
                   'path': REMOTE + '/models/' + remote_model['revision'], 'device': 'cuda',
                   'prefill_chunk_tokens': 1024,
                   'weights': {f['name']: f['sha256'] for f in remote_model['files'] if f['name'].endswith('.safetensors')}}
    for name, spec in [('local', local_spec), ('remote', remote_spec)]:
        (directory / f'{name}.spec.json').write_text(json.dumps(spec, indent=2))
    remote_dir = REMOTE + '/minions-probe/' + directory.name
    subprocess.run(SSH + ['mkdir -p ' + shlex.quote(remote_dir)], check=True)
    for src, dst in [(ROOT / 'minions_model_worker.py', 'worker.py'), (directory / 'remote.spec.json', 'spec.json')]:
        with src.open('rb') as f:
            subprocess.run(SSH + ['cat > ' + shlex.quote(remote_dir + '/' + dst)], stdin=f, check=True)
    evidence = {'status': 'running', 'case_order': CASES, 'max_rounds': 2,
                'local_backend': args.local_backend,
                'protocol': json.loads((ROOT / 'minions_3b_protocol.json').read_text()),
                'protocol_sha256': hashlib.sha256((ROOT / 'minions_3b_protocol.json').read_bytes()).hexdigest(),
                'selection': 'All selected before stronger-model results: prior code failure, named-entity paper lookup, numeric annual-report lookup',
                'source': upstream, 'task_source': tasks, 'cases': {},
                'privacy_shield': False, 'cross_request_cache': False,
                'prefill': 'Native backend caches; all input retained; local 512/CUDA 1024 chunks',
                'scripts': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                            ['probe_minions_3b_suite.py', 'probe_minions_collaboration.py', 'minions_model_worker.py']}}
    if args.local_backend == 'mlx':
        evidence['scripts']['minions_mlx_worker.py'] = hashlib.sha256((ROOT / 'minions_mlx_worker.py').read_bytes()).hexdigest()
    for name in evidence['scripts']:
        (directory / name).write_bytes((ROOT / name).read_bytes())
    save(directory, evidence)
    endpoints = []
    try:
        python = ROOT / ('.venv-minions-mlx/bin/python' if args.local_backend == 'mlx' else '.venv-stage-a/bin/python')
        worker = ROOT / ('minions_mlx_worker.py' if args.local_backend == 'mlx' else 'minions_model_worker.py')
        local = LongContextEndpoint('local', [str(python), '-u',
                         str(worker), '--spec', str(directory / 'local.spec.json')],
                         directory, Usage, directory / 'calls.jsonl')
        endpoints.append(local)
        command = ['env', 'CUDA_VISIBLE_DEVICES=GPU-f3f0fa0c-05fc-93c6-b661-14d0457a2cda',
                   'TMPDIR=' + REMOTE, REMOTE + '/runtime/bin/python3.13', '-u',
                   remote_dir + '/worker.py', '--spec', remote_dir + '/spec.json']
        remote = LongContextEndpoint('remote', SSH + [shlex.join(command)], directory, Usage, directory / 'calls.jsonl')
        endpoints.append(remote)
        evidence['endpoints'] = {e.name: e.ready for e in endpoints}
        # Bound the full-vs-chunked diagnostic to the original short code example.
        task = json.loads((SOURCE / 'minions/examples/code/task.json').read_text())['question']
        context = (SOURCE / 'minions/examples/code/sample.txt').read_text()
        evidence['chunking_checks'] = {}
        for endpoint in endpoints:
            request = {'messages': [{'role': 'user', 'content': task + '\n\n' + context}],
                       'max_new_tokens': 1, 'validate_chunking': True}
            endpoint.proc.stdin.write(json.dumps(request) + '\n')
            endpoint.proc.stdin.flush()
            evidence['chunking_checks'][endpoint.name] = endpoint.receive(300)
            print('CHUNK_CHECK', endpoint.name, evidence['chunking_checks'][endpoint.name], flush=True)
        save(directory, evidence)
        for case in CASES:
            print('CASE_START', case, flush=True)
            task = json.loads((SOURCE / f'minions/examples/{case}/task.json').read_text())['question']
            context = (SOURCE / f'minions/examples/{case}/sample.txt').read_text()
            record = {'status': 'running', 'task': task, 'context_chars': len(context),
                      'context_sha256': hashlib.sha256(context.encode()).hexdigest(), 'direct': {},
                      'no_document': {}}
            evidence['cases'][case] = record
            save(directory, evidence)
            for endpoint in endpoints:
                endpoint.phase = case + '/minion'
            try:
                run = Minion(local_client=local, remote_client=remote, max_rounds=2,
                             log_dir=str(directory / 'upstream_logs'))
                record['minion_result'] = run(task=task, context=[context], logging_id=case)
                record['status'] = 'protocol_returned'
            except Exception:
                record['status'] = 'protocol_failed'
                record['protocol_error'] = traceback.format_exc()
            save(directory, evidence)
            for endpoint in endpoints:
                endpoint.phase = case + '/direct'
                response = endpoint.chat([{'role': 'user', 'content': task + '\n\n' + context}])
                record['direct'][endpoint.name] = response[0][0]
                save(directory, evidence)
            if case != 'code':
                for endpoint in endpoints:
                    endpoint.phase = case + '/no_document'
                    response = endpoint.chat([{'role': 'user', 'content': task}])
                    record['no_document'][endpoint.name] = response[0][0]
                    save(directory, evidence)
            print('CASE_DONE', case, record['status'], flush=True)
        evidence['status'] = 'execution_completed'
    except Exception:
        evidence['status'] = 'execution_failed'
        evidence['execution_error'] = traceback.format_exc()
        raise
    finally:
        for endpoint in reversed(endpoints):
            try:
                endpoint.close()
            except Exception:
                evidence.setdefault('cleanup_errors', []).append(traceback.format_exc())
        evidence['closed_processes'] = {e.name: e.proc.poll() for e in endpoints}
        save(directory, evidence)
        print('STATUS', evidence['status'], flush=True)


if __name__ == '__main__':
    main()
