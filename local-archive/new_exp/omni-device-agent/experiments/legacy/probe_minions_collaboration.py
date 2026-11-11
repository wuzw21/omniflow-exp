"""Run checksum-pinned upstream Minion unchanged with two real-model adapters."""
import argparse
import ast
import copy
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import shlex
import subprocess
import sys
import time
import traceback
import types
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
SOURCE = Path.home() / '.agent-reach/research/minions-a87d0ee'
REMOTE = '/dev/shm/edgecloud-stage-a-wuzewen-01a15efc'
SSH = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15',
       '-o', 'ControlPath=/tmp/edgecloud-stage-a-ssh-%C', '9207']


def lcp(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def load_upstream(source=SOURCE):
    manifest = json.loads((ROOT / 'minions_source_expected.json').read_text())
    verified = {}
    for f in manifest['files']:
        raw = (source / f['path']).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(raw)).encode() + b'\0' + raw).hexdigest()
        if actual != f['git_blob_sha1']:
            raise ValueError(f"Source checksum mismatch: {f['path']}")
        verified[f['path']] = hashlib.sha256(raw).hexdigest()
    mod = types.ModuleType('_pinned_minions_probe')
    sys.modules[mod.__name__] = mod
    env = mod.__dict__
    env.update({k: globals()[k] for k in ('json', 're', 'os', 'time', 'datetime',
                                       'List', 'Dict', 'Any', 'Optional')})
    env['dataclass'] = dataclasses.dataclass
    # Provider type markers select upstream's generic client path. No fake inference.
    for name in ('OpenAIClient', 'TogetherClient', 'GeminiClient', 'SambanovaClient',
                 'SyncMCPClient'):
        env[name] = type(name, (), {})
    def definitions(path, allowed):
        tree = ast.parse((source / path).read_text(), filename=path)
        nodes = [n for n in tree.body if isinstance(n, allowed)]
        module = ast.Module(body=nodes, type_ignores=[])
        exec(compile(module, str(source / path), 'exec'), env)
    definitions('minions/usage.py', (ast.ClassDef,))
    for name in ('minion', 'minion_mcp', 'multi_turn'):
        definitions(f'minions/prompts/{name}.py', (ast.Assign, ast.AnnAssign))
    definitions('minions/minion.py', (ast.FunctionDef, ast.ClassDef))
    return env['Minion'], env['Usage'], {**manifest, 'verified_sha256': verified,
        'loading': 'Unmodified AST function/class bodies; original prompt assignments; optional provider imports excluded',
        'disabled_paths': ['MCP', 'privacy shield', 'multi-turn history']}


class Endpoint:
    def __init__(self, name, cmd, directory, usage, trace):
        self.name, self.usage, self.trace = name, usage, trace
        self.transport_failed = False
        self.last = {}
        self.phase = 'minion'
        self.log = open(directory / f'{name}.stderr.log', 'w')
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.log, text=True, bufsize=1)
        try:
            self.ready = self.receive(300)
            if not self.ready.get('ready'):
                raise RuntimeError(self.ready)
        except Exception:
            if self.proc.poll() is None:
                self.proc.terminate()
                self.proc.wait(timeout=15)
            self.log.close()
            raise
        print(f'READY {name} pid={self.ready["pid"]}', flush=True)

    def receive(self, timeout):
        with selectors.DefaultSelector() as select:
            select.register(self.proc.stdout, selectors.EVENT_READ)
            if not select.select(timeout):
                self.transport_failed = True
                raise TimeoutError(f'{self.name}: no result in {timeout}s; pid={self.proc.pid}')
            line = self.proc.stdout.readline()
        if not line:
            self.transport_failed = True
            raise RuntimeError(f'{self.name}: process closed stdout; exit={self.proc.poll()}')
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            self.transport_failed = True
            raise

    def chat(self, messages, **kwargs):
        if self.transport_failed:
            raise RuntimeError(f'{self.name}: transport unusable after failed receive; late responses cannot be reassigned')
        snapshot = copy.deepcopy(messages)
        request = {'messages': snapshot, 'max_new_tokens': 512 if self.name == 'local' else 768}
        started = time.perf_counter()
        self.proc.stdin.write(json.dumps(request) + '\n')
        self.proc.stdin.flush()
        result = self.receive(300)
        record = {'endpoint': self.name, 'phase': self.phase, 'messages': snapshot,
                  'max_new_tokens': request['max_new_tokens'], 'result': result,
                  'request_wall_seconds': time.perf_counter() - started,
                  'eligible_own_prefix_tokens': lcp(self.last.get(self.phase, []), result['input_ids']),
                  'measured_cache_hits': 0}
        # generate does not forward the final generated token: do not count it as cached.
        self.last[self.phase] = result['input_ids'] + result['generated_ids'][:-1]
        with self.trace.open('a') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f'CALL {self.phase}/{self.name}: input={len(result["input_ids"])} output={len(result["generated_ids"])} eligible={record["eligible_own_prefix_tokens"]} end={result["finish_reason"]}', flush=True)
        usage = self.usage(prompt_tokens=len(result['input_ids']),
                           completion_tokens=len(result['generated_ids']))
        response = ([result['text']], usage)
        return (*response, result['finish_reason']) if self.name == 'local' else response

    def close(self):
        if self.proc.poll() is None:
            self.proc.stdin.write('{"shutdown":true}\n')
            self.proc.stdin.flush()
            self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                # This process was created by this harness; never touch other jobs.
                self.proc.terminate()
                self.proc.wait(timeout=10)
        self.log.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    args = p.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    Minion, Usage, source = load_upstream()
    local_spec = {'repo': 'Qwen/Qwen2.5-0.5B-Instruct',
                  'revision': '7ae557604adf67be50417f59c2c2f167def9a775',
                  'path': str(Path.home() / '.agent-reach/edgecloud-stage-a/models/7ae557604adf67be50417f59c2c2f167def9a775'),
                  'device': 'mps', 'weights': {'model.safetensors':
                  'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe'}}
    remote_model = json.loads((ROOT / 'stage_a_7b_models.json').read_text())['models']['instruct']
    remote_spec = {'repo': remote_model['repo'], 'revision': remote_model['revision'],
                   'path': REMOTE + '/models/' + remote_model['revision'], 'device': 'cuda',
                   'weights': {f['name']: f['sha256'] for f in remote_model['files'] if f['name'].endswith('.safetensors')}}
    for name, spec in [('local', local_spec), ('remote', remote_spec)]:
        (directory / f'{name}.spec.json').write_text(json.dumps(spec, indent=2))
    remote_dir = REMOTE + '/minions-probe/' + directory.name
    subprocess.run(SSH + ['mkdir -p ' + shlex.quote(remote_dir)], check=True)
    for src, dst in [(ROOT / 'minions_model_worker.py', 'worker.py'),
                     (directory / 'remote.spec.json', 'spec.json')]:
        with src.open('rb') as f:
            subprocess.run(SSH + ['cat > ' + shlex.quote(remote_dir + '/' + dst)], stdin=f, check=True)
    trace = directory / 'calls.jsonl'
    endpoints = []
    evidence = {'source': source, 'max_rounds': 2, 'privacy_shield': False,
                'status': 'running', 'adapters': 'fresh prefill per request, decode cache within generate',
                'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'worker_sha256': hashlib.sha256((ROOT / 'minions_model_worker.py').read_bytes()).hexdigest()}
    try:
        local = Endpoint('local', [str(ROOT / '.venv-stage-a/bin/python'), '-u',
                         str(ROOT / 'minions_model_worker.py'), '--spec', str(directory / 'local.spec.json')], directory, Usage, trace)
        endpoints.append(local)
        command = ['env', 'CUDA_VISIBLE_DEVICES=GPU-f3f0fa0c-05fc-93c6-b661-14d0457a2cda',
                   'TMPDIR=' + REMOTE, REMOTE + '/runtime/bin/python3.13', '-u',
                   remote_dir + '/worker.py', '--spec', remote_dir + '/spec.json']
        remote = Endpoint('remote', SSH + [shlex.join(command)], directory, Usage, trace)
        endpoints.append(remote)
        evidence['endpoints'] = {e.name: e.ready for e in endpoints}
        task = json.loads((SOURCE / 'minions/examples/code/task.json').read_text())['question']
        context = (SOURCE / 'minions/examples/code/sample.txt').read_text()
        try:
            run = Minion(local_client=local, remote_client=remote, max_rounds=2,
                         log_dir=str(directory / 'upstream_logs'))
            result = run(task=task, context=[context], logging_id='public_code')
            evidence['minion_result'] = result
            evidence['status'] = 'completed'
        except Exception:
            evidence['status'] = 'protocol_failed'
            evidence['protocol_error'] = traceback.format_exc()
        # Independent full-context reference answers; labels are never passed to models.
        evidence['direct'] = {}
        for endpoint in endpoints:
            endpoint.phase = 'direct'
            response = endpoint.chat([{'role': 'user', 'content': task + '\n\n' + context}])
            evidence['direct'][endpoint.name] = response[0][0]
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
        (directory / 'evidence.json').write_text(json.dumps(evidence, indent=2, ensure_ascii=False,
              default=lambda x: x.to_dict() if hasattr(x, 'to_dict') else str(x)))
        print('STATUS', evidence['status'], flush=True)


if __name__ == '__main__':
    main()
