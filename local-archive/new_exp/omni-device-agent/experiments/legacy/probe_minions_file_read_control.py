"""Exploratory public-file control for the observed request to provide code.

Preserves the upstream supervisor loop. The one supported local operation returns
the public fixture verbatim, with no local model. It fails on any other request.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import traceback
from probe_minions_collaboration import ROOT, SOURCE, REMOTE, SSH, Endpoint, load_upstream

EXPECTED_REQUEST = 'Please provide the code so I can analyze it and identify the methods containing errors.'


class PublicFileClient:
    def __init__(self, Usage, directory, max_file_tokens=0):
        self.Usage, self.directory = Usage, directory
        self.max_file_tokens = max_file_tokens
        self.calls = []

    def chat(self, messages):
        if messages[-1]['content'] != EXPECTED_REQUEST or self.calls:
            raise ValueError('File-read control only supports the observed first request; no semantic worker emulation')
        content = (SOURCE / 'minions/examples/code/sample.txt').read_text()
        if self.max_file_tokens:
            from transformers import AutoTokenizer
            root = Path.home() / '.agent-reach/edgecloud-stage-a/models/aa8e72537993ba99e69dfaafa59ed015b17504d1'
            tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
            content = tokenizer.decode(tokenizer.encode(content, add_special_tokens=False)[:self.max_file_tokens])
        self.calls.append({'request': messages[-1]['content'], 'operation': 'read_public_source_file',
                           'max_file_tokens': self.max_file_tokens or None,
                           'utf8_bytes': len(content.encode()), 'content_sha256': hashlib.sha256(content.encode()).hexdigest(),
                           'local_model_calls': 0})
        return [content], self.Usage(), 'file_read'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--max-file-tokens', type=int, default=0)
    args = p.parse_args()
    directory = Path(args.output).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    Minion, Usage, source = load_upstream()
    spec = json.loads((ROOT / 'results/minions_3b_001/remote.spec.json').read_text())
    spec['selected_cuda_uuid'] = 'GPU-e1b48613-19fb-5e25-d089-f58f8d6efa42'
    (directory / 'remote.spec.json').write_text(json.dumps(spec, indent=2))
    remote_dir = REMOTE + '/minions-probe/' + directory.name
    subprocess.run(SSH + ['mkdir -p ' + shlex.quote(remote_dir)], check=True)
    for src, dst in [(ROOT / 'minions_model_worker.py', 'worker.py'), (directory / 'remote.spec.json', 'spec.json')]:
        with src.open('rb') as f:
            subprocess.run(SSH + ['cat > ' + shlex.quote(remote_dir + '/' + dst)], stdin=f, check=True)
    evidence = {'status': 'running', 'source': source, 'scope': 'post-hoc public code fixture control; not a general local agent',
                'max_file_tokens': args.max_file_tokens or None,
                'selection': 'Observed source copying in 3B run motivated this additional control',
                'scripts': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in
                            ['probe_minions_file_read_control.py', 'probe_minions_collaboration.py', 'minions_model_worker.py']}}
    local = PublicFileClient(Usage, directory, args.max_file_tokens)
    remote = None
    try:
        command = ['env', 'CUDA_VISIBLE_DEVICES=' + spec['selected_cuda_uuid'], 'TMPDIR=' + REMOTE,
                   REMOTE + '/runtime/bin/python3.13', '-u', remote_dir + '/worker.py', '--spec', remote_dir + '/spec.json']
        remote = Endpoint('remote', SSH + [shlex.join(command)], directory, Usage, directory / 'calls.jsonl')
        evidence['endpoint'] = remote.ready
        # A genuine remote split check: the earlier 931-token check used no 1024-token split.
        context = (SOURCE / 'minions/examples/code/sample.txt').read_text()
        question = json.loads((SOURCE / 'minions/examples/code/task.json').read_text())['question']
        diagnostic = {'messages': [{'role': 'user', 'content': context + '\n\n' + context + '\n\n' + question}],
                      'validate_chunking': True, 'max_new_tokens': 1}
        remote.proc.stdin.write(json.dumps(diagnostic) + '\n')
        remote.proc.stdin.flush()
        evidence['remote_chunk_check'] = remote.receive(300)
        print('REMOTE_CHUNK_CHECK', evidence['remote_chunk_check'], flush=True)
        run = Minion(local_client=local, remote_client=remote, max_rounds=2,
                     log_dir=str(directory / 'upstream_logs'))
        evidence['result'] = run(task=question, context=[context], logging_id='code_file_read')
        evidence['status'] = 'protocol_returned'
    except Exception:
        evidence['status'] = 'failed'
        evidence['error'] = traceback.format_exc()
    finally:
        if remote:
            remote.close()
            evidence['process_exit'] = remote.proc.poll()
        evidence['file_operations'] = local.calls
        (directory / 'evidence.json').write_text(json.dumps(evidence, ensure_ascii=False, indent=2,
                   default=lambda x: x.to_dict() if hasattr(x, 'to_dict') else str(x)))
        print('STATUS', evidence['status'], flush=True)


if __name__ == '__main__':
    main()
