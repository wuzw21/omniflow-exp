"""Physical Mac/9207 stdio deadline regression, not a model-quality test."""
import json
import hashlib
from pathlib import Path
import shlex
import sys
import tempfile
from probe_minions_collaboration import Endpoint, ROOT, SSH, REMOTE, load_upstream

CHILD = '''import json,os,sys,time,platform
print(json.dumps({'ready':True,'pid':os.getpid(),'python':platform.python_version(),'platform':platform.platform()}),flush=True)
first=json.loads(sys.stdin.readline())
time.sleep(0.3)
print(json.dumps({'late_response_to_first':True}),flush=True)
second=json.loads(sys.stdin.readline())
assert second=={'shutdown':True}, 'A new request was incorrectly sent after a receive deadline'
'''


def main():
    _, Usage, _ = load_upstream()
    output = []
    commands = [('physical_mac', [sys.executable, '-u', '-c', CHILD]),
                ('physical_9207_ssh', SSH + [shlex.join([REMOTE + '/runtime/bin/python3.13', '-u', '-c', CHILD])])]
    for name, cmd in commands:
        with tempfile.TemporaryDirectory(prefix='minions-transport-') as d:
            directory = Path(d)
            e = Endpoint(name, cmd, directory, Usage, directory / 'trace.jsonl')
            try:
                e.proc.stdin.write('{}\n')
                e.proc.stdin.flush()
                try:
                    e.receive(0.02)
                    raise AssertionError('Expected a real receive deadline')
                except TimeoutError:
                    pass
                assert e.transport_failed
                try:
                    e.chat([{'role': 'user', 'content': 'must not be submitted'}])
                    raise AssertionError('Late response could be reassigned')
                except RuntimeError as exc:
                    assert 'transport unusable' in str(exc)
                assert not (directory / 'trace.jsonl').exists()
            finally:
                e.close()
            assert e.proc.returncode == 0
            output.append({'device': name, 'worker_pid': e.ready['pid'], 'real_receive_deadline': True,
                           'runtime': e.ready,
                           'adapter_sha256': hashlib.sha256((ROOT / 'probe_minions_collaboration.py').read_bytes()).hexdigest(),
                           'subsequent_request_rejected': True, 'late_response_not_recorded_as_next_result': True,
                           'worker_shutdown_exit': e.proc.returncode})
    path = ROOT / 'results/minions_transport_verification.json'
    path.write_text(json.dumps(output, indent=2))
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
