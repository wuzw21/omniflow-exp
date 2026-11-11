"""Native MLX server lifecycle; no agent loop, tokenizer or cache replacement."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent


def local_request(base, endpoint, payload=None, timeout=180):
    # Loopback must not depend on machine-wide HTTP proxy settings.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    data = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(base + endpoint, data=data,
                                     headers={'Content-Type': 'application/json'})
    with opener.open(request, timeout=timeout) as response:
        return json.load(response)


@contextmanager
def native_server(model_path, trace_dir, evidence):
    trace_dir = Path(trace_dir)
    trace_dir.mkdir(parents=True, exist_ok=True)
    cache = trace_dir / 'hf-hub'
    cache.mkdir(exist_ok=True)
    env = os.environ.copy()
    env['HF_HUB_CACHE'] = str(cache)
    env['NO_PROXY'] = env['no_proxy'] = '127.0.0.1,localhost'
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    base = f'http://127.0.0.1:{port}'
    command = [str(ROOT / '.venv-minions-mlx/bin/python'), '-m', 'mlx_lm', 'server',
               '--model', model_path, '--host', '127.0.0.1', '--port', str(port),
               '--prefill-step-size', '512', '--prompt-cache-size', '4',
               '--prompt-concurrency', '1', '--decode-concurrency', '1', '--log-level', 'INFO']
    with (trace_dir / 'mlx_server.log').open('w') as log:
        proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
        evidence.update(pid=proc.pid, port=port, command=command, hf_hub_cache=str(cache))
        try:
            for _ in range(150):
                if proc.poll() is not None:
                    raise RuntimeError('Native MLX server exited before readiness')
                try:
                    local_request(base, '/v1/models', timeout=2)
                    evidence['ready'] = True
                    break
                except OSError:
                    time.sleep(1)
            else:
                raise TimeoutError('Native MLX server not ready')
            yield base
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=10)
            evidence['exit_code'] = proc.poll()

