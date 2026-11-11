"""Original PAPILLON pilot, then one paired cloud/local scheduling diagnostic.

Real Qwen endpoints only; no paid API. Sensitive/public dataset text and raw
generations stay in the separate research trace directory. Output is sanitized.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback

from probe_minions_collaboration import Endpoint, ROOT, REMOTE, SSH


@dataclass
class Usage:
    prompt_tokens: int
    completion_tokens: int


class PreparedEndpoint(Endpoint):
    use_prepared_next = False

    def prepare(self, messages):
        if self.transport_failed:
            raise RuntimeError('Unusable endpoint')
        self.proc.stdin.write(json.dumps({'prepare_prefix': True, 'messages': messages}) + '\n')
        self.proc.stdin.flush()
        return self.receive(300)

    def chat(self, messages, **kwargs):
        if not self.use_prepared_next:
            return super().chat(messages, **kwargs)
        if self.transport_failed:
            raise RuntimeError('Unusable endpoint')
        self.use_prepared_next = False
        req = {'messages': messages, 'max_new_tokens': 512, 'use_prepared': True}
        began = time.perf_counter()
        self.proc.stdin.write(json.dumps(req) + '\n'); self.proc.stdin.flush()
        result = self.receive(300)
        with self.trace.open('a') as f:
            f.write(json.dumps({'endpoint': self.name, 'phase': self.phase, 'messages': messages,
                                'max_new_tokens': 512, 'result': result,
                                'request_wall_seconds': time.perf_counter() - began}) + '\n')
        return [result['text']], Usage(len(result['input_ids']), len(result['generated_ids'])), result['finish_reason']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--case-index', type=int, default=0)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.trace.mkdir(parents=True, exist_ok=False)
    protocol = json.loads((ROOT / 'papillon_overlap_protocol.json').read_text())
    selected = protocol['cases'][args.case_index]
    source = Path.home() / '.agent-reach/research/papillon-594e7eb'
    dataset = source / 'pupa' / protocol['file']
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == protocol['file_sha256']
    with dataset.open() as f:
        q = list(csv.DictReader(f))[selected['row']]['user_query']
    assert hashlib.sha256(q.encode()).hexdigest() == selected['query_sha256']
    upstream = json.loads((source / 'manifest.json').read_text())
    for item in upstream['files']:
        assert hashlib.sha256((source / item['path']).read_bytes()).hexdigest() == item['sha256']
    # This run uses owned model processes, never a paid provider. Avoid the
    # unrelated price-map HTTP fetch through LiteLLM's supported offline option.
    os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
    import dspy
    from litellm import ModelResponse
    spec = importlib.util.spec_from_file_location('pinned_papillon', source / 'papillon/run_llama_dspy.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    local_spec = json.loads((ROOT / 'results/minions_3b_mlx_001/local.spec.json').read_text())
    remote_spec = json.loads((ROOT / 'results/minions_3b_mlx_001/remote.spec.json').read_text())
    for name, s in [('local', local_spec), ('remote', remote_spec)]:
        (args.output / f'{name}.spec.json').write_text(json.dumps(s, indent=2))
    scripts = ['probe_papillon_overlap.py', 'papillon_mlx_worker.py', 'probe_minions_collaboration.py',
               'minions_model_worker.py', 'papillon_overlap_protocol.json']
    for name in scripts:
        (args.output / name).write_bytes((ROOT / name).read_bytes())
    result = {'status': 'running', 'scope': 'One real dataset-length pilot; not a quality/privacy benchmark or novel mechanism',
              'protocol': protocol, 'selected': selected, 'source': upstream,
              'dspy': dspy.__version__, 'trace_directory': str(args.trace),
              'scripts_sha256': {n: hashlib.sha256((ROOT / n).read_bytes()).hexdigest() for n in scripts}}
    def save():
        temp = args.output / 'summary.tmp'
        temp.write_text(json.dumps(result, indent=2)); temp.replace(args.output / 'summary.json')
    save()
    endpoints = []
    try:
        remote_dir = REMOTE + '/papillon/' + args.output.name
        subprocess.run(SSH + ['mkdir -p ' + shlex.quote(remote_dir)], check=True)
        for src, name in [(ROOT / 'minions_model_worker.py', 'worker.py'),
                          (args.output / 'remote.spec.json', 'spec.json')]:
            with src.open('rb') as f:
                subprocess.run(SSH + ['cat > ' + shlex.quote(remote_dir + '/' + name)], stdin=f, check=True)
        local = PreparedEndpoint('local', [str(ROOT / '.venv-minions-mlx/bin/python'), '-u',
                     str(ROOT / 'papillon_mlx_worker.py'), '--spec', str(args.output / 'local.spec.json')],
                     args.trace, Usage, args.trace / 'calls.jsonl')
        endpoints.append(local)
        command = ['env', 'CUDA_VISIBLE_DEVICES=GPU-f3f0fa0c-05fc-93c6-b661-14d0457a2cda',
                   'TMPDIR=' + REMOTE, REMOTE + '/runtime/bin/python3.13', '-u',
                   remote_dir + '/worker.py', '--spec', remote_dir + '/spec.json']
        remote = Endpoint('remote', SSH + [shlex.join(command)], args.trace, Usage, args.trace / 'calls.jsonl')
        endpoints.append(remote)
        result['endpoints'] = {e.name: e.ready for e in endpoints}; save()
        failures = []
        class LM(dspy.BaseLM):
            def forward(self, prompt=None, messages=None, **kwargs):
                try:
                    replies, usage, finish = local.chat(messages or [{'role': 'user', 'content': prompt}])
                    return ModelResponse(model='qwen-local', choices=[{'index': 0,
                        'message': {'role': 'assistant', 'content': replies[0]}, 'finish_reason': finish}],
                        usage={'prompt_tokens': usage.prompt_tokens, 'completion_tokens': usage.completion_tokens,
                               'total_tokens': usage.prompt_tokens + usage.completion_tokens})
                except Exception as exc:
                    failures.append(type(exc).__name__); raise
        lm = LM(model='qwen-local', temperature=0, max_tokens=512, cache=False, num_retries=0)
        cloud_times = []
        def cloud(prompt):
            began = time.perf_counter()
            replies, _ = remote.chat([{'role': 'user', 'content': prompt}])
            cloud_times.append(time.perf_counter() - began)
            return replies
        flow = module.PAPILLON(cloud)
        local.phase = remote.phase = 'baseline'
        began = time.perf_counter()
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(), disable_history=True):
            baseline = flow(user_query=q)
        result['baseline_wall_seconds'] = time.perf_counter() - began
        result['baseline_nonempty'] = bool(baseline.prompt and baseline.output and baseline.gptResponse)
        result['adapter_failures'] = failures
        (args.trace / 'baseline.json').write_text(json.dumps(dict(baseline), ensure_ascii=False, indent=2))
        baseline_calls = [json.loads(line) for line in (args.trace / 'calls.jsonl').read_text().splitlines()]
        result['baseline_call_count'] = len(baseline_calls); save()
        if not result['baseline_nonempty'] or len(baseline_calls) != 3:
            result['status'] = 'integration_failed_or_extra_calls'
            return
        # This known prefix is constructed without any cloud response, before its second call.
        known = dspy.ChatAdapter().format(flow.info_aggregator.signature, [],
                                         {'userQuery': q, 'modelExampleResponses': ''})
        local.phase = remote.phase = 'overlap'
        began = time.perf_counter()
        with ThreadPoolExecutor(max_workers=1) as executor:
            preparation = executor.submit(local.prepare, known)
            remote_reply = cloud(baseline.prompt)[0]
            result['prepared'] = preparation.result()
        local.use_prepared_next = True
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(), disable_history=True):
            overlapped = flow.info_aggregator(userQuery=q, modelExampleResponses=remote_reply)
        result['overlap_cloud_plus_aggregation_wall_seconds'] = time.perf_counter() - began
        (args.trace / 'overlap.json').write_text(json.dumps(dict(overlapped), ensure_ascii=False, indent=2))
        # Post-hoc control motivated by first-use timing in pilot 002: a serial
        # path after both models have already run the identical requests.
        local.phase = remote.phase = 'serial_warm_control'
        began = time.perf_counter()
        serial_reply = cloud(baseline.prompt)[0]
        with dspy.context(lm=lm, adapter=dspy.ChatAdapter(), disable_history=True):
            serial = flow.info_aggregator(userQuery=q, modelExampleResponses=serial_reply)
        result['warm_serial_cloud_plus_aggregation_wall_seconds'] = time.perf_counter() - began
        result['warm_serial_same_cloud_response'] = serial_reply == baseline.gptResponse
        result['warm_serial_same_parsed_output'] = serial.finalOutput == baseline.output
        (args.trace / 'serial_warm.json').write_text(json.dumps(dict(serial), ensure_ascii=False, indent=2))
        calls = [json.loads(line) for line in (args.trace / 'calls.jsonl').read_text().splitlines()]
        def compact(call):
            data = call['result']
            return {'endpoint': call['endpoint'], 'phase': call['phase'],
                    'wall_seconds': call['request_wall_seconds'], 'input_tokens': len(data['input_ids']),
                    'output_tokens': len(data['generated_ids']), 'finish_reason': data['finish_reason'],
                    'prefill_seconds': data['total_prefill_seconds'],
                    'model_generation_seconds': data['generation_seconds'],
                    'input_sha256': hashlib.sha256(json.dumps(data['input_ids']).encode()).hexdigest(),
                    'output_sha256': hashlib.sha256(json.dumps(data['generated_ids']).encode()).hexdigest(),
                    'reused_prefix_tokens': data.get('reused_prefix_tokens', 0),
                    'prepared_prefix_mismatch': data.get('prepared_prefix_mismatch', False)}
        result['calls'] = [compact(c) for c in calls]
        result['baseline_cloud_plus_aggregation_wall_seconds'] = sum(c['request_wall_seconds'] for c in baseline_calls[1:])
        result['same_cloud_response'] = remote_reply == baseline.gptResponse
        a = baseline_calls[-1]['result']
        b = next(c['result'] for c in calls if c['endpoint'] == 'local' and c['phase'] == 'overlap')
        warm_serial = calls[-1]['result']
        result['same_aggregation_input_ids'] = a['input_ids'] == b['input_ids']
        result['same_aggregation_generated_ids'] = a['generated_ids'] == b['generated_ids']
        result['same_parsed_final_output'] = baseline.output == overlapped.finalOutput
        result['warm_serial_same_input_ids'] = warm_serial['input_ids'] == a['input_ids']
        result['warm_serial_same_generated_ids'] = warm_serial['generated_ids'] == a['generated_ids']
        result['status'] = 'completed'
        result['limits'] = 'Serial/overlap/serial order, one run each; weights already loaded but initial kernel warming uncontrolled. No quality/privacy evaluator or statistical speedup claim; matching inputs required.'
    except Exception:
        result['status'] = 'failed'
        result['exception'] = traceback.format_exc()
        raise
    finally:
        for endpoint in endpoints:
            endpoint.close()
        result['process_exits'] = {e.name: e.proc.returncode for e in endpoints}
        save()
        print(json.dumps({k: v for k, v in result.items() if k in ['status', 'baseline_nonempty',
                          'same_aggregation_generated_ids', 'process_exits']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
