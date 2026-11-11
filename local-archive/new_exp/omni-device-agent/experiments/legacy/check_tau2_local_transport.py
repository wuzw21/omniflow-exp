"""Actual M4 tool-call/continuation/restart acceptance; NOT tau2 task success."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import time
import traceback

from tau2_local_transport import ROOT, local_request, native_server


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.trace.mkdir(parents=True, exist_ok=False)
    spec = json.loads((ROOT / 'results/papillon_overlap_003/local.spec.json').read_text())
    report = {'status': 'initializing', 'scope': __doc__, 'task_success_claim_permitted': False,
              'speedup_claim_permitted': False, 'runs': [], 'scripts_sha256': {},
              'hardware': {'machine': platform.machine(), 'os': platform.platform(),
                           'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()},
              'model': spec, 'trace': str(args.trace)}
    for name in ['check_tau2_local_transport.py', 'tau2_local_transport.py']:
        shutil.copy2(ROOT / name, args.output / name)
        report['scripts_sha256'][name] = digest(args.output / name)
    def save():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    save()
    try:
        for name, expected in spec['weights'].items():
            assert digest(Path(spec['path']) / name) == expected
        report['weights_verified'] = True
        report['runtime'] = json.loads(subprocess.check_output([
            str(ROOT / '.venv-minions-mlx/bin/python'), '-c',
            'import json,importlib.metadata as m; print(json.dumps({n:m.version(n) for n in ["mlx","mlx-lm","transformers"]}))'], text=True))
        tools = [{'type': 'function', 'function': {'name': 'lookup_order',
                  'description': 'Look up the shipping state of the specified order.',
                  'parameters': {'type': 'object', 'properties': {'order_id': {'type': 'string'}},
                                 'required': ['order_id'], 'additionalProperties': False}}}]
        for cycle in range(2):
            row = {'cycle': cycle, 'server': {}, 'calls': []}
            report['runs'].append(row)
            trace = args.trace / f'cycle-{cycle}'
            with native_server(spec['path'], trace, row['server']) as base:
                messages = [{'role': 'system', 'content': 'Use the supplied tool when asked. Never invent tool results.'},
                            {'role': 'user', 'content': 'Call lookup_order with order_id TEST-001, then tell me its status and tracking code. Do not answer before getting the tool result.'}]
                def call():
                    payload = {'model': spec['path'], 'messages': messages, 'tools': tools,
                               'tool_choice': 'auto', 'temperature': 0, 'max_tokens': 128}
                    started = time.perf_counter()
                    response = local_request(base, '/v1/chat/completions', payload)
                    with (trace / 'calls.jsonl').open('a') as f:
                        f.write(json.dumps({'request': payload, 'response': response}) + '\n')
                    row['calls'].append({'seconds': time.perf_counter() - started,
                                         'usage': response.get('usage'),
                                         'finish_reason': response['choices'][0]['finish_reason']})
                    save()
                    return response['choices'][0]['message']
                first = call()
                calls = first.get('tool_calls') or []
                row['one_correct_tool_call'] = len(calls) == 1 and calls[0]['function']['name'] == 'lookup_order' and json.loads(calls[0]['function']['arguments']) == {'order_id': 'TEST-001'}
                assert row['one_correct_tool_call'], 'Native parsed tool call mismatch'
                messages.append(first)
                # Public deterministic fixture; not a retail benchmark DB or a real order.
                messages.append({'role': 'tool', 'tool_call_id': calls[0]['id'],
                                 'content': json.dumps({'status': 'shipped', 'tracking_code': 'TRACK-42'})})
                final = call()
                row['used_tool_result'] = 'TRACK-42' in (final.get('content') or '') and 'shipped' in (final.get('content') or '').lower() and not final.get('tool_calls')
                assert row['used_tool_result'], 'Continuation did not use the tool result'
            row['raw_calls_sha256'] = digest(trace / 'calls.jsonl')
            save()
        report['status'] = 'completed'
    except Exception as exc:
        report.update(status='failed', error_type=type(exc).__name__)
        (args.trace / 'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        save()
        print(json.dumps({'status': report['status'], 'summary': str(args.output / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
