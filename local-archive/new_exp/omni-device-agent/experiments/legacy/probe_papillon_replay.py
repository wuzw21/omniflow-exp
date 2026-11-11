"""Replay real archived PAPILLON cloud replies; compare native local controls."""
import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--trace', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.trace.mkdir(parents=True, exist_ok=False)
    here = Path(__file__).resolve().parent
    source = Path.home() / '.agent-reach/research/papillon-594e7eb'
    manifest = json.loads((source / 'manifest.json').read_text())
    for item in manifest['files']:
        assert digest(source / item['path']) == item['sha256']
    import dspy
    spec = importlib.util.spec_from_file_location('pinned_papillon', source / 'papillon/run_llama_dspy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    signature = module.PAPILLON(None).info_aggregator.signature
    adapters = {'chat': dspy.ChatAdapter(), 'json': dspy.JSONAdapter()}
    protocol = json.loads((here / 'papillon_overlap_protocol.json').read_text())
    csv_path = source / 'pupa' / protocol['file']
    assert digest(csv_path) == protocol['file_sha256']
    with csv_path.open() as handle:
        dataset = list(csv.DictReader(handle))
    cases, provenance = [], []
    # Long-case batch-boundary diagnostic first, then median formatting control.
    for label, number, selected in [('p95', '003', protocol['cases'][1]),
                                    ('median', '002', protocol['cases'][0])]:
        archived = args.trace.parent / ('overlap-' + number)
        baseline_path = archived / 'baseline.json'
        calls_path = archived / 'calls.jsonl'
        baseline = json.loads(baseline_path.read_text())
        calls = [json.loads(x) for x in calls_path.read_text().splitlines()]
        query = dataset[selected['row']]['user_query']
        assert hashlib.sha256(query.encode()).hexdigest() == selected['query_sha256']
        inputs = {'userQuery': query, 'modelExampleResponses': baseline['gptResponse']}
        messages = adapters['chat'].format(signature, [], inputs)
        assert messages == calls[2]['messages']
        cases.append({'case': label, 'chat': messages,
                      'json': adapters['json'].format(signature, [], inputs),
                      'empty': adapters['chat'].format(signature, [], {**inputs, 'modelExampleResponses': ''}),
                      'archived_input_ids': calls[2]['result']['input_ids'],
                      'archived_generated_ids': calls[2]['result']['generated_ids']})
        provenance.append({'case': label, 'selected': selected,
                           'baseline_path': str(baseline_path), 'baseline_sha256': digest(baseline_path),
                           'calls_path': str(calls_path), 'calls_sha256': digest(calls_path),
                           'cloud_finish_reason': calls[1]['result']['finish_reason']})
    raw_input = args.trace / 'input.json'
    raw_output = args.trace / 'output.json'
    raw_input.write_text(json.dumps({'cases': cases}, indent=2))
    scripts = {}
    for name in ['probe_papillon_replay.py', 'papillon_replay_mlx.py', 'minions_model_worker.py']:
        shutil.copy2(here / name, args.output / name)
        scripts[name] = digest(args.output / name)
    shutil.copy2(here / 'results/papillon_overlap_003/local.spec.json', args.output / 'local.spec.json')
    report = {'status': 'running', 'scope': 'Local replay of actual archived cloud responses; no new cloud calls',
              'pid': os.getpid(), 'source': manifest, 'dspy': dspy.__version__,
              'provenance': provenance, 'scripts_sha256': scripts,
              'trace': str(args.trace), 'speedup_claim_permitted': False,
              'quality_evaluated': False,
              'hardware': subprocess.check_output(['system_profiler', 'SPHardwareDataType', '-json'], text=True)}
    # Avoid serial numbers/UUIDs from system_profiler in retained metadata.
    hardware = json.loads(report.pop('hardware'))['SPHardwareDataType'][0]
    report['hardware'] = {k: hardware[k] for k in ['chip_type', 'physical_memory', 'machine_model'] if k in hardware}
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    command = [str(here / '.venv-minions-mlx/bin/python'), str(args.output / 'papillon_replay_mlx.py'),
               '--spec', str(args.output / 'local.spec.json'), '--input', str(raw_input), '--output', str(raw_output)]
    with (args.output / 'worker.log').open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    report['worker_exit_code'] = result.returncode
    if result.returncode:
        report['status'] = 'failed'
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
        raise RuntimeError('Native replay failed; see worker.log. Do not restart a live job.')
    raw = json.loads(raw_output.read_text())
    report['mlx'] = raw['mlx']
    report['mlx_lm'] = raw['mlx_lm']
    report['worker_pid'] = raw['pid']
    report['runs'] = []
    for row in raw['runs']:
        cleaned = {k: v for k, v in row.items() if k not in ('text', 'generated_ids')}
        adapter = adapters['json' if row['mode'] == 'json' else 'chat']
        try:
            parsed = adapter.parse(signature, row['text'])['finalOutput']
            cleaned.update({'parsed': True, 'final_chars': len(parsed),
                            'final_sha256': hashlib.sha256(parsed.encode()).hexdigest()})
        except Exception as exc:
            cleaned.update({'parsed': False, 'parse_error_type': type(exc).__name__})
        cleaned['begins_with_input_field'] = row['text'].lstrip().startswith('[[ ## userQuery ## ]]')
        cleaned['raw_contains_input_field_markers'] = any(x in row['text'] for x in
                                                        ['[[ ## userQuery ## ]]', '[[ ## modelExampleResponses ## ]]'])
        report['runs'].append(cleaned)
    report['raw_input_sha256'] = digest(raw_input)
    report['raw_output_sha256'] = digest(raw_output)
    report['status'] = raw['status']
    (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({'status': report['status'], 'worker_exit_code': result.returncode,
                      'runs': len(report['runs']), 'result': str(args.output / 'summary.json')}))


if __name__ == '__main__':
    main()
