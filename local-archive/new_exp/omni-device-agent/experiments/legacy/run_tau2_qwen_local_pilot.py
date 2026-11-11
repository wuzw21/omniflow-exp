"""Run unchanged tau2 interaction/evaluator with actual Qwen inference transports.

Agent: native MLX-LM HTTP server and its Qwen tool parser on physical Mac.
User simulator: actual Qwen 7B via the previously verified SSH/stdio worker.
Only completion transport is bound here; no new agent/user loop or gold actions.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import traceback
from types import SimpleNamespace

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
from check_tau2_native_environment import SOURCE, verify_source
from minions_model_worker import digest
from probe_minions_collaboration import Endpoint, SSH
from tau2_local_transport import native_server
from tau2_remote_preflight import remote_preflight

ROOT = Path(__file__).resolve().parent
AGENT_MODEL = 'openai/Qwen2.5-3B-Instruct-local'
USER_MODEL = 'edgecloud/Qwen2.5-7B-Instruct-user'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--task-id', default='66')
    p.add_argument('--remote-root', required=True, help='Existing remote experiment directory')
    p.add_argument('--remote-python', required=True, help='Pinned remote Python runtime')
    p.add_argument('--preflight-only', action='store_true')
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    args.trace.mkdir(parents=True, exist_ok=False)
    protocol = verify_source()
    assert args.task_id in {x['id'] for x in protocol['selected']}
    local_spec = json.loads((ROOT / 'results/papillon_overlap_003/local.spec.json').read_text())
    remote_spec = json.loads((ROOT / 'results/papillon_overlap_003/remote.spec.json').read_text())
    remote_spec['path'] = args.remote_root.rstrip('/') + '/models/' + remote_spec['revision']
    for name, expected in local_spec['weights'].items():
        assert digest(Path(local_spec['path']) / name) == expected
    for name, spec in [('local', local_spec), ('remote', remote_spec)]:
        (args.output / f'{name}.spec.json').write_text(json.dumps(spec, indent=2))
    scripts = {}
    for name in ['run_tau2_qwen_local_pilot.py', 'check_tau2_native_environment.py',
                 'probe_minions_collaboration.py', 'minions_model_worker.py', 'tau2_capability_protocol.json',
                 'tau2_local_transport.py', 'tau2_remote_preflight.py']:
        shutil.copy2(ROOT / name, args.output / name)
        scripts[name] = digest(args.output / name)
    report = {'status': 'initializing', 'scope': 'One actual native tau2 development task; capability pilot, no collaboration optimization',
              'pid': os.getpid(), 'task_id': args.task_id, 'scripts_sha256': scripts,
              'source_revision': protocol['source']['revision'], 'trace': str(args.trace),
              'model_roles': {'agent': local_spec['repo'], 'user_simulator': remote_spec['repo']},
              'settings': {'max_steps': 40, 'max_errors': 5, 'agent_max_tokens': 512,
                           'user_max_tokens': 768, 'seed': 20261024, 'temperature': 0,
                           'evaluation': 'ENV', 'enforce_communication_protocol': True,
                           'local_prefill_step_size': 512, 'local_prompt_cache_size': 4},
              'quality_generalization_permitted': False, 'speedup_claim_permitted': False,
              'calls': [], 'endpoints': {}}
    def save():
        (args.output / 'summary.json').write_text(json.dumps(report, indent=2))
    save()
    local_context = remote = None
    try:
        report['remote_preflight'] = remote_preflight(SSH, args.remote_root, args.remote_python, remote_spec)
        save()
        if report['remote_preflight']['status'] != 'ready':
            report['status'] = 'environment_unavailable'
            raise SystemExit(2)
        if args.preflight_only:
            report['status'] = 'preflight_ready'
            return
        from loguru import logger
        logger.remove()
        logger.add(str(args.trace / 'native_harness.log'), level='DEBUG')
        import litellm
        from litellm import ModelResponse, Usage
        from tau2.run import run_task, get_tasks
        from tau2.evaluator.evaluator import EvaluationType
        import tau2.utils.llm_utils as llm_utils
        report['runtime'] = {name: importlib.metadata.version(name) for name in ['tau2', 'litellm', 'pydantic']}
        report['endpoints']['local_server'] = {}
        local_context = native_server(local_spec['path'], args.trace, report['endpoints']['local_server'])
        base = local_context.__enter__()
        os.environ['NO_PROXY'] = os.environ['no_proxy'] = '127.0.0.1,localhost'
        save()
        remote_dir = args.remote_root.rstrip('/') + '/tau2-probe/' + args.output.name
        subprocess.run(SSH + ['mkdir -p ' + shlex.quote(remote_dir)], check=True)
        for src, dst in [(ROOT / 'minions_model_worker.py', 'worker.py'),
                         (args.output / 'remote.spec.json', 'spec.json')]:
            with src.open('rb') as handle:
                subprocess.run(SSH + ['cat > ' + shlex.quote(remote_dir + '/' + dst)], stdin=handle, check=True)
        remote_command = ['env', 'CUDA_VISIBLE_DEVICES=GPU-f3f0fa0c-05fc-93c6-b661-14d0457a2cda',
                          'TMPDIR=' + args.remote_root, args.remote_python, '-u',
                          remote_dir + '/worker.py', '--spec', remote_dir + '/spec.json']
        remote = Endpoint('remote', SSH + [shlex.join(remote_command)], args.trace,
                          SimpleNamespace, args.trace / 'user_worker_calls.jsonl')
        remote.phase = 'native_tau2_user_simulator'
        report['endpoints']['user_simulator'] = remote.ready
        report['status'] = 'running'
        save()
        original_completion = llm_utils.completion
        def completion(*, model, messages, tools=None, tool_choice=None, **kwargs):
            start = time.perf_counter()
            if model == AGENT_MODEL:
                result = original_completion(
                    model='openai/' + local_spec['path'], messages=messages,
                    tools=tools, tool_choice=tool_choice, api_base=base + '/v1',
                    api_key='local-self-hosted', temperature=0, max_tokens=512,
                    timeout=300, num_retries=0)
                role = 'agent'
            elif model == USER_MODEL:
                if tools:
                    raise ValueError('Retail user simulator must not use tools through text-only worker')
                texts, usage = remote.chat(messages)
                # This is an envelope around real generated tokens, not a mocked response.
                last = json.loads((args.trace / 'user_worker_calls.jsonl').read_text().splitlines()[-1])
                result = ModelResponse(model=model, choices=[{'index': 0,
                    'message': {'role': 'assistant', 'content': texts[0]},
                    'finish_reason': last['result']['finish_reason']}],
                    usage=Usage(prompt_tokens=usage.prompt_tokens, completion_tokens=usage.completion_tokens,
                                total_tokens=usage.prompt_tokens + usage.completion_tokens))
                role = 'user_simulator'
            else:
                raise ValueError('Unapproved external/model transport: ' + model)
            record = {'role': role, 'request': {'model': model, 'messages': messages, 'tools': tools},
                      'response': result.model_dump(), 'seconds': time.perf_counter() - start}
            with (args.trace / 'completion_calls.jsonl').open('a') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            summary = {'role': role, 'seconds': record['seconds'],
                       'input_tokens': result.usage.prompt_tokens, 'output_tokens': result.usage.completion_tokens,
                       'finish_reason': result.choices[0].finish_reason,
                       'tool_calls': [c.function.name for c in result.choices[0].message.tool_calls or []]}
            report['calls'].append(summary)
            save()
            print(json.dumps({'call': len(report['calls']), **summary}), flush=True)
            return result
        llm_utils.completion = completion
        task = get_tasks('retail', 'train', [args.task_id])[0]
        started = time.perf_counter()
        simulation = run_task(domain='retail', task=task, agent='llm_agent', user='user_simulator',
            llm_agent=AGENT_MODEL, llm_user=USER_MODEL,
            llm_args_agent={'temperature': 0, 'max_tokens': 512, 'num_retries': 0},
            llm_args_user={'temperature': 0, 'max_tokens': 768, 'num_retries': 0},
            max_steps=40, max_errors=5, evaluation_type=EvaluationType.ENV,
            seed=20261024, enforce_communication_protocol=True)
        (args.trace / 'simulation.json').write_text(simulation.model_dump_json(indent=2))
        report.update({'status': 'completed', 'simulation_seconds': time.perf_counter() - started,
                       'termination_reason': str(simulation.termination_reason),
                       'reward_info': simulation.reward_info.model_dump(),
                       'messages': len(simulation.messages),
                       'simulation_sha256': digest(args.trace / 'simulation.json')})
    except Exception as exc:
        report.update({'status': 'execution_failed', 'error_type': type(exc).__name__})
        (args.trace / 'exception.txt').write_text(traceback.format_exc())
        raise
    finally:
        if remote:
            try:
                remote.close()
            except Exception as exc:
                report.setdefault('cleanup_errors', []).append({'endpoint': 'remote', 'type': type(exc).__name__})
            report['remote_exit_code'] = remote.proc.poll()
        if local_context:
            local_context.__exit__(None, None, None)
            report['local_server_exit_code'] = report['endpoints']['local_server'].get('exit_code')
        save()
        print(json.dumps({'status': report['status'], 'task_id': args.task_id,
                          'result': str(args.output / 'summary.json')}), flush=True)


if __name__ == '__main__':
    main()
