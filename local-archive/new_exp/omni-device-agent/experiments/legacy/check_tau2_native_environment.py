"""Native sandbox/database self-checks. Reference replay is NOT agent success."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'
os.environ['PYTHON_DOTENV_DISABLED'] = '1'
SOURCE = Path.home() / '.agent-reach/research/tau2-verified-checkout'
os.environ['TAU2_DATA_DIR'] = str(SOURCE / 'data')


def verify_source():
    rev = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    assert rev == '864350a8971a8f8ee9e7b8472e2edc380a806b0c'
    subprocess.run(['git', '-C', str(SOURCE), 'diff', '--quiet', 'HEAD', '--'], check=True)
    untracked = subprocess.check_output(['git', '-C', str(SOURCE), 'ls-files', '--others', '--exclude-standard'], text=True).splitlines()
    assert all('__pycache__' in Path(name).parts and name.endswith('.pyc') for name in untracked), untracked
    protocol = json.loads((Path(__file__).parent / 'tau2_capability_protocol.json').read_text())
    for item in protocol['source']['files']:
        assert hashlib.sha256((SOURCE / item['path']).read_bytes()).hexdigest() == item['sha256']
    return protocol


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    protocol = verify_source()
    from loguru import logger
    import sys
    logger.remove()
    logger.add(sys.stderr, level='WARNING')
    from tau2.registry import registry
    from tau2.data_model.message import AssistantMessage, UserMessage, ToolCall
    from tau2.evaluator.evaluator_env import EnvironmentEvaluator
    constructor = registry.get_env_constructor('retail')
    tasks = {t.id: t for t in registry.get_tasks_loader('retail')(task_split_name='train')}
    report = {'scope': 'Native environment/evaluator self-check; reference actions, no model or agent run',
              'source_revision': protocol['source']['revision'], 'cases': []}
    for selected in protocol['selected']:
        task = tasks[selected['id']]
        init = task.initial_state
        history = list(init.message_history or []) if init else []
        empty = EnvironmentEvaluator.calculate_reward(constructor, task, history)
        env = constructor()
        env.set_state(init.initialization_data if init else None,
                      init.initialization_actions if init else None, history)
        errors = []
        for index, action in enumerate(task.evaluation_criteria.actions):
            call = ToolCall(id=f'reference_{index}', name=action.name,
                            arguments=action.arguments, requestor=action.requestor)
            response = env.get_response(call)
            message_cls = AssistantMessage if action.requestor == 'assistant' else UserMessage
            history.extend([message_cls(role=action.requestor, tool_calls=[call]), response])
            errors.append(response.error)
        reference = EnvironmentEvaluator.calculate_reward(constructor, task, history)
        row = {'task_id': task.id, 'noop_reward': empty.reward,
               'noop_db_match': empty.db_check.db_match,
               'reference_reward': reference.reward, 'reference_db_match': reference.db_check.db_match,
               'reference_action_count': len(task.evaluation_criteria.actions),
               'reference_tool_errors': errors}
        report['cases'].append(row)
        args.output.write_text(json.dumps(report, indent=2))
        print(json.dumps(row), flush=True)
        assert empty.reward == 0 and not empty.db_check.db_match
        assert reference.reward == 1 and reference.db_check.db_match and not any(errors)
    report['status'] = 'completed'
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
