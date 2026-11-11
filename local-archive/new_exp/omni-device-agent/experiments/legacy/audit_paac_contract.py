"""Execute pinned PAAC registry primitives; inspect scheduling/evaluation AST.

Controlled proposal inputs are supplied directly to deterministic operators.
No LLM response, agent trace, task success, or deployment is simulated.
"""
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import re
import subprocess


REVISION = 'f9298318bf169a5432a0bb563f54b10f527ff9db'


def get_method(tree, class_name, name):
    owner = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    return next(n for n in owner.body if isinstance(n, ast.FunctionDef) and n.name == name)


def audit(source):
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['revision'] == REVISION
    for item in manifest['files']:
        assert hashlib.sha256((source / item['path']).read_bytes()).hexdigest() == item['sha256']
    text = (source / 'src/main.py').read_text()
    tree = ast.parse(text)
    spec = importlib.util.spec_from_file_location('paac_pinned_config', source / 'src/config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'PrivacySanitizer')
    namespace = {'re': re, 'json': json, 'config': config}
    # Execute the exact unmodified class AST, without importing main's tool/client
    # stack. The exercised methods use only re/json/config, never an LM wrapper.
    module = ast.Module(body=[node], type_ignores=[])
    exec(compile(module, str(source / 'src/main.py'), 'exec'), namespace)
    cls = namespace['PrivacySanitizer']
    outcomes = []
    for case, content, mappings, proposed in [
        ('old_value_omitted_from_new_proposal', 'Ada needs a receipt.', {}, 'Ada needs a receipt.'),
        ('new_value_present_old_value_omitted', 'Ada booked Paris.',
         {'LOCATION_CITY_1': 'Paris'}, 'Ada booked LOCATION_CITY_1.'),
        ('old_value_proposed_again', 'Ada needs a receipt.',
         {'NAME_FIRST_7': 'Ada'}, 'NAME_FIRST_7 needs a receipt.'),
    ]:
        sanitizer = cls(2, 'tau2_retail')
        sanitizer.update_mapping({'NAME_FIRST_1': 'Ada'})
        before = sanitizer.get_mapping()
        aligned = sanitizer._verify_alignment(content, proposed, mappings)
        result = sanitizer._apply_mappings(content, proposed, mappings, aligned)
        old_registry_applied = sanitizer.sanitize(content)
        outcomes.append({'case': case, 'controlled_content': content,
                         'controlled_proposed_mapping': mappings,
                         'alignment_passed': aligned, 'registry_before': before,
                         'registry_after': sanitizer.get_mapping(),
                         'proposal_path_output': result,
                         'explicit_registry_sanitize_output': old_registry_applied,
                         'old_value_remains_in_proposal_path': 'Ada' in result,
                         'old_value_remains_in_explicit_registry_path': 'Ada' in old_registry_applied})

    run = get_method(tree, 'AgentStrategy', 'run')
    sanitizer = get_method(tree, 'AgentStrategy', 'device_sanitize')
    call_sites = []
    for n in ast.walk(run):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr in (
                'device_judgment', 'tau2_device_judgment', 'device_sanitize'):
            call_sites.append({'line': n.lineno, 'expression': ast.unparse(n)})
    dispatch = [{'line': n.lineno, 'expression': ast.unparse(n)} for n in ast.walk(sanitizer)
                if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                and n.func.attr in ('sanitize', 'generate_sanitized_content')]
    utils_tree = ast.parse((source / 'src/utils.py').read_text())
    evaluator = next(n for n in utils_tree.body if isinstance(n, ast.FunctionDef)
                     and n.name == '_process_tau2_final_answer')
    judge_sites = [{'line': n.lineno, 'expression': ast.unparse(n)} for n in ast.walk(evaluator)
                   if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                   and n.func.id == 'evaluate_result']
    tree_manifest = json.loads((source / 'tree.json').read_text())
    licenses = [n['path'] for n in tree_manifest['tree']
                if Path(n['path']).name.lower().startswith(('license', 'copying'))]
    return {'scope': 'Pinned deterministic operator experiment and static call-site audit; no LLM or agent run',
            'source': manifest, 'source_tree_license_files': licenses,
            'python': platform.python_version(), 'platform': platform.platform(),
            'chip': subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip(),
            'operator_cases': outcomes, 'run_call_sites': sorted(call_sites, key=lambda x: x['line']),
            'sanitizer_dispatch': dispatch, 'tau2_evaluator_override_calls': judge_sites,
            'scientific_limits': {'observed_real_llm_omission_rate': None,
                                  'task_quality_measured': False, 'latency_measured': False,
                                  'full_pipeline_leakage_measured': False,
                                  'upstream_modified': False, 'bug_fix_acceptance': False}}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    result = audit(args.source)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ('scope', 'chip', 'operator_cases', 'run_call_sites',
                                            'sanitizer_dispatch', 'tau2_evaluator_override_calls')}, indent=2))


if __name__ == '__main__':
    main()
