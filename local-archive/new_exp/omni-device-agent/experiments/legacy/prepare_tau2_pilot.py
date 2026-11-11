"""Freeze a development-only task subset before model execution; no task solving."""
import argparse
import hashlib
import json
from pathlib import Path

SEED = 'edgecloud-tau2-capability-20261024'
FAMILIES = ['cancel_pending_order', 'return_delivered_order_items',
            'exchange_delivered_order_items', 'modify_pending_order_items']


def build(source):
    manifest = json.loads((source / 'manifest.json').read_text())
    assert manifest['revision'] == '864350a8971a8f8ee9e7b8472e2edc380a806b0c'
    for item in manifest['files']:
        assert hashlib.sha256((source / item['path']).read_bytes()).hexdigest() == item['sha256']
    directory = source / 'data/tau2/domains/retail'
    rows = json.loads((directory / 'tasks.json').read_text())
    split = json.loads((directory / 'split_tasks.json').read_text())
    assert len({r['id'] for r in rows}) == len(rows)
    assert set(split['train']).isdisjoint(split['test'])
    assert set(split['base']) == {r['id'] for r in rows}
    selected, eligibility = [], {}
    for family in FAMILIES:
        candidates = []
        for row in rows:
            criteria = row['evaluation_criteria']
            names = [a['name'] for a in criteria.get('actions', [])]
            if row['id'] not in split['train'] or criteria.get('communicate_info') or criteria.get('nl_assertions'):
                continue
            if set(names) & set(FAMILIES) != {family}:
                continue
            candidates.append(row)
        eligibility[family] = len(candidates)
        assert candidates, family
        chosen = min(candidates, key=lambda r: hashlib.sha256(f"{SEED}:{r['id']}".encode()).hexdigest())
        selected.append({'id': chosen['id'], 'family': family,
                         'task_sha256': hashlib.sha256(json.dumps(chosen, sort_keys=True).encode()).hexdigest(),
                         'gold_action_count_for_selection_only': len(chosen['evaluation_criteria']['actions'])})
    return {'status': 'protocol_frozen_not_executed', 'source': manifest,
            'counts': {'total': len(rows), **{k: len(v) for k, v in split.items()}},
            'seed': SEED, 'eligibility': eligibility, 'selected': selected,
            'selection': 'One train task per listed mutation family, no communication/NL grading requirements; SHA256(seed:id) minimum. Other mutation types may coexist; gold actions are not agent inputs.',
            'native_user_protocol': 'Keep original user simulator, policy, confirmation requirements and messages. Do not expose the full hidden user_scenario to the agent or strip confirmation clauses.',
            'evaluation': 'Use pinned native environment/database evaluator with original termination checks; separately retain exact action/policy/confirmation outcomes. Never overwrite DB mismatch using LLM judgment. No real-world transactions.',
            'staged_execution': [
                'First validate native Qwen2.5-3B local-only and Qwen2.5-7B server-only capability under the same native harness and user protocol.',
                'Server-only full-observation run is a capability reference, not privacy-compliant deployment. Keep public synthetic data separate from user data.',
                'Only after reliable task completion, compare policy-aware structured filtering/placeholder registry + cloud against local semantic processing + cloud; retain exact failures and per-call costs.',
                'Measure native per-end prefix reuse, shared token spans, serial local processing, network bytes and outputs. Do not train a KV translator unless actual reusable work survives these baselines.'
            ],
            'local_necessity_unproven': True, 'edge_cloud_speedup_unproven': True,
            'no_model_outputs_used_for_selection': True, 'held_out_test_touched_by_training': False,
            'raw_gold_inputs_copied_into_protocol': False}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert not args.output.exists()
    result = build(args.source)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: result[k] for k in ['status', 'counts', 'eligibility', 'selected']}, indent=2))


if __name__ == '__main__':
    main()
