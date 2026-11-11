"""Exact state swaps isolate reconstruction drift from mixed-layer effects.

All target KV comes from a full target forward. None is a cheap online repair.
Original layer masks, both prefixes and all token positions are recorded.
"""
import argparse
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from fetch_stage_a_data import EVAL_SHA
from fetch_stage_a_models import sha256
from run_stage_a_development import load_cases
from stage_a_core import (assemble, capture, continuation_logits, distribution_metrics,
                         greedy, require_compatible, rebuild_interval, relative_error)


def layer_mask(layers, start, width, lower_target, upper_target):
    if not (0 <= start < start + width <= layers):
        raise ValueError('Invalid interval')
    return tuple(bool(lower_target) if i < start else
                 True if i < start + width else bool(upper_target) for i in range(layers))


def splice(source, target, mask):
    if len(mask) != len(target.kv) or any(type(v) is not bool for v in mask):
        raise ValueError('Mask must contain exactly one boolean per layer')
    p = target.prefix_length
    replacements = {i: (k[:, :, p:], v[:, :, p:])
                    for i, (k, v) in enumerate(target.kv) if mask[i]}
    return assemble(source, target, replacements)


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Never overwrite or automatically restart a run')
    protocol = json.loads(Path(args.protocol).read_text())
    manifest, cases = load_cases(args.data_manifest, args.cases, protocol['limit'], ['P2'])
    if [c['id'] for c in cases] != protocol['question_ids']:
        raise ValueError('Frozen question order differs')
    starts, width = protocol['starts'], protocol['width']
    torch.manual_seed(protocol['seed'])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model_manifest = json.loads(Path(args.model_manifest).read_text())['models']
    models, tokenizers = {}, {}
    for role, device in [('base', args.source_device), ('instruct', args.target_device)]:
        item = model_manifest[role]
        folder = Path(args.model_root) / item['revision']
        if json.loads((folder / 'verified_manifest.json').read_text()) != item:
            raise ValueError('Verified model manifest differs')
        models[role] = AutoModelForCausalLM.from_pretrained(
            folder, local_files_only=True, trust_remote_code=False,
            dtype=torch.bfloat16, attn_implementation='sdpa').to(device).eval()
        tokenizers[role] = AutoTokenizer.from_pretrained(folder, local_files_only=True)
        print(json.dumps({'loaded': role, 'device': device}), flush=True)
    a, b = models['base'], models['instruct']
    require_compatible(a, b)
    tok = tokenizers['instruct']
    vocab_hash = hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()
    if tok.get_vocab() != tokenizers['base'].get_vocab() or vocab_hash != manifest['tokenizer_vocab_sha256']:
        raise ValueError('Token vocabularies differ')
    if len(starts) != len(set(starts)) or any(s < 0 or s + width > len(b.model.layers) for s in starts):
        raise ValueError('Invalid intervals')
    if sha256(args.evaluator) != EVAL_SHA:
        raise ValueError('Official evaluator differs')
    spec = importlib.util.spec_from_file_location('hotpot_evaluator', args.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    stop = b.generation_config.eos_token_id
    stop = set(stop if isinstance(stop, list) else [stop])
    report = {
        'status': 'running', 'scope': 'Post-hoc same-process repetition of two observed exact/recomputed cases; not a quality benchmark',
        'protocol': protocol, 'protocol_sha256': sha256(args.protocol), 'settings': vars(args),
        'models': model_manifest, 'runtime': {
            'pid': os.getpid(), 'host': platform.node(), 'platform': platform.platform(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
        'source_sha256': {f: sha256(f) for f in [
            'stage_a_core.py', 'probe_stage_a_splice_rounding.py', 'run_stage_a_development.py',
            'fetch_stage_a_models.py', 'fetch_stage_a_data.py']},
        'data_manifest_sha256': sha256(args.data_manifest), 'cases': []}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(output)

    previous = json.loads(Path(args.previous).read_text())
    if previous['status'] != 'completed' or previous['protocol']['question_ids'] != protocol['question_ids']:
        raise ValueError('Previous factorial run is incomplete or incompatible')
    if previous['models'] != model_manifest:
        raise ValueError('Previous model pair differs')
    report['previous_sha256'] = sha256(args.previous)
    previous_rows = {r['id']: r for r in previous['cases']}
    save()
    selected = {'5ab915ad55429919ba4e239c', '5a89d36e554299515336132a'}
    report['audit_case_ids'] = sorted(selected)
    report['audit_start'] = 12
    report['repeats_per_variant'] = 3
    cases = [c for c in cases if c['id'] in selected]
    if {c['id'] for c in cases} != selected:
        raise ValueError('Missing predefined post-hoc cases')
    for case in cases:
        hs, ht, x, q = [case[k] for k in ('prefix_source_ids', 'prefix_target_ids', 'shared_ids', 'suffix_ids')]
        source = capture(a, hs, x, [])
        target = capture(b, ht, x, [12])
        rep = rebuild_interval(b, target, target.entries[12], 12, 4)
        exact = splice(source, target, layer_mask(28, 12, 4, False, False))
        reconstructed = assemble(source, target, rep)
        native_ids = greedy(b, target.kv, q, protocol['max_new_tokens'], stop)
        reference = continuation_logits(b, target.kv, q, native_ids)
        errors = [relative_error(got, full[:, :, len(ht):]) for i, pair in rep.items()
                  for got, full in zip(pair, target.kv[i])]
        results = {}
        for label, kv in [('exact', exact), ('recomputed', reconstructed)]:
            trials = []
            for repeat in range(3):
                ids = greedy(b, kv, q, protocol['max_new_tokens'], stop)
                logits = continuation_logits(b, kv, q, native_ids)
                text = tok.decode(ids, skip_special_tokens=True).strip()
                trials.append({'generated_ids': ids, 'prediction': text,
                               'f1': evaluator.f1_score(text, case['answer'])[0],
                               **distribution_metrics(logits, reference, native_ids)})
            results[label] = trials
        report['cases'].append({'id': case['id'], 'gold': case['answer'],
                                'max_interval_kv_relative_error': max(errors),
                                'same_generated_ids': results['exact'][0]['generated_ids'] == results['recomputed'][0]['generated_ids'],
                                'within_variant_identical': {k: all(v['generated_ids'] == trials[0]['generated_ids'] for v in trials) for k, trials in results.items()},
                                'variants': results})
        save()
        print(json.dumps({'case': case['id'], 'max_kv_error': max(errors), 'predictions': {k: v[0]['prediction'] for k, v in results.items()}}), flush=True)
    report['status'] = 'completed'
    save()
    print('ROUNDING_AUDIT_COMPLETED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', default='stage_a_splice_protocol.json')
    p.add_argument('--data-manifest', default='stage_a_data_manifest.json')
    p.add_argument('--model-manifest', default='stage_a_7b_models.json')
    p.add_argument('--cases', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--evaluator', required=True)
    p.add_argument('--previous', required=True)
    p.add_argument('--source-device', default='cuda:0')
    p.add_argument('--target-device', default='cuda:1')
    p.add_argument('--output', required=True)
    main(p.parse_args())
