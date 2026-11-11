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
                         greedy, require_compatible)


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
        'status': 'running', 'scope': 'Exact KV splice diagnostic; full forward oracles, no speed claim',
        'protocol': protocol, 'protocol_sha256': sha256(args.protocol), 'settings': vars(args),
        'models': model_manifest, 'runtime': {
            'pid': os.getpid(), 'host': platform.node(), 'platform': platform.platform(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
        'source_sha256': {f: sha256(f) for f in [
            'stage_a_core.py', 'run_stage_a_splice.py', 'run_stage_a_development.py',
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
    for case in cases:
        began = time.monotonic()
        hs, ht, x, q = [case[k] for k in ('prefix_source_ids', 'prefix_target_ids', 'shared_ids', 'suffix_ids')]
        if len(hs) != len(ht) or hs == ht:
            raise ValueError('Require different equal-length histories')
        s, t = capture(a, hs, x, []), capture(b, ht, x, [])
        layers = len(t.kv)
        native_ids = greedy(b, t.kv, q, protocol['max_new_tokens'], stop)
        reference = continuation_logits(b, t.kv, q, native_ids)
        old = previous_rows[case['id']]
        scored = {}

        def score(mask):
            key = ''.join('T' if v else 'S' for v in mask)
            if key in scored:
                return key
            kv = splice(s, t, mask)
            if all(mask):
                if not all(torch.equal(u, v) for pair, native in zip(kv, t.kv) for u, v in zip(pair, native)):
                    raise AssertionError('All-target splice differs from native KV')
                ids, logits = native_ids, reference
            else:
                ids = greedy(b, kv, q, protocol['max_new_tokens'], stop)
                logits = continuation_logits(b, kv, q, native_ids)
            prediction = tok.decode(ids, skip_special_tokens=True).strip()
            scored[key] = {'target_layers': sum(mask), 'prediction': prediction, 'generated_ids': ids,
                           'tokens': len(ids), 'finish_reason': 'stop' if ids[-1] in stop else 'length',
                           'f1': evaluator.f1_score(prediction, case['answer'])[0],
                           'em': float(evaluator.exact_match_score(prediction, case['answer'])),
                           **distribution_metrics(logits, reference, native_ids)}
            return key

        native_key, source_key = score((True,) * layers), score((False,) * layers)
        row = {'id': case['id'], 'split': 'dev', 'condition': 'P2', 'gold_answer': case['answer'],
               'prefix_tokens': len(ht), 'shared_tokens': len(x), 'layers': layers,
               'input_sha256': hashlib.sha256(json.dumps([hs, ht, x, q]).encode()).hexdigest(),
               'native_key': native_key, 'source_key': source_key, 'scored_masks': scored,
               'baseline_reproduction': {
                   'native_generated_ids_equal': native_ids == old['cache_factorial']['Bt']['generated_ids'],
                   'source_generated_ids_equal': scored[source_key]['generated_ids'] == old['cache_factorial']['As']['generated_ids']},
               'intervals': [], 'status': 'running'}
        if row['input_sha256'] != old['input_sha256']:
            raise ValueError('Previous actual inputs differ')
        report['cases'].append(row)
        save()
        for start in starts:
            variants = {}
            for label, lower, upper in [('source_both', False, False), ('target_lower', True, False),
                                         ('target_upper', False, True), ('target_both', True, True)]:
                variants[label] = score(layer_mask(layers, start, width, lower, upper))
            exact = scored[variants['source_both']]
            recomputed = next(i['scores']['joint_oracle'] for i in old['intervals'] if i['start'] == start)
            row['intervals'].append({
                'start': start, 'width': width, 'variants': variants,
                'exact_vs_previous_recomputed': {
                    'same_generated_ids': exact['generated_ids'] == recomputed['generated_ids'],
                    'f1_delta': exact['f1'] - recomputed['f1'],
                    'previous_prediction': recomputed['prediction'], 'previous_f1': recomputed['f1']}})
            save()
        row['status'] = 'completed'
        row['unique_masks'] = len(scored)
        row['wall_seconds_not_a_benchmark'] = time.monotonic() - began
        save()
        print(json.dumps({'case': len(report['cases']), 'id': case['id'], 'unique_masks': len(scored),
                          'baseline_reproduction': row['baseline_reproduction']}), flush=True)
        del s, t, reference
        gc.collect()
    report['status'] = 'completed'
    report['interpretation'] = 'Exact KV swaps from full forwards; interventions replacing extra layers are not equal-compute alternatives.'
    save()
    print('SPLICE_COMPLETED', flush=True)


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
