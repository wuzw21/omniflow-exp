"""Fixed-target 2x2 cache/entry diagnostic. Oracles are NOT online methods.

Uses existing stage-A operators and official Transformers decoder blocks.
All cache interventions keep Bt's prefix, shared tokens, positions and suffix.
No adapter training, network transfer, compression, or agent runtime.
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
from stage_a_core import (additive_entry, assemble, capture, continuation_logits,
                         distribution_metrics, greedy, rebuild_interval,
                         relative_error, require_compatible)


ENTRY_NAMES = ('source', 'context_oracle', 'model_oracle', 'additive_oracle', 'joint_oracle')


def entry_variants(states, start):
    a_s, a_t, b_s, b_t = [states[k].entries[start] for k in ('As', 'At', 'Bs', 'Bt')]
    return dict(zip(ENTRY_NAMES, (a_s, a_t, b_s, additive_entry(a_s, a_t, b_s), b_t)))


def factorial_interaction(scores, metric):
    """Signed paired interaction; not a claim about learned mapping methods."""
    return scores['As'][metric] - scores['At'][metric] - scores['Bs'][metric] + scores['Bt'][metric]


def select_shard(cases, protocol):
    """Partition the fixed development order without duplicating or selecting by quality."""
    if [c['id'] for c in cases] != protocol['question_ids']:
        raise ValueError('Frozen question order differs')
    shard = protocol.get('shard', {'index': 0, 'count': 1})
    index, count = shard['index'], shard['count']
    if type(index) is not int or type(count) is not int or not 0 <= index < count <= len(cases):
        raise ValueError('Invalid shard')
    return cases[index::count]


def main(args):
    output = Path(args.output)
    if output.exists():
        raise FileExistsError('Never overwrite or automatically restart a run')
    protocol = json.loads(Path(args.protocol).read_text())
    manifest, cases = load_cases(args.data_manifest, args.cases, protocol['limit'], ['P2'])
    cases = select_shard(cases, protocol)
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
        'status': 'running', 'scope': 'Fixed-target development diagnostic; no held-out or speed claim',
        'protocol': protocol, 'protocol_sha256': sha256(args.protocol), 'settings': vars(args),
        'models': model_manifest, 'runtime': {
            'pid': os.getpid(), 'host': platform.node(), 'platform': platform.platform(),
            'python': platform.python_version(), 'torch': torch.__version__,
            'transformers': transformers.__version__, 'cuda': torch.version.cuda,
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
        'source_sha256': {f: sha256(f) for f in [
            'stage_a_core.py', 'run_stage_a_factorial.py', 'run_stage_a_development.py',
            'fetch_stage_a_models.py', 'fetch_stage_a_data.py']},
        'data_manifest_sha256': sha256(args.data_manifest), 'cases': []}
    output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
        temporary.replace(output)

    save()
    for case in cases:
        began = time.monotonic()
        hs, ht, x, q = [case[k] for k in ('prefix_source_ids', 'prefix_target_ids', 'shared_ids', 'suffix_ids')]
        if len(hs) != len(ht) or hs == ht:
            raise ValueError('This diagnostic requires differing equal-length P2 prefixes')
        states = {key: capture(model, prefix, x, starts)
                  for key, model, prefix in [('As', a, hs), ('At', a, ht), ('Bs', b, hs), ('Bt', b, ht)]}
        s, t = states['As'], states['Bt']
        with torch.inference_mode():
            embedding = b.get_input_embeddings()(torch.tensor([x], device=b.device)).cpu()
        if not torch.equal(embedding, t.entries[0]):
            raise AssertionError('Layer-zero target entry differs from embedding lookup')
        native_ids = greedy(b, t.kv, q, protocol['max_new_tokens'], stop)
        reference = continuation_logits(b, t.kv, q, native_ids)
        repeated = continuation_logits(b, t.kv, q, native_ids)
        if not torch.equal(reference, repeated):
            raise AssertionError('Repeated native continuation changed')
        with torch.inference_mode():
            dense = b(torch.tensor([ht + x + q + native_ids[:-1]], device=b.device),
                      logits_to_keep=len(native_ids)).logits[0].float().cpu()

        def score(kv, native=False):
            ids = native_ids if native else greedy(b, kv, q, protocol['max_new_tokens'], stop)
            logits = reference if native else continuation_logits(b, kv, q, native_ids)
            prediction = tok.decode(ids, skip_special_tokens=True).strip()
            return {'prediction': prediction, 'generated_ids': ids, 'tokens': len(ids),
                    'finish_reason': 'stop' if ids[-1] in stop else 'length',
                    'f1': evaluator.f1_score(prediction, case['answer'])[0],
                    'em': float(evaluator.exact_match_score(prediction, case['answer'])),
                    **distribution_metrics(logits, reference, native_ids)}

        # Only this separate H0 control removes Ht from both prefill and decode.
        direct = {key: score(assemble(state, t), key == 'Bt') for key, state in states.items()}
        row = {'id': case['id'], 'condition': 'P2', 'split': 'dev', 'type': case['type'],
               'gold_answer': case['answer'], 'prefix_tokens': len(ht), 'shared_tokens': len(x),
               'input_sha256': hashlib.sha256(json.dumps([hs, ht, x, q]).encode()).hexdigest(),
               'cache_factorial': direct, 'native_source_history': score(states['Bs'].kv),
               'calibration': {'split_dense_relative_logit_error': relative_error(reference, dense),
                               **distribution_metrics(reference, dense, native_ids)},
               'factorial_interaction': {m: factorial_interaction(direct, m)
                                        for m in ('f1', 'em', 'kl_reference_to_variant')},
               'intervals': [], 'status': 'running'}
        report['cases'].append(row)
        save()
        for start in starts:
            entries = entry_variants(states, start)
            if start == 0:
                entries['joint_oracle'] = embedding
                if not torch.equal(entries['source'], entries['context_oracle']):
                    raise AssertionError('Layer-zero A entry depends on context')
                if not all(torch.equal(entries[k], embedding) for k in ('model_oracle', 'additive_oracle')):
                    raise AssertionError('Layer-zero target entry identities failed')
            results, diagnostics = {}, {}
            for label, entry in entries.items():
                rep = rebuild_interval(b, t, entry, start, width)
                results[label] = score(assemble(s, t, rep))
                errors = [relative_error(got, full[:, :, len(ht):])
                          for layer, pair in rep.items() for got, full in zip(pair, t.kv[layer])]
                results[label]['interval_kv_error_max'] = max(errors)
                if label == 'joint_oracle' and max(errors) > 0.03:
                    raise AssertionError('Native-entry interval reconstruction error exceeds 3%')
                if label in ('source', 'joint_oracle'):
                    # Isolate interval damage from stale KV outside the interval.
                    diagnostics[label + '_native_outside'] = score(assemble(s, t, rep, native_outside=True))
            if start == 0:
                if results['source'] != results['context_oracle'] or not (
                    results['model_oracle'] == results['additive_oracle'] == results['joint_oracle']):
                    raise AssertionError('Equal layer-zero entries produced different outcomes')
            if protocol.get('exact_joint_diagnostic', False):
                # Native target state, not the numerically approximate reconstruction.
                exact = {i: (k[:, :, len(ht):], v[:, :, len(ht):])
                         for i, (k, v) in enumerate(t.kv) if start <= i < start + width}
                diagnostics['exact_joint_source_outside'] = score(assemble(s, t, exact))
            row['intervals'].append({
                'start': start, 'width': width, 'scores': results, 'diagnostics': diagnostics,
                'entry_error': {label: relative_error(entry, entries['joint_oracle']) for label, entry in entries.items()},
                'joint_entry_provenance': 'target_embedding_lookup' if start == 0 else 'full_target_forward_oracle'})
            save()
            print(json.dumps({'case': len(report['cases']), 'id': case['id'], 'start': start,
                              'elapsed_seconds': time.monotonic() - began}), flush=True)
        row['status'] = 'completed'
        row['wall_seconds_not_a_benchmark'] = time.monotonic() - began
        save()
        del states, s, t, entries, reference, repeated, dense
        gc.collect()
    report['status'] = 'completed'
    report['interpretation'] = 'Development diagnostic only; oracles require unavailable full forwards. No mapping was trained.'
    save()
    print('FACTORIAL_COMPLETED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', default='stage_a_factorial_protocol.json')
    p.add_argument('--data-manifest', default='stage_a_data_manifest.json')
    p.add_argument('--model-manifest', default='stage_a_7b_models.json')
    p.add_argument('--cases', required=True)
    p.add_argument('--model-root', required=True)
    p.add_argument('--evaluator', required=True)
    p.add_argument('--source-device', default='cuda:0')
    p.add_argument('--target-device', default='cuda:1')
    p.add_argument('--output', required=True)
    main(p.parse_args())
