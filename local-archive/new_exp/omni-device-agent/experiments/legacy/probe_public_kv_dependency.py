"""Synthetic canary check of public-only KV export, using existing stage-A code.

No real secrets, remote inference, learned attack, defense or agent framework.
An eight-candidate fingerprint check assumes known model/template/public text.
It tests distinguishability, NOT arbitrary private prompt reconstruction.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from fetch_stage_a_models import CACHE, MODELS, sha256
from stage_a_core import capture


CORE_SHA = 'f8546e0de169a3e8a42c088607812cad330d17fda8e798943d9fae438566bb6a'
PREFIX_TOKENS = 96
PUBLIC_TOKENS = 128
LABELS = [str(i) for i in range(8)]
PUBLIC_TEXTS = {
    'build_notes': ('The public build guide describes a reproducible workspace. '
                    'Source files are checked before compilation. The build command writes '
                    'artifacts into a separate directory. A failed check returns a diagnostic '
                    'message. Completed artifacts can be inspected without changing the source. '),
    'device_guide': ('The public device guide describes routine maintenance. '
                     'Disconnect external accessories before cleaning the enclosure. '
                     'Use a soft dry cloth and keep the ventilation openings clear. '
                     'The status display reports the current operating mode. '
                     'Read the troubleshooting section when a warning remains visible. '),
}


def tensor_bytes(value):
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def public_slice(snapshot, count, layer, last_only=False):
    if count <= 0 or count > snapshot.shared_length:
        raise ValueError('Public slice exceeds shared segment')
    p = snapshot.prefix_length
    start = p + count - 1 if last_only else p
    k, v = snapshot.kv[layer]
    return k[:, :, start:p+count].clone(), v[:, :, start:p+count].clone()


def fingerprint(values):
    h = hashlib.sha256()
    for value in values:
        h.update(str((tuple(value.shape), str(value.dtype))).encode())
        h.update(tensor_bytes(value))
    return h.hexdigest()


def candidate_matches(reference, packet_fingerprint):
    return [label for label, digest in reference.items() if digest == packet_fingerprint]


def padded(tokenizer, text, count):
    ids = tokenizer.encode(text, add_special_tokens=False)
    filler = tokenizer.encode('\nWorkspace context is unchanged.', add_special_tokens=False)
    if len(ids) > count or not filler:
        raise ValueError('Prefix too long or filler empty')
    return (ids + filler * (count+1))[:count]


def summarize(records, layers):
    return {str(layer): {field: len({r['layers'][str(layer)][field] for r in records})
                        for field in ('kv_public_digest','kv_last_public_digest','entry_public_digest')}
            for layer in layers}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=['mps','cpu'], default='mps')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Preserve existing evidence')
    if sha256('stage_a_core.py') != CORE_SHA:
        raise ValueError('Existing capture implementation changed')
    repo, revision, weight_sha = MODELS['instruct']
    model_path = CACHE/'models'/revision
    if sha256(model_path/'model.safetensors') != weight_sha:
        raise ValueError('Weight checksum mismatch')
    torch.manual_seed(20261024)
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(model_path,local_files_only=True)
    label_ids = {label: tokenizer.encode(label,add_special_tokens=False) for label in LABELS}
    if any(len(ids)!=1 for ids in label_ids.values()) or len({ids[0] for ids in label_ids.values()})!=len(LABELS):
        raise ValueError('Canaries must be distinct single tokens')
    neutral = padded(tokenizer,'Workspace note: the internal routing label is unspecified.\n',PREFIX_TOKENS)
    prefixes = {label:padded(tokenizer,f'Private workspace note: the internal routing label is {label}.\n',PREFIX_TOKENS)
                for label in LABELS}
    # Every private-before prefix differs in exactly the canary token, never length.
    base = prefixes[LABELS[0]]
    for label in LABELS[1:]:
        if sum(a!=b for a,b in zip(base,prefixes[label]))!=1:
            raise ValueError('Canary altered other prefix tokens')
    public_ids = {name:tokenizer.encode(text*30,add_special_tokens=False)[:PUBLIC_TOKENS]
                  for name,text in PUBLIC_TEXTS.items()}
    if any(set(ids)&{v[0] for v in label_ids.values()} for ids in public_ids.values()):
        raise ValueError('Public text contains a canary token')
    suffixes = {label:padded(tokenizer,f'\nPrivate workspace label: {label}.\n',32) for label in LABELS}
    protocol = {'model':repo,'revision':revision,'labels':LABELS,'public_texts':PUBLIC_TEXTS,
                'prefix_tokens':PREFIX_TOKENS,'public_tokens':PUBLIC_TOKENS,
                'conditions':['private_before_public','neutral_before_public','public_before_private'],
                'challenge_order':list(reversed(LABELS)),
                'criterion':'Count byte-distinct public states; candidate identification is conditional on known template and source model',
                'not_claimed':['arbitrary secret recovery','security proof','new attack','cross-model transfer','performance improvement']}
    protocol_path = args.output.with_suffix('.protocol.json')
    if protocol_path.exists():
        raise FileExistsError('Preserve existing protocol')
    protocol_path.write_text(json.dumps(protocol,indent=2)+'\n')
    result = {'status':'running','protocol_sha256':sha256(protocol_path),
              'source_sha256':sha256(__file__),'capture_sha256':CORE_SHA,
              'model':{'repo':repo,'revision':revision,'weight_sha256':weight_sha},
              'runtime':{'python':platform.python_version(),'torch':torch.__version__,
                         'transformers':transformers.__version__,'device':args.device,'dtype':'bfloat16',
                         'platform':platform.platform()},
              'documents':{},'completed_forwards':0,
              'limits':['Only synthetic three-bit canaries, two fixed synthetic public passages',
                        'Candidate attacker knows model, exact prefix template and public text',
                        'Exact fingerprints assume same runtime/dtype; no robustness claim',
                        'Public suffix K/V rows only; no private prefix rows or private token IDs exported',
                        'Entry E is separately diagnosed; candidate matcher uses K/V only',
                        'Same-model local test; no cloud API, transport or task accuracy experiment',
                        'Before/after reordering changes prompt semantics; not a validated defense']}
    def save():
        tmp=args.output.with_suffix('.tmp')
        tmp.write_text(json.dumps(result,indent=2)+'\n')
        tmp.replace(args.output)
    save()
    began=time.monotonic()
    try:
        model=AutoModelForCausalLM.from_pretrained(model_path,local_files_only=True,
                  trust_remote_code=False,dtype=torch.bfloat16,attn_implementation='sdpa').to(args.device).eval()
        layers=list(range(model.config.num_hidden_layers))
        result['layers']=len(layers)
        for document, ids in public_ids.items():
            item={'public_ids':ids,'public_text_decoded':tokenizer.decode(ids),
                  'public_ids_sha256':hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
                  'conditions':{},'challenge':[]}
            result['documents'][document]=item
            for condition in protocol['conditions']:
                records=[]
                baseline=None
                for label in LABELS:
                    prefix=prefixes[label] if condition=='private_before_public' else neutral
                    shared=ids+suffixes[label] if condition=='public_before_private' else ids
                    snapshot=capture(model,prefix,shared,layers)
                    result['completed_forwards']+=1
                    row={'synthetic_label':label,'layers':{}}
                    for layer in layers:
                        public=public_slice(snapshot,PUBLIC_TOKENS,layer)
                        last=public_slice(snapshot,PUBLIC_TOKENS,layer,last_only=True)
                        entry=snapshot.entries[layer][:,:PUBLIC_TOKENS]
                        row['layers'][str(layer)]={
                            'kv_public_digest':fingerprint(public),
                            'kv_last_public_digest':fingerprint(last),
                            'entry_public_digest':fingerprint((entry,)),
                            'kv_last_public_bytes':sum(v.numel()*v.element_size() for v in last)}
                    if baseline is None:
                        baseline=snapshot
                    row['last_layer_public_kv_relative_delta_vs_first_label']=float(
                        torch.cat([(a-b).float().reshape(-1) for a,b in zip(
                            public_slice(snapshot,PUBLIC_TOKENS,layers[-1]),
                            public_slice(baseline,PUBLIC_TOKENS,layers[-1]))]).norm() /
                        torch.cat([b.float().reshape(-1) for b in public_slice(baseline,PUBLIC_TOKENS,layers[-1])]).norm())
                    records.append(row)
                item['conditions'][condition]={'records':records,'distinct_states':summarize(records,layers)}
                save()
                print(json.dumps({'document':document,'condition':condition,'forwards':result['completed_forwards'],
                                  'first_layer':item['conditions'][condition]['distinct_states']['0'],
                                  'last_layer':item['conditions'][condition]['distinct_states'][str(layers[-1])]}),flush=True)
            references={str(layer):{r['synthetic_label']:r['layers'][str(layer)]['kv_last_public_digest']
                         for r in item['conditions']['private_before_public']['records']} for layer in layers}
            # New independent forwards in reverse order; matcher gets only a
            # public-row fingerprint and candidate dictionary, never true label.
            for label in reversed(LABELS):
                snapshot=capture(model,prefixes[label],ids,[])
                result['completed_forwards']+=1
                match={str(layer):candidate_matches(references[str(layer)],fingerprint(
                    public_slice(snapshot,PUBLIC_TOKENS,layer,last_only=True))) for layer in layers}
                item['challenge'].append({'true_synthetic_label':label,'matches_by_layer':match})
            item['unique_correct_candidate_counts_by_layer']={str(layer):sum(
                r['matches_by_layer'][str(layer)]==[r['true_synthetic_label']] for r in item['challenge']) for layer in layers}
            save()
        result['status']='completed_synthetic_state_dependency_probe'
    except Exception as exc:
        result['status']='failed'
        result['error']=f'{type(exc).__name__}: {exc}'
        save()
        raise
    finally:
        result['wall_seconds']=time.monotonic()-began
        save()
    print(json.dumps({'status':result['status'],'forwards':result['completed_forwards'],
                      'wall_seconds':result['wall_seconds']}),flush=True)


if __name__=='__main__':
    main()
