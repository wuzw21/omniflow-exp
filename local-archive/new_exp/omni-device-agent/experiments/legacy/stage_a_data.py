"""Deterministic question-level split and equal-position prefix interventions."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer
from fetch_stage_a_data import DATA_REVISION, DATA_SHA
from fetch_stage_a_models import CACHE, MODELS, sha256

SEED = 20261024
CONDITIONS = ("P0", "P1", "P2")


def key(text):
    return hashlib.sha256(f"{SEED}|{text}".encode()).hexdigest()


def select_split(rows, total=512, dev=128):
    if not 0 < dev < total <= len(rows):
        raise ValueError("Invalid sample / development sizes")
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError("Duplicate question IDs")
    kinds = sorted({r['type'] for r in rows})
    groups = {t: sorted((r for r in rows if r['type'] == t), key=lambda r:key(r['id'])) for t in kinds}
    def allocate(size, counts):
        exact = {t: size*counts[t]/sum(counts.values()) for t in kinds}
        n = {t: int(exact[t]) for t in kinds}
        for t in sorted(kinds,key=lambda t:(-(exact[t]-n[t]),t))[:size-sum(n.values())]:
            n[t] += 1
        return n
    picked = allocate(total, {t:len(g) for t,g in groups.items()})
    dev_count = allocate(dev, picked)
    output = []
    for t in kinds:
        for i,row in enumerate(groups[t][:picked[t]]):
            output.append(("dev" if i < dev_count[t] else "test", row))
    return sorted(output, key=lambda x:(x[0],key(x[1]['id'])))


def paired_prefixes(tokenizer, a, b):
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    # Visible filler, never masked padding; exact token concatenation is the input contract.
    a_ids,b_ids = enc(a+"\nUnrelated filler (ignore):\n"),enc(b+"\nUnrelated filler (ignore):\n")
    length = max(len(a_ids),len(b_ids))
    filler = enc("Background note: this item is unrelated to the current question.\n")
    tail = enc("\nEnd of unrelated filler.\nShared observations:\n")
    def pad(ids):
        gap = length-len(ids)
        return ids + (filler * ((gap+len(filler)-1)//len(filler)))[:gap] + tail
    a_ids,b_ids = pad(a_ids),pad(b_ids)
    assert len(a_ids)==len(b_ids)
    return a_ids,b_ids


def build_case(row, tokenizer, condition, shared_budget=2048):
    if condition not in CONDITIONS:
        raise ValueError("Unknown prefix condition")
    enc = lambda text: tokenizer.encode(text, add_special_tokens=False)
    titles = row['context']['title']
    docs = dict(zip(titles,row['context']['sentences']))
    support = sorted(set(row['supporting_facts']['title']),key=lambda t:key(row['id']+'|'+t))
    if len(support)!=2 or not set(support).issubset(docs):
        raise ValueError(f"Expected two available support documents: {row['id']}")
    d1 = support[0]
    render = lambda title: f"Document: {title}\n"+''.join(docs[title])+"\n\n"
    neutral = "No previous evidence has been collected."
    lead = "<|im_start|>user\nAnswer the question using the observations.\nQuestion: "+row['question']+"\n"
    role_s = "Role: organize the evidence accurately.\n"
    role_t = role_s if condition=='P0' else "Role: verify the evidence accurately.\n"
    history_s = neutral
    history_t = render(d1) if condition=='P2' else neutral
    hs,ht = paired_prefixes(tokenizer,lead+role_s+"Previous observations:\n"+history_s,
                           lead+role_t+"Previous observations:\n"+history_t)
    required = set(support) - ({d1} if condition=='P2' else set())
    selected = set(required)
    n = sum(len(enc(render(t))) for t in selected)
    if n > shared_budget:
        raise ValueError(f"Support documents exceed budget; do not silently truncate: {row['id']}")
    for title in titles:
        if title in selected or (condition=='P2' and title==d1):
            continue
        count = len(enc(render(title)))
        if n+count <= shared_budget:
            selected.add(title)
            n += count
    # Keep original document order rather than placing all answer evidence first.
    shared = [token for t in titles if t in selected for token in enc(render(t))]
    suffix = enc("\nGive only the short answer, without explanation.<|im_end|>\n<|im_start|>assistant\n")
    return {"id":row['id'], "type":row['type'], "condition":condition,
            "answer":row['answer'], "prefix_source_ids":hs, "prefix_target_ids":ht,
            "shared_ids":shared, "suffix_ids":suffix,
            "shared_titles":[t for t in titles if t in selected],
            "history_support_title":d1 if condition=='P2' else None,
            "support_titles":support, "prefix_tokens":len(hs), "shared_tokens":len(shared)}


def prepare(output):
    path = CACHE/'data'/DATA_REVISION/'validation.parquet'
    if sha256(path) != DATA_SHA:
        raise ValueError("Dataset checksum mismatch")
    rows = pq.read_table(path).to_pylist()
    tokenizer = AutoTokenizer.from_pretrained(CACHE/'models'/MODELS['instruct'][1], local_files_only=True)
    selected = select_split(rows)
    output = Path(output)
    derived = CACHE/'data'/DATA_REVISION/'stage_a_cases.jsonl'
    metadata, lengths = [], {p:[] for p in CONDITIONS}
    with derived.open('w') as f:
        for split,row in selected:
            metadata.append({'id':row['id'],'type':row['type'],'split':split})
            for p in CONDITIONS:
                case = build_case(row,tokenizer,p)
                case['split'] = split
                f.write(json.dumps(case,ensure_ascii=False)+'\n')
                lengths[p].append(case['shared_tokens'])
    vocab_hash = hashlib.sha256(json.dumps(tokenizer.get_vocab(),sort_keys=True).encode()).hexdigest()
    report = {"status":"prepared; no quality experiment", "seed":SEED,
              "data_revision":DATA_REVISION,"data_sha256":DATA_SHA,
              "tokenizer_model_revision":MODELS['instruct'][1],"tokenizer_vocab_sha256":vocab_hash,
              "cases_file":str(derived),"cases_sha256":sha256(derived),
              "split_counts":dict(Counter(x['split'] for x in metadata)),
              "strata_counts":dict(Counter(x['split']+'/'+x['type'] for x in metadata)),
              "shared_tokens":{p:{'min':min(v),'median':sorted(v)[len(v)//2],'max':max(v)} for p,v in lengths.items()},
              "questions":metadata}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='questions'},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='results/stage_a_data_manifest.json')
    prepare(parser.parse_args().output)
