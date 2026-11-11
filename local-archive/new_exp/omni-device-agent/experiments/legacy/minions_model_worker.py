"""Text-only, persistent model process; no agent loop or cross-call KV cache."""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path


def prefill_ranges(length, chunk_tokens):
    """Leave at least one token for generate; never drop or reorder input."""
    if chunk_tokens <= 0:
        return []
    return [(start, min(start + chunk_tokens, length - 1))
            for start in range(0, max(0, length - chunk_tokens), chunk_tokens)]


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--spec', required=True)
    args = parser.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    root = Path(spec['path']).expanduser()
    hashes = {}
    for name, expected in spec['weights'].items():
        actual = digest(root / name)
        if actual != expected:
            raise ValueError(f'Weight checksum mismatch: {name}')
        hashes[name] = actual
    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = spec['device']
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        root, local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa').to(device).eval()
    def sync():
        if device == 'mps':
            torch.mps.synchronize()
        elif device.startswith('cuda'):
            torch.cuda.synchronize()
    sync()
    print(json.dumps({'ready': True, 'pid': os.getpid(), 'model': spec,
                      'weights_verified': hashes, 'python': platform.python_version(),
                      'torch': torch.__version__, 'transformers': transformers.__version__,
                      'config': model.config.to_dict(),
                      'tokenizer_sha256': digest(root / 'tokenizer.json'),
                      'template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                      'cross_call_cache': False}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request.get('shutdown'):
            break
        messages = []
        for m in request['messages']:
            if m.get('images') is not None:
                raise ValueError('Image inputs unsupported in this text-only probe')
            messages.append({k: m[k] for k in ('role', 'content')})
        ids = tok.apply_chat_template(messages, add_generation_prompt=True,
                                      return_tensors='pt').to(device)
        if ids.shape[-1] + request['max_new_tokens'] > model.config.max_position_embeddings:
            raise ValueError('Full request exceeds model context; probe does not truncate')
        chunk_tokens = spec.get('prefill_chunk_tokens', 0)
        ranges = prefill_ranges(ids.shape[-1], chunk_tokens)
        def build_prefix():
            past = None
            for index, (start, end) in enumerate(ranges):
                output = model(input_ids=ids[:, start:end],
                               attention_mask=torch.ones((1, end), device=device, dtype=torch.long),
                               past_key_values=past, use_cache=True, logits_to_keep=1)
                past = output.past_key_values
                if index % 8 == 0:
                    print(f'prefill {end}/{ids.shape[-1]}', file=sys.stderr, flush=True)
            return past
        if request.get('validate_chunking'):
            with torch.inference_mode():
                full = model(ids, attention_mask=torch.ones_like(ids), use_cache=False,
                             logits_to_keep=1).logits[:, -1].float()
                past = build_prefix()
                start = ranges[-1][1] if ranges else 0
                split = model(input_ids=ids[:, start:], attention_mask=torch.ones_like(ids),
                              past_key_values=past, use_cache=True, logits_to_keep=1).logits[:, -1].float()
                relative = ((split-full).norm()/full.norm()).item()
                kl = (full.softmax(-1)*(full.log_softmax(-1)-split.log_softmax(-1))).sum().item()
            sync()
            print(json.dumps({'validation': True, 'input_tokens': ids.shape[-1],
                              'prefix_ranges': ranges, 'logits_relative_l2': relative,
                              'kl_full_to_split': kl,
                              'full_argmax': full.argmax(-1).item(),
                              'split_argmax': split.argmax(-1).item()}), flush=True)
            del full, split, past
            continue
        first = {}
        def before(module, a, kw):
            if 'start' not in first:
                sync()
                first['start'] = time.perf_counter()
                first['input_tokens'] = int(kw['input_ids'].shape[-1])
        def after(module, a, kw, out):
            if 'seconds' not in first:
                sync()
                first['seconds'] = time.perf_counter() - first['start']
        pre = model.register_forward_pre_hook(before, with_kwargs=True)
        post = model.register_forward_hook(after, with_kwargs=True)
        sync()
        start = time.perf_counter()
        try:
            with torch.inference_mode():
                prefill_start = time.perf_counter()
                past = build_prefix()
                sync()
                preceding_prefill_seconds = time.perf_counter() - prefill_start if ranges else 0.0
                # The hooks have already seen an earlier chunk; reset for generate's final chunk.
                first.clear()
                out = model.generate(ids, attention_mask=torch.ones_like(ids),
                                     do_sample=False, max_new_tokens=request['max_new_tokens'],
                                     past_key_values=past, use_cache=True, pad_token_id=tok.eos_token_id)
            sync()
            seconds = time.perf_counter() - start
        finally:
            pre.remove()
            post.remove()
        generated = out[0, ids.shape[-1]:].tolist()
        eos = model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        print(json.dumps({'text': tok.decode(generated, skip_special_tokens=True),
                          'input_ids': ids[0].tolist(), 'generated_ids': generated,
                          'finish_reason': 'stop' if generated[-1] in eos else 'length',
                          'generation_seconds': seconds,
                          'first_forward_seconds': first['seconds'],
                          'preceding_prefill_seconds': preceding_prefill_seconds,
                          'total_prefill_seconds': preceding_prefill_seconds + first['seconds'],
                          'prefill_chunk_tokens': chunk_tokens,
                          'prefill_ranges': ranges,
                          'first_forward_input_tokens': first['input_tokens']}), flush=True)
        del past, out, ids


if __name__ == '__main__':
    main()
