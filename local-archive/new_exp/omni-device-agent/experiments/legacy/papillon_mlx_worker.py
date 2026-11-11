"""Native MLX-LM inference plus one explicitly prepared, single-use prefix slot.

No cache translation/compression or agent loop. Prefix validation against the
eventual full token sequence is mandatory; mismatch discards the prepared state.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

from minions_model_worker import digest


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--spec', required=True)
    args = p.parse_args()
    spec = json.loads(Path(args.spec).read_text())
    root = Path(spec['path'])
    for name, expected in spec['weights'].items():
        assert digest(root / name) == expected
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.generate import generate_step
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    model, tok, config = load(str(root), return_config=True)
    mx.synchronize()
    print(json.dumps({'ready': True, 'pid': os.getpid(), 'model': spec,
                      'mlx': mx.__version__, 'mlx_lm': importlib.metadata.version('mlx-lm'),
                      'transformers': importlib.metadata.version('transformers'),
                      'weights_verified': spec['weights'],
                      'tokenizer_sha256': digest(root / 'tokenizer.json'),
                      'template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                      'cache_policy': 'One explicit single-use prefix; exact full-token validation; no other cross-call cache'}), flush=True)
    held = None
    for line in sys.stdin:
        req = json.loads(line)
        if req.get('shutdown'):
            break
        messages = req['messages']
        if req.get('prepare_prefix'):
            if held is not None:
                raise RuntimeError('Prepared cache must be consumed or explicitly discarded first')
            text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            marker = '[[ ## modelExampleResponses ## ]]'
            prefix_text = text[:text.rindex(marker)]
            # Hold back a conservative boundary region. Correctness comes from validation,
            # not from assuming a fixed number of BPE tokens always guarantees stability.
            ids = tok.encode(prefix_text, add_special_tokens=False)[:-16]
            if not ids:
                raise ValueError('No prefix to prepare')
            cache = make_prompt_cache(model)
            mx.synchronize(); began = time.perf_counter()
            for _ in generate_step(mx.array(ids), model, max_tokens=0, prompt_cache=cache,
                                   prefill_step_size=spec['prefill_chunk_tokens']):
                raise AssertionError('A prefill-only native call generated output')
            mx.synchronize()
            offsets = [c.offset for c in cache]
            assert all(n == len(ids) for n in offsets)
            held = (ids, cache)
            print(json.dumps({'prepared': True, 'input_tokens': len(ids),
                              'seconds': time.perf_counter() - began,
                              'all_layer_offsets': offsets,
                              'known_prefix_sha256': hashlib.sha256(json.dumps(ids).encode()).hexdigest()}), flush=True)
            continue
        ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
        if len(ids) + req['max_new_tokens'] > config['max_position_embeddings']:
            raise ValueError('No silent truncation')
        cache, reused, mismatch = None, 0, False
        if req.get('use_prepared'):
            if held is None:
                raise RuntimeError('Missing prepared state')
            prefix, candidate = held
            held = None  # Never let a mutated cache survive a second request.
            if ids[:len(prefix)] == prefix:
                cache, reused = candidate, len(prefix)
            else:
                mismatch = True
            del candidate, prefix
        elif held is not None:
            raise RuntimeError('Unrelated request while prepared state is pending')
        pending = ids[reused:]
        assert pending
        mx.synchronize(); mx.reset_peak_memory(); began = time.perf_counter()
        generated, fragments = [], []
        for response in stream_generate(model, tok, pending, max_tokens=req['max_new_tokens'],
                                        sampler=make_sampler(temp=0.0), prompt_cache=cache,
                                        prefill_step_size=spec['prefill_chunk_tokens']):
            generated.append(response.token); fragments.append(response.text)
        mx.synchronize()
        elapsed = time.perf_counter() - began
        assert response.generation_tokens == len(generated)
        print(json.dumps({'text': ''.join(fragments), 'input_ids': ids, 'generated_ids': generated,
                          'finish_reason': response.finish_reason, 'generation_seconds': elapsed,
                          'total_prefill_seconds': response.prompt_tokens / response.prompt_tps,
                          'prefill_measurement': 'Native MLX-LM remaining prompt_tokens/prompt_tps including first-token preparation',
                          'reused_prefix_tokens': reused, 'prepared_prefix_mismatch': mismatch,
                          'pending_prompt_tokens': len(pending), 'mlx_peak_memory_gb': response.peak_memory}), flush=True)
        del response, cache
        mx.clear_cache()


if __name__ == '__main__':
    main()
