"""Thin MLX-LM adapter: original weights, native generation/cache, no compression."""
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
    from mlx.utils import tree_flatten
    from mlx_lm import load, stream_generate
    from mlx_lm.sample_utils import make_sampler
    model, tok, config = load(str(root), return_config=True)
    mx.synchronize()
    print(json.dumps({'ready': True, 'pid': os.getpid(), 'model': spec,
                      'backend': 'MLX-LM', 'mlx': mx.__version__,
                      'mlx_lm': importlib.metadata.version('mlx-lm'),
                      'transformers': importlib.metadata.version('transformers'),
                      'config': config, 'weight_dtypes': sorted(set(str(p.dtype) for _, p in tree_flatten(model.parameters()))),
                      'weights_verified': spec['weights'], 'tokenizer_sha256': digest(root / 'tokenizer.json'),
                      'template_sha256': hashlib.sha256(tok.chat_template.encode()).hexdigest(),
                      'quantization': None, 'kv_compression': None, 'cross_call_cache': False}), flush=True)
    for line in sys.stdin:
        req = json.loads(line)
        if req.get('shutdown'):
            break
        messages = []
        for m in req['messages']:
            if m.get('images') is not None:
                raise ValueError('Text-only adapter')
            messages.append({k: m[k] for k in ('role', 'content')})
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)
        if len(ids) + req['max_new_tokens'] > config['max_position_embeddings']:
            raise ValueError('Context overflow; input will not be truncated')
        def run(step, maximum, progress=None):
            return stream_generate(model, tok, ids, max_tokens=maximum, sampler=make_sampler(temp=0.0),
                                   prefill_step_size=step, prompt_progress_callback=progress)
        if req.get('validate_chunking'):
            # Native generation API; short fixture only. Compare first-token log-prob vectors.
            full = list(run(len(ids), 1))[-1]
            split = list(run(spec['prefill_chunk_tokens'], 1))[-1]
            a, b = full.logprobs.astype(mx.float32), split.logprobs.astype(mx.float32)
            rel = mx.linalg.norm(a-b) / mx.linalg.norm(a)
            mx.eval(rel)
            print(json.dumps({'validation': True, 'metric': 'first-token log probabilities',
                              'input_tokens': len(ids), 'full_step_size': len(ids),
                              'split_step_size': spec['prefill_chunk_tokens'],
                              'logprobs_relative_l2': rel.item(), 'full_argmax': full.token,
                              'split_argmax': split.token}), flush=True)
            del full, split, a, b, rel
            mx.clear_cache()
            continue
        progress_points = []
        def progress(n, total):
            progress_points.append(n)
            if n == total or n % 4096 == 0:
                print(f'mlx prefill evaluated {n}/{total}', file=sys.stderr, flush=True)
        mx.reset_peak_memory()
        started = time.perf_counter()
        generated, fragments = [], []
        for response in run(spec['prefill_chunk_tokens'], req['max_new_tokens'], progress):
            generated.append(response.token)
            fragments.append(response.text)
        mx.synchronize()
        seconds = time.perf_counter() - started
        assert response.generation_tokens == len(generated)
        prompt_seconds = response.prompt_tokens / response.prompt_tps
        boundaries = [n for n in progress_points if n < len(ids)]
        ranges = list(zip(boundaries[:-1], boundaries[1:]))
        print(json.dumps({'text': ''.join(fragments), 'input_ids': ids, 'generated_ids': generated,
                          'finish_reason': response.finish_reason, 'generation_seconds': seconds,
                          'total_prefill_seconds': prompt_seconds,
                          'prefill_measurement': 'MLX-LM reported prompt_tokens/prompt_tps; includes first-token preparation, not isolated forward hooks',
                          'preceding_prefill_seconds': None, 'first_forward_seconds': None,
                          'prefill_chunk_tokens': spec['prefill_chunk_tokens'], 'prefill_ranges': ranges,
                          'first_forward_input_tokens': len(ids) - boundaries[-1],
                          'mlx_prompt_tps': response.prompt_tps, 'mlx_generation_tps': response.generation_tps,
                          'mlx_peak_memory_gb': response.peak_memory}), flush=True)
        del response
        mx.clear_cache()


if __name__ == '__main__':
    main()
