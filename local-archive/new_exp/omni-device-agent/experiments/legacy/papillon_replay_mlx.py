"""Local replay controls using native MLX-LM; no cloud calls or agent loop."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import time

from minions_model_worker import digest


def sha(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--spec', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    payload = json.loads(args.input.read_text())
    spec = json.loads(args.spec.read_text())
    root = Path(spec['path'])
    for filename, expected in spec['weights'].items():
        assert digest(root / filename) == expected
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.generate import generate_step, generation_stream
    from mlx_lm.models.cache import make_prompt_cache
    from mlx_lm.sample_utils import make_sampler
    model, tok, config = load(str(root), return_config=True)
    chunk = spec['prefill_chunk_tokens']
    report = {'status': 'running', 'pid': os.getpid(), 'spec': spec,
              'mlx': mx.__version__, 'mlx_lm': importlib.metadata.version('mlx-lm'),
              'input_sha256': digest(args.input), 'runs': []}
    args.output.write_text(json.dumps(report, indent=2))
    for case in payload['cases']:
        ids = tok.apply_chat_template(case['chat'], tokenize=True, add_generation_prompt=True)
        assert ids == case['archived_input_ids'], 'Replay must match the actual archived input'
        template = tok.apply_chat_template(case['empty'], tokenize=False, add_generation_prompt=True)
        boundary = template.rindex('[[ ## modelExampleResponses ## ]]')
        prefix = tok.encode(template[:boundary], add_special_tokens=False)[:-16]
        assert ids[:len(prefix)] == prefix, 'Cannot reuse a mismatched prefix'
        modes = (['full_before', 'unaligned', 'aligned', 'full_after', 'json']
                 if case['case'] == 'p95' else ['full_before', 'json'])
        for mode in modes:
            full_ids = (tok.apply_chat_template(case['json'], tokenize=True, add_generation_prompt=True)
                        if mode == 'json' else ids)
            assert len(full_ids) + 512 <= config['max_position_embeddings']
            cache, reused, prepare_seconds, offsets, chunks = None, 0, 0.0, [], []
            if mode in ('aligned', 'unaligned'):
                reused = len(prefix) if mode == 'unaligned' else len(prefix) // chunk * chunk
                assert 0 < reused < len(ids)
                cache = make_prompt_cache(model)
                mx.synchronize()
                start = time.perf_counter()
                if mode == 'unaligned':
                    for _ in generate_step(mx.array(prefix[:reused]), model, max_tokens=0,
                                           prompt_cache=cache, prefill_step_size=chunk):
                        raise AssertionError('Prefill-only call generated output')
                else:
                    # Same model/cache, stream, batch shapes and evaluation boundaries
                    # as native generate_step's prefix loop. Do not call max_tokens=0:
                    # that would split its last chunk and recreate the confound.
                    with mx.stream(generation_stream):
                        for begin in range(0, reused, chunk):
                            part = prefix[begin:begin + chunk]
                            assert len(part) == chunk
                            model(mx.array([part]), cache=cache)
                            mx.eval([c.state for c in cache])
                            chunks.append(len(part))
                            mx.clear_cache()
                mx.synchronize()
                prepare_seconds = time.perf_counter() - start
                offsets = [c.offset for c in cache]
                assert all(x == reused for x in offsets)
            mx.synchronize()
            mx.reset_peak_memory()
            start = time.perf_counter()
            generated, fragments = [], []
            for response in stream_generate(model, tok, full_ids[reused:], max_tokens=512,
                                            sampler=make_sampler(temp=0.0), prompt_cache=cache,
                                            prefill_step_size=chunk):
                generated.append(response.token)
                fragments.append(response.text)
            mx.synchronize()
            row = {'case': case['case'], 'mode': mode, 'input_tokens': len(full_ids),
                   'input_sha256': sha(full_ids), 'output_tokens': len(generated),
                   'generated_sha256': sha(generated), 'generated_ids': generated,
                   'text': ''.join(fragments), 'finish_reason': response.finish_reason,
                   'generation_seconds': time.perf_counter() - start,
                   'remaining_prefill_seconds': response.prompt_tokens / response.prompt_tps,
                   'prepared_tokens': reused, 'prepare_seconds': prepare_seconds,
                   'offsets': offsets, 'prepared_chunk_sizes': chunks,
                   'eligible_prefix_tokens': len(prefix),
                   'aligned_eligible_tokens': len(prefix) // chunk * chunk,
                   'peak_memory_gb': response.peak_memory,
                   'matches_archived_generation': generated == case['archived_generated_ids']}
            assert response.generation_tokens == len(generated)
            report['runs'].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps({k: v for k, v in row.items() if k not in ('text', 'generated_ids', 'offsets')}), flush=True)
            del response, cache
            mx.clear_cache()
    report['status'] = 'completed'
    args.output.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
