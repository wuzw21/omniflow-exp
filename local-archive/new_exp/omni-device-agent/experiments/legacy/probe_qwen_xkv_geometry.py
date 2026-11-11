"""Offline structural probe of unchanged upstream XKV, not a model-quality test.

Uses pinned real model configs but synthetic caches. No base-model weights load.
Also guards against silently reusing a homogeneous translator for a new pair.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upstream', type=Path, required=True)
    parser.add_argument('--sharer-config', type=Path, required=True)
    parser.add_argument('--receiver-config', type=Path, required=True)
    parser.add_argument('--old-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sys.path.insert(0, str(args.upstream / 'src'))
    import torch
    import transformers
    from transformers import AutoConfig
    from kvloom.adapters import adapter_for
    from kvloom.cache.bundle import CacheBundle
    from kvloom.errors import CheckpointError
    from kvloom.training.checkpoint import load_translator
    from kvloom.translator.xkv.model import XKVConfig, XKVTranslator

    torch.manual_seed(20261024)
    torch.set_num_threads(2)
    configs = [AutoConfig.from_pretrained(str(p.parent), local_files_only=True)
               for p in (args.sharer_config, args.receiver_config)]
    geometries = [adapter_for(c).geometry(c) for c in configs]
    sharer_g, receiver_g = geometries
    assert configs[0].model_type == configs[1].model_type == 'qwen3'
    assert sharer_g.num_layers == 28 and receiver_g.num_layers == 36
    config = XKVConfig.for_receiver(receiver_g)
    translator = XKVTranslator(sharer_geometry=sharer_g,
                              receiver_geometry=receiver_g, config=config)

    def bundle(g, length):
        shape = (1, g.num_kv_heads, length, g.head_dim)
        return CacheBundle.from_tensors(
            keys=[torch.randn(shape) for _ in range(g.num_layers)],
            values=[torch.randn(shape) for _ in range(g.num_layers)])

    source, target = bundle(sharer_g, 11), bundle(receiver_g, 7)
    before = [source.content_hash(), target.content_hash()]
    residual = translator(source, target)
    assert residual.num_layers == receiver_g.num_layers
    for deltas, tensors in ((residual.delta_keys, target.keys),
                            (residual.delta_values, target.values)):
        assert all(d.shape == t.shape for d, t in zip(deltas, tensors, strict=True))
        assert all(torch.isfinite(d).all() for d in deltas)
    assert residual.is_zero, 'upstream zero-identity initialization changed'
    assert before == [source.content_hash(), target.content_hash()]
    # A nonzero synthetic target tests graph connectivity, NOT task learning.
    loss = sum((d - 1).square().mean()
               for d in (*residual.delta_keys, *residual.delta_values))
    loss.backward()
    gradients = [p.grad for p in translator.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    nonzero = sum(bool(g.count_nonzero()) for g in gradients)
    assert nonzero > 0
    try:
        load_translator(args.old_checkpoint, sharer_geometry=sharer_g,
                        receiver_geometry=receiver_g)
    except CheckpointError as exc:
        rejection = str(exc)
        assert 'receiver_geometry' in rejection
    else:
        raise AssertionError('homogeneous checkpoint was incorrectly accepted')
    result = {
        'status': 'passed',
        'scope': 'Real model configs; synthetic KV only. No pretrained 4B inference, translator training, accuracy or speed claim.',
        'upstream_commit': subprocess.check_output(
            ['git', '-C', str(args.upstream), 'rev-parse', 'HEAD'], text=True).strip(),
        'versions': {'python': sys.version, 'torch': torch.__version__,
                     'transformers': transformers.__version__},
        'configs': [{'path': str(p), 'sha256': hashlib.sha256(p.read_bytes()).hexdigest(),
                     'geometry': asdict(g)}
                    for p, g in zip((args.sharer_config, args.receiver_config), geometries)],
        'translator_config': asdict(config),
        'trainable_parameters': translator.trainable_parameters(),
        'synthetic_input_lengths': [11, 7],
        'residual_layer_count': residual.num_layers,
        'residual_layer_shape': list(residual.delta_keys[0].shape),
        'zero_identity': residual.is_zero,
        'finite_gradient_tensor_count': len(gradients),
        'nonzero_gradient_tensor_count': nonzero,
        'input_hashes_unchanged': True,
        'old_checkpoint_rejected': str(args.old_checkpoint),
        'rejection': rejection,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ('configs', 'rejection')}, indent=2))


if __name__ == '__main__':
    main()
