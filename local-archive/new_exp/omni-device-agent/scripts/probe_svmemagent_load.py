"""Attempt native Transformers config loading; never invent a missing model class.

This is an executable prerequisite probe, NOT a writer/QA benchmark.
The input config is an unverified public candidate, not an official release.
"""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError('Use a new result path; do not overwrite evidence.')
    import transformers
    from transformers import AutoConfig
    cfg = args.config_dir / 'config.json'
    original = json.loads(cfg.read_text())
    result = {
        'scope': 'native config loading only; no weights or video inference',
        'candidate_official_provenance': 'unverified',
        'python': sys.version,
        'platform': platform.platform(),
        'transformers': transformers.__version__,
        'config_sha256': hashlib.sha256(cfg.read_bytes()).hexdigest(),
        'model_type': original.get('model_type'),
        'architectures': original.get('architectures'),
        'auto_map': original.get('auto_map'),
        'writer_measurements': None,
        'qa_measurements': None,
    }
    try:
        loaded = AutoConfig.from_pretrained(
            args.config_dir, local_files_only=True, trust_remote_code=False)
        result.update(status='config_loaded_only', config_class=type(loaded).__name__)
        code = 0
    except (ValueError, OSError, ImportError) as exc:
        result.update(status='blocked_at_config_loading',
                      exception_type=type(exc).__name__, error=str(exc))
        code = 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
    return code

if __name__ == '__main__':
    raise SystemExit(main())
