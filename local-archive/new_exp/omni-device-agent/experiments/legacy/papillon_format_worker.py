"""Use unchanged PAPILLON signatures and installed DSPy to render prompts offline.

No LM is called and no output is fabricated. Empty external response marks an
input-layout lower bound, not an executed PAPILLON interaction.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    manifest = json.loads((args.source / 'manifest.json').read_text())
    for item in manifest['files']:
        raw = (args.source / item['path']).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == item['sha256']
    import dspy
    spec = importlib.util.spec_from_file_location('pinned_papillon', args.source / 'papillon/run_llama_dspy.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    flow = module.PAPILLON(None)
    adapter = dspy.ChatAdapter()
    print(json.dumps({'ready': True, 'dspy': dspy.__version__, 'source': manifest}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        q = request['query']
        creation = adapter.format(flow.prompt_creater.predict.signature, [], {'userQuery': q})
        aggregation = adapter.format(flow.info_aggregator.signature, [],
                                     {'userQuery': q, 'modelExampleResponses': ''})
        print(json.dumps({'creation': creation, 'aggregation_empty_response': aggregation}), flush=True)


if __name__ == '__main__':
    main()
