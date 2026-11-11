"""Cross-check saved placement evidence against the earlier scalar evaluator."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from fetch_public_trace import CACHE
from feasibility import evaluate_fixed_residency
from remote_expert_dependencies import evaluate
from streaming_expert_bounds import load_verified


def main():
    result = json.loads(Path('results/remote_expert_dependencies.json').read_text())
    assert result['source_sha256'] == hashlib.sha256(Path('remote_expert_dependencies.py').read_bytes()).hexdigest()
    _, routes, segments = load_verified(CACHE)
    assert result['calibration_ranges'] == [list(x) for x in segments[::2]]
    assert result['evaluation_ranges'] == [list(x) for x in segments[1::2]]
    assert result['calibration_tokens'] + result['evaluation_tokens'] == len(routes)
    start, end = result['evaluation_ranges'][0]
    tiny = routes[start:min(start+16, end)]
    max_float_delta = 0.
    checked = 0
    for row in result['results'].values():
        for values in row['policies'].values():
            resident = values['resident_expert_ids']
            assert sum(map(len, resident)) == row['slots']
            mask = np.zeros((48, 512), dtype=bool)
            for layer, ids in enumerate(resident):
                mask[layer, ids] = True
            scalar = evaluate_fixed_residency(tiny.tolist(), resident, 30, expert_count=512)
            vector = evaluate(tiny, mask)
            hits = int(mask[np.arange(48)[None,:,None], tiny].sum())
            assert hits + scalar['remote_expert_selections'] == tiny.size
            delta = abs(scalar['expert_hit_rate'] - vector['local_expert_hit_rate'])
            max_float_delta = max(max_float_delta, delta)
            # hit/total versus 1-miss/total can differ at binary64 rounding.
            assert math.isclose(scalar['expert_hit_rate'], vector['local_expert_hit_rate'], rel_tol=0, abs_tol=1e-15)
            assert sum(scalar['blocking_rpc_per_token'])/len(tiny) == vector['remote_layers_per_token']['mean']
            hist = values['remote_layers_per_token']['histogram']
            assert sum(hist) == result['evaluation_tokens']
            assert sum(i*n for i,n in enumerate(hist))/sum(hist) == values['remote_layers_per_token']['mean']
            checked += 1
    report = {'source_sha256': result['source_sha256'], 'verified_split': True,
              'equal_capacity_placements': checked, 'legacy_vector_integer_count_agreement_placements': checked,
              'token_histogram_consistency': True, 'max_hit_rate_rounding_delta': max_float_delta,
              'ratio_absolute_tolerance': 1e-15,
              'evidence': 'CPU trace analysis, no inference or networking',
              'initial_verification': 'Exact float equality failed; integer counts agree. No placement/result changes.'}
    Path('results/remote_expert_dependency_verification.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
