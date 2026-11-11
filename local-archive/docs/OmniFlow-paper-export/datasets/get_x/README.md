# get_x Mapping Dataset

Generated on 2026-06-10 from:

- `/Users/wuzewen/Projects/Omni/omniflow-paper-export/goals.json`
- `/Users/wuzewen/Projects/Omni/OmniFlow/Benchmark Dataset_Blur/Benchmark Dataset`

## Files

- `get_x_cases.json`: full XML TestCase-compatible dataset for
  `tests/vector/benchmarks/action_accuracy.py --input-cases`.
- `get_x_calls.json`: compact call-oriented view of
  `get_x(x, y, src, target) -> {x, y}`. `src.page_ref` and
  `target.page_ref` point to XML files in the OmniFlow workspace.
- `summary.json`: counts, skip reasons, source paths, and usage notes.
- `fixed_baseline_results.json`: sanity run for raw fixed-coordinate replay.
- `text_feature_baseline_results.json`: selector-like text/resource/class
  baseline.
- `omniflow_anchor_results.json`: current OmniFlow `get_new_coordinates(...)`
  action anchor result.
- `anchor_accuracy_summary.json`: compact comparison table for the three
  methods above.

## Schema

Each case maps:

```text
x = source_x
y = source_y
src = {page: source_xml, element: metadata.source_action_element}
target = {page: target_xml}
expected = {x: target_x, y: target_y, bounds: target_bounds}
```

The current set keeps only raw-step-aligned source/target point actions with
the same action name: `click`, `long-click`, or `repeat-click`. Labels are
`silver_raw_step_aligned`; they are suitable for benchmark/debug use, not
human-reviewed gold.

## Validation

Loaded with:

```bash
PYTHONPATH=. python -m tests.vector.benchmarks.action_accuracy \
  --input-cases /Users/wuzewen/Projects/Omni/omniflow-paper-export/datasets/get_x/get_x_cases.json \
  --adapter fixed \
  --output /Users/wuzewen/Projects/Omni/omniflow-paper-export/datasets/get_x/fixed_baseline_results.json
```

Results on 212 positive silver cases:

| Method | Mapped / valid | Failure | TA@50 | Node match / case acc. |
|---|---:|---:|---:|---:|
| fixed coordinate | 212 | 0.00% | 61.79% | 64.15% |
| text feature | 145 | 31.60% | 75.17% | 51.89% |
| OmniFlow action anchor | 118 | 44.34% | 79.66% | 44.81% |

For OmniFlow action anchor, mapped-click precision is 95/118 = 80.51%.
The remaining 94 cases fail closed: 68 `absent`, 14 `ambiguous`, and
12 `occluded`.
