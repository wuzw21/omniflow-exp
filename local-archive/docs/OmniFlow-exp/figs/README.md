# Evaluation figures

This directory is the single plotting entry point for the paper evaluation.
The scripts never invent missing measurements: blank CSV fields remain blank,
and a panel with no measurements is rendered as `Results pending`.

## Accounting policy

- Task success uses every benchmark task in the declared workload.
- Online cost is primarily averaged over every episode that actually entered
  execution, whether it succeeded or failed.
- A cell rejected before execution is reported as missing coverage, not as a
  zero-cost run.
- Success-only cost is a secondary statistic and must be shown beside success
  rate and execution coverage.
- Offline Function construction is reported separately from recurring online
  cost.

## Data flow

During paper layout development, all figures are mocks by default. Run:

```bash
./.venv/bin/python figs/plot_all.py
```

Every output contains an explicit `MOCK — NOT RESULTS` marker and uses only
illustrative values. No experiment table is read in this mode.

After the protocols and visual layouts are frozen, real data can be imported
and plotted explicitly:

```bash
./.venv/bin/python figs/prepare_existing_results.py
./.venv/bin/python figs/plot_all.py --results
```

`prepare_existing_results.py` imports the authoritative AndroidWorld matrix,
the checked-in B-MoCA alignment corpus, and the frozen canonical OmniTransfer
reports. It preserves manually entered rows in the protocol templates. The
`--results` mode must not be used for a paper draft until the corresponding
experiment block has been declared complete.

The generated paper assets are written to `figs/generated/` as PDF and PNG.
CSV schemas live in `figs/data/`; later experiment runs can fill these files
without changing the plotting code.

## OmniTransfer evidence

OmniTransfer has one authoritative experiment record:
`/Users/wuzewen/Projects/Omni/OmniTransfer/docs/omnitransfer_v10_all_experiments_report_20260829.md`.
The files in this directory are derived supporting artifacts and must not be
used to introduce a competing final result. The focused paper evaluation
contains two claims: page embedding is compared with screenshot and compressed
XML embeddings on frozen PAPT-heldout-128, and action transfer is evaluated on
the current ASE Test correspondence split. The paper-facing comparison keeps
  one row for the latest OmniTransfer checkpoint and four directly measured
  baselines on the same ASE Test: Text matching, Layout scaling, Text + Layout
  (handcrafted), and a fresh
LightGlue/SuperPoint run.
Historical internal checkpoints and earlier-paper methods remain in the
authoritative report or baseline inventory, not in the current ASE Test
comparison table.

Run the focused OmniTransfer pipeline from this directory:

```bash
./.venv/bin/python figs/sync_omnitransfer_results.py
./.venv/bin/python figs/test_omnitransfer_results.py
./.venv/bin/python figs/plot_omnitransfer.py
```

`sync_omnitransfer_results.py` reads only the canonical checkout and records
source paths plus SHA-256 hashes in `data/omnitransfer_result_manifest.json`.
It writes page-recall and current-protocol action-transfer comparisons,
baseline taxonomy, graph-depth, and latency tables. The plot uses measured
rows only: no mock bars, no `TBD` values, and no cloud result is inferred when
credentials or raw measurements are absent. `test_omnitransfer_results.py`
is a small regression suite for the exact metrics and provenance contract.
The action comparison CSV contains the latest OmniTransfer row, four existing
ASE baselines, and the freshly measured LightGlue row. The baseline inventory
additionally records the earlier-paper matcher ladder, historical LightGlue,
Qwen-VL-Plus/Max,
normalized-coordinate replay, UTG identity/attribute matching, Fixed Replay,
MobileGPT, AppAgent, T3A, AutoDroid, M3A, MobiMem, V-Droid, and ShowUI. Methods
without current ASE measurements are explicitly labeled rather than assigned
fabricated scores.

## Experiment files

- `androidworld_success.csv`: success and coverage by method, target, and
  official Easy/Medium/Hard difficulty.
- `androidworld_episodes.csv`: one row per executed episode; this is the source
  for calls, tokens, latency, action, and fallback aggregates.
- `bmoca_alignment_corpus.csv`: existing offline corpus statistics.
- `bmoca_replay.csv`: replay-only outcomes with planning disabled.
- `omnitransfer_page_recall.csv`: screenshot, compressed XML, and latest
  OmniTransfer page recall.
- `omnitransfer_step_match.csv`: latest OmniTransfer action transfer and its
  same-checkpoint descriptor diagnostic.
- `omnitransfer_action_comparisons.csv`: direct ASE Test identity baselines,
  latest OmniTransfer, and the complete historical Action Transfer comparison.
- `omnitransfer_baseline_inventory.csv`: reader-facing baseline taxonomy with
  method role, input/output, protocol, measurement status, and evidence path.
- `omnitransfer_ablation.csv`: fixed-weight H² versus H¹ graph-depth diagnostic.
- `omnitransfer_latency.csv`: page encoding, dense-path comparison, and two-head decomposition.
- `omnitransfer_embedding_deployment.csv`: user-provided remote `9207` deployment-feasibility measurements for HTML-like and screenshot embedding.
- `omnitransfer_action_errors.csv`: exact ASE rank buckets for Top-1 misses.
- `omnitransfer_result_manifest.json`: supporting-artifact provenance and source hashes.
- `real_device_dynamic.csv`: cross-device dynamic-task outcomes.
- `energy_power.csv`: component and end-to-end device measurements.
- `execution_overhead.csv`: measured OnePlus replay-step component latency,
  incremental power, and active energy used by
  `execution_overhead_breakdown.pdf`.
- `function_scaling.csv`: Function Store scale, lookup latency, and memory.
- `failure_breakdown.csv`: first unrecovered failure categories.
