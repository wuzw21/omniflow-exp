# OmniFlow-exp Rules

This repository exists only for the paper's AndroidWorld experiment. Do not add
product code, historical campaigns, ablations, exploratory methods, raw data,
or compatibility layers.

## OmniFlow / OmniTransfer boundary

- OmniTransfer is responsible only for mapping and mapped points.
- OmniFlow-exp owns the experiment's core logic, including the native 512D
  pole encoding and page encoding. Do not substitute OmniTransfer matcher
  embeddings for these OmniFlow embeddings.

## Current phase

The current phase is code migration and script maintenance only. Do not launch
formal experiments or revise method implementations unless the user explicitly
opens a new execution phase.

## Formal protocol

- Run task-major: one task across every method and both devices before the next task.
- The exact method set is `fixed_replay`, `ours`,
  `mobilegpt_offline_retrieval`, `appagent_demo`, and `t3a_hint`.
- Source seed is `111`; target seed is `113`.
- Targets are SmallPhone and Pixel Fold in unfolded state `2`.
- Use AndroidWorld native observation/action and its official validator. Do not use OOB.
- Record validator result, model calls, prompt/completion/total tokens, actions,
  episode duration, and outer wall time for every cell.
- Results and attempts are immutable and live outside this repository.

## Method boundary

The migrated method implementations are frozen during this phase. Script,
environment, installation, preflight, accounting, and registration repairs may
not change a method's prompt, memory, demonstration, action policy, parser,
retry policy, step budget, model, or output.

## Code layout

- Put contracts and shared data types in `omniflow/core/`.
- Put Function lifecycle code in `omniflow/functions/`.
- Put replay orchestration and execution in `omniflow/runtime/`.
- Put transfer orchestration and evidence tooling in `omniflow/transfer/`.
- Put VLM-specific planning and adaptation in `omniflow/vlm/`.
- Keep `omniflow/bridge.py` as the external bridge entry point.
- Keep `omniflow/vlm_coordinates.py` at its shared-contract path.
- Do not recreate the retired flat module layout.
- Keep `scripts/exp/run_androidworld.sh` as the only repository script.
- Put experiment implementation in `src/experiment/` or `src/integrations/`.

## Data boundary

Never commit RunLogs, screenshots, XML dumps, model weights, APKs, emulator
images, baseline memory, credentials, runtime attempts, or result tables. All
such assets must be supplied through explicit absolute paths.

The only RunLog contract is `omniflow.run_log.v1`. Its observation fields are
the AndroidWorld `State` fields `pixels`, `forest`, `ui_elements`, and
`auxiliaries`; its actions use only fields and action types accepted by
AndroidWorld `JSONAction`. OmniFlow persists `pixels` as an immutable screenshot
reference instead of embedding the array. Never encode an OmniFlow action alias
or a private device command as an AndroidWorld action.
This contract is defined by OmniFlow source and the checked-in schema, never by
`current.json`; experiment memory only indexes immutable assets and results.

OmniTransfer failures must return to the normal OmniFlow fallback path. Never
replay source-device coordinates directly on a target device.

## Long-term experiment memory

- `OMNIFLOW_EXP_MEMORY_ROOT/current.json` is the only canonical entry point for
  existing AndroidWorld RunLogs, converted Function assets, and registered
  results. The experiment script must resolve source and Store indexes from it
  before preflight or execution.
- Maintain the memory only through
  `scripts/exp/run_androidworld.sh --refresh-memory`. Function conversion must
  update the same memory before reporting success, and result registration must
  update it immediately after writing an immutable formal result.
- Store immutable evidence by exact SHA-256. Identical content has one object
  and any additional locations are aliases; never copy it into another
  logical version or regenerate it.
- Keep every original attempt immutable. Deduplication changes only canonical
  indexes and content-addressed storage; it never deletes or rewrites evidence.
- Classify memory by AndroidWorld task. Preserve unclassified evidence in the
  registry instead of dropping it or guessing a task.
- The master source index is authoritative for the canonical source RunLog.
  A converted Function Store is canonical only when the v2 Store, transfer
  states, provenance hashes, and no-target-input audit all verify. Equal-quality
  conflicting Stores are an error and must not be selected silently.
- For a task/method/device result cell, use the earliest immutable registered
  result with an official-validator conclusion. Never rank or replace results
  by success, token count, duration, or a later retry.
- If `current.json` already contains a canonical asset or formal result, reuse
  it or skip the completed cell. Do not call a model, rebuild an asset, rerun a
  completed cell, or reason about which historical directory to use.
- A missing, corrupt, or ambiguous memory entry is a preflight failure. Report
  it explicitly; do not fall back to path guessing or automatic generation.
