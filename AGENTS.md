# OmniFlow-exp Rules

This repository exists only for the paper's AndroidWorld experiment. Do not add
product code, historical campaigns, ablations, exploratory methods, raw data,
or compatibility layers.

## Current phase

The current phase is code migration and script maintenance only. Do not launch
formal experiments or revise method implementations unless the user explicitly
opens a new execution phase.

## Formal protocol

- Run task-major: one task across every method and both devices before the next task.
- The exact method set is `fixed_replay`, `ours`,
  `mobilegpt_offline_retrieval`, `appagent_demo`, and `mobile_agent_v3`.
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

OmniTransfer failures must return to the normal OmniFlow fallback path. Never
replay source-device coordinates directly on a target device.
