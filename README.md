# OmniFlow-exp

Clean, paper-only AndroidWorld evaluation code for OmniFlow.

This repository contains code and orchestration only. RunLogs, screenshots,
models, APKs, AndroidWorld checkouts, baseline memories, and evaluation results
must live outside the repository and are supplied through environment paths.

## Paper methods

- `fixed_replay` (RPA)
- `ours` (OmniFlow)
- `mobilegpt_offline_retrieval` (MobileGPT)
- `appagent_demo` (AppAgent)
- `mobile_agent_v3` (Mobile-Agent-V3)

The public entry point is:

```bash
OMNIFLOW_EXP_ASSET_ROOT=/absolute/path/to/external/assets \
OMNIFLOW_EXP_RESULTS_ROOT=/absolute/path/to/external/results \
PYTHON_BIN=/absolute/path/to/python \
OMNITRANSFER_ROOT=/absolute/path/to/versioned/omnitransfer \
bash scripts/exp/run_androidworld.sh
```

The scheduler is task-major: one task, all five methods, then SmallPhone and
the unfolded Pixel Fold cells. It does not launch a method-major campaign.

## Repository contents

- `omniflow/`: OmniFlow's public Python package.
  - `core/`: data models, configuration, schemas, and canonical RunLog handling.
  - `functions/`: Function artifacts, compilation, retrieval, storage, and management.
  - `runtime/`: runtime orchestration, action execution, and Checker recovery.
  - `transfer/`: OmniTransfer calls, page encoding, alignment, memory, and review.
  - `vlm/`: VLM planning, prompt construction, model adaptation, UI projection, and accounting.
  - `bridge.py`: external JSON-line bridge entry point.
  - `vlm_coordinates.py`: shared-contract owner for VLM coordinate conversion.
- `omnitransfer/`: the real OmniTransfer runtime code; learned checkpoints stay external.
- `src/experiment/`: formal task-major orchestration and immutable result registration.
- `src/integrations/`: AndroidWorld, baseline adapters, and their runtime helpers.
- `scripts/exp/run_androidworld.sh`: the only experiment script and public one-command entry point.
- `skills/`: AndroidWorld preflight and source RunLog collection instructions.
- `config/paper_androidworld.json`: the five-method paper configuration.

No experiment is run as part of code migration.
