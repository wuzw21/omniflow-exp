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
bash scripts/run_androidworld_single_task_server.sh
```

The scheduler is task-major: one task, all five methods, then SmallPhone and
the unfolded Pixel Fold cells. It does not launch a method-major campaign.

## Repository contents

- `omniflow/`: OmniFlow Function recall, transfer, execution, and accounting.
- `omnitransfer/`: the real OmniTransfer runtime code; learned checkpoints stay external.
- `src/integrations/`: AndroidWorld and the four baseline adapters.
- `scripts/`: source preparation, single-task execution, preflight support, and result registration.
- `skills/`: AndroidWorld preflight and source RunLog collection instructions.
- `config/paper_androidworld.json`: the five-method paper configuration.

No experiment is run as part of code migration.
