# AutoDroid 9207 Supplemental Comparison

This document defines the supplemental AutoDroid comparison. It does not add a
method or device to the formal AndroidWorld matrix. The formal matrix remains
the five-method, three-target-device protocol and its historical 116 × 10 =
1160 cells.

## Protocol

| Field | Supplemental value |
| --- | --- |
| Task set | The same 116 tasks enumerated by `data/current.json` |
| Method | `autodroid` only; explicitly selected, never included in `all` |
| Device | `autodroid9207:emulator-5590:5590` on host `9207` |
| AndroidWorld revision | `632ac95959ace58c8e2ed2db8e4209cc3d9c26ef` |
| Source seed | `111`, used for task/source lineage only |
| Evaluation seed | `113` |
| Task parameters | Fixed values from the canonical source index |
| Action budget | `20` replay events, matching the formal `max_steps` |
| Model calls | `0` by contract; no planner, VLM, embedding, or fallback |
| Validator | AndroidWorld official validator after replay |
| Initial state | Fresh task initialization/reset before every task |
| Replay implementation | Original AutoDroid/DroidBot UTG replay and original memory |
| Transfer/Function | Not used; no OmniFlow Function or OmniTransfer conversion |

AutoDroid memory is read-only. The local manifest and its SHA-256 are recorded
in the run evidence; the memory is never copied into `current.json` or a
Function Store.

## Fairness boundary

The comparison uses the same task names, task parameters, evaluation seed,
AndroidWorld revision, official validator, action budget, and per-task reset as
the formal results. Task parameters come from the canonical `data/current.json`
task reference; a source RunLog is not required because AutoDroid does not
consume OmniFlow source trajectories. It differs intentionally in the agent contract: AutoDroid
is an app-level UTG replay, not a task-conditioned planner. Therefore its
result is reported as a supplemental baseline and must not be interpreted as a
fifth planner method or inserted into the 116 × 10 main table.

Report at least these fields per task:

- official validator conclusion and validator coverage;
- replay event count and actions executed;
- wall time and episode time;
- model calls, tokens, and fallback steps (expected to be zero);
- task parameters hash, AutoDroid memory manifest hash, device serial, and
  immutable artifact paths.

For aggregate reporting, compute official success rate over tasks with a valid
validator conclusion, and report unavailable/environment failures separately.
Do not turn missing app memory, missing replay events, startup failure, or
missing validator evidence into a method failure.

## Result namespace

Supplemental artifacts are isolated under:

```text
data/androidworld_validator/supplemental/autodroid_9207/
```

This namespace contains attempts, result outcomes, replay logs, and a
supplemental summary. It is not scanned by the formal result registry and does
not refresh `data/current.json`.

Run one smoke task through the only public launcher:

```bash
OMNIFLOW_EXP_ASSET_ROOT=/absolute/OmniFlow-exp/data \
OMNIFLOW_EXP_RESULTS_ROOT=/absolute/OmniFlow-exp/data \
OMNIFLOW_EXP_MEMORY_ROOT=/absolute/OmniFlow-exp/data \
OMNIFLOW_AUTODROID_ROOT=/absolute/OmniFlow-exp/data/runtime/external/autodroid \
OMNIFLOW_AUTODROID_MEMORY_ROOT=/absolute/OmniFlow-exp/data/runtime/autodroid/androidworld_apps \
OMNITRANSFER_ROOT=/home/wuzewen/Projects/Omni/OmniTransfer \
bash scripts/exp/run_androidworld.sh \
  --e2e-task CameraTakePhoto \
  --e2e-method autodroid \
  --e2e-device autodroid9207:emulator-5590:5590 \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113
```

The full supplemental task run uses the same command once per task with
`--tasks` dispatching, but it must remain outside the formal `--e2e-method all`
campaign.
