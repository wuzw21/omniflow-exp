# OmniFlow-exp Architecture

This repository is the paper experiment and B-MoCA validation. It has one
public launcher, one scheduler, one AndroidWorld episode owner, one Function
writer, and one local artifact index. Keep those seams stable when changing
the implementation.

## Top-level flow

```text
scripts/exp/run_androidworld.sh
  -> src/experiment/e2e_task_pipeline.py
  -> src/experiment/androidworld.py
  -> src/integrations/android_world/launch.py
  -> src/integrations/android_world/methods.py
  -> omniflow/runtime/engine.py
  -> omniflow/runtime/execution.py
```

Function authoring is a separate, intentionally short path:

```text
omniflow/bridge.py
  -> omniflow/functions/assets.py::save_function
  -> omniflow/functions/recall.py
  -> data/current.json
```

`src/experiment/e2e_task_pipeline.py` is allowed to call `save_function` for
the B-MoCA source gate. It must not implement another compiler or executor.

The single detailed edit guide is [`docs/FILE_EDIT_GUIDE.md`](FILE_EDIT_GUIDE.md).
This document records the architecture; ownership changes belong in that guide.

Ownership and edit locations are maintained only in
[`docs/FILE_EDIT_GUIDE.md`](FILE_EDIT_GUIDE.md).

## Method names

AndroidWorld public selectors are exactly `fixed_replay`, `omniflow`,
`mobilegpt`, `appagent`, and `t3a_hint`. B-MoCA replay selectors are a
different benchmark contract and remain unchanged. New method code belongs in
one Method Adapter and is selected by the shared runner; it does not get a new
runner, lifecycle, result table, or artifact index.

## Artifact lifecycle

1. A successful source RunLog is validated.
2. `save_function` compiles exactly one Function and writes one Store.
3. `--refresh-memory` materializes `data/current.json` atomically.
4. The scheduler reads that index and skips immutable official conclusions.
5. Each task/device/method writes one classified evidence bundle and one result.

There is no packaged Function catalog. Transfer-state catalogs, RunLog
provenance, and method-native manifests are evidence inside their respective
contracts; they are not alternate Function registries.

## File map

Directory READMEs describe local usage only. The following production-file map
is architectural context; edit ownership is maintained only in
`docs/FILE_EDIT_GUIDE.md`.

### `omniflow/`

| File | Responsibility |
| --- | --- |
| `omniflow/__init__.py` | package exports |
| `omniflow/bridge.py` | JSON-line management interface and `run_gui` |
| `omniflow/runlog.py` | canonical RunLog evidence loading |
| `omniflow/vlm_coordinates.py` | VLM coordinate conversion |
| `omniflow/catalog/*` | removed; Functions come only from `save_function` |

### `omniflow/core/`

| File | Responsibility |
| --- | --- |
| `__init__.py` | core exports |
| `androidworld_accessibility.py` | native accessibility projection |
| `config.py` | runtime and experiment configuration types |
| `model.py` | shared domain models and interfaces |
| `schemas.py` | action and payload normalization |
| `trajectory.py` | canonical RunLog/trajectory validation |

### `omniflow/functions/`

| File | Responsibility |
| --- | --- |
| `__init__.py` | Function exports |
| `assets.py` | Function schema, compiler, validator, and sole Store writer |
| `recall.py` | Function candidate selection |

### `omniflow/runtime/`, `transfer/`, and `vlm/`

| File | Responsibility |
| --- | --- |
| `runtime/core.py` | one action execution primitive |
| `runtime/checker.py` | Function-local checker evaluation |
| `runtime/engine.py` | persistent Planner/Function lifecycle |
| `runtime/execution.py` | transfer, grounding, action dispatch, fallback |
| `runtime/semantic_grounding.py` | visible-target grounding |
| `transfer/page_embedding.py` | canonical OmniTransfer page encoder |
| `transfer/runtime.py` | transfer-state catalog loading |
| `vlm/context.py` | Planner context assembly |
| `vlm/model_config.py` | model endpoint resolution |
| `vlm/planner.py` | VLM tool selection and validation |
| `vlm/usage.py` | token accounting |

### `src/experiment/`

| File | Responsibility |
| --- | --- |
| `androidworld.py` | one AndroidWorld result and source evidence |
| `appagent_source.py` | AppAgent source-memory preparation |
| `batch_outcomes.py` | in-memory result summarization |
| `development_emulator.py` | bounded emulator development preflight |
| `e2e_task_pipeline.py` | only task/method/device scheduler |
| `emulator_processes.py` | emulator process inspection |
| `local_data.py` | sole `current.json` index builder/reader |
| `mobilegpt_contract.py` | MobileGPT evidence constants |
| `mobilegpt_source.py` | MobileGPT source-memory preparation |
| `observation_evidence.py` | observation evidence and metrics |
| `performance_metrics.py` | optional performance side-channel aggregation |
| `preflight.py` | static asset and contract gates |
| `protocol.py` | typed view over the canonical protocol |
| `result_registry.py` | immutable result registration |
| `result_schema.py` | public result-row field contract |
| `source_assets.py` | source evidence validation and conversion dispatch |

### `src/integrations/`

| File | Responsibility |
| --- | --- |
| `android_world/agent.py` | AndroidWorld OmniFlow Agent Adapter |
| `android_world/apps.py` | AndroidWorld app setup helpers |
| `android_world/environment.py` | official environment adapter |
| `android_world/host.py` | native observation/action Host |
| `android_world/launch.py` | native lifecycle and launcher |
| `android_world/methods.py` | method adapters |
| `android_world/mobilegpt_agent.py` | MobileGPT episode adapter |
| `android_world/oob_control.py` | explicit development/source transport adapter |
| `android_world/state.py` | native state normalization |
| `appagent_adapter.py` | AppAgent conversion/runtime adapter |
| `bmoca.py` | B-MoCA DeviceDriver adapter |
| `mobilegpt_converter.py` | one MobileGPT memory converter |
| `mobilegpt_runtime.py` | upstream MobileGPT runtime patch seam |
| `runlog.py` | legacy/source RunLog importer |
| `script_replay.py` | direct Function replay adapter |
| `skilldroid_replay.py` | official DroidRun replay adapter |

### `scripts/`, `schemas/`, `tools/`, and `tests/`

| Path | Responsibility |
| --- | --- |
| `scripts/exp/run_androidworld.sh` | only public experiment entry |
| `scripts/exp/README.md` | launcher contract and commands |
| `schemas/oob/*.json` | external payload schemas |
| `schemas/oob/README.md` | schema contract notes |
| `tools/manual_androidworld_harness.py` | human-only diagnosis |
| `tests/` | module-level contract tests; add tests beside the owner |

## Adding or changing code

To add a method, change the protocol, Method Adapter, shared scheduler dispatch,
and its contract tests together. Do not add a method-specific lifecycle or
result format. The optional OOB transport is selected only through the shared
launcher with `--control-backend oob` for bounded development, source
collection, or E2E runs; it does not create a second AndroidWorld lifecycle.
To add an action, change the shared action schema, compiler,
runtime execution, and one focused test. To add an artifact, extend the
canonical bundle/index contract and its validator before writing it; never add
a sidecar registry. To change a schema or public result field, make a separate
commit and update its schema tests and README in the same commit.

## Verification

Use `./.venv/bin/python` for all Python commands:

```bash
./.venv/bin/python -m pytest -q
./.venv/bin/python -m ruff check omniflow src tests
bash -n scripts/exp/run_androidworld.sh
git diff --check
```
