---
name: androidworld-runlog-harvester
description: "Author, enhance, save, convert, and source-qualify versioned semantic OmniFlow Functions from successful AndroidWorld RunLogs. Use for Function schema/binding failures, over-split workflows, MCP save_function calls, or immutable AndroidWorld Function asset refresh and replay."
---

# Semantic Function Authoring

Create reusable semantic Functions from successful RunLogs. This Skill authors
Function semantics only. OmniFlow remains the sole compiler, Store, runtime,
transfer implementation, and AndroidWorld adapter.

## Fixed Interfaces

- Public Function write API: MCP `save_function`.
- Function schema: `omniflow.function.v2`.
- Bundle schema: `omniflow.function-bundle.v2`.
- Paper manifest: `omniflow.function-agent-skill-manifest.v1`.
- RunLog schema: `omniflow.run_log.v1`.
- AndroidWorld entry: `scripts/exp/run_androidworld.sh`.

Do not add parallel create/update/convert APIs, compilers, Stores, runners, or
adapters. `save_function` accepts `run_id` or `run_log`, one complete Function,
its source `arguments`, and `agent_visible`. Discover its exact request schema
through MCP `tools/list`.

## Semantic Rule

One Function represents one reusable operation, even when it expands to many
GUI actions. Split only when contiguous operations are independently meaningful
and reusable. Never split merely because an argument selects another visible
item or that item appears at a different coordinate.

A full task may call the same Function repeatedly, call several Functions, and
continue planning afterward. Function success is a normal tool result, not task
completion.

## Authoring Rule

1. Read the goal, ordered successful actions, action metadata, and source state
   evidence. Never use target-device evidence for authoring.
2. Preserve recorded action order and canonical `source_state_id` values.
3. Expose task-varying values in `input_schema` and bind them directly:
   - entered text -> `$.steps[n].action.args.text`;
   - app -> `$.steps[n].action.args.package_name`;
   - visible choice -> `$.steps[n].action.args.target_description`.
4. Persist the intended visible label in `target_description`. Keep recorded
   coordinates only as immutable source evidence and source-layout fallback.
5. Require every source call to match a contiguous successful RunLog sequence.
   Multiple calls may prove one Function at different coordinates.
6. Require at least one source call to prove the stored fallback coordinates;
   every other semantic value must be visible in its corresponding source state.

At runtime OmniFlow grounds `target_description` on the current page, then uses
canonical OmniTransfer when needed. Never pass source coordinates through to a
target device.

## Enhancement Rule

`enhance_function()` may add, delete, modify, or reorder actions and change
parameters or bindings. The revised Function must remain completely grounded in
contiguous successful source evidence. Do not add task-specific planner rules,
hidden validator knowledge, mechanical templates, coordinate scaling, or target
observations.

A schema/compiler error means the authored Function violates the shared
contract. Correct the Function or the generic contract; do not special-case the
task.

## Versioned Workflow

For MCP/OOB use, call `save_function`; it compiles, validates, and persists the
complete Agent-authored Function internally. Saving the complete revised
Function under its stable `function_id` is the update operation.

For paper assets, emit one immutable authoring manifest containing exact source
index and RunLog SHA-256 values, a short semantic rationale, and the Function
bundle with ordered source calls. It must contain no target input, target state,
validator secret, or repaired model output.

Before AndroidWorld work, read repository `AGENTS.md` and
`scripts/exp/README.md`. Use only:

```bash
OMNIFLOW_OURS_AUTHORING_MANIFEST=/absolute/manifest.json \
OMNIFLOW_OURS_CONVERTED_ASSET_ROOT=/absolute/new-version \
bash scripts/exp/run_androidworld.sh --convert-ours-assets --tasks TASK

OMNIFLOW_MEMORY_FUNCTION_CATALOGS=/absolute/catalog.json \
bash scripts/exp/run_androidworld.sh --refresh-memory

bash scripts/exp/run_androidworld.sh \
  --check-only --tasks TASK --methods ours --devices small5554
```

Source qualification also enters through the unified script's E2E pipeline,
never through a Skill-owned or direct Python runner:

```bash
bash scripts/exp/run_androidworld.sh \
  --e2e-task TASK --source-backend reuse-only --source-qualification-only
```

## Acceptance

Conversion alone is insufficient. A version is executable only when ordered
seed-111 source calls succeed, AndroidWorld's official validator passes,
`model_calls=0`, and `fallback_steps=0`. Record immutable RunLog, Store,
transfer-state, provenance, runtime, and result hashes.

On failure, preserve the attempt, report the first failed boundary, author a new
version from source evidence, and repeat. Report Function IDs/count, source
calls, hashes, canonical memory identity, validator result, actions, model calls,
fallback steps, and duration.
