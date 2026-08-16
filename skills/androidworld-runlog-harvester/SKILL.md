---
name: androidworld-runlog-harvester
description: "Author, revise, save, and source-qualify versioned semantic OmniFlow Functions from successful AndroidWorld RunLogs. Use for Function schema or binding failures, over-split workflows, MCP save_function calls, and immutable Function asset refresh or replay."
---

# Semantic Function Authoring

Author reusable semantic Functions from successful RunLogs. Leave compilation,
storage, transfer, execution, and AndroidWorld adaptation to OmniFlow.

## Fixed Interfaces

- Function write API: MCP `save_function` only.
- Function schema: `omniflow.function.v2`.
- Bundle schema: `omniflow.function-bundle.v2`.
- Paper manifest: `omniflow.function-agent-skill-manifest.v1`.
- RunLog schema: `omniflow.run_log.v1`.
- AndroidWorld entry: `scripts/exp/run_androidworld.sh`.

Discover the current request with MCP `tools/list`. Submit one complete Function;
include `run_id` or `run_log`, source `arguments`, and `agent_visible` when
authoring from evidence. Do not create another API, compiler, Store, runner, or
adapter.

## Semantic Rule

One Function represents one reusable operation, even when it expands to many
GUI actions. Split only independently meaningful, reusable operations; never
split because an argument changes a visible choice or coordinate.

A full task may call the same Function repeatedly, call several Functions, and
continue planning afterward. Function success is a normal tool result, not task
completion.

## Authoring Rule

1. Use only the goal, contiguous successful source actions, action metadata,
   and source states. Preserve action order and canonical `source_state_id`.
2. Expose task-varying values in `input_schema` and bind them directly:
   - entered text -> `$.steps[n].action.args.text`;
   - app -> `$.steps[n].action.args.package_name`;
   - visible choice -> `$.steps[n].action.args.target_description`.
3. Persist the intended visible label in `target_description`. Keep recorded
   coordinates only as immutable source evidence and source-layout fallback.
4. Require every source call to match a contiguous successful RunLog sequence.
   Multiple calls may prove one Function at different coordinates.
5. Require at least one source call to prove stored fallback coordinates;
   every other semantic value must be visible in its corresponding source state.

At runtime OmniFlow grounds `target_description` on the current page, then uses
canonical OmniTransfer when needed. Never pass source coordinates through to a
target device.

## Enhancement Rule

Revise a Function by submitting its complete new version with the same stable
`function_id` to `save_function`. Actions and bindings may change only when the
new version remains grounded in contiguous successful source evidence. Do not
use task-specific planner rules, validator knowledge, templates, coordinate
scaling, or target observations.

A schema/compiler error means the authored Function violates the shared
contract. Correct the Function or the generic contract; do not special-case the
task.

## Versioned Workflow

`save_function` compiles, validates, and persists the submitted version. For
paper assets, emit one immutable authoring manifest with exact source-index and
RunLog SHA-256 values, a short rationale, the Function bundle, and ordered
source calls. Exclude target input, target state, validator secrets, and repaired
model output.

Before AndroidWorld work, read `AGENTS.md` and `scripts/exp/README.md`. Run every
conversion, memory refresh, check, and qualification through
`scripts/exp/run_androidworld.sh`; never add or invoke a Skill-owned runner.

## Acceptance

Accept a version only when ordered seed-111 source calls succeed, the official
validator passes, `model_calls=0`, and `fallback_steps=0`. Record immutable
RunLog, Store, transfer-state, provenance, runtime, and result hashes.

On failure, preserve the attempt, report the first failed boundary, author a new
version from source evidence, and repeat. Report Function IDs/count, source
calls, hashes, canonical memory identity, validator result, actions, model calls,
fallback steps, and duration.
