# File Edit Guide

This is the only maintained file-ownership guide for production changes.
`AGENTS.md` defines repository rules; this file identifies the one owner for
each concern. Tests change beside the owner they verify.

## Public path

```text
scripts/exp/run_androidworld.sh
  -> src/experiment/e2e_task_pipeline.py
  -> src/experiment/androidworld.py
  -> src/integrations/android_world/launch.py
  -> src/integrations/android_world/methods.py
  -> omniflow/runtime/engine.py
  -> omniflow/runtime/execution.py
```

The only AndroidWorld method names are `fixed_replay`, `omniflow`,
`mobilegpt`, `appagent`, and `t3a_hint`. Do not introduce aliases, suffixes,
or method-specific lifecycle names. B-MoCA selectors are external benchmark
inputs and remain owned by its launcher contract.

## One owner per concern

| Concern | Edit only here |
| --- | --- |
| Public flags and process entry | `scripts/exp/run_androidworld.sh` |
| Frozen methods, devices, seeds, and budgets | `config/paper_androidworld.json` |
| Typed protocol view | `src/experiment/protocol.py` |
| Task/method/device scheduling | `src/experiment/e2e_task_pipeline.py` |
| One AndroidWorld result | `src/experiment/androidworld.py` |
| Native lifecycle and launcher | `src/integrations/android_world/launch.py` |
| Method adapters | `src/integrations/android_world/methods.py` |
| OmniFlow AndroidWorld adapter | `src/integrations/android_world/agent.py` |
| Native observation and actions | `src/integrations/android_world/host.py` |
| Function validation and writing | `omniflow/functions/assets.py` |
| Function selection | `omniflow/functions/recall.py` |
| Planner lifecycle | `omniflow/runtime/engine.py` |
| Action transfer and fallback | `omniflow/runtime/execution.py` |
| OmniTransfer page encoding | `omniflow/transfer/page_embedding.py` |
| External JSON-line API | `omniflow/bridge.py` |
| Canonical local index | `src/experiment/local_data.py` |
| Result row schema | `src/experiment/result_schema.py` |
| Result registration | `src/experiment/result_registry.py` |
| Source evidence and conversion dispatch | `src/experiment/source_assets.py` |
| MobileGPT source preparation | `src/experiment/mobilegpt_source.py` |
| AppAgent source preparation | `src/experiment/appagent_source.py` |
| MobileGPT native conversion | `src/integrations/mobilegpt_converter.py` |
| AppAgent native conversion and runtime | `src/integrations/appagent_adapter.py` |
| B-MoCA device adapter | `src/integrations/bmoca.py` |

## Change rules

- Change a public name in the protocol, launcher, adapters, and focused tests
  together. Keep only the canonical simple name.
- Change a schema or public result field in its owner, its schema tests, and
  the relevant README in one separate commit.
- Add no second scheduler, executor, Function writer, index, manifest,
  provenance registry, checker registry, or lifecycle wrapper.
- If a helper has no production or test caller, delete it instead of adding an
  alias or preserving a compatibility branch.
- Keep B-MoCA's native baseline adapters separate from AndroidWorld's method
  names; do not copy their external selectors into the AndroidWorld protocol.
