---
name: androidworld-runtime-preflight
description: Deterministically gate and audit AndroidWorld MobileGPT environments before formal evaluation.
---

# AndroidWorld Runtime Preflight

Run formal episodes only after the deterministic host gate passes. Preserve the
same seed and task parameters when repairing an environment failure.

Use `--profile appagent` for AppAgent-only single-task episodes. It checks the
AppAgent deployment and AndroidWorld dependencies without requiring the
MobileGPT server or the full 116-task source pool. The default profile remains
`mobilegpt`; `--function-manifest` selects the Function profile.

## Host Gate

```bash
python skills/androidworld-runtime-preflight/scripts/preflight.py \
  --repo "$PWD" \
  --serial emulator-5554 \
  --expected-tasks 116 \
  --require-kvm \
  --require-device \
  --require-contacts-ready \
  --json-out runtime/evals/androidworld_validator/runtime_preflight.json
```

Do not start a formal task when the command exits nonzero. Repair every failed
check first. Never treat a warning from AndroidWorld app setup as harmless.

## Per-Task Audit

```bash
python skills/androidworld-runtime-preflight/scripts/classify_result.py \
  --summary one_task_summary.json \
  --log task.log \
  --initial-memory-condition native_memory \
  --frozen-manifest frozen_memory_manifest.json
```

Use `empty_memory` for a cold episode; it does not require a prebuilt frozen
manifest. `native_memory` and `function_transfer` require the immutable
manifest shown above.

- `success`: preserve the official success and continue.
- `method_failure`: preserve the failure and continue without changing memory.
- `environment_failure`: stop, repair the failed gate, and rerun the same task,
  frozen seed, and parameters.

`task_finished=0` alone is a method failure after one clean `task_started` and
official-validator coverage. Missing setup, device, OOB, required warm memory,
`task_started`, validator coverage, or result artifacts are environment failures.

## Formal Invariants

- Use one normal AndroidWorld episode per task and its official validator.
- Keep prebuilt memory immutable throughout warm evaluation.
- Never inspect target seed values, target XML, or validator state while
  authoring or changing memory.
- Separate setup costs from online warm calls, tokens, and latency.
- Preserve every attempt.
