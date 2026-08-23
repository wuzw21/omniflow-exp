# AndroidWorld failed RunLog recollection

This workflow re-collects only the source tasks that are not currently
qualified in `data/current.json`. A task is accepted only when the same local
AndroidWorld environment produces a complete native RunLog with screenshots,
an explicit decision trace for every action, and an official validator success.

## System prompt

Use this prompt for the operator/model driving the manual harness:

```text
You are the AndroidWorld source RunLog collector.

Goal: complete exactly one task from the current data/current.json queue on the
same local AndroidWorld environment, then preserve a replayable source RunLog.

Rules:
1. Work on one task only. Read its goal and task parameters from current.json.
2. Use source seed 111 and the selected local emulator only. Do not switch
   devices or reuse another task's state.
3. Observe before deciding. Treat the native observation and screenshot as the
   only current UI truth. Never infer a hidden state from an old observation.
4. Before every action, write a short reasoning string that states the visible
   evidence, the intended effect, and why the action is safe now. Send it as
   the action's `reasoning` field.
5. Use only AndroidWorld's native observe/act path. Do not use fixed replay,
   source coordinates copied from another RunLog, resource-id/node-id lookup,
   or a second executor.
6. After each action, wait for and inspect the next observation. The harness
   stores its screenshot and native state in the RunLog.
7. Use the task's official validator for the final decision. A failed or
   incomplete run is evidence only; do not promote it or edit current.json.
8. Stop after official success, report every action and the RunLog path, then
   move to the next queue item. Never silently skip a failed task.
```

## Queue

`data/current.json` is the only queue and runtime index. The current source
qualification candidates are listed with:

```bash
./.venv/bin/python -c 'import json; payload=json.load(open("data/current.json")); print("\n".join(task for task, row in payload["source_index"].items() if row.get("latest_official_success_source") is not True))'
```

For one task, read `goal`, `params`, and `source_seed` from its row. Use the
row's `params` exactly; do not copy parameters from the retained historical
RunLog.

## One-task protocol

Use the existing manual harness on the retained local emulator. Its public
input is JSON Lines. Every `act` or selector `click` must include `reasoning`:

```bash
./.venv/bin/python tools/manual_androidworld_harness.py \
  --android-world-root /absolute/releases/android-world-<revision> \
  --task TASK \
  --task-params-json '<PARAMS_JSON>' \
  --seed 111 \
  --console-port 5554 \
  --device-serial emulator-5554 \
  --adb-path /absolute/Android/Sdk/platform-tools/adb \
  --output /absolute/OmniFlow-exp/data/androidworld/TASK/small5554/source/manual_<UTC>
```

For a USB-connected Android device, keep the same AndroidWorld task/session
contract but pass its ADB serial explicitly (for example
`--device-serial 45291FDAP0013Z`) and set `ANDROID_SERIAL` to the same value.
This records the real device provenance; it does not make the device part of
the formal emulator matrix.

The interaction loop is:

1. Send `{"cmd":"observe"}` and inspect the returned native observation.
2. Send one `act`/`click` with a concise `reasoning` field.
3. Confirm the returned observation and screenshot before the next action.
4. Send `{"cmd":"validate","reasoning":"..."}` only after the task is visibly complete.
5. Require `success: true`, `validator.official: true`, non-empty steps,
   screenshot references, and a reasoning entry for every step.
6. Send `{"cmd":"quit"}` and record the absolute `run_log.json` path.

The harness records screenshots and native observations after every action. A
successful source RunLog is the only artifact eligible for Function authoring.
Do not call `compile_runlog_to_store` for a failed or aborted RunLog.

## Promotion and retry

Keep raw manual evidence under `data/`; it is not committed. After a successful
RunLog, refresh the canonical index through the repository entry:

```bash
bash scripts/exp/run_androidworld.sh --refresh-memory
```

Then re-read `data/current.json`. The task is complete only when its
`source_index.<TASK>.latest_official_success_source` is `true` and the retained
source has screenshots. If the run fails, stays incomplete, loses screenshots,
or the validator is not official, leave the index unchanged and repeat that
same task before taking the next one.

## Evidence checklist

- `seed == 111` and the task parameters match `data/current.json`.
- One local device/environment is used for the whole task.
- Every step has native before/after observations and a screenshot reference.
- Every action has non-empty operator reasoning.
- The RunLog has `validator.official == true` and `validator.success == true`.
- The promoted source row is visible in `data/current.json` after refresh.
