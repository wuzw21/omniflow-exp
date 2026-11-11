# AndroidWorld Offline Replay Audit 2026-06-19

Status: evidence note, not a replacement for the frozen paper result matrix.

This audit checks whether existing AndroidWorld-related trace fixtures and the
current latest-result index are still usable after OmniFlow runtime cleanup.
It does not run a live AndroidWorld device validator and does not supersede the
paper-facing frozen result policy in `UTG_papers/nsdi/eval_matrix.md`.

## Commands

Repository:

```text
/Users/wuzewen/Projects/Omni/OmniFlow
```

Offline replay fixture regression:

```bash
python -m pytest src/utg/testing/test_offline_replay_runner.py -q
```

Result:

```text
13 passed in 12.45s
```

Core offline replay / simple executor regression:

```bash
python -m pytest \
  tests/utg_core/testing/test_offline_replay.py \
  tests/utg_core/execution/test_executor_simple.py \
  -q
```

Result:

```text
57 passed in 0.73s
```

Latest AndroidWorld result index rebuild:

```bash
python scripts/androidworld_baseline_eval.py --organize-latest-results
```

Result generated at:

```text
2026-06-19T12:23:39
```

## Existing Trace Fixtures Exercised

| Fixture | Goal | Expected Function | Expected Actions |
|---|---|---|---|
| `tests/fixtures/offline_replay/androidworld_open_app_contacts/trace_001` | Open the contacts app and clear required permission pop-ups. | `aw_function_213a4dcca1` | `open_app`, `click` |
| `tests/fixtures/offline_replay/androidworld_open_app_contacts/trace_002` | Open the contacts app and clear required permission pop-ups. | `raw_replay_6d7643fd` | `open_app` |
| `tests/fixtures/offline_replay/demo_open_settings/trace_001` | 打开设置 | `path_open_settings` | `click`, `finished` |
| `tests/fixtures/offline_replay/demo_open_settings/trace_002` | 打开设置 | `path_open_settings` | `click`, `finished` |

The offline replay tests execute these fixtures through the current replay
runtime and verify that all replay events are consumed.

## Latest Index Snapshot

Current rebuilt manifest:

```text
/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/manifest.json
```

Snapshot:

| Metric | Value |
|---|---:|
| Latest rows | 72 |
| Tasks | 38 |
| Success rows | 45 |
| Failure rows | 27 |
| Official AndroidWorld validator successes | 29 |
| Official AndroidWorld validator failures | 13 |
| Source pool records | 391 |
| Source pool tasks | 58 |
| Missing local source refs | 0 |

Backend snapshot:

| Backend | Total | Success | Failure | Official Success | Official Failure |
|---|---:|---:|---:|---:|---:|
| `omniflow-native` | 32 | 29 | 3 | 29 | 3 |
| `autodroid` | 9 | 0 | 9 | 0 | 9 |
| `mobilegpt` | 1 | 0 | 1 | 0 | 1 |
| `mobilegpt-self` | 30 | 16 | 14 | 0 | 0 |

## Interpretation

- Existing offline replay fixtures are usable after the runtime cleanup.
- The offline source pool remains self-contained enough for audit use:
  `missing_local_source_refs=0`.
- The rebuilt latest index is close to, but not identical to, the older frozen
  manifest text. The older paper-facing manifest listed 73 latest rows and
  `omniflow-native` as 29/33 official successes. This rebuild lists 72 latest
  rows and `omniflow-native` as 29/32 official successes.
- Do not update headline paper numbers from this audit alone. Treat it as a
  reproducibility check and investigate the one-row index delta before changing
  `eval_matrix.md` or the paper text.

## Live Validator Preflight

A minimal live smoke was attempted with:

```bash
bash scripts/androidworld_run.sh \
  --tasks OpenAppTaskEval \
  --console-port 5554 \
  --max-steps 6 \
  --utg-store-path runtime/utg/utg_store.json \
  --output-path runtime/evals/androidworld_validator/live_smoke_20260619_openapp \
  --checkpoint-dir runtime/evals/androidworld_validator/live_smoke_20260619_openapp/checkpoints \
  --task-random-seed 42 \
  --fixed-task-seed \
  --no-emulator-setup
```

It did not reach policy execution or AndroidWorld validator scoring. The first
attempt attached to an already-running `SmallPhone` emulator on `emulator-5554`
and failed during AndroidWorld simulator initialization:

```text
grpc._channel._InactiveRpcError: StatusCode.UNAUTHENTICATED
details = "Missing the 'authorization' header with security credentials."
```

The AndroidWorld AVD restart helper was then invoked:

```bash
rm -f /Users/wuzewen/.android/avd/AndroidWorldAvd.avd/multiinstance.lock
bash /Users/wuzewen/Projects/Omni/OmniFlow/scripts/start_androidworld_avds.sh
adb devices -l
```

`AndroidWorldAvd` did not come online in adb within the checked window; the
emulator logs at `/tmp/omniflow_androidworld_avds/AndroidWorldAvd-5554.log`
contained only the emulator startup header. This is an AndroidWorld/emulator
environment blocker, not a replay-policy result.

## Source Paths

Offline replay fixtures:

```text
/Users/wuzewen/Projects/Omni/OmniFlow/tests/fixtures/offline_replay/
```

Latest AndroidWorld index:

```text
/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/
```

Offline source run-log pool:

```text
/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/offline_source_runlog_pool/
```
