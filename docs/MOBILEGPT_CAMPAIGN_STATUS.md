# MobileGPT AndroidWorld 116 × 3 Campaign

This document records only the MobileGPT campaign on host `9207` with source
seed `111` and evaluation seed `113`. A cell is complete when the official
AndroidWorld validator produced a boolean conclusion. Environment and memory
preparation failures remain retryable and are not recorded as completed cells.

The working spreadsheet is `/Users/wuzewen/Desktop/mobilegpt.xlsx`. It is a
separate copy based on `OmniFlow_AndroidWorld_116Tasks_10cell_extended.xlsx`;
the reference sheets are unchanged, and the `MobileGPT Results` sheet contains
all 116 tasks × Standard/Fold/Tablet cells.

## Progress

- Total cells: 348
- Completed conclusions: 13
- Official validator pass: 5
- Confirmed method failure: 8
- Remaining: 335
- Models: `GLM-4.6V`, `GLM-Embedding-2`
- Step budget: 20

## Recorded groups

### 9207-existing-20260823

| Task | Device | Conclusion | Actions | Episode calls | Total calls | Tokens | Memory hit | Fallback | Latency (s) | Attempt |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AudioRecorderRecordAudio | Standard / small5562 | method failure | 3 | 19 | 52 | 16,136 | 0/0 | 0 | 80.802 | attempt_022 |
| CameraTakePhoto | Fold / fold5564 | method failure | 20 | 81 | 126 | 88,114 | 0/0 | 0 | 362.023 | attempt_013 |
| CameraTakePhoto | Standard / small5562 | method failure | 20 | 102 | 147 | 107,056 | 0/0 | 0 | 317.941 | attempt_013 |
| ClockStopWatchRunning | Standard / small5562 | method failure | 4 | 17 | 35 | 11,955 | 0/0 | 0 | 160.343 | attempt_011 |
| ContactsAddContact | Fold / fold5564 | method failure | 2 | 10 | 63 | 6,999 | 0/0 | 0 | 202.976 | attempt_002 |
| ContactsAddContact | Tablet / small5554 | method failure | 3 | 13 | 66 | 9,602 | 0/0 | 0 | 294.030 | attempt_004 |
| ContactsAddContact | Standard / small5562 | pass | 9 | 38 | 91 | 42,759 | 0/0 | 0 | 368.155 | attempt_002 |
| MarkorCreateFolder | Fold / fold5564 | pass | 4 | 19 | 52 | 21,922 | 0/0 | 0 | 113.360 | handshake_fix_fold_perm |
| MarkorCreateFolder | Tablet / small5554 | pass | 5 | 23 | 56 | 27,423 | 0/0 | 0 | 132.804 | handshake_fix_tablet_main |
| MarkorCreateFolder | Standard / small5562 | pass | 15 | 51 | 84 | 66,269 | 0/0 | 0 | 276.281 | handshake_fix_small |
| SystemBluetoothTurnOn | Standard / small5562 | pass | 2 | 15 | 40 | 13,326 | 0/0 | 0 | 61.686 | attempt_001 |

### SystemBluetoothTurnOn / attempt_002

The Standard cell was skipped because `attempt_001` already contains an
official PASS. Fold and Tablet both passed runtime preflight and reached the
official validator; neither is an environment failure.

| Task | Device | Conclusion | Actions | Episode calls | Total calls | Tokens | Memory hit | Fallback | Latency (s) | Failure reason |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SystemBluetoothTurnOn | Fold / fold5564 | method failure | 5 | 17 | 42 | 17,351 | 0/0 | 0 | 135.924 | mobilegpt_step_timeout |
| SystemBluetoothTurnOn | Tablet / small5554 | method failure | 0 | 21 | 46 | 21,807 | 0/0 | 0 | 78.968 | mobilegpt_handshake_failed |

`0/0` means the official result emitted zero memory lookup events; it is not
rewritten as a successful memory hit. Missing energy measurements remain blank.

## Update contract

After each task group, add only cells with an official validator conclusion to
this document and `mobilegpt.xlsx`. Preserve environment/preparation failures
as retryable evidence, increment attempt IDs naturally, and skip every cell
already recorded here.
