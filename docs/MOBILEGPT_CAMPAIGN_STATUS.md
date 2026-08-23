# MobileGPT AndroidWorld 116 × 3 Campaign

This document records only the MobileGPT campaign on host `9207` with source
seed `111` and evaluation seed `113`. A failed cell is final only when the
official AndroidWorld validator produced a boolean conclusion and the sealed
memory passed the RunLog-teacher alignment audit. An official PASS remains
valid historical evidence. Environment, preparation, and legacy-memory
failures remain retryable and are not recorded as completed cells.

The working spreadsheet is `/Users/wuzewen/Desktop/mobilegpt.xlsx`. It is a
separate copy based on `OmniFlow_AndroidWorld_116Tasks_10cell_extended.xlsx`;
the reference sheets are unchanged, and the `MobileGPT Results` sheet contains
all 116 tasks × Standard/Fold/Tablet cells.

## Progress

- Total cells: 348
- Completed conclusions under the strong memory contract: 5
- Official validator pass: 5
- Confirmed method failure with authoritative memory: 0
- Legacy-memory attempts pending retry: 11
- Remaining: 343
- Models: `GLM-4.6V`, `GLM-Embedding-2`
- Step budget: 20

## Recorded groups

### 9207-existing-20260823

| Task | Device | Conclusion | Actions | Episode calls | Total calls | Tokens | Memory hit | Fallback | Latency (s) | Attempt |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| AudioRecorderRecordAudio | Standard / small5562 | legacy memory retry | 3 | 19 | 52 | 16,136 | 0/0 | 0 | 80.802 | attempt_022 |
| CameraTakePhoto | Fold / fold5564 | legacy memory retry | 20 | 81 | 126 | 88,114 | 0/0 | 0 | 362.023 | attempt_013 |
| CameraTakePhoto | Standard / small5562 | legacy memory retry | 20 | 102 | 147 | 107,056 | 0/0 | 0 | 317.941 | attempt_013 |
| ClockStopWatchRunning | Standard / small5562 | legacy memory retry | 4 | 17 | 35 | 11,955 | 0/0 | 0 | 160.343 | attempt_011 |
| ContactsAddContact | Fold / fold5564 | legacy memory retry | 2 | 10 | 63 | 6,999 | 0/0 | 0 | 202.976 | attempt_002 |
| ContactsAddContact | Tablet / small5554 | legacy memory retry | 3 | 13 | 66 | 9,602 | 0/0 | 0 | 294.030 | attempt_004 |
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
| SystemBluetoothTurnOn | Fold / fold5564 | legacy memory retry | 5 | 17 | 42 | 17,351 | 0/0 | 0 | 135.924 | mobilegpt_step_timeout |
| SystemBluetoothTurnOn | Tablet / small5554 | legacy memory retry | 0 | 21 | 46 | 21,807 | 0/0 | 0 | 78.968 | mobilegpt_handshake_failed |

### SystemBluetoothTurnOff / attempt_002

All three devices passed runtime preflight and reached the official validator,
but the reused `attempt_001` manifest explicitly had
`runlog_teacher_alignment=false`. These rows are retained as diagnostic
evidence and are not final cells under the strong memory contract.

| Task | Device | Status | Actions | Episode calls | Total calls | Tokens | Fallback | Latency (s) | Concrete failure |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| SystemBluetoothTurnOff | Standard / small5562 | legacy memory retry | 5 | 25 | 46 | 20,171 | 0 | 99.098 | Server `KeyError: 4`; mobilegpt_server_handler_failed |
| SystemBluetoothTurnOff | Fold / fold5564 | legacy memory retry | 0 | 19 | 40 | 16,739 | 0 | 78.211 | mobilegpt_handshake_failed |
| SystemBluetoothTurnOff | Tablet / small5554 | legacy memory retry | 0 | 22 | 43 | 21,876 | 0 | 82.748 | mobilegpt_handshake_failed |

`0/0` means the official result emitted zero memory lookup events; it is not
rewritten as a successful memory hit. Missing energy measurements remain blank.

## Update contract

After each task group, add final cells only when the sealed manifest passes the
strong RunLog-teacher alignment audit. Preserve environment, preparation, and
legacy-memory failures as retryable evidence, increment attempt IDs naturally,
and skip only final cells already recorded here.
