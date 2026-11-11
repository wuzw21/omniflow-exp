# Memory latency diagnosis — 2026-09-09

This is a single-task baseline profile, not an optimization or paper speed result.
All task executions used the existing `scripts/exp/run_androidworld.sh run` entry,
`SystemBluetoothTurnOn`, `omniflow`, `standard45562`, builtin Harness, evaluation
seed 113 and the configured Qwen3.6-Plus endpoint. Code revision: `a48ac384`.
Memory ON explicitly used
`data/androidworld/SystemBluetoothTurnOn/omniflow/OmniFlowSourceSmall_seed111/memory/current/store.json`;
OFF omitted `--memory`. Order: OFF → ON → ON → OFF. No runtime code was changed.

| Run | Official success | Execution seconds | Model calls | OOB observations | OOB observation seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| OFF-1 | No | 73.065 | 10 | 11 | 26.695 |
| ON-1 | Yes | 49.855 | 1 | 12 | 34.855 |
| ON-2 | Yes | 43.114 | 1 | 11 | 28.744 |
| OFF-2 | Yes | 74.200 | 10 | 11 | 26.793 |

OFF-2 reached official success but its runtime reported `max_steps_exceeded`;
these facts remain separate. OFF-1 must not enter success latency. Only the second
OFF/ON pair has both official successes, insufficient to establish a general
speed advantage. The OFF branch also omits the explicit 30-second model request
limit used by ON; align this policy before a controlled speed claim.

Exclusive component timing reconciled with the common execution owner to less
than 1 ms in all four episodes. The existing timing regression suite passed
(`.venv/bin/python -m pytest tests/test_wall_accounting.py -q`: 7 passed).
On the two Function success paths, OOB observation accounts for approximately
67–70% of execution time. ON-1 used three stable rechecks. OFF still performed
ten page encodings despite having no Functions. Per-episode image files occupied
approximately 4.8–6.4 MB in the first three runs; this measures disk bytes, not RAM.

A separate read-only OOB diagnostic varied the existing `includeScreenshot`
broadcast field in process, without changing production code or executing any
actions. Four image ON/OFF/OFF/ON reads produced 3.048/0.547/0.467/1.029 seconds
with identical XML hashes. Four subsequent image-enabled reads with stable-wait
ON/OFF/OFF/ON produced 4.654/0.901/0.877/2.950 seconds. These small samples have
warm-up and device noise; broadcast duration includes APK work and cannot isolate
capture, compression, or XML generation. They indicate that stable waiting and
image production deserve measurement before changing transport or mapping.

Next verification targets: redundant stable waits in the existing observation
owner; image retrieval only when current pixels are unnecessary; empty-candidate
recall without page encoding; aligned request budgets. Do not remove mandatory
post-action feedback or reuse stale pixels in canonical OmniTransfer.

Raw logs, per-run diagnostic snapshots, input hashes, microbenchmark timings,
immutable RunLog references/hashes, and calculated results are stored in
`data/runtime/validation/20260909-memory-latency/summary.json` and sibling files.
This run used a local emulator, not a physical phone. No acceleration fix has
been implemented or accepted by these tests.
