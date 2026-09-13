# Adaptive experiment scenarios

This is the maintained coverage contract for supplementary experiments. It is
not a scheduler, source selector, result registry, or replacement for the
116-task × 15-cell formal table. All AndroidWorld execution uses the existing
`scripts/exp/run_androidworld.sh run` entry and explicit Memory/Source paths.
Software regressions use pytest under `tests/`; simulated Host/Transfer inputs
prove control flow only, never device success or learned mapping accuracy.

## Coverage and executable owners

| Scenario | Existing executable regression/evaluation | Required live evidence |
|---|---|---|
| Stable reuse | `tests/test_completion_verification.py` | Real Function actions and official success |
| Changed task values | `tests/test_function_render_binding.py` | Explicit source/target parameters, bound actions and validator |
| Phone/Fold/Tablet layout changes | canonical OmniTransfer ASE/PAPT evaluators; `tests/test_transfer_candidate_admission.py` | Correct target selection, per-attempt rejection/acceptance and task outcome |
| Multiple local Functions | `test_incomplete_local_function_can_route_to_second_local_function` | Actual independent selections in one task, not a prescribed Function sequence |
| Failure → Planner → same Function | `test_recovery_preserves_prefix_and_records_actual_reuse[resume_failed_step]` | Failed step, recovery actions, resume index, no repeated prefix |
| Failure → Planner → another Function | `test_recovery_preserves_prefix_and_records_actual_reuse[select_other_function]` | Different Function selected after failure and remaining goal completed |
| Failure persists after recovery | `test_failed_resume_stays_failed_and_never_dispatches_source_point` | Explicit rejection and zero source-coordinate dispatch |
| Resume arguments changed | `test_resume_argument_change_cannot_repeat_prefix_or_change_remaining_effect` | Rejection before dispatch and preserved existing effects |
| Online observation / content decisions | `test_online_observation_hides_complete_replay_and_keeps_safe_local` | Fresh content read by Planner; hidden full evidence not executed as a static script |
| Repeated operations | `test_authoring_harness_registers_one_repeated_function_with_three_calls` | Runtime-selected invocations and task-specific count, not source_calls replay |
| Pop-up/package recovery | `tests/test_checker_restore.py`, `tests/test_checker_lifecycle.py` | Actual interruption, Checker action, resumed flow; record not_triggered if absent |
| Function success / task failure | `test_successful_resume_does_not_override_official_rejection` | Separate execution and official completion facts |
| Memory OFF | `test_memory_off_uses_same_kernel_but_never_recalls_or_exposes_functions` | Same Store/model/Host/configuration, zero Function invocation |
| Reentry OFF | `test_reentry_ablation_keeps_same_planner_actions_and_completion_gate` | First Function failure followed by Planner-only execution |
| New task after recovery | `test_reentry_ablation_and_recovery_counters_reset_for_new_task` | New task state/counters without replaying old effects |
| Cancellation, retry, reconnect, unknown effect | `tests/test_invocation_protocol.py` | Physical-device action sequence, stale-session rejection and zero duplicate effects |
| Paired latency and recovery costs | `tests/test_wall_accounting.py` | Complete paired episodes, usage and reconciled exclusive time |

Executable regression entry (repository Python):

```bash
.venv/bin/python -m pytest -q tests
```

## Frozen supplementary workload

The initial workload is selected by task semantics, before inspecting the new
results: CameraTakePhoto (stable control), RecipeAddSingleRecipe (parameter
binding), SimpleSmsSendClipboardContent (online observation),
RecipeAddMultipleRecipes (repeated operations), and RecipeDeleteDuplicateRecipes
(content-dependent selection). They are scenario probes, not a replacement for
all 116 tasks or evidence that every interruption/branch actually occurred.

Targets remain Standard, Fold, and Tablet. Every task uses the explicit paths:

```text
data/androidworld/<TASK>/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json
data/androidworld/<TASK>/omniflow/OmniFlowSourceSmall_seed111/memory/current/store.json
```

Do not replace a failed task or choose a different historical Memory after
seeing its outcome. Verify task parameters from actual evidence: equal seeds
alone do not certify equality. The source-only small phone is never a target.

Asset qualification precedes mechanism claims. The initial legacy multi-recipe
Store contains a monolithic three-recipe Function, and the duplicate-deletion
Store exposes an observation-dependent workflow as static replay. Neither is
evidence of the current local-composition/online-observation contract. Preserve
the initial batch as historical evidence, re-author through `convert-memory`
into a new explicit immutable bundle, and freeze a new
batch identity before evaluating those mechanisms. Do not patch old Stores or
mix their results with the new authoring condition.

The first re-authoring attempt exposed an invalid semantic classification of
live-discovered record titles as task parameters; the prompt now distinguishes
predicate-based discovery from goal-supplied values. The duplicate-record
regression exercises the boundary: live inspection/comparison remains with the
Planner and only the selected-record deletion suffix is executable. The compiler
reports the exact occurrence, source step, and missing Agent-declared parameter
when a binding is incomplete; it still does not infer or fill bindings.

For SMS and duplicate deletion, `memory/adaptive_authoring_002/` is the reviewed
new bundle used by `adaptive-v3-20260913`. SMS exposes paste/send; duplicate
deletion exposes menu/delete/confirm. Both full workflows are hidden evidence.
The multi-recipe conversion timed out twice; no accepted current bundle or
live repeated-composition success is claimed for those attempts.

## Experimental conditions

For each task, run the same three targets and explicit assets in this order:

| Condition | Memory | Reentry | Checker | Target seed |
|---|---|---|---|---|
| memory_off_1 | off | on | on | 113 |
| full_1 | on | on | on | 113 |
| full_2 | on | on | on | 113 |
| memory_off_2 | off | on | on | 113 |
| no_reentry | on | off | on | 113 |
| no_checker | on | on | off | 113 |
| no_reentry_no_checker | on | off | off | 113 |
| source_parameter_seed | on | on | on | 111 |

The first four runs use OFF/ON/ON/OFF ordering. Pair OFF1 with ON1 and OFF2
with ON2, preserving both replicates and clustering inference by task. The
seed-111 condition isolates target parameter generation relative to seed 113;
it still changes device/layout relative to the source and must not be labelled
same-device exact replay. Checker and reentry conditions only demonstrate the
mechanism if the relevant trigger actually occurred.

One concrete command, using the existing entry:

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto --method omniflow --device all \
  --source-run-log data/androidworld/CameraTakePhoto/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json \
  --memory data/androidworld/CameraTakePhoto/omniflow/OmniFlowSourceSmall_seed111/memory/current/store.json \
  --function-memory off --function-reentry on --checker on --evaluation-seed 113 \
  --output data/androidworld/.archive/adaptive-supplement/memory_off_1
```

Change only the task's explicit paths and the condition's listed switches.
Keep supplementary output in its explicitly supplied archive subtree; it must
not replace the frozen paper's visible results. No separate launcher or
task-specific action script is used. Baseline without `--memory` is a distinct
configuration and is not this matched Memory ablation.

## Action-adaptation evidence

Reuse the canonical OmniTransfer evaluation and its reviewed ASE/PAPT inputs.
`scripts/evaluate_association_depth.py` takes an explicit checkpoint, dataset,
split, device, and output; it performs inference evaluation, not training.
`scripts/evaluate_local_context_algorithm.py` evaluates explicit prepared
inputs/predictions. Do not introduce a second mapper or fall back to source
coordinates. Fixed-checkpoint inference ablation and retrained architecture
ablation are different experiments.

The current OmniFlow launcher does not expose alternate OmniTransfer
checkpoints/architectures for an end-to-end ablation. This remains **not
implemented at the E2E boundary**; existing canonical micro-benchmarks do not
close that gap. A new checkpoint condition must preserve the canonical matcher
interface, identify its own protocol and hash, and be validated before use.

## Evidence and analysis

`diagnostics.function_resume.events` records actual fresh/resumed Function
invocations and whether a previous invocation failed. Event success means
execution completion, not semantic correctness. An event's trace interval
locates its action evidence. Unknown/missing historical counts remain null.

Use `src.experiment.performance_metrics.summarize_paired_experiments` with an
explicit frozen expected pair list and evidence-derived samples. Required
shared identities are task, real device, parameters, seed, model/endpoint,
source, code, Transfer, protocol, Store and transfer-state catalog. Strict
Memory ON/OFF must share both artifact hashes as well as the model. Preserve
condition-specific policy hashes with each source receipt. RunLog paths/hashes are analysis
inputs, not runtime selection indexes.

Report all planned/completed/failed/unreached/missing counts. Success latency
uses `execution_duration_ms`, never `duration_ms`. Report own success sets,
paired full system, paired fast path, and paired recovered successes separately.
Missing timing is not zero. Never classify an episode as fast path from only
its last Function's success when an earlier invocation failed.

The time ledger's exclusive components must reconcile to its owner. OOB RPC
includes transport/device work; it is not pure network or stabilization time.
Missing component/usage fields stay unknown. Root causes requiring a target
label or task-state oracle remain unverified until reviewed against evidence.

## Acceptance status

Software, canonical mapping evaluation, AndroidWorld emulator execution, and
physical-device acceptance are separate levels. Record every scenario as
passed / failed / not_run / not_triggered / pending_physical_device, with the
actual test or RunLog reference. Do not fill a live scenario with simulated
success. A successful camera run with no failure does not validate reentry.
Physical acceptance records device identity, installed APK version, code and
asset hashes, operations, results, and applicable restart/repeat boundaries.

The 2026-09-13 implementation regression run passed 261 tests under `tests/`.
Unrestricted pytest discovery also collected untracked vendor AutoDroid tests
and stopped on their unavailable `gym` dependency; that is not a passing run.
The checked 4090 targets are emulators with OOB 0.6.1 (versionCode 7), not
physical-device acceptance. Environment receipts and live results are kept
under `data/runtime/validation/adaptive-v2-20260913/` and the explicit batch
archive. SSH failure is transport evidence, not an AndroidWorld task outcome;
check the atomic RunLog and process completion before proceeding or retrying.
