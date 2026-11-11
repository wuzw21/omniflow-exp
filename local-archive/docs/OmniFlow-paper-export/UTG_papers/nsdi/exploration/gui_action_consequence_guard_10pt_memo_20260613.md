# CommitGuard / Consequence-Gated GUI Functions: 10-point idea hardening memo

Status: exploration-only. This memo does not modify NSDI claims, eval numbers, or LaTeX text.

## Verdict

The broad EffectTrace idea is a good intuition, but it is not yet a 10/10 paper idea by itself. In its broad form it risks sounding like a coined concept: "every GUI action has an effect footprint." That statement is true, but reviewers will ask for labels, datasets, baselines, and measurable deltas.

The 10/10 version is narrower:

> **CommitGuard for GUI Functions:** before replaying or executing a mobile GUI `Function`, predict whether the next action crosses a real-world commit boundary, verify the post-action consequence with programmatic sensors, and gate execution to reduce harmful commits without losing task success.

This turns the problem from a concept into a measurable systems question:

> When a cache-first mobile agent reuses a GUI `Function`, can it avoid harmful real-world commits better than replay, UI similarity, terminal verification, and LLM safety judges, while preserving goal success and low overhead?

## Why this is stronger than generic EffectTrace

The core object should not be a new abstract "trace" alone. The core object should be a **commit decision** with externally measurable labels.

| Object | Measurable label | Why it is not vague |
|---|---|---|
| Candidate GUI action | `commit` / `non_commit` / `unknown` | Determined by pre/post state diff and task metadata |
| Commit class | `ui_only`, `local_data_write`, `device_setting_change`, `external_message`, `backend_state_change`, `privacy_exposure` | Each class has a sensor/verifier |
| Gate decision | `allow`, `confirm`, `fallback`, `abort` | Directly compared against ground-truth allowed/unsafe actions |
| Failure | harmful commit false accept, benign overblock, wrong-effect success | Directly scored per task and per step |

## Concrete problem

Input:

- Current observation: screenshot/XML/UI tree plus `PageVectorSet` evidence.
- Candidate action inside a recalled `Function`.
- Historical `run_log -> function asset` evidence.
- Programmatic effect sensors: UI state, app data, Android settings, permissions, files, test backend, message outbox, notification state.

Output:

- Commit-boundary prediction.
- Effect class prediction.
- Gate decision before execution.
- Post-action verifier evidence after execution.

Primary failure mode:

- The agent appears to complete the GUI route, but commits a wrong object, wrong recipient, wrong account, wrong permission, irreversible setting, or external message. A terminal screen verifier may still mark the task as successful.

## Datasets and evaluation substrate

Use public benchmarks where possible, and add a small programmatic sidecar only for labels that public datasets do not provide.

| Source | Role | What it gives | Limitation | How to use |
|---|---|---|---|---|
| AndroidWorld | Main live execution substrate | Live Android emulator, programmatic tasks, reward signals, dynamic task instantiation | Does not directly label every side effect | Add consequence sensors and commit labels to selected tasks |
| MobileSafetyBench | Main safety task source | Mobile safety/helpfulness tasks, daily scenarios, indirect prompt injection, emulator-based setting | Safety labels are task-level more than step-level | Re-run agents with CommitGuard and measure harmful commit false accepts |
| AITW | Offline pretraining / analysis source | Large human Android demonstrations across versions/devices; action traces and UI annotations | No live side-effect verifier | Use for offline commit-boundary classifier training/heuristics, not as final safety evidence |
| OS-Harm | Cross-domain supporting evidence | Computer-use safety tasks and misuse/injection/misbehavior framing | Desktop, not mobile | Use to motivate community attention and maybe test generality later |
| OmniFlow local Function assets | Direct system ablation | Cache-first replay, `Function`, `PageVectorSet`, control-aware execution | Current assets are not a public effect dataset | Use for ablation, not as the only evidence |

Minimum viable dataset for a strong paper:

- 60-100 live Android tasks.
- 5 consequence classes: local data, communication, device setting, file/content, backend/profile.
- At least 3 parameterized variants per task.
- Each task has:
  - allowed goal state,
  - unsafe/wrong-effect variants,
  - pre/post programmatic verifier,
  - cleanup/rollback script,
  - replayable seed `Function` where applicable.

The dataset claim should be modest:

> We contribute a consequence-labeled extension of live mobile GUI tasks, not a new foundation benchmark.

## Baselines

The paper only becomes credible if these baselines are included.

| Baseline | What it tests | Expected weakness |
|---|---|---|
| Unguarded GUI replay / OmniFlow without CommitGuard | Direct ablation | Fast but may execute harmful commits |
| Step-wise VLM/mobile agent | Current model-centric baseline | Can succeed but lacks consequence gating |
| Terminal verifier only | Whether end-screen success is enough | Misses wrong object and side effects |
| `PageVectorSet` / UI similarity threshold | Cheap state guard | Similar UI can have different consequences |
| UI tree diff only | Obvious effect proxy | Misses app data, settings, backend, privacy |
| LLM safety judge before action | Strong generic safety baseline | Costly, unstable, weak auditability |
| Safety prompt / MobileSafetyBench-style guard | Published safety-prompt baseline | May reduce harm but overblocks or misses concrete state diffs |
| Human-confirm all high-risk actions | Conservative safety baseline | Safe but high confirmation burden |
| Typed API only where available | Jump-beyond-GUI baseline | Low risk when available, poor coverage |

## Metrics

Primary metrics:

- **Harmful commit false accept rate:** unsafe commit actions that the system allows.
- **Goal success rate:** task completion under the same live verifier.
- **Benign overblock rate:** safe actions blocked or escalated unnecessarily.
- **Commit-boundary F1:** step-level `commit` vs `non_commit` classification.
- **High-risk recall:** recall on external messages, settings, backend, privacy exposure.

Secondary metrics:

- Effect-class macro F1.
- Wrong-effect success rate: terminal task success but wrong side effect.
- Cleanup/rollback success rate and cleanup time.
- Added latency and token/model calls.
- Confirmation burden: confirmations per completed task.
- Sensor coverage / missing verifier rate.

## What can be implemented first

P0: Build a consequence verifier layer for 3 safe Android classes.

- Contacts/calendar local data: use provider or app-local database diff.
- Device settings/permissions: use Android settings/AppOps/permission diff.
- File/content operations: use sandbox directory diff.

P1: Add communication/backend classes in controlled form.

- Message/email tasks go to test inbox or mock send service.
- Commerce/profile tasks use fake backend or AndroidWorld-style test app state.

P2: Integrate with OmniFlow.

- Store commit class and verifier obligations in `Function` metadata.
- Before cache hit execution, run CommitGuard.
- After execution, write consequence evidence into `run_log -> function asset`.

P3: Run baselines.

- Start with unguarded replay, terminal verifier only, UI similarity, LLM judge, safety prompt.
- Add typed API baseline only for tasks where a typed route exists.

## The strongest paper claim

Do not claim "we can understand all real-world effects of GUI actions."

Claim this:

> A cache-first GUI agent needs a commit-aware control plane. By adding programmatic consequence labels and a commit gate to reusable `Function` execution, the agent can substantially reduce harmful commits and wrong-effect successes while preserving most of the efficiency benefit of cache-first replay.

This claim is measurable and does not depend on inventing a new philosophical definition of GUI.

## Kill criteria

Do not promote this idea if any of the following happens:

- The evaluation relies mainly on LLM-as-judge instead of programmatic pre/post verifiers.
- The dataset has fewer than 3 consequence classes.
- Baselines do not include unguarded replay, terminal verifier only, UI similarity, and LLM safety judge.
- The method is mostly keyword rules over button text.
- The overhead is close to or worse than full replanning without a clear safety gain.
- The public benchmark link is weak and all evidence is private/self-collected.

## Score

Current broad EffectTrace: **7/10**.

Reason: important and well-motivated, but too broad and too easy to sound conceptual.

Narrow CommitGuard with AndroidWorld/MobileSafetyBench/AITW-backed evaluation: **9-10/10**.

Reason: clear failure mode, real cost, public evaluation substrate, strong baselines, measurable metrics, and direct fit with `Function`, `PageVectorSet`, `cache-first`, and `control-aware execution`.
