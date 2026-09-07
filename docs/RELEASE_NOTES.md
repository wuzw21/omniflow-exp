# Releases

## 2026-09-08 Harness refactor acceptance

The paper remains the design authority. The shared engine owns Recall/Execute,
control and Host access; `builtin_harness.py` owns the original native planning
loop. External hosts use the same Function runtime through two MCP tools.
The latest code checkpoint is `7945b101`; the project suite has 217 passing tests.

| Requested capability | Verification |
|---|---|
| One AndroidWorld entry, interchangeable Harness | `run_androidworld.sh run --harness builtin/codex/claude/module:factory`; official success in `data/runtime/validation/20260908-unified-harness/summary.json`, newer Codex/Claude RunLog hashes rechecked from `20260908-state-consistency/summary.json` |
| Simple public interface | `omniflow_recall` / `omniflow_execute`, Python SDK and packaged GUI Skill; real MCP tests and model-driven host evidence above |
| Local Harness iteration without a second runner | Factory capability provisioning tests; temporary fault-injection factories use the same episode and are preserved as evidence strings, not production task scripts |
| Configurable, measured recovery | Checker ON/OFF tests and `20260908-checker-policy/summary.json`; bounded native model retries in `20260908-model-request-recovery/evidence.json` |
| Complete execution/lifecycle accounting | Shared non-overlapping ledgers, failed-call and concurrency regression tests; native and external RunLogs reconcile with owner clocks |
| Failure pairs and original Planner fallback | Mapping rejection followed by official success in `20260908-model-request-recovery/evidence.json`; binding failure followed by official success in `20260908-binding-failure/evidence.json` |
| Faithful local mapping replay | Deduplicated, bounded, hash-verified failure assets; masked target input verified on emulator in `20260908-masked-pair/evidence.json`; preprocessing failures explicitly report unavailable replay |
| Stop and completion semantics | Shared cancellation/closed-session/deduplication tests, real in-flight OOB cancellation in `20260907-oob-inflight-cancel-003/summary.json`; no further Planner call after verified Function completion |

All abbreviated evidence directories above are beneath `data/runtime/validation/`.
Acceptance is on Android emulators as requested. It proves these integration and
recovery paths, not all 116 AndroidWorld tasks, physical-phone acceptance, or a
speed advantage. External CLI internals remain `host.external_unattributed`;
OOB RPC time includes transport and device work. Neither is reported as pure
model latency. Codex/Claude's two-tool integration requires the host to supply
its own primitive-action recovery channel; it does not secretly run the native
Planner. Historical releases and older evidence below retain their original scope.

## 1.1.0.dev1 — two Function services, 2026-09-07

Actual model-driven host acceptance now covers Codex CLI 0.153.4 and Claude
Code 2.1.237 on the local Pixel 6 Pro emulator. Each autonomously calls recall
and execute once, completing five OOB actions using canonical OmniTransfer:
Codex turns Bluetooth off; Claude turns it back on. Returned XML and independent
OS state agree on 1→0→1. The kernel returns task unknown and control to the host;
the host assesses the goal from observation. No kernel or adapter code changed.
Codex's initial read-only/never attempt stopped before recall execution because
MCP approval was required. After explicit user authorization, the temporary
session used on-request/auto-review with the read-only sandbox retained;
Claude allowed only the two MCP tools and disabled built-in tools. Neither test
changed global configuration or created a persistent host session.

Evidence: `data/runtime/validation/20260907-model-harness-acceptance/summary.json`.
These are successful-path model tests, not model-driven cancellation/recovery
acceptance or comparative performance measurements. Droidrun remains verified
at actual ToolRegistry dispatch; GUI-Owl/V-Droid are output adapters, not fully
validated upstream harnesses. All device bug acceptance remains **待真机验证**.

Follow-up model tests cover empty Memory and CLI interruption. Codex recalls
twice and Claude once with empty Memory, then both stop without execute or
device actions. In the interruption case, SIGINT after the first completed
open_app stops both CLIs; a five-second observation window adds no step facts,
the device lease can be reacquired, and OOB still shows Settings home with
Bluetooth unchanged. This is a completed-step boundary check, not proof of
draining an in-flight Android action. Codex returns no execute result; Claude
returns a generic rejected-tool error despite the completed first action.
The Skill now treats missing invocation feedback as unknown effects and
requires observation before recovery. No Function kernel or schema changed.
Evidence: `data/runtime/validation/20260907-model-empty-memory/summary.json`
and `data/runtime/validation/20260907-model-stop-acceptance/summary.json`.

A subsequent real OOB/MCP check cancels while the act broadcast subprocess is
still running, after Android am announces the request. The shared operation
lock stays held at cancellation; the current OOB call drains and records one
action before release. No second act is broadcast, the original request returns
cached cancellation facts, and a new request is rejected by the closed session.
Bluetooth stays on. Instrumentation records actual subprocess I/O without
delaying actions or replacing results, Functions, or the execution kernel.
Evidence: `data/runtime/validation/20260907-oob-inflight-cancel-003/summary.json`.
This verifies pending OOB I/O, not the exact Android gesture dispatch instant
or physical-device acceptance. An earlier diagnostic found no window after
broadcast return and completed the OFF Function; the same service restored ON.

The public MCP surface now consists of `omniflow_recall` and `omniflow_execute`.
Both use the existing recall and Check → Transfer → Act → Observe owners.
The host owns planning, primitive fallback, completion and cancellation; a
Function result never implicitly starts another Planner. Task identity keeps
repeated recall within one deadline; request identity deduplicates execution.
The development seven-tool MCP surface is superseded; invocation v1 and stored
Function/RunLog schemas are unchanged.

MCP, GUI-Owl, V-Droid and Mobilerun adapters share the two input definitions.
An explicit Host factory supports synchronous and asynchronous backends, while
formal AndroidWorld remains OOB-only. Optional Host application inventory is
resolved under the invocation deadline. The OOB backend queries real installed
packages before executing open_app; absent inventory and absent packages still
fail explicitly. This addresses a live integration failure, **待真机验证**.

The full suite passes 169 tests, with two optional upstream Droidrun registry
checks skipped because its dependency cannot import. The installed Droidrun
0.5.6 imports `mobilerun`, while installed mobilerun-sdk 5.1.0 exposes
`mobilerun_sdk`. Contract fixtures cover four harness adapters × three Host
forms × success/partial failure, plus application inventory, cancellation,
task retirement, request deduplication and real MCP stdio discovery. These
fixtures do not establish upstream LLM-driven or physical-device acceptance.

The dev1 wheel imports from an isolated installation outside the repository,
discovers exactly two tools and contains the distributable Skill. It excludes
screenshots, APKs and model weights. No physical device is available; 9207
and 4090 inventories contain emulators. Memory ON/OFF timing remains pending.

Live evidence under `data/runtime/validation/20260907-two-tools-live*` records
canonical 1024D OmniTransfer recall of the explicit Bluetooth Function on
9207/emulator-45562 with OOB 0.6.1 (versionCode 7). The first execution exposed
the missing inventory; subsequent runs encountered SSH timeouts. Preserve these
failed attempts as integration evidence, not benchmark success or latency.

Follow-up acceptance pins the optional Droidrun SDK dependency to
`mobilerun-sdk==2.1.0`, whose actual module exports match Droidrun 0.5.6.
The upstream ToolRegistry now passes both success and partial-failure calls;
no shim or vendor source modification is used. A real MCP ClientSession/Server
cancellation test verifies draining the first action, suppressing the second,
returning cached cancellation facts and rejecting new execution in the closed
session. Full suite: **172 passed**, including the two previously skipped SDK
checks. This covers actual protocol/SDK dispatch with controlled device facts,
not upstream model-driven or physical-device E2E.

Two additional live attempts use an SSH-forwarded existing ADB server instead
of one SSH shell per device command. Standard recall and the real 226-package
inventory succeed, but the next OOB observation times out before any action;
Tablet times out while preparing observation. Both return failure facts and
do not claim task completion. Evidence: live attempts 005 and 006. No remote
working-tree changes were overwritten and no APK was rebuilt or installed.

Live attempts 007–009 complete their Function actions on 4090 Standard (1440×3120), Fold (1768×2208)
and Tablet (1280×800), using the same explicit Bluetooth Store and canonical
OmniTransfer checkpoint. Each recalls `turn_bluetooth_on`, completes five OOB
actions, and serves the repeated request without device I/O. Final OOB XML shows
the Bluetooth switch checked only on Standard; Fold and Tablet remain unchecked,
so those runs do not establish target-goal success. MCP keeps task status `unknown`:
no official validator runs in this supplemental integration check. OOB is
0.6.1 / versionCode 7 on all three devices. Physical acceptance and formal
Memory ON/OFF latency are still pending. The machine-readable consolidated
evidence is `data/runtime/validation/20260907-harness-acceptance/cross_device.json`.

Step facts explain the Fold/Tablet goal failures: the stored toggle action
changed `checked=true` to `false`. This is not an idempotent "ensure on"
operation. The Skill now instructs the host to assess goal completion and
state-sensitive starting conditions before invocation.

Attempts 010–011 start with the switch observed off, using GUI-Owl and the
actual Droidrun ToolRegistry. They expose a shared Checker bug: package
restoration ran before explicit `open_app` and tried to launch the historical
source Launcher. The fix stays in `checker_rule_matches`, aligning the shared
package-mismatch condition with default_checker for open_app/press_key;
contextual actions still restore their expected source app. No mapper, Store
or action sequence is replaced. Three focused cases cover this distinction.

Attempts 012–013 pass with this fix: GUI-Owl/Fold and Droidrun/Tablet each
observe `checked=false`, recall and execute the same five-step Function, then
observe `checked=true`; transport retries do not repeat device actions. Full
suite: **175 passed**. Skill validation also passes. These are actual adapter/
SDK-to-OOB runs, without upstream LLM planning or an official task validator.
All device bug acceptance remains **待真机验证**.

User-requested local emulator verification also passes on
OmniFlowTargetPixel6Pro / emulator-45562 (Android 13, 1440×3120), reusing the
installed OOB 0.6.0.3 / versionCode 10 after enabling its accessibility service.
The real MCP stdio client/server recalls and executes the explicit OFF and ON
Bluetooth Functions, five actions each, with read-only OS state verification
of 1→0→1. Mid-Function cancellation stops after one completed action, rejects
new execution in the closed session, and permits an explicitly new task.
Restarted MCP rejects the prior process's session id; identical request retries
add no actions. Real empty-Memory recall returns no candidates and creates no
Store. All mappings use the same canonical OmniTransfer checkpoint. Evidence:
`data/runtime/validation/20260907-local-emulator/summary.json`. This remains
emulator integration evidence, without physical-device or official benchmark
acceptance; no APK was built or reinstalled. The emulator remains open.

## 1.1.0.dev0 — host protocol refactor, 2026-09-07

Observation evidence now uses immutable, content-addressed PNG files within
each existing bundle. Host and recorder share one capture; the post-action
Fast Pass reads the recorder's canonical XML rather than an undefined Host
attribute. The observation reporting index holds compact references. Raw
image resolution and the RunLog schema are unchanged.

The outer Planner loop and one-invocation execution now have separate entry
methods and reuse the same Function kernel. Completion checks, cooperative
cancellation, shared deadlines, Router iteration limits, and no-progress
stops use one control contract. External Function failure returns execution
facts and the current observation without starting another Planner.

The MCP stdio adapter and distributable `omniflow-gui` Skill expose canonical
tools, explicit Memory, session identity, request deduplication, cancellation,
and completion. Kernel import no longer loads AndroidWorld's experiment file.
An isolated wheel installation outside the checkout can import the kernel,
construct the MCP server, discover tools, and find the packaged Skill/schema.
No screenshots, APKs, or model weights are included in that wheel.

Focused lifecycle/protocol/packaging regression: 28 passed. The full repository
suite passes **131 tests** with `python -m pytest -q tests`; bare pytest additionally collects
an unrelated, untracked AutoDroid vendor RL suite whose optional `gym` dependency
is absent. Transport checks include a real MCP stdio subprocess, but its test
device is deliberately synthetic and does not validate Android execution.

All device behavior remains **待真机验证**. Local devices were disconnected at
the final inventory; 4090 and 9207 inventories contained emulators only. No
physical-device acceptance or comparative speed/RSS result is claimed. The
protocol is cooperative, has no durable public resume API, and does not yet
provide an HTTP service. Details: [protocol](HARNESS_PROTOCOL.md),
[MCP installation](HARNESS_MCP.md).

Local historical archive cleanup removed approximately 16.72 GB after protecting
731 dependency/unique-success files. The deduplicated deletion journal and audit
remain under `data/runtime/archive_cleanup/20260907/`, outside Git releases.

Follow-up cleanup removed the entire remaining local `.archive` directory
(790 files, 11.41 MB). The only live reference was an obsolete registry search
root, which was removed. The previously presumed unique source was proven
equivalent to its ordinary-directory copy after path normalization, with
byte-identical screenshots; ordinary evidence was unchanged. Final audit:
`data/runtime/archive_cleanup/20260907-final/`.

A subsequent live OOB protocol smoke passed on 9207 / emulator-45562 with OOB
0.6.1 (versionCode 7): capture, one wait action, request deduplication without
additional device I/O, post-cancel rejection, and session restart. This remains
supplemental emulator evidence; it does not validate physical-device behavior
or comparative performance.

## 1.0.0 — 2026-09-07

Preserved the pre-refactor working version, including Function semantic
authoring, shared Checker recovery, GUI-agent adapters, and runtime timing.
The imported working-tree schema snapshot is kept in Git history. Its initial
validation had 108 passing and 3 failing tests: the action and RunLog mirrors
did not match the existing swipe and compact-observation consumers. Release
preparation reconciled those mirrors with the established runtime contract and
made external scroll adapters emit canonical endpoints. The complete existing
suite plus the GUI-Owl swipe regression passes: **112 tests**. `git diff --check`
also passes. The raw imported changes remain available in earlier commits.

This release records a code baseline; device behavior remains **待真机验证**.
The local ADB inventory contained only emulator-5560. Historical physical-phone
measurements do not validate the current changes.

Experiment assets, source/target RunLogs, credentials, APKs, model weights,
paper build products, and temporary troubleshooting files are not Git release
contents. Reusable Memory remains an explicitly supplied external asset.
