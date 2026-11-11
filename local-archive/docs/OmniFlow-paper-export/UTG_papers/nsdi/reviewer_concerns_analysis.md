# NSDI Paper Risk And Open Issues

Status: Active merged risk ledger
Last Updated: 2026-05-22
Scope: Current NSDI paper only

This file is the single active reviewer-risk and open-issue ledger for the NSDI paper. It merges the useful content from the old pre-writing issue list and the reviewer concern list. It is not a claim source and not an evaluation source.

## Source Of Truth

- Claims: `UTG_papers/nsdi/core_claims.md`
- Evaluation matrix: `UTG_papers/nsdi/eval_matrix.md`
- English source: `UTG_papers/nsdi/overleaf/`
- Figure source: `UTG_papers/nsdi/figures/`
- File governance: `UTG_papers/nsdi/MAINLINE_FILE_CONSOLIDATION.md`
- Agent database: `docs/agent_context/paper_mainline_file_database.json`

Current paper vocabulary:

- `GUI Agent cache system with co-evolution`
- `Function` / `function asset`
- `PageVectorSet`
- `run_log -> function asset`
- `cache-first`
- `control-aware execution`

Do not use this file to revive `executable memory`, `dynamic UI state machine`, `Trajectory Cache`, `Logical Cache`, `Vector Automaton`, or `Path` as primary paper objects.

## Resolved Cleanup Decision

`pre_writing_issues.md` has been merged into this file and deleted from the active workspace. The useful parts were:

- write the paper around one cache-system identity, not parallel memory/cache identities;
- keep a formal cost and bootstrap discussion, but avoid unsupported complexity slogans;
- collapse many sub-contributions into the frozen C1-C5 claims;
- keep placeholders out of final claims until `eval_matrix.md` has frozen output files;
- use the system overview to map each component to Design, Implementation, and Evaluation.

## Critical Reviewer Risks

### R1. Offline Cost And Bootstrap

Reviewer concern:

- Cache-first execution benefits later users, but the first successful run still pays exploration and asset construction cost.
- The paper must not imply offline enrichment is free.
- New apps and long-tail tasks remain cache-miss dominated until enough trajectories exist.

Current mainline answer:

- Write this as amortization: first successful runs create `Function` / page assets; later cache hits skip online reasoning for matched segments.
- Attribute cost explicitly: online execution cost, offline asset construction cost, and shared reuse benefit are separate.
- Use asset-yield and multi-user experiments for evidence. Do not invent user-scale transition numbers that are not in `eval_matrix.md`.

Paper impact:

- Introduction can claim cost transfer and reuse, not universal zero-cost execution.
- Background or Discussion should include cold-start and long-tail limits.
- Evaluation should report available asset-yield and multi-user numbers only.

### R2. Source-to-Current Page And Action Reuse Boundary

Reviewer concern:

- PageVectorSet candidate recall is not the same as live execution success.
- Raw UI-tree baselines can be strong on same-device replay.
- Coordinate or anchor projection can be unsafe if the target element is absent or ambiguous.

Current mainline answer:

- Source-to-current transfer must compare against `B0 coordinate replay`, `B1 UI tree exact replay`, and `B2 UI tree + anchor replay`.
- `PageVectorSet` is the privacy-safe candidate recall object; target transfer still requires element/anchor mapping plus control-aware verification.
- If raw UI-tree replay wins in a narrow same-device setup, the paper should explain the privacy/shared-cache tradeoff instead of attacking the baseline.

Paper impact:

- Do not write source-to-current live success as frozen until device verify exists.
- Keep `recall@k` separate from `action_success_rate` and `false_positive_rate`.
- Figures should label PageVectorSet as candidate recall, not guaranteed execution.

### R3. Segment Or Function Granularity

Reviewer concern:

- Reuse only works if the cached unit is neither too small nor too coarse.
- Different users may produce alternate ways to satisfy the same goal.
- Conflicting or stale assets need ranking, alternatives, or deprecation.

Current mainline answer:

- The paper's active object is `Function` / `function asset`, with actions and evidence from canonical run logs.
- Multiple non-equivalent functions can coexist and be ranked by cache gate / quality metadata.
- Claim only the current implemented import/export and multi-user registry behavior; leave learned optimal granularity as future work unless evidence is added.

Paper impact:

- Design should explain why Function assets are the cache unit.
- Implementation should describe the current asset fields and evidence path.
- Evaluation should rely on asset compression and multi-user sharing, not unmeasured optimal segmentation.

### R4. Privacy Boundary

Reviewer concern:

- PageVectorSet still leaks some structural information.
- Embeddings and element-type catalogs are not formal differential privacy.
- Package identity or page type may remain visible depending on deployment.

Current mainline answer:

- State the threat model: no shared raw screenshots or raw XML as the paper object; text is hashed or removed in the vector page representation.
- Be precise that this is privacy-safe for the system's sharing boundary, not a formal DP guarantee.
- If formal privacy is not implemented, keep it in limitations or future work.

Paper impact:

- Related work and Discussion must avoid absolute privacy claims.
- C2 should remain "privacy-safe GUI state representation" within the stated threat model.

### R5. Control-Aware Execution

Reviewer concern:

- Raw replay is brittle; reviewers need to know what happens when the UI shifts, an action has no effect, or terminal verification fails.
- Recovery claims must not become scattered benchmark-specific if/else logic.

Current mainline answer:

- Use the implemented control bus semantics: `Observation + RouteContext -> ControlEngine.collect_effects(...) -> executor apply effects`.
- Cache hit is not bare replay; it remains wrapped by verification, recovery, and event logging.
- Do not claim a large recovery-coverage percentage unless it is backed by an output in `eval_matrix.md`.

Paper impact:

- Design must make control-aware execution central to C4.
- Evaluation should use end-to-end efficiency and case-study outputs and avoid unsupported recovery-rate numbers.

### R6. Related Work And Baseline Fairness

Reviewer concern:

- Comparisons can look like strawman replay or strawman memory agents.
- Citation mistakes are high-risk.
- Modern UI replay systems may use fuzzy matching, selectors, or anchors.

Current mainline answer:

- Related work must be rewritten in `overleaf/related.tex` with verified citations only.
- Baselines must be concrete: raw coordinate replay, UI-tree exact replay, UI-tree + anchor replay, and relevant agent baselines only when implemented or clearly marked as literature comparison.
- Never generate BibTeX from memory. Use verified sources or explicit placeholders.

Paper impact:

- Related work is a late-stage pass after claims and evaluation boundaries are stable.
- Unverified citations must stay marked as placeholders.

## Read-Only Core Code Review Notes (2026-05-22)

This section records current source-level risks found during a read-only review pass. Core code was not modified in this pass.

### CR1. Function Timeout Does Not Cover Start-Node Precondition Bridge

Code evidence:

- `src/utg/runtime/utg_runtime.py:613` defines `_ensure_function_start_precondition(...)`, which can perform stable observation, route-only bridge Function execution, and bridge verification observation before semantic execution.
- `src/utg/runtime/utg_runtime.py:935` calls `_ensure_function_start_precondition(...)` before the timeout block.
- `src/utg/runtime/utg_runtime.py:954` applies `asyncio.timeout(function_timeout_seconds)` only around the final `_executor.run_function(...)` call.

Risk:

- `function_timeout_seconds` is not currently a whole-Function wall-clock bound. Slow observe, route-only bridge, or bridge verify work can exceed the configured timeout before the timeout wrapper starts.
- Paper and OOB docs should not claim full end-to-end Function timeout enforcement until this is patched and tested.

Suggested fix boundary if core edits are later authorized:

- Wrap start-node precondition and target Function execution in one timeout budget, or account precondition time against a shared remaining budget and expose timeout provenance in `provider_detail`.

### CR2. Minimal Shared `PageVectorSet` Can Make Exact Action-Signature Guard Abstain

Code evidence:

- `src/utg/assets/embedding/vector_set.py:81` and `src/utg/assets/embedding/vector_set.py:196` default `to_dict(include_action_identity=False)`, omitting `resource_id`, `class_name`, `flags`, `text_hash`, and `content_desc_hash` from shared payloads.
- `src/utg/assets/embedding/vector_set.py:114` deserializes missing `flags` as `0`.
- `src/utg/assets/embedding/element_match.py:1132` handles `PageVectorSet` target pages by filtering candidates to actionable, enabled, and visible elements from those flags.
- `src/utg/assets/embedding/element_match.py:1136`-`1139` therefore drops every minimal shared-payload element whose action flags were omitted.
- The XML branch at `src/utg/assets/embedding/element_match.py:1188`-`1208` does not show the same failure mode because it reads `clickable` / `focusable` from XML and defaults enabled/visible to true.

Risk:

- A privacy-minimal shared `PageVectorSet` is safe as a page-recall object, but it does not by itself preserve enough action identity/state evidence for exact-signature guarded action transfer.
- Claims must keep the C2 boundary precise: current evidence supports privacy-safe page/family recall plus mapping/verify when the required local action context or identity-bearing diagnostic payload is available; it does not prove that the minimal uploadable shared payload alone can always drive exact action transfer.

Suggested fix boundary if core edits are later authorized:

- Decide whether shared execution payloads may include a limited hashed action-identity profile, or change guarded candidate filtering to distinguish "unknown flags" from "disabled/unactionable" while retaining a conservative abstain path.

### CR3. Retry Resume Is A Documented Full-Rerun Contract, Not Checkpoint Resume

Code evidence:

- `src/integrations/utg_api/_execute.py:99`-`121` parses `retry_context` and marks any nonzero `resume_from_step` as unsupported.
- `src/integrations/utg_api/_execute.py:140`-`222` returns `retry_resume_unsupported` before observing the host or executing actions.

Risk boundary:

- This is acceptable as a conservative contract, but paper/OOB wording must keep saying "full rerun only" and must not imply verified checkpoint resume.

### Code Boundary Map For Paper, Packaging, And OOB

This map records the read-only review boundary for the current paper goal.  It
does not authorize core edits.

| Boundary | Current source surface | Review classification | Why it matters now |
|---|---|---|---|
| UTG semantics and execution correctness | `src/utg/core/`, `src/utg/runtime/utg_runtime.py`, `src/utg/execution/`, `src/utg/assets/embedding/`, `src/utg/log/` | **Core** | This is the implementation substrate for Function recall/execution, `PageVectorSet`, action transfer, control effects, terminal/effect evidence, and run-event materialization. C1-C4 claims must follow this layer. |
| Repo high-level flow facade | `src/omniflow/core.py`, `src/omniflow/protocol.py` | **Core-adjacent facade** | This is the built-in host/planner orchestration over UTG semantics. It is paper-facing implementation, but its README says the public import contract is not frozen. Treat it as a facade, not the distributable artifact by itself. |
| Provider and public tool contract | `src/integrations/api/services.py`, `src/integrations/utg_api/_execute.py`, `src/integrations/utg_api/_import.py`, `src/integrations/mcp_server.py` | **Adapter / package seam** | `/recall`, `/functions/execute`, `/run_logs/ingest`, `/mcp`, standalone MCP, and bundle/import routes make the core callable by OOB and external agents. Contract/test/documentation changes belong here when they do not require new UTG semantics. |
| Evaluation and benchmark hosts | `scripts/nsdi_eval.py`, `scripts/nsdi_e2e_live.py`, `src/integrations/android_world/`, `tests/control_plane_benchmark/`, `tests/vector/` | **Experiment adapter** | These files produce the frozen manifest and registered summaries. They can prove paper evidence, but passing them does not by itself prove production OOB uptake or a new core capability. |
| OOB product host | `<OOB_ROOT>/ui/lib/services/assists_core_service.dart`, `<OOB_ROOT>/app/src/main/java/.../mcp/`, `<OOB_ROOT>/app/src/main/java/.../runlog/`, `<OOB_ROOT>/app/src/main/java/.../manager/` | **External host adapter** | The inspected Flutter service defines provider bridge/config calls and provider `/vlm/pre_hook` / `/functions/execute` clients, while the inspected native channel currently lacks those provider bridge MethodChannel handlers. OOB also exposes a separate native-local toolkit and MCP tools. This is integration evidence around OmniFlow, not a second paper core and not provider-backed cache-hit execution. |
| External skill/tool package | `skills/omniflow-agent-cache/`, `docs/agent_context/skills/omniflow-public-tool-surface/SKILL.md` | **Distributable artifact seam** | The packageable surface is the minimal `recall -> explicit call_function -> ingest_run_log` contract plus a host `observe/act` boundary. It should not ship paper files, AndroidWorld runners, or OOB UI as default tools. |

Read-only verdict:

1. Core review currently has the concrete CR1/CR2 risks above.  They remain
   source risks until core edits are explicitly authorized, patched, and tested.
2. The present packageable boundary is adapter-first: external users consume the
   skill/MCP contract and bring a host observation/action adapter.  Provider-only
   `/mcp` does not itself supply host `observe` / `act`, so package claims must
   keep that host dependency explicit.
3. The OOB bridge is real but not yet final artifact proof.  In the inspected
   OOB worktree, Flutter provider clients define `/vlm/pre_hook` and
   `/functions/execute` calls, but the active native MethodChannel surface does
   not register the provider bridge/config methods needed to supply those calls
   with a bridge token/context.  The native OOB MCP definitions expose
   `omniflow.call_tool` as the fixed new-client tool, keep
   `omniflow.call_function` executable for compatibility, and route both
   through a native-local `OobOmniFlowToolkitService` / replay layer.  That
   coexistence is an integration surface to reconcile or document before
   claiming one stable external OOB tool surface.
4. The inspected OOB source now resolves both `OpenOmniBot` and `Openomnibot`
   to the canonical `wzw-dev` worktree.  The branch mismatch is no longer the
   current blocker; the remaining blocker is missing native bridge
   token/context handlers and observed provider callbacks, so do not promote
   pre-execution artifacts into an OOB cache-hit execution claim.

### Branch-Aware OOB And Package Smoke Boundary

The follow-up smoke pass keeps the package and OOB evidence separate:

- The distributable `skills/omniflow-agent-cache` folder still validates, the
  MCP mapping selftest is ready, and the registered external-exposure
  experiment can be rerun inside the NSDI manifest.  This supports the public
  skill/MCP contract only.
- `bash scripts/start_oob_utg_host_bridge.sh emulator-5554` brought up the
  provider/ADB bridge on the current emulator.  Provider `/health` returned
  `omniflow_utg` with `observe` / `act` / `confirm` callbacks, and adb had the
  expected reverse `tcp:9417` and forward `tcp:8899` rules.
- The live OOB pipeline artifact
  `runtime/logs/androidworld_pipeline/nsdi_oob_branch_audit_20260522/summary.json`
  stopped at `status=oob_not_ready`: provider and bridge probes passed, but
  `com.example.bot.debug` was not running and the expected OOB accessibility
  component was not enabled on `emulator-5554`.
- The branch-scoped package-fix smoke
  `runtime/logs/androidworld_pipeline/nsdi_oob_one_click_package_fix_20260522/summary.json`
  passed OOB app readiness on `emulator-5554`: `workbench_check.ready=true`,
  `cn.com.omnimind.bot.debug` was running, the expected accessibility service
  was enabled, and provider/bridge probes passed.  The pipeline status was
  `fallback_required` because the active provider store had zero Functions, so
  this is app/bridge readiness evidence, not cache-hit execution evidence.
- The one-click script now resolves its default source worktree to
  `<PROJECT_ROOT>/../OpenOmniBot`, uses the documented `wzw-dev` branch guard,
  builds `developStandardDebug`, and checks `cn.com.omnimind.bot.debug`.  Older
  smoke artifacts that required a noncanonical branch override remain
  branch-scoped readiness evidence; the current canonical `wzw-dev` worktree
  still needs a fresh token/context callback run before it can support a product
  cache-hit execution claim.
- The follow-up OpenApp store-hit probe
  `runtime/logs/androidworld_pipeline/nsdi_oob_openapp_store_hit_bridge_token_blocked_20260522/summary.json`
  used `runtime/utg/openapp_official_success_5554_latest.json` and reached
  `/vlm/pre_hook` with `kind=hit`, `execution_route=utg`, and
  `function_id=aw_function_213a4dcca1` for the exact goal `Open the contacts
  app. Clear any pop-ups that may appear by granting all permissions that are
  required.`  `workbench_check.ready=true` confirmed `cn.com.omnimind.bot.debug`
  and the expected accessibility service were ready on `emulator-5554`.
  Execution was skipped with `status=missing_bridge_token_for_execute` because
  `OOB_UTG_BRIDGE_TOKEN` was not present; the forwarded bridge port existed, but
  direct bridge HTTP probes were not usable enough to treat `/utg/observe|act|confirm`
  as executed callbacks.
- The refreshed current probe
  `runtime/logs/androidworld_pipeline/nsdi_oob_openapp_store_hit_current_20260522/summary.json`
  reproduced the same boundary after starting the provider with
  `OMNIFLOW_UTG_STORE_PATH=runtime/utg/openapp_official_success_5554_latest.json`.
  `workbench_check.ready=true` again showed `cn.com.omnimind.bot.debug`
  running with the expected accessibility service on `emulator-5554`, and
  `/vlm/pre_hook` again returned `kind=hit`, `execution_route=utg`, and
  `function_id=aw_function_213a4dcca1` for the exact OpenApp goal.  The same
  artifact also shows `provider_health.store.function_count=0`; this should be
  preserved as a reporting nuance rather than interpreted as a list-route proof
  of executable readiness.  Execution was still skipped with
  `status=missing_bridge_token_for_execute` because `OOB_UTG_BRIDGE_TOKEN` was
  not present, so the artifact remains cache-gate evidence before execution,
  not cache-hit execution evidence.
- The latest bridge-ready OpenApp probe
  `runtime/logs/androidworld_pipeline/nsdi_oob_openapp_bridge_ready_no_token_20260522/summary.json`
  keeps the same execution boundary while strengthening bridge-readiness
  evidence: `workbench_check.ready=true`, adb reverse `tcp:9417 -> tcp:9417`,
  adb forward `tcp:8899 -> tcp:8899`, bridge probe open, `cn.com.omnimind.bot.debug`
  running, and the expected accessibility service enabled on `emulator-5554`.
  `/vlm/pre_hook` returned `kind=hit`, `execution_route=utg`, and
  `function_id=aw_function_213a4dcca1`; `/functions/execute` was skipped with
  `reason=bridge_token_missing` because `bridge_token_present=false`.  This is
  branch-scoped OOB app/bridge readiness plus cache-gate evidence, not
  cache-hit execution evidence.
- The current token-boundary rerun
  `runtime/logs/androidworld_pipeline/nsdi_oob_openapp_bridge_ready_token_boundary_20260522/summary.json`
  again confirms the same boundary on the inspected `OpenOmniBot` branch.
  The current provider is healthy on `runtime/utg/openapp_official_success_5554_latest.json`,
  `workbench_check.ready=true`, adb reverse `tcp:9417 -> tcp:9417` and forward
  `tcp:8899 -> tcp:8899` are bound, bridge probe is open, `cn.com.omnimind.bot.debug`
  is running, and the expected accessibility service is enabled on
  `emulator-5554`.  `/vlm/pre_hook` returned `kind=hit`,
  `execution_route=utg`, and `function_id=aw_function_213a4dcca1` for the exact
  OpenApp goal, but `/functions/execute` was still skipped with
  `reason=bridge_token_missing` because `bridge_token_present=false`.
  Direct bridge HTTP probes to `/utg/observe`, `/utg/act`, and `/utg/confirm`
  returned empty replies, so this remains readiness + cache-gate evidence, not
  observed callback execution.
- The continuation bridge smoke
  `runtime/logs/androidworld_pipeline/nsdi_oob_openapp_manual_bridge_ready_no_token_20260522/summary.json`
  keeps the same execution boundary while recording a launcher-side caveat.
  `OMNIFLOW_UTG_STORE_PATH=runtime/utg/openapp_official_success_5554_latest.json
  bash scripts/start_oob_utg_host_bridge.sh emulator-5554` failed before adb
  binding because the provider was already supervised by `launchctl` and the
  recorded raw-PID reclaim path could not reclaim port `9417`.  The provider
  itself remained healthy on the OpenApp store.  Manual `adb reverse tcp:9417
  tcp:9417` and `adb forward tcp:8899 tcp:8899` restored
  `workbench_check.ready=true`: bridge probe open, OOB app/accessibility ready,
  adb reverse/forward bound, and `/vlm/pre_hook` hit
  `aw_function_213a4dcca1`.  `/functions/execute` was still skipped with
  `reason=bridge_token_missing` because `bridge_token_present=false`, so this
  is bridge plumbing plus cache-gate evidence, not OOB cache-hit execution.

Consequence: current evidence says "public package/tool contract validated" and
"branch-scoped OOB app/bridge readiness passed on `emulator-5554`."  It also
says "a non-empty OpenApp Function store can produce a real OOB cache-gate hit
before execution."  It does not yet say "canonical `wzw-dev` OOB product smoke
passed", "cache-hit OOB execution passed", or "OOB exposes one final external
OmniFlow tool surface."  Those require a canonical-branch run, a valid
`OOB_UTG_BRIDGE_TOKEN`, working OOB `/utg/observe|act|confirm` callbacks, and
reconciliation of the native OOB MCP tool surface.

### Source-Level OOB Bridge Token Blocker

Follow-up read-only inspection of both local OOB worktrees,
`/Users/wuzewen/Projects/Omni/OpenOmniBot` and
`/Users/wuzewen/Projects/Omni/Openomnibot`, now resolves both paths to the same
filesystem inode on the documented `wzw-dev` branch; the current OmniFlow shell
still has `OOB_UTG_BRIDGE_TOKEN` absent, and `/Users/wuzewen/Projects/Omni/OOB`
is still absent.  The earlier bridge/cache-gate artifacts remain valid as
pre-execution evidence, but they should not be retroactively promoted to
canonical `wzw-dev` cache-hit execution without a fresh token/context callback
run.  A separate clean worktree exists at
`/Users/wuzewen/Projects/Omni/OpenOmniBot-omniflow-simple-utg` on
`codex/oob-omniflow-simple-utg`; it is branch-scoped simple-UTG work, not the
canonical OOB product branch.

Code evidence:

- Flutter calls `getUtgBridgeExecutionContext()` in
  `<OOB_ROOT>/ui/lib/services/assists_core_service.dart` and uses the returned
  `bridgeBaseUrl` / `bridgeToken` for both `replayUtgRunLog(...)` and
  `runUtgFunction(...)`.
- The same Flutter service also calls MethodChannel methods
  `getUtgBridgeConfig`, `saveUtgBridgeConfig`, `controlUtgProvider`, and
  `requestUtgJson` around its provider UI/API surface.  The inspected native
  `AssistsCoreChannel.kt` registers none of these provider bridge/config
  methods, so the Flutter provider surface should be treated as source-incomplete
  on this branch until the native bridge channel is restored or replaced.
- Native `AssistsCoreChannel.kt` in the inspected branch does not register a
  `getUtgBridgeExecutionContext` MethodChannel handler; unknown methods fall
  through to `result.notImplemented()`.
- Repository search over the active app / assists / omniintelligence source,
  excluding build and third-party outputs, found no active native
  `/utg/observe`, `/utg/act`, or `/utg/confirm` endpoint implementation.
- OOB native MCP and run-log code does provide a local deterministic toolkit:
  `OobOmniFlowToolkitService` returns `source=oob_native_omniflow_toolkit`,
  `McpToolDefinitions.fixedTools` exposes `omniflow.recall`,
  `omniflow.call_tool`, `omniflow.ingest_run_log`, and
  `omniflow.explore_replay`, and `McpRoutes` still routes
  `omniflow.call_function` to the same local toolkit for compatibility.  This
  proves a branch-local OOB toolkit surface, not provider `/functions/execute`
  with OmniFlow bridge callbacks.
- The separate simple-UTG branch has `OmniFlowSimpleProviderClient` calls to
  provider `/vlm/pre_hook` and `/functions/execute`, plus a local Android
  action runner.  It does not supply the canonical
  `getUtgBridgeExecutionContext` token/context handler or observed `/utg/*`
  callback execution, so it cannot be cited as OOB cache-hit execution evidence.

Risk boundary:

- The current `OOB_UTG_BRIDGE_TOKEN` blocker remains source-level, not merely a
  missing shell export in this environment.  The inspected `wzw-dev` source does
  not expose a discoverable token source for Flutter's manual Function
  execution or replay path, and its native channel does not currently expose the
  provider request bridge used by the Flutter service.
- Do not claim OOB cache-hit execution until a user-approved OOB branch provides
  the MethodChannel token/context and provider request handlers, exposes usable
  bridge callbacks, and a live run reaches `POST /functions/execute` with
  callback evidence.
- Do not patch OOB native bridge/token code from this paper audit unless the
  user explicitly authorizes external OOB source edits.

## Current Verification Snapshot (2026-05-22)

- Unified NSDI manifest `runtime/evals/nsdi/manifest.json` is refreshed at
  `2026-05-22T00:05:16.908859+00:00` and remains restored to
  `mode=all`, `device=emulator-5554`, `status=partial`; the remaining partial
  experiments are `representation-recall` and `case-study`.
- Re-running `python scripts/nsdi_eval.py --all --device emulator-5554` on
  2026-05-22 completed with exit code 0; the refreshed case-study summary is
  dated `2026-05-22T00:08:54.002154+00:00` and again leaves the manifest at
  `mode=all/status=partial`.  The
  registered case-study output at
  `runtime/evals/nsdi/rq6_case_study/summary.json` is a real nonzero
  `1968.3 ms` event stream, but it still observes only `cache_hit` and is
  missing `control` and `fallback`.
- Read-only audit of `runtime/logs/run_logs.db` after the rerun found 544 runs,
  280 steps, and 2,111 events.  Step rows include `hit|cache=163`,
  `miss|fallback=83`, `miss|agent=32`, and `recall|fallback=2`; there are 163
  runs with hit steps, 101 runs with fallback steps, and 0 runs with both.
  Event-type search also found 0 actual control/recovery/guard rows.  A broader
  raw payload substring scan returns only false positives such as an
  AndroidWorld action description (`control_before_step_open_app`) and UI/XML
  text, not control-plane events or control labels.  These fallback-only rows
  therefore cannot be promoted into the RQ6 hit-control-fallback figure.
- End-to-end live evidence is bounded to AndroidWorld `OpenAppTaskEval`.
  The latest top-level unified output `runtime/evals/nsdi/e2e_live/summary.json`
  is a single-device `emulator-5554` run `20260522T000809Z` with 1/1 success,
  1 cache hit, and 1 zero-token hit.  The unified manifest E2E section points
  to run dir `runtime/evals/nsdi/e2e_live/20260522T000516Z`; cite run dirs
  explicitly because later resource/time runs overwrite the top-level summary.
  The separate run
  `runtime/evals/nsdi/e2e_live/20260522T000228Z/summary.json` passed 2/2 on
  `emulator-5554,emulator-5556` with 2 cache hits and 2 zero-token hits; cite
  that run directory explicitly because later unified runs overwrite the
  top-level summary.
- Source-to-current live verify remains bounded: `runtime/evals/nsdi/page_replay_baselines/headline/live_verify.json`
  covers 16 safe Contacts cases on `emulator-5554`, with 15 positive cases,
  1 negative case, positive action success `14/15`, and negative false positive
  `1/1`.  This supports bounded transfer evidence only, not broad live safety.
- Resource/energy is secondary accounting only:
  `runtime/evals/nsdi/resource_energy/summary.json` contains one-run emulator
  counters such as `device_active_time_ms=40720.0`, `battery_delta_pct=0.0`,
  `network_total_bytes=5043`, `host_child_cpu_time_ms=4541.5`, and
  `host_children_max_rss_bytes_observed=228081664`; it does not support an
  energy-saving claim.
- Public package/tool surface evidence remains contract-level:
  `runtime/evals/nsdi/external_exposure/summary.json` was refreshed at
  `2026-05-22T00:08:53.999095+00:00` and records 22 passed MCP/workbench/external
  compatibility tests plus 6 passed selected public-route tests
  (`81 deselected`, `1 warning`).  The exposed surface is
  `omniflow.recall`, `omniflow.call_function`, `omniflow.ingest_run_log`,
  `POST /mcp`, `POST /vlm/pre_hook`, `POST /functions/execute`,
  `POST /run_logs/ingest`, and `GET /functions`.  This proves
  recall/call-function/ingest contract readiness, not production scale,
  package-distribution readiness, or OOB host execution.
- A refreshed local distributable skill artifact was generated at
  `/private/tmp/omniflow-agent-cache-package/omniflow-agent-cache-20260522T0545.skill`
  with:
  `cd skills && zip -r /private/tmp/omniflow-agent-cache-package/omniflow-agent-cache-20260522T0545.skill omniflow-agent-cache -x '*/__pycache__/*'`.
  The verified artifact size is `27998` bytes, last modified
  `2026-05-22 05:45:54`, with SHA-256
  `ac684e9ed3a8ac777cc0689391b5c1ba630ab9694c7679c5341c36b81a0462c2`.
  The package preserves the `omniflow-agent-cache/` directory layout, contains
  `SKILL.md`, `agents/openai.yaml`, `references/`, `scripts/`, and `assets/`,
  and excludes `__pycache__`.  This proves the current skill folder is
  reproducibly packageable.
- A named fake-host package trace was generated at
  `runtime/evals/nsdi/external_host_trace/20260521T211121Z/summary.json`.
  It exercises `POST /mcp` `omniflow.recall`, `POST /mcp`
  `omniflow.call_function`, `POST /run_logs/ingest`, and standalone
  `omniflow.ingest_run_log` against a trace-local store copy.  It proves the
  external-host contract can reach call + writeback with a working
  `observe/act/verify_effect` adapter; it does not prove OOB, real-device,
  production-scale, or shared-registry readiness.
- OOB remains pre-execution only.  `/Users/wuzewen/Projects/Omni/OpenOmniBot`
  and `/Users/wuzewen/Projects/Omni/Openomnibot` now resolve to the same
  canonical `wzw-dev` worktree, while the expected
  `/Users/wuzewen/Projects/Omni/OOB` path is absent and
  `OOB_UTG_BRIDGE_TOKEN` is currently unset.  Source inspection still found no
  native handler for Flutter's `getUtgBridgeExecutionContext` call, no native
  handlers for the Flutter provider bridge/config methods
  `requestUtgJson` / `getUtgBridgeConfig` / `saveUtgBridgeConfig` /
  `controlUtgProvider`, and no active `/utg/observe|act|confirm` callback
  endpoints in the inspected `wzw-dev` source.  OOB cache-hit execution is
  therefore still blocked even though earlier smoke has app/bridge readiness,
  open bridge probe, and OpenApp cache-gate hit evidence.  The native-local
  `omniflow.call_tool` / replay toolkit is useful OOB capability, but it does
  not close the provider bridge callback proof gap.

## Implementation Text Code-Truth Audit (2026-05-22)

Read-only source inspection found that `overleaf/implementation.tex` needed a
paper-side correction, not a core-code change.  The current code supports:

- local model-free `PageVectorSet` encoding with structural signatures and
  deterministic page fingerprints;
- an L1 recall hash cache over normalized goal/package/node context, with
  direct hits limited to registered no-argument Functions;
- a separate in-process `LiteVectorDB` for page/node localization, followed by
  recall candidate scoping;
- posterior action transfer with a null target and Bayes-risk gate;
- explicit `import_run_log` ingestion/write-back and public
  `omniflow.recall -> omniflow.call_function -> omniflow.ingest_run_log`
  tool contracts.

The inspected code does not support paper claims of a cross-observation
per-session subtree incremental encoder cache, an online dense-vector semantic
Function search level that automatically selects parameterized Functions, a
target-spatial prefilter plus early-exit action-transfer path, or a bounded
background ingestion queue/worker.  The Implementation section has therefore
been rewritten to describe the implemented recall cache, page-localization,
posterior transfer, explicit ingest, and MCP contract only.  Keep these
boundaries unless core code and registered evaluation outputs change.

### Recall Metric Wording Audit (2026-05-22)

The active recall summary splits the 38 labeled checks into 21 top-1
exact-reuse cases, 11 candidate-coverage cases, and 6 context-isolation cases.
Paper text may report 100% overall accuracy across all 38 checks, but "top-1
exact reuse" applies only to the 21 exact-reuse cases.  Do not collapse the
coverage and isolation checks into a 38-case top-1 accuracy claim.

### Source-to-Current Baseline Wording Audit (2026-05-22)

The active source-to-current transfer evidence does not support the stale
sentence that coordinate replay falls to near-zero action success under a
frozen cross-device transfer result.  The registered offline-oracle B0 row is
`100.0%` positive action success with `16.7%` false positives, while the bounded
live verifier exercises B3 safe Contacts cases and reports `14/15` positive
success plus a `1/1` negative false positive.  Paper text may use this as a
safety/privacy-boundary argument against static coordinate replay, but not as a
near-zero cross-device success claim.

### Accessibility-Observable Boundary Audit (2026-05-22)

Paper wording must distinguish rendered UI from accessibility-observable UI.
`PageVectorSet` currently encodes pages and elements exposed through the
accessibility tree; it does not prove coverage for fully opaque Flutter,
WebView, Unity, canvas, or custom-rendered surfaces unless the host provides an
observable element tree.  The 32.5% attribute-absent statistic is a lower-bound
risk inside the MobileViews accessibility-observed touch-target set, not a
claim that every rendered element can be encoded or transferred.

### Active Text Evidence-Boundary Audit (2026-05-22)

A fresh read-only pass over the active Overleaf text found no new paper-side
patch required.  The current abstract, introduction, implementation,
evaluation, and limitations sections already keep the latest evidence boundary:
the top-level live E2E result is the single-device `emulator-5554`
OpenAppTaskEval `1/1` cache-hit run; the separate `20260522T000228Z` run is the
only current 5554/5556 two-device E2E evidence; recall wording separates the
21 top-1 exact-reuse checks from the 38 total labeled checks; bounded
source-to-current live verification still includes a `1/1` negative false
positive; resource counters remain secondary one-run emulator accounting; and
package/OOB wording does not promote the fake-host trace or `/vlm/pre_hook`
hits into OOB cache-hit execution.

## Objective-To-Evidence Audit (2026-05-22)

This table maps the current user objective to evidence that can be cited in the
paper or artifact handoff.  It is deliberately conservative: an item is
complete only when the current repository state proves the full requirement.

| Objective requirement | Current proof | Status | Remaining proof needed |
|---|---|---|---|
| Review the paper without fake experiment claims | `core_claims.md` contains the C1-C5 mathematics/evidence audit; `eval_matrix.md` registers output paths and statuses; this ledger records overclaim boundaries and current verification snapshots | Partially complete | Final pass over active Overleaf text and figures after the remaining experiments are no longer partial |
| Run every registered experiment at least to runnable status | `runtime/evals/nsdi/manifest.json` is `mode=all`, `device=emulator-5554`, `status=partial`, with no failed experiments and partial items limited to `representation-recall` and `case-study`; the latest 2026-05-22 rerun confirmed the registered case-study is still a real cache-hit-only stream, and the refreshed 544-run DB audit still has no hit+fallback run or actual control/recovery/guard event rows | Runnable but partial | Full representation-recall 30 goals x 3 device fingerprints; case-study event stream with hit, control/recovery, and fallback regimes |
| Give every claim mathematical grounding | `core_claims.md` has formulas for cost amortization, payload reduction, compression ratio, verified success, and co-evolution token reduction, plus an eight-item table for C1-C5 | Mostly complete | Keep formulas synchronized with any future result or system-semantics change; do not add new claims outside C1-C5 without the same audit |
| Produce something packageable for other agents | `skills/omniflow-agent-cache` validates; `mcp_tool_mapping.py selftest` is ready; `runtime/evals/nsdi/external_exposure/summary.json` records 22 passed MCP/workbench/external compatibility tests and 6 passed selected public-route tests; refreshed local package artifact `/private/tmp/omniflow-agent-cache-package/omniflow-agent-cache-20260522T0545.skill` was generated with the expected skill directory layout, size `27998` bytes, and SHA-256 `ac684e9ed3a8ac777cc0689391b5c1ba630ab9694c7679c5341c36b81a0462c2`; fake-host trace `runtime/evals/nsdi/external_host_trace/20260521T211121Z/summary.json` reaches `recall -> call_function -> ingest` and standalone `omniflow.ingest_run_log` | Skill package generated, fake-host trace available | A real external host or OOB branch smoke with observed host callbacks and cache-hit execution; production packaging policy |
| Mark core code and distinguish adapter/package layers | The code boundary map in this ledger marks `src/utg/**` as core, `src/omniflow` as facade, provider/MCP/AndroidWorld as adapter/experiment surfaces, OOB as external host, and `skills/omniflow-agent-cache` as distributable seam | Complete for paper review | If core fixes are later authorized, re-run core tests and update CR1/CR2 risk status; do not treat current dirty core worktree as accepted by this audit |
| Integrate with OOB at `~/Projects/Omni/Openomnibot` | Existing OOB evidence shows app/accessibility/bridge readiness, bound adb reverse/forward, open bridge probe, and OpenApp `/vlm/pre_hook` hit `aw_function_213a4dcca1` in `runtime/logs/androidworld_pipeline/nsdi_oob_openapp_bridge_ready_token_boundary_20260522/summary.json` and the continuation artifact `runtime/logs/androidworld_pipeline/nsdi_oob_openapp_manual_bridge_ready_no_token_20260522/summary.json`; the continuation also records that `start_oob_utg_host_bridge.sh` failed to reclaim a launchctl-supervised provider before manual adb binding restored readiness; current resume audit shows `OpenOmniBot` / `Openomnibot` are the same worktree on canonical `wzw-dev`, but source inspection still found no native handler for Flutter `getUtgBridgeExecutionContext`, no native handlers for Flutter provider bridge/config methods such as `requestUtgJson`, and no active `/utg/observe|act|confirm` endpoints; OOB native-local `omniflow.call_tool` / replay exists but is not provider bridge execution; the separate clean `OpenOmniBot-omniflow-simple-utg` worktree has simple provider calls but no canonical bridge token/context or callback evidence | Partial | Fresh `wzw-dev` runtime evidence with implemented/discoverable bridge token context plus provider request bridge; observed `/utg/observe|act|confirm` callbacks; `/functions/execute` cache-hit execution success |
| Do not modify core code for this thread | This continuation only refreshes runtime evaluation outputs and edits paper/reviewer/iteration documents; source-level CR1/CR2 remain risks, not patched claims, and `src/utg/**` remains untouched by this pass | Satisfied for this continuation | Keep future work docs-only unless the user explicitly authorizes core edits |

## Open Issue Table

| Issue | Current Source | Paper Section | Status | Required Evidence |
|---|---|---|---|---|
| Cold start and offline cost attribution | `core_claims.md` C5, asset-yield and multi-user experiments | Intro, Discussion, Evaluation | Partially covered | Abstract/Intro/Background/Design/Conclusion now state amortization for compatible repeated hits, not unconditional `$O(m)` or every-user reuse; keep using frozen asset-yield and multi-user outputs only |
| Source-to-current live execution verify | `eval_matrix.md` source-to-current transfer | Evaluation | Partially covered | `emulator-5554` bounded live verify exists, but includes 1/1 negative false positive; broader negative set / human-live gold still required |
| PageVectorSet privacy wording | `core_claims.md` C2, `VECTOR.md` | Design, Discussion | Partially covered | Paper wording now says operational privacy boundary, accessibility-observable scope, and no formal DP claim; still needs final threat-model pass |
| Function granularity and alternatives | `core_claims.md` C1/C3/C5 | Design, Implementation | Open | Asset-yield compression evidence and current Function schema |
| Control-aware hit execution | `core_claims.md` C4 | Design, Evaluation | Partially covered | End-to-end and boundary-ablation outputs support the mechanism; the registered case-study timeline remains cache-hit-only after the latest all-run and DB audit, so the final multi-regime figure still needs a real hit + control/recovery + fallback event stream |
| Placeholder result discipline | `eval_matrix.md` | Abstract, Intro, Evaluation | Partially covered | Latest wording consumes the 2026-05-21/22 manifest only: single-device E2E live 1/1, bounded live transfer with a negative false positive, resource sidecar as secondary accounting, and case study partial |
| Claim math and amortization boundary | `core_claims.md` Claim Mathematics And Evidence Audit | Abstract, Intro, Background, Design, Evaluation, Related Work, Conclusion | Partially covered | C1-C5 now have explicit formulas and output-path evidence; active Overleaf wording uses conditional amortization, matched-segment zero online model tokens, bounded live evidence, and latest registered timing/resource numbers instead of unconditional zero-cost / `$O(m)` total cost |
| Related-work citation verification | `overleaf/related.tex`, `overleaf/refs.bib` | Related Work | Partially covered | Active citation keys now match BibTeX keys and known bad active refs (`vhm`, `backstage`, `hermes`, `federated-gui`, `infinigen`) were removed or deactivated; final camera-ready pass still needs a full source-by-source bibliography audit |
| OOB live packaging evidence | OOB board, `OpenOmniBot` run-log contract, `skills/omniflow-agent-cache` | Implementation, Evaluation, Artifact Appendix | Partially covered | Public skill/tool validation, branch-scoped one-click app/bridge readiness, a non-empty OpenApp store cache-gate hit, and a fake-host `recall -> call_function -> ingest` trace exist; canonical `wzw-dev` smoke or user-approved branch, native bridge token/context support, usable bridge callbacks, and OOB cache-hit execution evidence remain required |
| Core / adapter / package boundary | `src/utg/README.md`, `src/omniflow/README.md`, `src/integrations/README.md`, this review map | Artifact Appendix, reviewer response | Partially covered | Boundary map, release checklist, packaged skill artifact, and fake-host package trace exist; final handoff still needs branch-aware OOB live evidence and a real-host end-to-end trace |

## Paper Creation Workflow

1. Freeze vocabulary and source files: read `CORE_CONCEPTS.md`, paper concept boards, `core_claims.md`, and `eval_matrix.md`.
2. Define the one-sentence contribution: `OmniFlow is a GUI Agent cache system with co-evolution through trajectory reuse`.
3. For each paper claim, check the eight required elements: actual example, prior-work gap, mechanism, paper mapping, code outline, tests, scalability, and future tests.
4. Check `core_claims.md` Claim Mathematics And Evidence Audit before writing cost, payload, compression, transfer, verification, or co-evolution formulas.
5. Freeze evaluation before writing result text: only numbers registered in `eval_matrix.md` and backed by output files may enter the paper.
6. Draft Design, Implementation, and Evaluation first, because they are constrained by code and data.
7. Build figures from code truth and frozen claims; figures cannot introduce new terms or claims.
8. Draft Introduction and Abstract after the technical sections have stabilized.
9. Draft Related Work with verified citations only; mark any unverified citation as a placeholder.
10. Run a reviewer-risk pass using this file before finalizing claims, numbers, and limitations.

## Current Non-Goals

- Do not add a new research-note directory.
- Do not restore old Chinese iterations, NeurIPS files, storybook, stale strategy docs, or duplicate figures.
- Do not add unverified result numbers or new experiment categories directly into overleaf.
- Do not create another parallel issue tracker for paper risks.
