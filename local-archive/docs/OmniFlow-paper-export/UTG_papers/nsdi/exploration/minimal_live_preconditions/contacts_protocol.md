# Contacts Minimal Live Preconditions Protocol

Status: Draft
Last Updated: 2026-05-24
Scope: First executable protocol for the minimal live preconditions direction. This is a research protocol, not a frozen NSDI claim.

## Purpose

Turn the current Contacts evidence into the first small reuse-admission dataset:

> Given the historical `ContactsAddContact` Function, decide whether the current live state satisfies the minimal conditions needed to reuse it safely.

This protocol is deliberately narrow. It should produce predicate/outcome data before any runtime integration or broad "all-agent" framing.

## Seed Asset

Function under study:

```text
runtime/utg/contacts_official_success_5554_latest.json#aw_function_515be7bf7e
```

Existing evidence:

- Bounded success sidecar:
  `runtime/evals/nsdi/harder_tasks_contacts_add_20260524/summary.json`
- Entry-gap probe:
  `runtime/evals/nsdi/harder_tasks_contacts_add_20260524/entry_gap_probe_20260524T0410Z.json`

Current known facts:

- The Function is a parameterized multi-action `ContactsAddContact` workflow.
- Three successful bounded replays reached terminal provider verification with 0 online model tokens.
- Successful rows used `__utg_skip_start_node_precondition__=1`.
- Direct full admission from launcher failed with `EXEC_START_NODE_PRECONDITION_FAIL`.
- A later direct-admission diagnostic probe from HOME/launcher also failed before any device action with
  `EXEC_START_NODE_PRECONDITION_FAIL` (`aw_node_8225087450 -> aw_node_22653dacf0`) and terminal verifier
  `contacts_record_not_found`. The required start node `aw_node_22653dacf0` is a recorded launcher node whose first
  sequence action is `open_app`; the matcher selected `aw_node_8225087450`, a Contacts form node in the same Function,
  for the live launcher observation. Treat this as node/matcher instability diagnostic evidence, not a positive row.
- A VLM planner could select `aw_function_515be7bf7e` and fill `first_name`, `last_name`, and `phone`; the remaining blocker was live start-node applicability, not parameter extraction.

## Candidate Predicate Vocabulary

The first version should keep predicates deterministic and inspectable.

| Predicate id | Meaning | Source | Expected role |
|---|---|---|---|
| `p_app_contacts` | foreground app package is `com.google.android.contacts` or route-only bridge can open it | observation / app state | necessary unless route bridge is allowed |
| `p_start_node_match` | current state matches `Function.start_node_id` or an admissible route-only bridge exists | UTG node match / route graph | necessary for direct execution |
| `p_create_entry_available` | create-contact entry point is visible/actionable or reachable by route-only navigation | XML/actionability | likely necessary |
| `p_required_fields_bindable` | first name, last name, and phone fields can be grounded after entering create-contact flow | XML/action target evidence | necessary |
| `p_no_blocking_overlay` | permission dialog, keyboard trap, or modal overlay does not block required actions | observation/control evidence | necessary for safe dispatch |
| `p_args_valid` | required arguments are present and pass simple format checks | Function arguments | necessary |
| `p_terminal_verifier_available` | Contacts provider terminal verifier can check inserted record | verifier registry / host capability | necessary for research label quality |
| `p_account_state_compatible` | account/default-save target does not require an unresolved user choice | observation / dialog evidence | likely necessary |
| `p_action_transfer_confident` | target actions retain unambiguous element/anchor mapping under current state | PageVectorSet / element match | sufficient evidence, not always necessary |

Do not include raw screenshots, raw user data, or hidden chain-of-thought as predicates.

## Perturbation Table

The first suite should contain at least these cases. The labels below are expected labels; actual labels must come from execution or a stronger offline oracle.

| Case id | State perturbation | Predicate intervention | Expected outcome | Why it matters |
|---|---|---|---|---|
| `c0_correct_create_flow` | start from a state known to satisfy the create-contact path | all key predicates true | `success` | positive control |
| `c1_launcher_wrong_start` | start from launcher with no bridge into Function start node | `p_start_node_match=false` | `precondition_failure` | current observed blocker |
| `c2_contacts_home_reachable` | start from Contacts home where create entry is reachable | `p_start_node_match=false`, route bridge possibly true | `repair_or_success` | distinguishes strict start match from admissible route repair |
| `c3_missing_create_entry` | Contacts UI lacks or hides create-contact entry | `p_create_entry_available=false` | `reject` or `precondition_failure` | prevents page-family-only false accept |
| `c4_overlay_blocks_target` | dialog/keyboard/modal intercepts target action | `p_no_blocking_overlay=false` | `reobserve_or_repair` | tests stale/blocked observation |
| `c5_missing_phone_field` | phone field cannot be grounded | `p_required_fields_bindable=false` | `reject` | checks semantic field precondition |
| `c6_args_missing_or_invalid` | missing phone or invalid argument format | `p_args_valid=false` | `reject_before_execution` | separates argument validation from GUI state |
| `c7_terminal_verifier_missing` | action may execute but provider verifier is skipped, disabled, or unavailable | `p_terminal_verifier_available=false` | `abstain` | prevents weak success labels |
| `c8_similar_non_contacts_page` | visually/textually similar list page in another app | `p_app_contacts=false`, page similarity possibly high | `reject` | cross-app false positive control |

## Labels

Use these labels consistently:

- `success`: terminal/business-effect verifier confirms the expected contact row.
- `precondition_failure`: execution is blocked before semantic actions because required live state predicates are false.
- `unsafe_or_wrong_effect`: action dispatch occurs but terminal/business effect is wrong or ambiguous.
- `reject`: admission policy refuses reuse before execution.
- `reobserve_or_repair`: admission policy requires a bounded observation or route-only repair before execution.
- `abstain`: label cannot be trusted because the verifier is missing.

Low-level `act=true` is not a success label.

## Label Acquisition And Promotion

The runtime package started as `planned_skeleton_available` and is now `partial`. It becomes stronger evidence only by replacing planned labels with oracle-backed or execution-backed labels.

Label source hierarchy:

```text
planned_expected_only < offline_oracle < execution
```

- `planned_expected_only`: protocol expectation only. It can define cases, but it cannot score baselines or train a contract.
- `offline_oracle`: a deterministic replay fixture, existing run artifact, or manually curated observation pair proves the predicate/outcome relation without a fresh live run. It must include source path, observation evidence, oracle rule, and why terminal/business effect is or is not decidable offline.
- `execution`: a fresh live or replay command ran against a concrete device or host. It must include command, device, first observation, final observation, terminal verifier result, error code if any, and elapsed/check cost.

Promotion rules:

1. A case moves from `unlabeled` to `oracle_labeled` only when its `outcomes.json` row uses `label_source=offline_oracle` and its evidence reference points to a readable artifact.
2. A case moves from `unlabeled` or `oracle_labeled` to `execution_labeled` only when its `outcomes.json` row uses `label_source=execution`, `execution.ran=true`, and either terminal evidence or an explicit pre-execution reject/precondition error is present.
3. `success` requires terminal/business-effect evidence. For Contacts, this means provider verification for the inserted row or an equivalent host verifier. XML navigation success is not enough.
4. `reject` and `reobserve_or_repair` labels can be policy decisions, but they still need predicate evidence explaining which live condition triggered the refusal.
5. `abstain` is required when the verifier is missing or the evidence path cannot distinguish success from unsafe/wrong effect.
6. LLM applicability judgment can be recorded as a baseline decision, never as `label_source`.

Per-case acquisition plan:

| Case id | Preferred label source | Required evidence | Notes |
|---|---|---|---|
| `c0_correct_create_flow` | `execution`, or existing terminal sidecar promoted as `offline_oracle` | replay command or sidecar, `emulator-*` serial, first/final observation when available, Contacts provider verifier hit for the inserted row | Positive control; cannot be promoted from planned without terminal verifier output. If using the existing sidecar, keep the start-node-skip caveat explicit and do not treat it as direct admission evidence. |
| `c1_launcher_wrong_start` | `execution` or existing sidecar promoted as `offline_oracle` | command or sidecar showing `EXEC_START_NODE_PRECONDITION_FAIL`, first observation from launcher, no semantic actions executed | This is the current known admission blocker; if using existing sidecar, mark it offline-oracle and link the sidecar. |
| `c2_contacts_home_reachable` | `offline_oracle` for route/form evidence, `execution` for terminal success | first observation or fixture at Contacts home, route-only repair or reobserve decision, final terminal verifier if execution proceeds | Separates strict start-node match from admissible route repair. Offline route evidence can label `reobserve_or_repair`, but cannot label `success`. |
| `c3_missing_create_entry` | `offline_oracle` first, then `execution` if feasible | curated observation without actionable create entry, predicate matrix evidence for `p_create_entry_available=false` | Reject is valid only if the missing entry is not a transient stale observation. |
| `c4_overlay_blocks_target` | `offline_oracle` for overlay-present repair label, `execution` for action obstruction/failure trace | first observation or fixture with modal/keyboard/dialog, overlay predicate evidence, reobserve/repair or failure trace if execution proceeds | XML overlay evidence can label `reobserve_or_repair`; live execution is still required to claim action obstruction/failure or terminal effect. |
| `c5_missing_phone_field` | `offline_oracle` first, then `execution` if feasible | create-contact observation where phone field cannot be grounded, predicate evidence for missing required field | Use `abstain` if the field may be reachable after an unmodeled expansion action. |
| `c6_args_missing_or_invalid` | `offline_oracle` | argument payload, validation rule, no device action required | This case should be rejected before GUI execution. |
| `c7_terminal_verifier_missing` | `offline_oracle` | control/API/CLI/test/docs evidence showing skipped or disabled terminal verify yields `verifiable=false` and `terminal_page_reached=null` | Must not be scored as success or failure; label should remain `abstain`. |
| `c8_similar_non_contacts_page` | `offline_oracle` plus optional `execution` | observation from non-Contacts app, page similarity evidence if available, `p_app_contacts=false` | Main false-accept control for similarity-only baselines. |

## Positive Evidence Acquisition Protocol

The current CMLP-001 package lacks a terminal-verified positive row that proves direct admission or route-repair admission.
The next positive row must be one of the two forms below. A stdout success, XML-only navigation success, skipped start-node
success, or missing terminal verifier is not enough.

### Direct-admission positive

A direct-admission positive proves that the historical Function is safe to execute from the current state without bypassing
the Function start-node precondition.

Required conditions:

- `label_source=execution`
- `label=success`
- seed Function is `runtime/utg/contacts_official_success_5554_latest.json#aw_function_515be7bf7e`
- the command does not set `__utg_skip_start_node_precondition__`
- the first observation and matcher evidence show `p_start_node_match=true`
- `p_args_valid=true` for `first_name`, `last_name`, and `phone`
- terminal verifier is enabled, available, and confirms the inserted Contacts provider row
- execution records command, device serial, first observation, final observation, terminal evidence, elapsed time, and success
  or null error code
- predicate matrix and outcomes rows include evidence references for the start-node match, arguments, terminal verifier, and
  final business effect
- if measured, check/replay/replan cost fields are recorded rather than inferred from stdout

Do not promote this row if any successful replay uses `__utg_skip_start_node_precondition__=1`, if the start-node predicate is
unknown, if terminal verification is skipped, or if the row only proves that the create-contact form can be reached.

### Route-repair positive

A route-repair positive proves that reuse can start from a non-start state only after a bounded route-only bridge moves the
state into an admissible Function start or create-flow state, followed by terminal-verified semantic execution.

Required conditions:

- `label_source=execution`
- `label=success`; if terminal verification is unavailable, the row cannot be promoted as success
- the first observation does not direct-match the Function start node, so `p_start_node_match=false` is recorded explicitly
- route predicate evidence shows that a route-only bridge, goto edge, or reobserve/repair step reaches an admissible start or
  create-flow state before semantic form actions begin
- the bridge path does not include semantic, input, save, or business-effect edges
- semantic Function execution begins only after the bridge evidence is recorded
- terminal verifier confirms the target contact row after bridge plus semantic execution
- evidence separates first observation, bridge observations/actions, semantic Function start observation, final observation,
  terminal evidence, route cost, semantic replay cost, and any replan baseline cost

Do not collapse this into direct allow. The intended predicate pattern is:

```text
p_start_node_match = false
route_only_bridge_admissible = true
terminal_business_effect = success
```

This row should support a `reobserve_or_repair -> success` policy decision, not a rule that any similar Contacts page can run
the semantic Function directly.

Minimum evidence fields for every promoted row:

- `case_id`
- `label_source`
- `label_status`
- command or oracle rule
- device/host id when applicable
- first observation reference
- final observation reference when execution ran
- terminal/business-effect evidence reference or explicit reason it is unavailable
- error code or refusal reason
- observe/check cost and execution/replan cost if measured

## Derived Candidate Materialization Protocol

The planned Contacts-home candidate is a future asset-writing step, not current evidence. This section applies to
`contacts_home_create_contact_candidate_20260524` in
`runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`.

Before asset materialization:

- require explicit confirmation to create a new file;
- keep the seed UTG store immutable;
- use the reserved path
  `runtime/evals/preconditions/contacts_minimal_live_preconditions/derived_candidates/contacts_home_create_contact_candidate_20260524.utg.json`;
- recompute the canonical `candidate_blueprint` digest and compare it with
  `manifest.json#planned_candidates[0].candidate_blueprint.digest` before writing the asset;
- require `materialization_preflight.current_checks` to pass and keep
  `materialization_preflight.status=blocked_pending_explicit_asset_creation_confirmation` until the user explicitly approves the
  new candidate asset file;
- treat `dry_run_materialization_audit` and `materialization_readiness` as planning metadata, not labels.

Materialization may only write the separate candidate asset. It must not:

- mutate `runtime/utg/contacts_official_success_5554_latest.json`;
- update `outcomes.json`;
- update `baseline_report.json`;
- update `learned_contract.json`.

Post-write checks required before any live probe:

- candidate asset JSON parses;
- candidate `function_id` is unique in the written asset;
- candidate blueprint digest still matches the manifest blueprint when reserialized with stable JSON key sorting;
- candidate `start_node_id` is `aw_node_51b4969e57`;
- candidate action sequence contains exactly the retained semantic actions:
  `aw_seq_51b4969e57`, `aw_seq_8225087450`, `aw_seq_8492568afe`, `aw_seq_d113a7485e`, `aw_seq_5577160d3a`;
- candidate contains no `aw_seq_22653dacf0` launcher/open-app prefix action;
- required arguments remain `first_name`, `last_name`, and `phone`;
- seed UTG file has no diff.

The first live probe is allowed only after those checks pass. Its command shape is:

```bash
python scripts/device_cli.py --target <serial> replay-function \
  --utg-store-path runtime/evals/preconditions/contacts_minimal_live_preconditions/derived_candidates/contacts_home_create_contact_candidate_20260524.utg.json \
  --function-id contacts_home_create_contact_candidate_20260524 \
  --argument first_name=<name> \
  --argument last_name=<name> \
  --argument phone=<phone> \
  --no-skip-terminal-verify
```

The live probe can promote a positive row only if it records `label_source=execution`, `label=success`, no
`__utg_skip_start_node_precondition__`, first-observation evidence for `p_start_node_match=true` on `aw_node_51b4969e57`,
valid arguments, terminal provider verification for the inserted row, and complete row-level cost fields.

If the probe has a start-node mismatch, missing terminal verifier, missing provider row, missing cost fields, or uses
start-node skip, it remains diagnostic-only.

## Cost Measurement Protocol

Cost measurement is part of the reuse-admission label, not an optional afterthought. A precondition contract is useful only if
the admission path is cheaper and safer than simply replanning or reobserving. Static checks, node matching, route repair, and
terminal verification are not free.

Required measured fields for every future execution-labeled row and every learned or selected contract:

| Field | Meaning |
|---|---|
| `observation_capture_ms` | Wall time to obtain the live observation used by admission. |
| `predicate_check_ms` | Total wall time spent evaluating candidate predicates before semantic replay. |
| `node_match_ms` | Time spent matching the current observation to the Function start node or candidate route node. |
| `argument_validation_ms` | Time spent validating required Function arguments before GUI actions. |
| `route_bridge_check_ms` | Time spent deciding whether a route-only bridge is admissible. |
| `route_bridge_execution_ms` | Time spent executing route-only bridge actions before semantic replay. Use `0` only for direct admission. |
| `semantic_replay_ms` | Time spent executing the semantic Function actions after admission or bridge. |
| `terminal_verify_ms` | Time spent running terminal/business-effect verification. |
| `total_admission_check_ms` | `observation_capture_ms + predicate_check_ms`; include node match, argument validation, and bridge admissibility checks. |
| `total_reuse_path_ms` | Admission check plus route bridge, semantic replay, and terminal verification. |
| `replan_wall_ms` | Wall time for the baseline replanning or rerun path on the same task/case. |
| `replan_model_tokens` | Online model tokens consumed by the replanning baseline, or `null` with an explicit local-agent proxy reason. |
| `replan_model_cost_proxy` | A normalized model/tool cost proxy when exact provider cost is unavailable. |
| `timing_source` | Timer source, command, host/device id, and whether the timing came from runner events or manual sidecar fields. |

Direct-admission cost must not use skipped-start rows. A direct-admission cost row requires the same no-skip start-node
conditions as the direct positive protocol above.

Route-repair cost must separate route checking, route execution, semantic replay, and terminal verification. Do not collapse a
successful route-repair row into a direct replay timing number.

If any required cost field is unavailable, set it to `null` and mark the cost gate unresolved. Do not infer timing from stdout
success, XML-only navigation, or a single final status line.

Cost gate:

```text
admit_reuse iff
  C_total_reuse_path + L_FA * Pr(false_accept)
  <
  C_replan + L_replan_failure * Pr(replan_failure)
```

where `C_total_reuse_path` includes admission checking, route repair if any, semantic replay, and terminal verification. If
`C_total_reuse_path` plus false-accept risk is not lower than the measured replan/rerun path, record
`learned_contract.json.status=negative_result` even if the predicate contract reduces false accepts.

Scoring order after labels exist:

1. Fill `outcomes.json` and `predicate_matrix.json` with evidence references.
2. Score `always_replay` and `always_replan` first to define safety and cost bounds.
3. Score `pagevectorset_threshold` and `exact_start_node` before any learned contract.
4. Score `hand_written_guard` and `llm_judge`.
5. Train or select `learned_minimal_preconditions`.
6. Only then compute false accept, false reject, allowed success, predicate count, and cost metrics.

Negative result handling:

- If precondition checking costs more than replanning for most Contacts rows, mark `learned_contract.json.status=negative_result`.
- If the learned contract collapses to page-family similarity or exact start-node match, record that as a negative or weak result rather than renaming it as learning.
- If terminal verifier coverage is insufficient, keep the experiment below `evidence_available`.
- If replan/rerun cost is not measured, keep the cost gate unresolved and do not claim the learned or selected contract is useful.

## Baselines

The first experiment should compare:

1. **Always replay**
   Execute the Function whenever it is retrieved.
2. **Always replan**
   Never reuse; this is the cost upper bound and false-accept lower bound.
3. **PageVectorSet threshold**
   Allow reuse if page/family similarity exceeds a fixed threshold.
4. **Exact start-node match**
   Allow only if `Function.start_node_id` matches.
5. **Hand-written guard**
   Use the obvious predicates manually selected from the table.
6. **LLM judge**
   Ask a model whether the Function is reusable from the state, but score it only by grounded execution labels.
7. **Learned minimal preconditions**
   Learn a compact predicate set from perturbation-labeled outcomes.

## Metrics

Primary:

- false accept rate: policy allows reuse but outcome is not `success`;
- false reject rate: policy rejects a state where reuse would succeed;
- unsafe/wrong-effect count;
- success rate among allowed reuses.

Secondary:

- predicate count in the learned contract;
- serialized contract length;
- observe/check cost before execution;
- saved online model tokens versus replanning;
- live execution time versus replanning;
- terminal verifier coverage.

## First Minimal Learners

Start with interpretable models:

1. Decision tree with max depth 2-4.
2. Sparse logistic regression over binary predicates.
3. MDL-style selector:

```text
score(P') = false_accept_cost * FA(P')
          + false_reject_cost * FR(P')
          + lambda_pred * |P'|
          + lambda_check * check_cost(P')
```

The expected paper contribution is not the specific learner. The contribution is the perturbation-driven framing that learns minimal applicability contracts instead of relying on similarity thresholds.

Cost-weighted selector diagnostic:

```text
score(P') =
  C_check(P')
  + C_reuse * allow_rate(P')
  + C_replan * reject_rate(P')
  + L_FA * false_accept_rate(P')
  + L_FR * false_reject_rate(P')
  + lambda_complexity * |P'|
```

The first Contacts result should report this as a decision/accounting score only. If `C_check(P')` plus required reobserve,
route-repair, and terminal-verifier cost is not lower than replanning for the target case, mark the selector result
`negative_result`.

## Expected First Contract

A plausible learned contract may look like:

```text
allow_reuse(ContactsAddContact) iff
  p_args_valid
  and p_terminal_verifier_available
  and (
    p_start_node_match
    or (p_app_contacts and p_create_entry_available and p_required_fields_bindable)
  )
  and p_no_blocking_overlay
```

This is a hypothesis. Do not report it as learned until the perturbation suite produces labels.

## Implementation Boundary

First implementation should be test-only:

```text
tests/preconditions/
  fixtures/
  perturbation_cases.json
  predicate_table.json
  evaluate_contacts_preconditions.py
```

Do not modify `src/utg/execution/control/` until the test-only protocol proves that precondition learning improves false accept / false reject over simple baselines.

If this later enters runtime, the likely integration point is between Function recall and `run_function(...)` admission, before semantic action dispatch and before terminal verification.

## Minimum Evidence Package

The first evidence package should write:

```text
runtime/evals/preconditions/contacts_minimal_live_preconditions/
  cases.jsonl
  predicate_matrix.json
  outcomes.json
  baseline_report.json
  learned_contract.json
  summary.md
```

The `summary.md` must state whether the learned contract actually beats the baselines. Negative results are valid.

## Promotion Gate

Do not promote this into `core_claims.md` or `eval_matrix.md` until:

- at least one positive and four negative/repair perturbations are labeled;
- terminal/business-effect labels are available for all success claims;
- false accept improves over PageVectorSet threshold and LLM judge, or false reject improves at fixed false accept;
- precondition checking cost is lower than replanning cost for the target case;
- the learned contract is smaller than a hand-written exhaustive guard.
