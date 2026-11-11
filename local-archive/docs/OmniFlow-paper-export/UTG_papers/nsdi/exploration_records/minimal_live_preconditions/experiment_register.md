# Minimal Live Preconditions Experiment Register

Status: Active
Last Updated: 2026-05-24

## Register Rules

- Every experiment entry must point to a protocol document and an output directory.
- Planned experiments must not be described as evidence.
- Completed experiments must point to generated files and state whether promotion gates passed.
- Runtime outputs belong under `runtime/evals/preconditions/`; paper claims still require promotion through official claim/eval sources.

## CMLP-001 - Contacts Minimal Live Preconditions

Status: partial

Protocol:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`

Seed asset:

- `runtime/utg/contacts_official_success_5554_latest.json#aw_function_515be7bf7e`

Input evidence already available:

- `runtime/evals/nsdi/harder_tasks_contacts_add_20260524/summary.json`
- `runtime/evals/nsdi/harder_tasks_contacts_add_20260524/entry_gap_probe_20260524T0410Z.json`

Planned output directory:

```text
runtime/evals/preconditions/contacts_minimal_live_preconditions/
```

Required output files:

- `manifest.json`
- `cases.jsonl`
- `predicate_matrix.json`
- `outcomes.json`
- `baseline_report.json`
- `learned_contract.json`
- `summary.md`

Planned sub-gate evidence schema:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/reasoning_subgate_evidence_schema.md`

Planned first PABench sub-gate row:

- `pabench_subgate_parser_missing_guard_original`

This row remains planned-only until a parser fixture exists and the row has ProjectState, VerifierObservation, cached skeleton,
executable predicate values, patch application evidence, verifier results, baseline decisions, cost measurement, risk metrics,
and a promotion gate. The row cannot update ProjectGraph or choose `delegate / verify / rollback / stop / escalate`; those
remain PABench Project Control responsibilities.

Minimum planned cases:

- `c0_correct_create_flow`
- `c1_launcher_wrong_start`
- `c2_contacts_home_reachable`
- `c3_missing_create_entry`
- `c4_overlay_blocks_target`
- `c5_missing_phone_field`
- `c6_args_missing_or_invalid`
- `c7_terminal_verifier_missing`
- `c8_similar_non_contacts_page`

Minimum planned baselines:

- `always_replay`
- `always_replan`
- `pagevectorset_threshold`
- `exact_start_node`
- `hand_written_guard`
- `llm_judge`
- `learned_minimal_preconditions`

Gate to move from `planned_skeleton_available` to `evidence_available`:

- All required output files exist.
- At least one positive and four negative/repair cases are labeled.
- All `success` rows have terminal/business-effect evidence.
- `baseline_report.json` includes false accept and false reject for every baseline.
- `learned_contract.json` is marked either `learned`, `not_learned_insufficient_data`, or `negative_result`.
- Row-level admission/reuse/terminal-verifier cost and a measured or explicitly proxied replan/rerun cost are recorded for
  any execution row used by a selected contract. If this is missing, the experiment remains `partial`.

Gate to move from `evidence_available` to `promotion_candidate`:

- Learned or selected contract improves false accept over PageVectorSet threshold and LLM judge, or improves false reject at fixed false accept.
- Checking cost is lower than replanning cost for the Contacts target.
- Contract is smaller than a hand-written exhaustive guard.
- The output `summary.md` explicitly lists failures and negative results.

Current partial evidence:

- `c0_correct_create_flow`: `offline_oracle` success label from
  `runtime/evals/nsdi/harder_tasks_contacts_add_20260524/summary.json`.
  The sidecar has terminal/business-effect provider evidence, but successful rows used
  `__utg_skip_start_node_precondition__=1`, so it is bounded create-flow success evidence rather than direct start-node admission evidence.
- `c1_launcher_wrong_start`: `execution` precondition-failure label from
  `runtime/evals/preconditions/contacts_minimal_live_preconditions/live_attempts/c1_launcher_wrong_start_execution_20260524.json`
  plus matcher diagnostic
  `runtime/evals/preconditions/contacts_minimal_live_preconditions/live_attempts/c1_launcher_wrong_start_match_diagnostic_20260524.json`.
  The replay-function command ran on `emulator-5554`, failed before semantic actions with
  `EXEC_START_NODE_PRECONDITION_FAIL`, left `device_actions=[]`, and terminal verification reported
  `contacts_record_not_found`. The matcher selected live launcher node `aw_node_8225087450` rather than required
  start node `aw_node_22653dacf0`, so this is execution-backed precondition failure, not direct admission evidence.
- Diagnostic direct-admission probe, not a promoted row:
  `runtime/evals/preconditions/contacts_minimal_live_preconditions/live_attempts/c0_direct_admission_probe_20260524.json`.
  This attempted `aw_function_515be7bf7e` from HOME/launcher with valid `first_name`, `last_name`, and `phone`, terminal
  verification enabled, and no start-node skip. It failed before any device action with
  `EXEC_START_NODE_PRECONDITION_FAIL` (`aw_node_8225087450 -> aw_node_22653dacf0`), and terminal verification reported
  `contacts_record_not_found`. The required start node `aw_node_22653dacf0` is the recorded launcher entry node whose
  first sequence action is `open_app`; the selected live node `aw_node_8225087450` is a Contacts form node in the Function.
  This is diagnostic evidence for node/matcher instability around direct admission, not the missing positive row.
- `c2_contacts_home_reachable`: `offline_oracle` reobserve/repair label from
  `runtime/page_collections/contacts_5554_targeted_20260506_151100_clean/index.json` plus
  `raw/root.xml` and `raw/create_contact.xml`.
  Contacts home has an actionable create-contact entry, and the route fixture reaches a create-contact form with Save, First name, Last name, and Phone fields. This is route/form evidence only, not terminal success evidence.
- Diagnostic route-repair probe, not a promoted row:
  `runtime/evals/preconditions/contacts_minimal_live_preconditions/live_attempts/c2_contacts_home_route_repair_probe_20260524.json`.
  This attempted `aw_function_515be7bf7e` from Contacts home with valid `first_name`, `last_name`, and `phone`, terminal
  verification enabled, and no start-node skip. It failed before any device action with
  `EXEC_START_NODE_PRECONDITION_FAIL` because no route-only bridge exists from Contacts home node `aw_node_51b4969e57` to
  the required Function start node `aw_node_22653dacf0`. Terminal verification reported `contacts_record_not_found`. This is
  diagnostic evidence for the current seed asset's route-repair gap, not a terminal success row.
- Diagnostic seed-asset audit, not a promoted row:
  `runtime/evals/preconditions/contacts_minimal_live_preconditions/asset_audits/p0_existing_function_start_state_audit_20260524.json`.
  This read-only graph audit found only one top-level Function in the seed UTG: `aw_function_515be7bf7e`, whose required start
  node is `aw_node_22653dacf0`. Contacts home `aw_node_51b4969e57` has no path back to that start node and no route-only bridge.
  The only non-route hop from Contacts home to the first-name form node comes from the semantic top-level Function, and
  node-scoped functions are atomic steps rather than terminal-verified top-level Functions. This artifact confirms that the
  current seed asset cannot itself supply the missing P0 positive row.
- Planned derived candidate, not evidence:
  `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json#planned_candidates/contacts_home_create_contact_candidate_20260524`.
  This candidate removes the original launcher/open-app prefix action from `aw_function_515be7bf7e` and starts the retained
  create-entry/form-fill/save sequence at Contacts home `aw_node_51b4969e57`. It tests a direct-from-Contacts-home Function
  hypothesis, not a route-only bridge back to the original start node. It remains `planned_only_not_evidence` until a separate
  candidate Function asset is materialized, executed from live Contacts home without start-node skip, terminal provider
  verification confirms the inserted row, and row-level cost fields are recorded.
  Its dry-run materialization audit is `dry_run_materializable_not_written`: the seed Function's 6 actions can become a
  5-action candidate by removing `aw_seq_22653dacf0`, while preserving the required arguments and end node. No seed or
  candidate asset file was written, and no outcome/baseline/learned-contract state changed.
	  Its materialization readiness gate is `ready_for_explicit_asset_materialization_not_promotion`: a future candidate asset path
	  is reserved under `runtime/evals/preconditions/contacts_minimal_live_preconditions/derived_candidates/`, but
	  `candidate_asset_written=false`, `asset_write_allowed_without_explicit_confirmation=false`, and `promotion_allowed=false`.
	  Function-id uniqueness, live Contacts-home execution, terminal provider verification, and row-level cost fields remain
	  blockers before the candidate can affect outcomes, baselines, or learned-contract status.
	  Its deterministic candidate blueprint is recorded with SHA-256
	  `ed993c650c5cfb214e7b20b8a41a202e4b5b61537cdc0773b802259d5f3c93c2` over the canonical planned-candidate blueprint. This
	  digest freezes the source Function lookup, removed prefix action, retained semantic actions, derived start/end nodes,
	  required arguments, and reserved asset path; it is not a candidate asset and must not promote evidence.
	  Its materialization preflight is `blocked_pending_explicit_asset_creation_confirmation`: digest recomputation, seed Function
	  shape, retained action sequence, reserved asset absence, and seed-file immutability are recorded, but the separate candidate
	  asset still cannot be written without explicit confirmation and cannot affect outcomes, baselines, or learned-contract status.
	  Its candidate materialization protocol is `planned_protocol_only`: a future agent may only write the reserved separate
	  candidate asset after explicit confirmation, may not mutate the seed UTG, and must pass post-write checks before any live
  probe. A live probe remains diagnostic-only unless it records no start-node skip, `p_start_node_match=true` for
  `aw_node_51b4969e57`, terminal provider verification, and complete row-level cost fields.
- `c4_overlay_blocks_target`: `offline_oracle` reobserve/repair label from
  `runtime/page_collections/contacts_5554_targeted_20260506_151100_clean/raw/chip_company.xml`.
  The fixture is a Contacts package observation with `com.google.android.contacts:id/touch_outside` covering the full screen
  and `com.google.android.contacts:id/design_bottom_sheet` covering the bottom of the display, with visible `Company` sheet text.
  This is overlay/blocking-observation evidence only, not terminal success or live replay failure evidence.
- `c6_args_missing_or_invalid`: `offline_oracle` reject label from
  `runtime/utg/contacts_official_success_5554_latest.json` plus `Function.validate_arguments`.
  The Function schema requires `first_name`, `last_name`, and `phone`; missing required arguments are rejected before GUI execution.
- `c7_terminal_verifier_missing`: `offline_oracle` abstain label from
  `src/integrations/utg_api/_execute.py`, `scripts/device_cli.py`,
  `tests/integrations/test_utg_api.py`, `tests/controller/test_terminal_verify.py`, and
  `src/omniflow/README.md`.
  Skipped, disabled, or unavailable terminal verification yields `verifiable=false` and
  `terminal_page_reached=null`, so this row is a verifier-boundary abstain label rather than success/failure evidence.
- `c8_similar_non_contacts_page`: `offline_oracle` reject label from
  `runtime/manual/usability_after_param_contact_baseline.xml`.
  The fixture is a Dialer page (`com.google.android.dialer`) with Contacts-like text such as `Create new contact`,
  `Search contacts & places`, and `Contacts`, but it is not the Contacts app and the current observation lacks
  the Contacts create-contact form anchors `First name`, `Last name`, `Phone`, and `Save`.

Current next action:

Use the label acquisition plan, Positive Evidence Acquisition Protocol, and Cost Measurement Protocol in `contacts_protocol.md`
to complete baseline scoring beyond the first deterministic rows and collect stronger positive evidence. The first deterministic
baseline pass is partial: `always_replay` has 6/7 false accepts on the promoted rows, while `always_replan`,
`exact_start_node`, and the current `hand_written_guard` avoid false accepts but reject, repair, or abstain on the only success
row. An explicit selector diagnostic has been rerun and `learned_contract.json` is still `not_learned_insufficient_data`, not
learned: apparent perfect subsets either exploit unknown predicate values or require `p_start_node_match` while the only
success row has that predicate unknown. The fresh c1 execution row, c4 overlay row, c7 abstain boundary, and c0
direct-admission diagnostic do not change this conclusion because they are negative/repair, abstain, or diagnostic evidence. A
machine-readable blocker pass now records why `pagevectorset_threshold`, `llm_judge`, and
`learned_minimal_preconditions` remain unscored: PageVectorSet lacks per-case scores and a frozen threshold; LLM judge lacks a
frozen prompt, per-case decisions, and token/cost records; learned preconditions lack direct-admission or route-repair positive
evidence, scored similarity/LLM baselines, and a resolved cost gate. A follow-up PageVectorSet diagnostic audit computed only
partial, non-scoring scores from existing artifacts: `c1` existing matcher top-1 is `0.810096263885498`, `c2` best-over-seed-node
`xml_similarity` is `0.944586`, `c4` is `0.569046`, and `c8` is `0.907145`. This audit keeps the baseline blocked because
`c0` lacks a direct-admission or route-repair first/current XML score tied to terminal-verified positive evidence, `c6` is a
schema-level pre-GUI row, `c7` is a verifier-boundary abstain row, and no comparator or threshold rule has been frozen. These
are missing-evidence blockers and diagnostic warning signs, not measured baseline failures. A
structured XML audit found no promotable `c5_missing_phone_field` fixture and no stable/non-transient
`c3_missing_create_entry` negative in the current artifacts: create/form-like Contacts pages with Save / First name / Last name
all had Phone, while pages without Create contact were routeable or page-local states such as detail, search, tab, menu,
account, or overlay pages. The next priority is either a direct-admission positive row with `label_source=execution`, no
`__utg_skip_start_node_precondition__`, `p_start_node_match=true`, valid arguments, terminal provider verification, and cost
fields, or a route-repair positive row where `p_start_node_match=false`, a route-only bridge reaches an admissible
start/create-flow state before semantic form actions, terminal verification confirms the inserted row, and route/replay/verify
costs are separated. If c3/c5 remain useful, they need a new curated/live fixture rather than reusing the current XML
collection. The experiment is not `evidence_available` because PageVectorSet / LLM baselines remain unscored, no learned
contract exists, and checking/reuse cost has not been measured against replanning.

P0 feasibility update:

- The live Contacts-home route-repair probe did not collect a positive row. Current `run_function(...)` semantics reject the
  state before device actions because the seed graph lacks a route-only bridge from `aw_node_51b4969e57` to `aw_node_22653dacf0`.
  The next positive row therefore needs either a different Function whose start/create-flow state is admissible from the live
  observation, or independently collected route-only bridge evidence.
- The seed-asset audit confirms there is no existing alternative top-level Function in `contacts_official_success_5554_latest.json`
  that starts at Contacts home or create-flow. Do not try to promote node-scoped atomic functions as the missing positive row;
  they lack the complete Function argument schema and terminal-verifier evidence required by `contacts_protocol.md`.
- The planned derived candidate `contacts_home_create_contact_candidate_20260524` is the next concrete hypothesis to test. It
  should be materialized as a separate candidate asset before live execution and must not mutate the seed UTG store or update
  `outcomes.json` until terminal/provider and cost gates pass.
- The dry-run audit shows the candidate is mechanically derivable in memory, so the next step is asset materialization plus live
  execution, not another route-search audit over the old seed Function.

Evidence acquisition queue:

1. P0: collect exactly one terminal-verified positive row, either direct-admission with `p_start_node_match=true` and no
   start-node skip, or route-repair with an explicit route-only bridge before semantic form actions.
2. P1: score `pagevectorset_threshold` only after every scored row has score provenance and the comparator plus threshold rule
   are frozen before scoring.
3. P2: run `llm_judge` only as a frozen baseline with prompt/model/decision/token records; do not use it as an oracle.
4. P3: if the next positive row or selected contract lacks required cost fields, keep the cost gate unresolved; if reuse cost
   plus false-accept risk is not lower than measured/proxied replan, record `negative_result`.

## CMLP-002 - Reasoning Trace Minimal Validity Conditions

Status: planned_skeleton_available

Protocol:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`

Seed asset:

- planned public code-agent skeleton only; no hidden chain-of-thought is allowed.

Planned output directory:

```text
runtime/evals/preconditions/reasoning_minimal_validity_conditions/
```

Required output files:

- `manifest.json`
- `cases.jsonl`
- `predicate_matrix.json`
- `outcomes.json`
- `baseline_report.json`
- `learned_contract.json`
- `summary.md`

Minimum planned cases:

- `r0_original_failure`
- `r1_failure_signature_changed`
- `r2_anchor_removed`
- `r3_anchor_semantics_changed`
- `r4_patch_slot_ambiguous`
- `r5_verifier_missing`
- `r6_regression_scope_missing`
- `r7_tool_version_changed`
- `r8_conflicting_local_changes`

Minimum planned baselines:

- `always_reuse_skeleton`
- `always_rerun_code_agent`
- `retrieval_score_threshold`
- `exact_anchor_match`
- `hand_written_guard`
- `llm_judge`
- `learned_minimal_validity_conditions`

Gate to move from `planned_skeleton_available` to `evidence_available`:

- All required output files exist.
- At least one positive and four negative/repair cases are labeled by tests/checkers.
- All `success` rows have verifier command output.
- Each promoted row records whether the patch/action was applied, verifier command, exit code, stdout/stderr reference or digest,
  and verifier wall time.
- `baseline_report.json` includes false accept and false reject for every baseline.
- `learned_contract.json` is marked either `learned`, `not_learned_insufficient_data`, or `negative_result`.
- `summary.md` records `predicate_static_check_ms`, `verifier_probe_ms`, `patch_apply_ms`,
  `total_admission_check_ms`, `rerun_code_agent_wall_ms`, and rerun model tokens or a stated local-agent proxy.
- No hidden chain-of-thought is stored, consumed, or evaluated.

Gate to move from `evidence_available` to `promotion_candidate`:

- Learned or selected contract improves false accept over retrieval-score threshold and LLM judge, or improves false reject at fixed false accept.
- Checking cost is lower than rerunning the code agent for the target case.
- Contract is smaller than a hand-written exhaustive guard.
- The output `summary.md` explicitly lists failures and negative results.

Current next action:

Keep this behind CMLP-001 unless the user explicitly wants the code-agent branch first. The output directory and required skeleton files now exist, but the experiment is not `evidence_available` because there is no code-agent fixture, patch application, verifier command output, learner, or baseline scoring. The next concrete step is to add one tiny test/checker-backed code fixture and replace `planned_expected_only` rows with verifier-backed labels.

If this branch is pulled forward, the first fixture must be a code/repo-state validity task, not generic reasoning:

```text
failing pytest output + stable AST anchor + patch slot + patch action + verifier obligations
```

The current runtime manifest includes a machine-readable `domain_triage` block:

- `accepted_now`: only code repair, bound to `code_fix_parser_missing_guard_v0`, still `planned_only_not_evidence`.
- `parked_later`: SQL repair, proof/program synthesis, and build/config repair.
- `rejected`: open-ended chat reasoning, reflection/explanation memory, hidden chain-of-thought reuse, and LLM-agreement labels.

The current runtime manifest also includes `pre_fixture_readiness_audit`:

- `status`: `not_ready_to_execute_or_promote`.
- `decision`: `keep_planned_skeleton_available`.
- `materialization_gate`: `planned_only`; fixture source files, public skeleton JSON, and failing-before verifier output are missing.
- `predicate_gate`: `planned_only`; static/test/tool/diff-derived predicate values are missing.
- `verifier_gate`: `missing`; pytest output, patch-applied flags, exit codes, and stdout/stderr digests are missing.
- `cost_gate`: `missing`; predicate/check/verifier/patch/rerun timing and token/proxy fields are missing.
- `must_not_infer`: planned seed bindings are not fixture evidence, planned labels are not verifier-backed labels, and the manual
  hypothesis is not a learned contract.

The first executable pass must create exactly one parser fixture after explicit confirmation and then produce fixture, predicate,
verifier, and cost artifacts. Its acceptable outcomes are `evidence_available`, `not_learned_insufficient_data`, or
`negative_result`. Promotion still requires measuring checker cost against rerunning a code agent. If the checker path is more
expensive, the correct outcome is `negative_result`, not a broader reasoning-memory system.

Planned PABench sub-gate execution checklist:

- Source: `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md#PABench Sub-Gate Execution Checklist`.
- Future row root:
  `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/`.
- Required future artifact classes: public skeleton, row envelope, ProjectState, VerifierObservation, executable predicate
  values, patch application evidence, verifier logs, baseline decisions, cost measurement, risk metrics, and promotion gate.
- Current status: planned-only; none of the row-root artifacts, parser fixture files, patch diff, verifier output, or cost
  measurements exist yet.
- Promotion blocker: the row cannot leave planned/diagnostic status until every artifact class is backed by executable evidence
  and the row remains a reuse-admission gate rather than a `delegate / verify / rollback / stop / escalate` controller policy.
- Non-overlap audit: the row must pass
  `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md#PABench Non-Overlap Audit`. If it chooses a
  Project Control action, updates or folds ProjectGraph, marks the transition resolved, owns rollback/stop/escalation, or turns
  transition-quality metrics into the primary controller objective, it is no longer a CMLP row and must be moved back to PABench.
- Subsumption / replacement test: the row must also pass
  `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md#PABench Subsumption / Replacement Test`.
  Under the same `ProjectState + VerifierObservation`, changing only the cached skeleton must be able to change the correct
  admission decision through skeleton-specific predicates, verifier obligations, write-set binding, or cost/risk. If the cached
  skeleton is irrelevant and the decision collapses to `delegate / verify / rollback / stop / escalate`, the artifact belongs to
  PABench rather than CMLP.

Cost/risk accounting rule for the first fixture:

```text
admit_reuse iff
  E[C_check(P*) + C_patch + C_verify + L_FA * Pr(false_accept | P*)]
  <
  E[C_rerun_code_agent + L_rerun_failure * Pr(rerun_failure)]
```

The first pass must report this as an accounting gate. It is not a new mathematical claim, and it must not be used to promote
CMLP-002 without verifier-backed labels and measured costs.

Parking rule:

Keep `CMLP-002` parked unless explicitly pulled forward. If it is pulled forward, the first executable pass is only
`code_fix_parser_missing_guard_v0`; it must produce fixture, predicate, verifier, and cost artifacts. If checker plus patch plus
verify cost is not lower than rerunning a strong code/tool agent, record `negative_result` and do not broaden the branch.
