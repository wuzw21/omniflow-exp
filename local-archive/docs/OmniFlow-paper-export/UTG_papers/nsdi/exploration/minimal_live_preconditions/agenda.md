# Minimal Live Preconditions Agenda

Status: Draft
Last Updated: 2026-05-24

## Current Research Spine

The spine is reuse admission:

```text
historical successful trace + current live state
  -> candidate predicates
  -> perturbation-labeled outcomes
  -> minimal precondition contract
  -> allow / reject / reobserve / repair
```

Do not broaden this into "all-agent self-evolution" until the admission problem has hard evidence.

## Boundary With PABench

Keep the two research lines separated by the decision they make:

- PABench / Project Control decides the next project-control action under `ProjectState + VerifierObservation`:
  `delegate`, `verify`, `rollback`, `stop`, or `escalate`.
- CMLP decides whether one already-known trace, Function, or verifier-backed skeleton is admissible now:
  `allow_reuse`, `reject`, `reobserve`, or `repair_precondition`.

Allowed composition:

```text
Project Control transition
  -> optional CMLP gate for a cached repair skeleton
  -> controller still decides delegate / verify / rollback / stop / escalate
```

Disallowed broadening:

```text
CMLP becomes the project-level policy that chooses the whole transition.
```

If a future step combines the two, the artifact must record the interface as `ProjectState + VerifierObservation + cached
skeleton -> reuse-admission decision`, not as a replacement for PABench's controller policy.

Replacement/subsumption test:

- Hold `ProjectState` and `VerifierObservation` fixed.
- Vary only the cached trace, Function, or verifier-backed skeleton.
- If admission decisions differ because the cached object has different anchors, slots, verifier obligations, or cost/risk,
  the row is a CMLP row.
- If the cached object is irrelevant and the only meaningful decision is the outer control action, the row belongs to PABench.
- If a row needs both decisions, record it as Project Control calling a CMLP sub-gate; do not merge the action spaces.

Current comparison verdict:

- PABench's own research object is the controller input interface and policy over verifier-observed project transitions.
- CMLP's own research object is the admission contract for one reusable object.
- Therefore GUI CMLP is independent enough to keep because the cached Function is the object being admitted or rejected.
- CMLP-002 is fragile: without skeleton-specific admission evidence, it collapses into PABench.
- The first parser row must include either two cached skeleton choices, or one cached skeleton plus a predicate perturbation that
  changes only skeleton validity while `ProjectState + VerifierObservation` stay fixed. Otherwise keep it parked or move it to
  PABench records.
- The first planned skeleton-causality contrast is now fixed in `reasoning_protocol.md`: under the same parser
  `ProjectState + VerifierObservation`, `code_fix_parser_missing_guard_v0` is the only candidate that may become admissible,
  while `code_fix_parser_delimiter_rewrite_v0` is the planned contrasting skeleton to reject for skeleton-specific anchor,
  patch-slot, write-set, regression-verifier, or cost reasons. This remains planned-only until executable predicate and verifier
  evidence exists.

## Immediate Next Steps

1. Keep `contacts_protocol.md` as the current protocol for the first evidence package.
2. Continue following the label acquisition plan in `contacts_protocol.md` to replace the remaining planned-only rows under
   `runtime/evals/preconditions/contacts_minimal_live_preconditions/`
   with execution-labeled or offline-oracle-labeled outcomes.
3. Measure row-level admission, reuse, terminal-verifier, and replan/rerun cost fields before claiming any learned or selected
   contract is useful.
4. Decide the first learner:
   start with decision tree and sparse logistic; keep MDL as the paper framing if it gives clearer explanation.
5. Establish labels:
   `success`, `terminal_failure`, `precondition_failure`, `unsafe_or_wrong_effect`, `abstain`.
6. Add two baselines first:
   PageVectorSet threshold and hand-written guard.

## Evidence Acquisition Queue

Use this queue to avoid broadening the exploration before the first reuse-admission result is actually grounded.

P0 - collect one terminal-verified positive row for `CMLP-001`.

- Preferred direct-admission positive: `label_source=execution`, no `__utg_skip_start_node_precondition__`,
  `p_start_node_match=true`, valid `first_name` / `last_name` / `phone`, terminal provider verification, and full cost fields.
- Acceptable route-repair positive: `p_start_node_match=false`, a route-only bridge reaches an admissible start/create-flow state
  before semantic form actions, terminal provider verification confirms the inserted row, and route/check/semantic/verify costs
  are separated.
- Stop condition: if the row cannot provide terminal/business-effect evidence or required cost fields, keep it diagnostic rather
  than promoting it.
- Latest feasibility diagnostic: `runtime/evals/preconditions/contacts_minimal_live_preconditions/live_attempts/c2_contacts_home_route_repair_probe_20260524.json`
  attempted the seed Function from Contacts home with terminal verification enabled, but current runtime rejected before
  device actions with `EXEC_START_NODE_PRECONDITION_FAIL` because no route-only bridge exists from `aw_node_51b4969e57` to
  `aw_node_22653dacf0`. Terminal verification found no inserted contact. This is not a positive row; it says the current seed
  asset cannot supply P0 route-repair evidence without a new route bridge or a different start/create-flow Function.
- Follow-up asset audit: `runtime/evals/preconditions/contacts_minimal_live_preconditions/asset_audits/p0_existing_function_start_state_audit_20260524.json`
  checked the stored seed UTG directly. It found one top-level Function, no alternative top-level Function starting at Contacts
  home or create-flow, no path from `aw_node_51b4969e57` back to required start node `aw_node_22653dacf0`, and no route-only
  bridge for that pair. This is diagnostic-only and must not update `outcomes.json`.
- Planned derived candidate: `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json` now records
  `contacts_home_create_contact_candidate_20260524` as `planned_only_not_evidence`. It removes the original
  launcher/open-app prefix action and starts from Contacts home `aw_node_51b4969e57`, then keeps the semantic create-entry,
  first-name, last-name, phone, and save steps. This turns the P0 gap into a concrete direct-from-Contacts-home Function
  hypothesis, but it must not update outcomes, baselines, or learned-contract status until a separate candidate asset is
  materialized, live-executed with terminal provider verification, and measured with row-level cost fields.
- Dry-run materialization audit: the same manifest now records `dry_run_materialization_audit.status=dry_run_materializable_not_written`.
  The in-memory derivation confirms that the source Function's 6 actions become a 5-action Contacts-home candidate after
  removing `aw_seq_22653dacf0`; parameters and end node are preserved, and no seed or candidate asset file was written. This
  makes the next implementation step concrete, but it still does not promote any label or prove live success.
- Materialization readiness gate: the manifest now records
  `materialization_readiness.status=ready_for_explicit_asset_materialization_not_promotion`. The gate reserves a future
  candidate path under `runtime/evals/preconditions/contacts_minimal_live_preconditions/derived_candidates/` but records
  `candidate_asset_written=false`, `asset_write_allowed_without_explicit_confirmation=false`, and
  `promotion_allowed=false`. This is a no-new-file readiness checkpoint: it says the candidate can be materialized next, not
  that it exists, succeeded, or can update outcomes, baselines, or learned-contract status.
- Candidate materialization protocol: the manifest also records
  `candidate_materialization_protocol.status=planned_protocol_only`, and `contacts_protocol.md` now defines the matching
  future procedure. The protocol requires explicit confirmation before creating the candidate asset, forbids seed mutation and
  outcome/baseline/contract updates from materialization alone, and separates post-write schema checks from the later live
  Contacts-home probe. A live probe remains diagnostic unless it has no start-node skip, `p_start_node_match=true` for
  `aw_node_51b4969e57`, terminal provider verification, and complete row-level cost fields.
- Candidate blueprint digest: the same manifest now records
  `candidate_blueprint.status=canonical_digest_recorded_not_asset` with SHA-256
  `ed993c650c5cfb214e7b20b8a41a202e4b5b61537cdc0773b802259d5f3c93c2` over the canonical planned-candidate blueprint.
  This freezes the derivation identity for future materialization checks, but it is still not a Function asset, not live
  execution evidence, and not a reason to update outcomes, baselines, or learned-contract status.
- Materialization preflight: the same manifest now records
  `materialization_preflight.status=blocked_pending_explicit_asset_creation_confirmation`. The preflight recomputes the
  blueprint digest, checks the seed Function shape, confirms the reserved candidate asset is still absent, and confirms the seed
  file is unchanged. It remains blocked because creating the reserved asset is a new-file write that requires explicit
  confirmation.

P1 - score PageVectorSet only after its evidence is complete.

- Required before scoring: score provenance for every scored row, a frozen comparator rule, a frozen threshold rule, and grounded
  labels for every row in the denominator.
- Current warning sign: partial diagnostic scores already show that best-over-Function-nodes similarity can admit non-success
  rows `c2_contacts_home_reachable` and `c8_similar_non_contacts_page`.
- Stop condition: if comparator or threshold choices are selected after seeing labels, keep the baseline blocked.

P2 - run LLM-judge baseline only if it is treated as a baseline, not an oracle.

- Required before scoring: frozen prompt, model id, per-case decisions, token/cost records, and grounded outcome labels.
- Stop condition: LLM agreement cannot promote labels and cannot replace terminal/business-effect verification.

P3 - keep `CMLP-002` parked unless explicitly pulled forward.

- The only allowed first executable pass is one parser fixture for `code_fix_parser_missing_guard_v0`.
- Stop condition: if `C_check + C_patch + C_verify` is not lower than rerunning a strong code/tool agent, record
  `negative_result` and do not expand into broad reasoning-memory work.
- PABench composition path: if Project Control uses CMLP-002 later, start with the `PABench Sub-Gate Interface Draft` in
  `reasoning_protocol.md`. The first artifact must be a row-level admission decision for one cached parser skeleton, not a new
  controller policy. The row must keep `may_apply_cached_skeleton=false` until executable labels and cost fields exist.
- First sub-gate row plan: `reasoning_protocol.md` now defines
  `pabench_subgate_parser_missing_guard_original` as the only planned first row. It must compare always-reuse,
  retrieval-score reuse, LLM judge, always-rerun code agent, and hand-written guard under the same verifier-backed labels.
  Kill the branch as `negative_result` if admission cost is not below rerun cost or if the row becomes a project-level
  `delegate / verify / rollback / stop / escalate` decision.
- Evidence template: the same first row must follow
  `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/reasoning_subgate_evidence_schema.md`. A row-level result is
  not promotable unless it includes ProjectState, VerifierObservation, cached skeleton, executable predicate values, patch
  application evidence, verifier results, baseline decisions, cost measurement, risk metrics, and a promotion gate.
- Execution checklist: `reasoning_protocol.md#PABench Sub-Gate Execution Checklist` now maps the planned row to exact future
  artifact paths under
  `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/`.
  This is still planned-only metadata. Do not create those files, the parser fixture, patch evidence, or verifier logs without
  explicit confirmation.
- Non-overlap audit: `reasoning_protocol.md#PABench Non-Overlap Audit` now defines the rejection test for drift. A row remains
  CMLP only if it admits or rejects one cached skeleton; if it chooses `delegate`, `verify`, `rollback`, `stop`, or `escalate`,
  updates ProjectGraph, marks a transition resolved, or owns rollback/stop/escalation, it must move back to PABench rather than
  expanding the CMLP action space.
- Skeleton-causality contrast: `reasoning_protocol.md#First Skeleton-Causality Contrast Plan` records the planned A/B contrast
  for the first parser row. It freezes the fixed outer state and two cached skeleton choices, but it does not create fixture
  evidence, verifier labels, baseline decisions, or cost measurements.

## Research Maturity Gate

Use this gate before starting any next slice. Its purpose is to keep the line evidence-first and prevent it from expanding into
PABench, generic reasoning memory, or an expensive guard layer.

Every proposed slice must be classified as exactly one of:

- `explore`: clarify boundary, protocol, schema, or a planned row without changing runtime evidence status.
- `collect_evidence`: create or run one bounded evidence package with a named oracle, baseline set, and cost fields.
- `park`: leave the branch unchanged because it needs explicit confirmation, a missing fixture, a missing verifier, or a missing
  cost baseline.
- `negative_result`: stop expanding that branch because labels are not grounded, the cached object is not the causal variable, or
  checking plus reuse is not cheaper or safer than rerun/replan.

Promotion to `collect_evidence` requires all of the following:

- one named trace, Function, candidate asset, or public skeleton;
- one protocol section that says how labels are acquired;
- terminal/business-effect evidence for GUI rows or verifier-backed labels for artifact rows;
- baseline decisions for always-reuse/replay, always-rerun/replan, similarity or retrieval score, LLM judge if used, and a
  hand-written guard if available;
- row-level cost fields for admission checking, reuse or patching, terminal/verifier checks, and rerun/replan comparison;
- a PABench boundary statement if `ProjectState + VerifierObservation` appears in the row.

`CMLP-001` next-slice rule:

- Continue only with a terminal-verified positive row with full cost fields, or with explicit materialization of the reserved
  Contacts-home candidate followed by a no-skip live probe.
- Keep any run diagnostic if it lacks terminal provider verification, uses a skipped start-node precondition, lacks cost fields,
  or mutates only metadata.
- Mark `negative_result` if the measured `total_admission_check_ms + total_reuse_path_ms` plus false-accept risk is not lower
  than measured or explicitly proxied replan/rerun cost.

`CMLP-002` next-slice rule:

- Keep parked unless the user explicitly confirms creation of the single parser fixture for `code_fix_parser_missing_guard_v0`.
- Once confirmed, create only that fixture and require fixture, predicate, verifier, baseline, risk, and cost artifacts before
  changing status.
- Mark `negative_result` if `C_check + C_patch + C_verify` is not lower than rerunning a strong code/tool agent, if labels depend
  on LLM agreement, or if the row becomes a project-level controller decision.

PABench sub-gate rule:

- A CMLP/PABench row may continue only if the cached trace, Function, or skeleton remains a necessary variable under fixed
  `ProjectState + VerifierObservation`.
- If all useful decisions collapse to `delegate / verify / rollback / stop / escalate`, move the artifact to PABench records
  rather than broadening CMLP.

## Current Next-Slice Classification

This table applies the research maturity gate to the current candidates. It is a record-only classification; it does not
promote labels, create assets, or change runtime evidence status.

| Candidate slice | Classification | Reason | Allowed next move |
|---|---|---|---|
| `CMLP-001` P0 terminal-positive through the reserved Contacts-home candidate | `park` | The candidate is only a planned blueprint/preflight entry. The separate Function asset is not written, no no-skip live probe exists, and terminal/cost evidence is missing. | After explicit asset-creation confirmation, materialize only the reserved candidate asset, then run a no-skip live probe with terminal provider verification and full row-level cost fields. |
| `CMLP-001` positive row from any other existing admissible Function/start state | `park` | The current seed asset audit found no alternative top-level Function starting at Contacts home/create-flow and no route-only bridge from `aw_node_51b4969e57` to `aw_node_22653dacf0`. | Continue only if a named executable Function/start state or independently collected route-only bridge evidence is identified before the live probe. |
| `CMLP-001` PageVectorSet threshold baseline | `park` | Existing scores are diagnostic only; the comparator, threshold rule, complete score provenance, direct-positive page score, and grounded denominator are not frozen. | Reopen after a terminal-verified positive row exists and all rows in the scoring denominator have grounded labels and score provenance. |
| `CMLP-001` LLM-judge applicability baseline | `park` | It has no frozen prompt, model id, per-case decision table, token/cost record, or grounded-label denominator. LLM agreement cannot label success. | Reopen only as a baseline after labels are grounded and the prompt/model/cost ledger is frozen before scoring. |
| `CMLP-002` parser fixture for `code_fix_parser_missing_guard_v0` | `park` | The branch has planned skeleton metadata only. Creating the parser fixture, verifier outputs, patch evidence, and cost artifacts requires explicit confirmation. | If explicitly pulled forward, create exactly one parser fixture and end the first pass as `evidence_available`, `not_learned_insufficient_data`, or `negative_result`. |
| `pabench_subgate_parser_missing_guard_original` | `park` | The PABench sub-gate row depends on the same missing parser fixture, executable predicate values, verifier labels, baseline decisions, rerun-cost baseline, and a skeleton-causality check. The planned contrast is `code_fix_parser_missing_guard_v0` versus `code_fix_parser_delimiter_rewrite_v0` under fixed parser `ProjectState + VerifierObservation`. | Reopen only after fixture evidence exists and the cached skeleton remains causal under fixed `ProjectState + VerifierObservation`; otherwise move the row to PABench. |
| Any CMLP proposal whose decision is `delegate / verify / rollback / stop / escalate` | `negative_result` for CMLP scope | That is Project Control action selection, not reuse admission for one cached object. | Move the artifact to PABench records instead of expanding the CMLP action space. |

## Planned-Only Freeze And Go/No-Go Gate

The current exploration has enough protocol, schema, and parked-row text. Do not add another planned row, another schema, or
another contrast unless it replaces an existing one with stronger evidence. The next slice must be decided by this gate:

| Candidate next move | Go condition | No-Go outcome |
|---|---|---|
| Materialize `contacts_home_create_contact_candidate_20260524` | The user explicitly confirms writing the reserved candidate asset, and the materialization step writes only that separate asset without mutating seed UTG or outcomes/baselines/contracts. | Keep CMLP-001 parked. Do not add more Contacts candidate designs or selector text. |
| Run a CMLP-001 live positive probe | A materialized candidate or another named admissible Function/start state exists, terminal provider verification is enabled, no start-node skip is used, and row-level cost fields can be recorded. | Keep the run diagnostic-only, or record `negative_result` if the cost/risk gate fails. |
| Create the CMLP-002 parser fixture | The user explicitly confirms creating exactly one parser fixture for `code_fix_parser_missing_guard_v0`, and the pass also records predicate, verifier, baseline, risk, and cost artifacts. | Keep CMLP-002 parked. Do not add another reasoning domain, skeleton family, or prompt-memory variant. |
| Promote a CMLP/PABench sub-gate row | Both cached skeleton choices have executable predicate values, verifier-backed labels, baseline decisions, and cost fields under the same outer `ProjectState + VerifierObservation`. | Move the row to PABench if the cached skeleton is not causal, or keep it planned if evidence is missing. |
| Score PageVectorSet or LLM judge baselines | The denominator has grounded labels, positive evidence exists, the comparator/prompt/model is frozen before scoring, and per-case cost/provenance is recorded. | Keep the baseline blocked; do not use similarity or LLM agreement as an oracle. |

If none of the Go conditions holds, the correct next action is not more CMLP planning. It is either:

- answer the user with the current parked status;
- collect one of the missing evidence artifacts after explicit confirmation;
- or move a project-control idea to PABench.

## Open Questions

- Is "minimal" measured by predicate count, serialized contract length, observe/check cost, or all three?
- Should the learner optimize false accept first, or a cost-weighted risk score?
- Can one successful trace plus generated perturbations produce enough signal, or do we need multiple successful/failed traces per Function?
- Which predicates are host-independent enough to survive beyond GUI?
- How should learned preconditions be written back into a Function asset without polluting the current NSDI claim surface?

## Cost And Risk Accounting

The current decision-theoretic rule is deliberately simple:

```text
admit_reuse iff
  E[C_check(P*) + C_reuse + L_FA * Pr(false_accept | P*)]
  <
  E[C_replan + L_replan_failure * Pr(replan_failure)]
```

Candidate contracts should be selected with a cost-weighted diagnostic score:

```text
score(P') =
  C_check(P')
  + C_reuse * allow_rate(P')
  + C_replan * reject_rate(P')
  + L_FA * false_accept_rate(P')
  + L_FR * false_reject_rate(P')
  + lambda_complexity * |P'|
```

Use this as accounting, not as a novelty claim. If the check path costs more than rerunning or reobserving, the right outcome is
`negative_result`, not a broader precondition system.

## Reasoning Extension Parking Lot

The reasoning extension is promising only if it is verifier-grounded. It should be framed as minimal validity conditions for a public reasoning skeleton, not as replaying hidden chain-of-thought:

- Code-agent trace: reuse a bug-fix skeleton only if test failure invariant, code structure, patch location, and verifier obligations still hold.
- SQL-agent trace: reuse query reasoning only if schema relations, key constraints, and expected result invariants still hold.
- Proof/program synthesis: reuse proof or program sketch only if assumptions and checker obligations still hold.

Minimum first protocol for this branch:

- one public skeleton, such as `failure signature -> AST anchor -> patch slot -> verifier command`;
- candidate predicates over repo state, tool versions, anchors, verifier availability, and regression checks;
- perturbations that remove or alter one predicate at a time;
- labels from tests/checkers/tool results, never from LLM self-judgment;
- baselines: retrieval-score reuse, LLM judge, always rerun code agent, and hand-written guard.

Current planned package:

- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/`
- status: `planned_skeleton_available`
- next evidence step: add one tiny code-agent bug-fix fixture with verifier command output, then replace `planned_expected_only` labels with test/checker-backed labels.
- concrete first fixture plan: `code_fix_parser_missing_guard_v0`, a tiny parser bug where `parse_record` should reject empty input, verified by one targeted failing pytest and one valid-record regression pytest.
- current package tightening: all planned runtime `cases.jsonl` and `predicate_matrix.json` rows are now bound to that parser seed, with explicit
  `planned_only_not_evidence` status.

Do not use chain-of-thought text as the reusable object. If the branch cannot produce executable predicates and verifier-backed outcomes, it is not research for this direction.

Decision for the current reasoning question:

- Keep the branch only as verifier-grounded artifact-state validity learning.
- Treat "reasoning" as a misleading surface word unless it is reduced to public artifacts: failure signatures, anchors, patch/query/proof slots, tool outputs, verifier obligations, and verifier results.
- Do not broaden it to all-agent memory, self-evolution, reflection reuse, or explanation retrieval.
- A model upgrade can improve raw reasoning, but it does not remove the artifact-validity question: a reused skeleton is still unsafe when the current repo/schema/test/proof state violates its minimal predicates.
- The first action remains one tiny code-agent fixture with real verifier labels and cost comparison.

Current answer to "what if it is reasoning":

- Yes, but the object changes from reasoning text to a verifier-grounded dependency skeleton.
- The state changes from GUI live state to artifact live state: repository files, AST anchors, tool versions, tests, schema, and verifier availability.
- The method is still reuse admission: learn the minimal predicates under which reuse is safe.
- The first experiment must be one tiny code-agent fixture, not open-ended chat reasoning or prompt-memory reuse.
- The kill condition remains cost-based: if checking the validity predicates costs more than rerunning a strong code agent, record a negative result.
- Working score: 8/10, below the GUI branch's 9/10, because public verifiers make it researchable but strong code-agent reruns make the cost advantage harder to prove.
- Best first seed: a tiny bug-fix skeleton `failing pytest output -> AST anchor -> patch slot -> patch action -> verifier obligations`, with perturbations for changed failure signature, missing anchor, ambiguous patch slot, missing verifier, and conflicting local changes.
- Do not pitch it as "reasoning cache". Pitch it as validity-condition learning for reusable verifier-grounded skeletons.
- One-line definition: learn when a previous verifier-backed artifact-edit skeleton is still valid in the current artifact state, under a cost gate against rerunning the agent.

Reasoning-domain triage:

- Accept as CMLP-002 targets only domains with public skeletons and verifier labels: code repair first, then SQL repair,
  proof/program synthesis, and build/config repair if the parser fixture works.
- Reject open-ended chat reasoning, reflection memory, hidden chain-of-thought reuse, and any variant whose labels come from
  model agreement rather than tests, checkers, constraints, or tool outcomes.
- Do not add another reasoning domain before `code_fix_parser_missing_guard_v0` produces verifier-backed labels and measured
  checking cost.
- The runtime package now mirrors this rule in `runtime/evals/preconditions/reasoning_minimal_validity_conditions/manifest.json`
  as `domain_triage`, so downstream agents do not need to infer the accepted/rejected domain set from prose.

Sharper CMLP-002 rule:

- Acceptable reasoning object: public verified dependency skeleton.
- Unacceptable reasoning object: hidden CoT, reflection, explanation memory, or an LLM's agreement with the old rationale.
- Four-gate litmus: materialized public skeleton, pre/post-checkable predicates, verifier-backed labels, and checking cost below rerunning a strong code/tool agent.
- First fixture size: one missing-guard or parser/schema bug with one targeted failing test and one nearby regression verifier.
- First fixture seed: `code_fix_parser_missing_guard_v0`, with planned anchor `parser.py::parse_record`, patch slot before split/parse, patch action "insert empty-input guard", and verifier obligations `test_rejects_empty_record` plus `test_parse_valid_record`.
- First success condition: learned or selected validity predicates reduce false accepts against retrieval/LLM-judge baselines while checking cost stays below rerunning a strong code agent.
- First negative condition: if the guard degenerates into a hand-written checklist or checker cost dominates, keep the result as `negative_result` and do not broaden the branch.
- First cost accounting fields: `predicate_static_check_ms`, `verifier_probe_ms`, `patch_apply_ms`,
  `total_admission_check_ms`, `rerun_code_agent_wall_ms`, and rerun tokens or a local-agent proxy. If
  `total_admission_check_ms` plus model/tool cost is not lower than rerun, stop the branch as cost-dominated.
- First executable pass protocol: after explicit confirmation to add fixture files, create only one parser fixture and require four
  artifact classes before status can change: fixture artifacts, predicate artifacts, verifier artifacts, and cost artifacts.
  The only acceptable first-pass outcomes are `evidence_available`, `not_learned_insufficient_data`, or `negative_result`.
  A negative result is useful and should stop expansion rather than trigger a broader reasoning-memory benchmark.
- Current pre-fixture audit: `runtime/evals/preconditions/reasoning_minimal_validity_conditions/manifest.json` now records
  `pre_fixture_readiness_audit.status=not_ready_to_execute_or_promote`. Materialization and predicate gates are `planned_only`;
  verifier and cost gates are `missing`. This is intentional: it prevents planned seed bindings, expected labels, or the manual
  hypothesis from being mistaken for evidence.

## Next Evidence Package

The next useful package should contain:

- one concrete Function id or trace id;
- one current-state fixture;
- at least five perturbation states;
- predicate table;
- replay outcomes;
- baseline admission decisions;
- a first learned `P*`;
- failure analysis for false accept and false reject.

Current protocol:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md` is secondary and should not displace the Contacts evidence package.

Registered experiment and schema:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`

Current label acquisition plan:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- hierarchy: `planned_expected_only < offline_oracle < execution`
- current promoted rows: `c1_launcher_wrong_start` is execution-backed; `c0_correct_create_flow`, `c2_contacts_home_reachable`, `c4_overlay_blocks_target`, `c6_args_missing_or_invalid`, `c7_terminal_verifier_missing`, and `c8_similar_non_contacts_page` are offline-oracle labeled in
  `runtime/evals/preconditions/contacts_minimal_live_preconditions/`.
- caveat: `c0_correct_create_flow` uses previous terminal-verified sidecar evidence where successful rows skipped the start-node precondition, so it proves bounded create-flow terminal success rather than direct admission.
- caveat: `runtime/evals/preconditions/contacts_minimal_live_preconditions/live_attempts/c0_direct_admission_probe_20260524.json`
  is diagnostic-only. It attempted direct admission from HOME/launcher with valid arguments, failed before device actions with
  `EXEC_START_NODE_PRECONDITION_FAIL` (`aw_node_8225087450 -> aw_node_22653dacf0`), and found no target contact. It does not
  provide the missing direct positive; it reframes the gap as recorded start-node / matcher instability.
- caveat: `c1_launcher_wrong_start` is now a fresh execution row, but it is still a negative/precondition-failure row: replay stopped before semantic actions with `EXEC_START_NODE_PRECONDITION_FAIL`, terminal verification found no target contact, and matcher diagnostics selected live launcher node `aw_node_8225087450` rather than required start node `aw_node_22653dacf0`.
- caveat: `c2_contacts_home_reachable` uses Contacts page-collection route/form evidence only; it proves a repair/reobserve admission state, not terminal contact creation.
  The live P0 route-repair probe from Contacts home failed before device actions because the stored UTG has no route-only
  bridge from the matched Contacts-home node to the Function's required launcher start node; keep that probe diagnostic-only.
- caveat: `c4_overlay_blocks_target` uses a Contacts XML bottom-sheet fixture only; it proves an overlay-present reobserve/repair state, not terminal contact creation or live replay failure.
- caveat: `c6_args_missing_or_invalid` is a schema-level admission label only; it does not make any claim about live GUI state.
- caveat: `c7_terminal_verifier_missing` is a verifier-boundary abstain label from control/API/CLI/test/docs evidence; skipped or disabled terminal verify yields `verifiable=false` and `terminal_page_reached=null`, so it must not be scored as success or failure.
- caveat: `c8_similar_non_contacts_page` uses a Dialer XML fixture with Contacts-like surface text; it proves a cross-app reject condition, not live replay failure.
- first deterministic baselines on the seven promoted rows are now partially scored: `always_replay` has 6/7 false accepts, while `always_replan`, `exact_start_node`, and the current `hand_written_guard` avoid false accepts but reject, repair, or abstain on the only success row under direct-allow scoring.
- explicit selector diagnostic result: `learned_contract.json.status=not_learned_insufficient_data`. The current seven rows are useful as a tradeoff demo, but not enough to learn a defensible direct-allow contract because the only success row has unknown `p_start_node_match`, while the Contacts-home and overlay rows are repair-only, and c7 is abstain-only rather than terminal success/failure. The fresh c1 execution row, c4 overlay row, c7 verifier boundary, and c0 direct-admission diagnostic do not change this because they are negative/repair, abstain, or diagnostic evidence.
- c3/c5 evidence-gap audit: existing Contacts XML artifacts do not support promotion of `c3_missing_create_entry` or
  `c5_missing_phone_field`. Pages without a visible create entry are page-local routeable states rather than stable
  missing-entry negatives, and no create/form-like Contacts page with Save / First name / Last name lacks Phone.
- next useful actions: follow the Positive Evidence Acquisition Protocol in `contacts_protocol.md`. The next positive row must be
  either a direct-admission positive with `label_source=execution`, no `__utg_skip_start_node_precondition__`,
  `p_start_node_match=true`, valid arguments, and terminal provider verification, or a route-repair positive where the first
  state has `p_start_node_match=false`, a route-only bridge reaches an admissible start/create-flow state before semantic form
  actions, and terminal verification confirms the inserted row. Then score PageVectorSet threshold and LLM-judge baselines if
  evidence can be obtained, and measure row-level admission/reuse/terminal-verifier cost versus replanning. If
  `total_reuse_path_ms` plus false-accept risk is not lower than the measured replan/rerun path, record `negative_result`
  rather than broadening the system. If `c3_missing_create_entry` or
  `c5_missing_phone_field` are still needed, collect a new curated/live fixture instead of reusing the current XML audit.

## Promotion Gate

This exploration can move toward a formal paper claim only after:

- at least one executable perturbation suite exists;
- success labels are grounded in terminal/business-effect evidence;
- learned preconditions beat similarity-only and LLM-judge baselines on false accept or false reject;
- checking cost is measured against replanning cost;
- the result is registered in `UTG_papers/nsdi/eval_matrix.md` or a future-paper equivalent.
