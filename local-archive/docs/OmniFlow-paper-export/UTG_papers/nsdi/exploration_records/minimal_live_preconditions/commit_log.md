# Minimal Live Preconditions Exploration Commit Log

Status: Active
Last Updated: 2026-05-24

## mlp-20260524-077 - 2026-05-24

Type: continuation audit / schema-aware parked-status / record-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Keep CMLP parked and do not add planned material. The current state still fails every Go condition: the reserved
Contacts-home candidate asset is absent, the parser fixture is absent, the PABench sub-gate row root is absent, CMLP-001 has
no terminal-positive row with complete cost fields, and CMLP-002 has no verifier-backed fixture evidence.

Reason:

The active objective is to maintain the separate exploration workspace and record trail while moving evidence forward only when
an explicit evidence path exists. A schema-aware audit confirmed the authoritative runtime files still encode the same state:
Contacts `manifest.json.status=partial`, `baseline_report.json.status=partial_scored`,
`learned_contract.json.status=not_learned_insufficient_data`, `learned_contract.cost_gate.status=unresolved`, and reasoning
`manifest.json.status=planned_skeleton_available` with `pre_fixture_readiness_audit.status=not_ready_to_execute_or_promote`.
Therefore another protocol, schema, skeleton family, or contrast would be planned-only expansion rather than progress.

Validation:

Structured Node audit parsed Contacts and reasoning manifests, outcomes, baselines, and learned contracts. Contacts outcomes
remain 9 rows: one `execution`, six `offline_oracle`, and two `planned_expected_only`; reasoning outcomes remain 9 rows, all
`planned_expected_only`. Absence checks confirmed the reserved Contacts candidate asset, parser fixture directory, and
reasoning subgate row root do not exist. No runtime artifacts, candidate assets, fixtures, baselines, learned contracts, parser
files, or NSDI claim/eval files were changed.

Next:

Continue from parked status unless the user selects one evidence path. The only evidence-moving actions remain: explicitly
materialize the reserved Contacts-home candidate asset, collect one terminal/cost positive CMLP-001 row, explicitly create the
single CMLP-002 parser fixture, produce verifier-backed skeleton-causality evidence, or ground PageVectorSet / LLM-judge
baseline scoring with frozen provenance and cost.

## mlp-20260524-076 - 2026-05-24

Type: continuation audit / parked-status / record-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Continue the CMLP line, but keep it parked. The current audit still finds no Go condition: the reserved Contacts-home
candidate asset is absent, CMLP-001 has no terminal-positive row with full cost fields, the CMLP-002 parser fixture is absent,
the PABench sub-gate row root is absent, and PageVectorSet / LLM-judge baseline scoring remains blocked.

Reason:

The active goal is to maintain the exploration thread and independent record trail, not to broaden planned-only material. The
runtime truth has not changed: Contacts remains `partial`, reasoning remains `planned_skeleton_available`,
`learned_contract.json` remains `not_learned_insufficient_data`, and the cost gate remains unresolved. Adding another protocol,
skeleton family, schema, or contrast would violate the planned-only freeze without producing evidence.

Validation:

Structured Node audit read the Contacts manifest, baseline report, learned contract, and reasoning manifest. It confirmed all
five Go conditions are false and that the reserved candidate asset, parser fixture directory, and reasoning subgate row root are
absent. No runtime artifacts, fixtures, outcomes, baselines, learned contracts, parser files, or NSDI claim/eval files were
changed.

Next:

Keep answering from parked status unless one evidence path is explicitly selected. Allowed evidence-moving actions remain:
materialize only the reserved Contacts-home candidate asset, collect a terminal/cost positive CMLP-001 row, create exactly the
single parser fixture for CMLP-002, produce verifier-backed skeleton-causality evidence, or ground PageVectorSet / LLM-judge
baseline scoring.

## mlp-20260524-075 - 2026-05-24

Type: Go-condition audit / parked-status / record-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Keep the CMLP line parked. The current Go-condition audit found no allowed promotion path: the Contacts candidate asset is not
materialized, CMLP-001 has no terminal execution positive row with cost fields, the CMLP-002 parser fixture does not exist, no
sub-gate row or verifier-backed skeleton-causality evidence exists, and PageVectorSet / LLM-judge grounded baseline scoring
remains blocked.

Reason:

The planned-only freeze says the next slice must either collect a confirmed evidence artifact, report the parked status, or
move project-control work to PABench. Current runtime manifests still report `contacts_status=partial` and
`reasoning_status=planned_skeleton_available`; `learned_contract.json.status=not_learned_insufficient_data` with
`cost_gate_status=unresolved`. Therefore adding more planned rows, schemas, skeleton families, or contrasts would violate the
freeze without improving evidence.

Validation:

Structured audit read the Contacts and reasoning runtime manifests, outcomes, baseline reports, and learned-contract files.
It confirmed all five Go conditions are false, the reserved Contacts candidate asset path is absent, the parser fixture
directory is absent, and the reasoning subgate row root is absent. No runtime artifacts, fixtures, baselines, learned
contracts, or NSDI claim files were changed.

Next:

If no explicit Go condition is satisfied, continue answering from parked status. The only evidence-moving next steps are:
explicitly materialize the reserved Contacts-home candidate asset, collect a terminal/cost positive CMLP-001 row, explicitly
create the single parser fixture for CMLP-002, produce verifier-backed skeleton-causality evidence, or ground PageVectorSet /
LLM-judge baseline scoring.

## mlp-20260524-074 - 2026-05-24

Type: CMLP / PABench equivalence audit / record-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

The current CMLP/reasoning proposal is not identical to PABench, but its reasoning branch is only distinct as a sub-gate when
the cached object remains causal. PABench asks which project-control action to take under `ProjectState + VerifierObservation`
(`delegate / verify / rollback / stop / escalate`). CMLP asks whether one retrieved trace, Function, or verifier-backed
skeleton is admissible now (`allow_reuse / reject / reobserve / repair_precondition / abstain`).

Reason:

Reading the PABench framing shows that Project Control owns ProjectState, VerifierSchema, write set, rollback, stopping, and
verified update authority. Therefore any row that chooses the next controller action, updates ProjectGraph, owns rollback/stop,
or optimizes transition regret is PABench. CMLP remains separate only when swapping the cached trace/Function/skeleton under the
same outer state changes the admission decision for anchor, slot, verifier-obligation, write-set, or cost reasons.

Validation:

Record-only audit. Read PABench README and paper draft, CMLP README and reasoning protocol, and confirmed the existing boundary
sections already encode this verdict. No runtime artifacts, parser fixtures, subgate rows, baselines, learned contracts, or
NSDI claim files were changed.

Next:

Do not present CMLP-002 as a separate Project Control paper. If the next artifact is about controller action selection, move it
to PABench. If the next artifact is about cached-object admission, require the fixed-state cached-skeleton replacement test and
cost gate before promotion.

## mlp-20260524-001 - 2026-05-24

Type: direction freeze / workspace initialization

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Freeze the next exploration direction as "Learning Minimal Live Preconditions for Reusable GUI Agent Traces" and keep the reasoning-trace version as a verifier-grounded secondary branch.

Reason:

The broader "self-evolving all-agent system" framing is too engineering-heavy. The smaller research question is sharper: given a retrieved historical trace, learn the minimal current-state predicates required for safe reuse. This sits directly between OmniFlow recall and control-aware execution without changing the current NSDI claim surface.

Validation:

Initial validation is documentation-level only. No code or experiment was run for the method itself in this entry. Follow-up validation must create a perturbation-labeled Contacts suite and compare against similarity-only, hand-written guard, and LLM-judge baselines.

Next:

Build the first Contacts predicate vocabulary and perturbation table, then produce a tiny executable evidence package with false accept / false reject metrics.

## mlp-20260524-002 - 2026-05-24

Type: protocol refinement

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Use `ContactsAddContact` and Function `aw_function_515be7bf7e` as the first concrete protocol for minimal live precondition learning.

Reason:

The current repo already has the right boundary evidence: three bounded terminal-verified successful replays, start-node precondition failures, planner-side Function selection and argument filling, and explicit caveats around `__utg_skip_start_node_precondition__=1`. This makes Contacts a better seed than OpenApp because it includes parameters, multi-action form entry, and terminal/business-effect verification.

Validation:

This entry is still protocol-level. It reuses existing evidence paths but does not run the perturbation suite yet. The protocol now specifies candidate predicates, perturbations, labels, baselines, metrics, learners, and a promotion gate.

Next:

Create `runtime/evals/preconditions/contacts_minimal_live_preconditions/` with `cases.jsonl`, `predicate_matrix.json`, `outcomes.json`, `baseline_report.json`, `learned_contract.json`, and `summary.md`.

## mlp-20260524-003 - 2026-05-24

Type: evidence schema / experiment registration

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/README.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Register the first experiment as `CMLP-001 - Contacts Minimal Live Preconditions` and freeze the runtime evidence schema before collecting new data.

Reason:

The previous protocol named the research object, but a future agent still needed to know exactly which files to write, which fields are mandatory, and how to avoid mistaking planned labels for evidence. Freezing the schema first keeps the work falsifiable and prevents paper-claim drift.

Validation:

Schema-level only. No device replay, learner, or new runtime evidence was run in this entry. The register and schema explicitly mark the experiment as `planned`.

Next:

Create the runtime output directory and write skeleton `manifest.json`, `cases.jsonl`, `predicate_matrix.json`, `outcomes.json`, `baseline_report.json`, `learned_contract.json`, and `summary.md` using planned labels only.

## mlp-20260524-004 - 2026-05-24

Type: reasoning branch boundary

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Keep the reasoning-trace variant, but define it as learning minimal validity conditions for public verifier-grounded reasoning skeletons, not as storing or replaying hidden chain-of-thought.

Reason:

The user asked whether the direction can be "reasoning" instead of GUI. The answer is yes only in domains where the reasoning object has external evidence: code, SQL, proof, or similar verifier-rich tasks. A valid reusable reasoning trace must expose claims, premises, tool results, actions, verifier obligations, and checker outcomes. Otherwise the work collapses into prompt reuse or LLM self-judgment and is not defensible as research.

Validation:

Documentation-level only. This entry tightens scope and kill conditions; it does not add a reasoning benchmark or runtime evidence.

Next:

Do not start with open-ended chat reasoning. If this branch is pursued, create a code-agent protocol with one bug-fix skeleton, predicate perturbations over repo state, and labels from tests/checkers.

## mlp-20260524-005 - 2026-05-24

Type: secondary protocol registration

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Register the reasoning branch as `CMLP-002 - Reasoning Trace Minimal Validity Conditions`, but keep it `planned_secondary` behind the Contacts GUI protocol.

Reason:

The reasoning variant is promising only if it becomes an executable admission protocol. A code-agent bug-fix skeleton gives the smallest defensible unit: failure signature, source anchor, patch slot, patch action, verifier obligations, and verifier results. The protocol makes the branch falsifiable with perturbation cases, baselines, false accept / false reject metrics, and a no-hidden-CoT gate.

Validation:

Protocol-level only. No code-agent fixture, verifier run, learned contract, or runtime evidence package exists yet.

Next:

If the branch is pursued, create `runtime/evals/preconditions/reasoning_minimal_validity_conditions/` with planned-only labels first, then add a tiny test/checker-backed code fixture.

## mlp-20260524-006 - 2026-05-24

Type: planned runtime skeleton

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/cases.jsonl`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/predicate_matrix.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/baseline_report.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`

Decision:

Create the `CMLP-001` runtime evidence skeleton and mark it `planned_skeleton_available`, not `evidence_available`.

Reason:

The protocol needed an inspectable output package before any device replay or learner is run. The skeleton fixes case ids, predicate rows, expected labels, planned baselines, and the manual contract hypothesis while preventing planned labels from being mistaken for evidence.

Validation:

JSON files parse, required files exist, there are 9 cases and 9 outcome rows, every outcome row uses `label_source=planned_expected_only`, no execution row is marked as run, and manifest status is `planned_skeleton_available`.

Next:

Replace planned-only labels with execution-labeled or offline-oracle-labeled outcomes, then score baselines and learn or reject a minimal contract.

## mlp-20260524-007 - 2026-05-24

Type: secondary planned runtime skeleton

Files:

- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/manifest.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/cases.jsonl`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/predicate_matrix.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/outcomes.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/baseline_report.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/learned_contract.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Create the `CMLP-002` reasoning runtime skeleton and mark it `planned_skeleton_available`, not `evidence_available`.

Reason:

The reasoning branch needed the same evidence hygiene as Contacts: case ids, predicate rows, expected labels, planned baselines, and a manual contract hypothesis must exist before any verifier-backed fixture is built. This keeps the branch falsifiable while preserving the hard boundary that hidden chain-of-thought is not a reusable object.

Validation:

JSON files parse, required files exist, there are 9 cases and 9 outcome rows, every outcome row uses `label_source=planned_expected_only`, no execution row is marked as run, and manifest status is `planned_skeleton_available`.

Next:

Add one tiny code-agent bug-fix fixture with a real verifier command, then replace planned-only labels with verifier-backed outcomes before scoring baselines or learning a contract.

## mlp-20260524-008 - 2026-05-24

Type: label acquisition protocol

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Freeze the `CMLP-001` label acquisition and promotion path before running any learner or promoting planned cases as evidence.

Reason:

The Contacts skeleton already fixed cases, predicates, expected labels, and baselines, but it did not specify how a row graduates from planned expectation to usable evidence. Without a promotion protocol, a future run could accidentally score baselines on planned labels or treat low-level action dispatch as success. The new rule hierarchy is `planned_expected_only < offline_oracle < execution`.

Validation:

Documentation-level only. This entry defines the evidence fields, per-case acquisition method, scoring order, and negative-result handling. It does not run device replay, offline oracle labeling, baseline scoring, or a learner.

Next:

Promote `c0_correct_create_flow` with terminal/business-effect execution evidence and `c1_launcher_wrong_start` with execution or offline-oracle evidence for `EXEC_START_NODE_PRECONDITION_FAIL`, then update `outcomes.json` and `predicate_matrix.json` with evidence references.

## mlp-20260524-009 - 2026-05-24

Type: partial offline-oracle label promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/cases.jsonl`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/predicate_matrix.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Promote `CMLP-001` from planned skeleton to partial evidence by labeling `c0_correct_create_flow` and
`c1_launcher_wrong_start` with `label_source=offline_oracle` from existing Contacts sidecars.

Reason:

The existing harder Contacts sidecars already contain two useful admission signals. The success sidecar proves
bounded multi-action create-contact replay reaches Contacts provider terminal evidence with 0 online model tokens.
The entry-gap probe proves the VLM planner can select and parameterize `aw_function_515be7bf7e`, but launcher
admission fails before semantic actions because the start-node precondition is not satisfied and no route-only
bridge exists.

Validation:

This entry is partial evidence only. `c0_correct_create_flow` keeps `p_start_node_match` unknown because the
successful sidecar rows used `__utg_skip_start_node_precondition__=1`; it must not be read as direct start-node
admission evidence. `c1_launcher_wrong_start` records the exact sidecar error string rather than inventing a new
execution result. No fresh live replay, baseline scoring, learner, or cost comparison was run in this entry.

Next:

Label at least three more negative or repair rows, starting with `c2_contacts_home_reachable`,
`c3_missing_create_entry`, `c4_overlay_blocks_target`, and `c5_missing_phone_field`, then score baselines before
learning or rejecting a minimal contract.

## mlp-20260524-010 - 2026-05-24

Type: schema-level offline-oracle label promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/cases.jsonl`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/predicate_matrix.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Promote `c6_args_missing_or_invalid` to `label_source=offline_oracle` using the Function argument schema and
`Function.validate_arguments` as a deterministic pre-GUI oracle.

Reason:

The seed Function `aw_function_515be7bf7e` declares `first_name`, `last_name`, and `phone` as required JSON Schema
arguments, and the runtime `Function.validate_arguments(...)` path rejects a call missing a required property before
action rendering or device execution. This makes `c6` a clean admission label that does not require a live GUI state.

Validation:

This is a schema-level label only. It does not prove anything about Contacts UI reachability, required field
binding, overlays, account state, or action transfer. Baseline metrics remain null because three oracle labels are
still insufficient for false accept / false reject scoring.

Next:

Collect at least two more GUI-state negative or repair labels, preferably from `c2_contacts_home_reachable`,
`c3_missing_create_entry`, `c4_overlay_blocks_target`, or `c5_missing_phone_field`, before scoring baselines.

## mlp-20260524-011 - 2026-05-24

Type: reasoning boundary refinement

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Answer the reasoning variant as artifact-state validity learning: the reusable object is a public verifier-grounded
skeleton, not hidden chain-of-thought or generic prompt memory.

Reason:

The user asked whether the same idea can be "reasoning". It can, but only if the problem remains reuse admission:
which current repo/test/tool predicates make a previous reasoning skeleton safe to reuse. This preserves the same
research core as GUI preconditions while avoiding a broad code-agent memory system.

Validation:

Documentation-level only. No code-agent fixture, verifier run, learner, baseline scoring, or checking-cost comparison
was executed in this entry. The branch remains `planned_skeleton_available`.

Next:

If CMLP-002 is pulled forward, create one tiny failing-test code fixture with public skeleton fields, verifier-backed
labels, baseline decisions, and a cost comparison against rerunning a code agent.

## mlp-20260524-012 - 2026-05-24

Type: route/form offline-oracle label promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/cases.jsonl`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/predicate_matrix.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Promote `c2_contacts_home_reachable` to `label_source=offline_oracle` as a `reobserve_or_repair` row using the
Contacts page collection route and XML fixtures.

Reason:

The page collection proves a bounded route/form admission condition: Contacts root is in
`com.google.android.contacts`, `raw/root.xml` has an enabled clickable `Create contact` floating action button,
`index.json#routes/create_contact` reaches `page_013_create_contact_08e213ee` via tap `[943,2011]`, and
`raw/create_contact.xml` exposes `Save`, `First name`, `Last name`, and `Phone`. This distinguishes strict
start-node matching from a repairable Contacts-home state without claiming terminal contact creation.

Validation:

This is offline route/form evidence only. No fresh live replay, Contacts provider terminal verification, baseline
scoring, learner, or checking-cost comparison was run. `c2` is not a `success` row.

Next:

Collect at least one more negative or repair row, preferably `c3_missing_create_entry`, `c4_overlay_blocks_target`,
`c5_missing_phone_field`, or `c8_similar_non_contacts_page`, before baseline scoring.

## mlp-20260524-013 - 2026-05-24

Type: cross-app offline-oracle label promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/cases.jsonl`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/predicate_matrix.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Promote `c8_similar_non_contacts_page` to `label_source=offline_oracle` as a cross-app `reject` row.

Reason:

The manual XML fixture `runtime/manual/usability_after_param_contact_baseline.xml` is a Dialer page with Contacts-like
surface signals: `Create new contact`, `Search contacts & places`, and a `Contacts` tab. However, the app package is
`com.google.android.dialer`, not `com.google.android.contacts`, and the current observation lacks the Contacts
create-contact form anchors `First name`, `Last name`, `Phone`, and `Save`. This is the first explicit false-positive
control for similarity-only reuse admission.

Validation:

This is offline XML evidence only. No fresh live replay, terminal provider verification, baseline scoring, learner,
or checking-cost comparison was run. The row proves a policy reject condition for cross-app observations; it does not
prove that a live replay would fail after dispatch.

Next:

Score the first deterministic baselines on the five oracle rows, or collect one more GUI-state negative/repair row
from `c3_missing_create_entry`, `c4_overlay_blocks_target`, or `c5_missing_phone_field` before learner selection.

## mlp-20260524-014 - 2026-05-24

Type: partial deterministic baseline scoring

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/baseline_report.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Score the first deterministic baselines on the five offline-oracle CMLP-001 rows while leaving PageVectorSet threshold,
LLM judge, and learned contract unscored.

Reason:

After `c8_similar_non_contacts_page` the minimum label-count gate is met: one positive row and four negative/repair rows.
This is enough to expose the first reuse-admission tradeoff without claiming a learned result. `always_replay` false-accepts
the wrong-start, repair-only, invalid-argument, and cross-app reject rows. `always_replan`, `exact_start_node`, and the
current hand-written guard avoid false accepts, but reject or repair the only success row under direct-allow scoring.

Validation:

This is partial scoring only. It uses offline-oracle labels, not fresh execution. No PageVectorSet similarity threshold,
LLM applicability judge, learner, or checking-cost comparison was run. `learned_contract.json` remains a planned manual
hypothesis.

Next:

Score PageVectorSet / LLM baselines if their evidence can be collected, run an explicit selector for a minimal contract,
and measure predicate checking cost against replanning before considering `evidence_available`.

## mlp-20260524-015 - 2026-05-24

Type: reasoning branch score / boundary tightening

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Score the reasoning variant as an 8/10 secondary branch and keep it behind the GUI protocol.

Reason:

The branch is researchable only when "reasoning" is converted into artifact-state validity learning over public verifier-grounded
skeletons. Code, SQL, proof, build, schema, and config domains can expose predicates and verifier-backed labels; open-ended chat
reasoning cannot. The main risk is cost: a strong code agent plus tests is already a hard baseline, so the branch must show that
predicate checking plus skeleton reuse is cheaper than rerunning the agent.

Validation:

Documentation-level only. No code-agent fixture, verifier run, baseline scoring, learner, or checking-cost comparison was run.
`CMLP-002` remains `planned_skeleton_available`.

Next:

If reasoning is prioritized, build exactly one tiny failing-test fixture with public skeleton fields, perturbation rows, patch
application, verifier output, baseline scoring, and checking-cost comparison. If checker cost exceeds rerunning a code agent, record
`negative_result` rather than expanding the branch.

## mlp-20260524-016 - 2026-05-24

Type: selector diagnostic / insufficient-data result

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Mark CMLP-001's learned contract as `not_learned_insufficient_data` after an explicit exhaustive selector diagnostic on
the five offline-oracle rows.

Reason:

The current rows expose the reuse-admission tradeoff but cannot support a learned direct-allow contract. The only success row,
`c0_correct_create_flow`, has unknown `p_start_node_match` because the successful sidecar skipped the start-node precondition.
The Contacts-home row `c2_contacts_home_reachable` has the same app/form/argument/verifier predicates as the positive row, but
it is only `reobserve_or_repair`, not terminal success. Candidate subsets that appear perfect either exploit unknown predicate
values or require `p_start_node_match` without positive-row evidence for that predicate.

Validation:

Ran an inline Node exhaustive selector over `predicate_matrix.json` and `outcomes.json`. With unknown values treated as false,
192 zero-false-accept / zero-false-reject subsets exist, but they are invalid because they separate rows by missing evidence.
With unknown values ignored, 96 apparent perfect subsets exist, but many depend on `p_start_node_match` while the only success
row has that predicate unknown. No fresh device replay, PageVectorSet scoring, LLM judge, or checking-cost measurement was run.

Next:

Collect a terminal-verified direct-admission positive row with `p_start_node_match=true`, or a terminal-verified route-repair
positive row with explicit route predicate semantics. Then score PageVectorSet / LLM baselines and measure checking cost versus
replanning before trying another selector.

## mlp-20260524-017 - 2026-05-24

Type: reasoning boundary / first-fixture rule

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Tighten the answer to "what if the object is reasoning" into the shortest acceptable CMLP-002 claim: private reasoning must
be erased, the reusable object must be a public verified dependency skeleton, and reuse admission must be learned from
artifact-state validity predicates rather than retrieval score, prompt memory, or LLM agreement with the old rationale.

Reason:

Without this boundary the branch collapses into explanation memory or hidden-chain-of-thought reuse, which has no externally
checkable state, no pre-action admission oracle, and no stable labels beyond model self-judgment. The smallest defensible
experiment is therefore a tiny missing-guard or parser/schema bug fixture with public skeleton JSON, repo-state perturbations,
real patch/action execution, verifier output, baseline decisions, and checking cost compared against rerunning a strong code agent.

Validation:

Documentation-level only. No code-agent fixture, verifier run, baseline scoring, learner, or checking-cost comparison was run.
`CMLP-002` remains `planned_skeleton_available`, not evidence.

Next:

If CMLP-002 is prioritized, build one tiny failing-test fixture first. If the learned/selected guard becomes a long hand-written
checklist or checker cost exceeds rerunning a strong code agent, record `negative_result` and do not broaden the branch.

## mlp-20260524-018 - 2026-05-24

Type: reasoning verdict / scope hardening

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Answer the current "what if it is reasoning" question by hardening the branch boundary: generic reasoning is not the research
object. CMLP-002 remains viable only for verifier-grounded artifact domains where private reasoning can be erased into a public
dependency skeleton and reuse admission is learned from current artifact-state predicates.

Reason:

Open-ended reasoning reuse would be dominated by model upgrades and lacks externally checkable predicates or labels. Code, SQL,
proof, build/config, and schema repair can expose failure signatures, anchors, patch/query/proof slots, tool outputs, verifier
obligations, and verifier results. That makes the problem falsifiable and keeps it aligned with the GUI reuse-admission method.

Validation:

Documentation-level only. No code-agent fixture, verifier run, baseline scoring, learner, or checking-cost comparison was run.
`CMLP-002` remains `planned_skeleton_available`, not evidence.

Next:

If this branch is prioritized, build the first tiny code-agent fixture and measure whether predicate checking plus skeleton reuse
beats rerunning a strong code agent under verifier-backed labels. Otherwise keep CMLP-002 secondary behind the GUI evidence path.

## mlp-20260524-019 - 2026-05-24

Type: live execution evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/cases.jsonl`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/predicate_matrix.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/baseline_report.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Promote `c1_launcher_wrong_start` from offline-oracle evidence to execution-backed evidence using the fresh replay-function
artifact and matcher diagnostic.

Reason:

The fresh `emulator-5554` replay command attempted `aw_function_515be7bf7e` from launcher with valid contact arguments and
terminal verification enabled. It failed before semantic actions with `EXEC_START_NODE_PRECONDITION_FAIL`, recorded
`device_actions=[]`, and terminal verification reported `contacts_record_not_found`. The matcher diagnostic shows the live
launcher observation selected `aw_node_8225087450`, not required start node `aw_node_22653dacf0`. This strengthens the c1
negative row without turning it into a positive or direct-admission example.

Validation:

This entry changes evidence strength, not the label. Deterministic baseline counts remain 5 promoted rows with `always_replay`
at 4/5 false accepts. The selector result remains `not_learned_insufficient_data` because the missing evidence is still a
positive row with `p_start_node_match=true` or an explicit terminal-verified route-repair positive.

Next:

Collect a terminal-verified direct-admission positive row or a route-repair positive row with explicit route predicate
semantics. Then score PageVectorSet threshold and LLM-judge baselines and measure checking cost against replanning.

## mlp-20260524-020 - 2026-05-24

Type: offline oracle evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/cases.jsonl`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/predicate_matrix.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/baseline_report.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Promote `c4_overlay_blocks_target` from planned-only to offline-oracle reobserve/repair evidence using the Contacts
`chip_company.xml` bottom-sheet fixture.

Reason:

The fixture is a Contacts package observation containing a full-screen `touch_outside` scrim and bottom
`design_bottom_sheet`, with visible `Company` and `No contacts in this account have company info` sheet text. This supports
`p_no_blocking_overlay=false` and a `reobserve_or_repair` label: normal Function action dispatch should not proceed until the
overlay is dismissed or a fresh observation is taken. It does not prove terminal contact creation, terminal failure, or live
replay behavior.

Validation:

The deterministic baseline denominator now has six promoted rows. `always_replay` has 5/6 false accepts; `always_replan`,
`exact_start_node`, and the current `hand_written_guard` still avoid false accepts but reject or repair the only success row
under direct-allow scoring. The exhaustive selector diagnostic was rerun: unknown-as-false still yields 192 invalid perfect
subsets, and unknown-ignored yields 48 apparent perfect subsets that remain not promotable because the only positive row lacks
direct `p_start_node_match` evidence.

Next:

Collect a terminal-verified direct-admission positive row with `p_start_node_match=true`, or a terminal-verified route-repair
positive row with explicit route predicate semantics. Then score PageVectorSet threshold and LLM-judge baselines and measure
checking cost against replanning.

## mlp-20260524-021 - 2026-05-24

Type: reasoning litmus test

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Add a four-gate litmus test for the CMLP-002 reasoning branch: public skeleton materialization, pre/post-checkable predicates,
verifier-backed labels, and checking cost below rerunning a strong code or tool agent.

Reason:

The user asked what happens if the object is reasoning. The defensible answer needs an operational rejection test, not another
name. A reasoning trace is researchable here only after private reasoning is erased into artifact-bound obligations that can be
checked before and after reuse. Otherwise the branch collapses into prompt memory, retrieval, or LLM self-judgment.

Validation:

Documentation-level only. No code-agent fixture, verifier run, learner, baseline scorer, or checking-cost comparison was run.
`CMLP-002` remains `planned_skeleton_available`, not evidence.

Next:

If CMLP-002 is prioritized, build one tiny verifier-backed fixture and apply the four gates before any baseline or learner claim.

## mlp-20260524-022 - 2026-05-24

Type: planned first fixture specification

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/manifest.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Fix the first planned CMLP-002 seed as `code_fix_parser_missing_guard_v0`: a tiny Python parser bug where
`parser.py::parse_record` should reject empty input, the patch slot is before the first split/parse operation, and verifier
obligations are one targeted failing pytest plus one valid-record regression pytest.

Reason:

The previous reasoning branch boundary was correct but still too abstract. A concrete planned seed lets later work implement
exactly one fixture instead of drifting into broad code-agent memory, hidden reasoning reuse, or a benchmark-first protocol.
The seed is deliberately small so each validity predicate can be perturbed independently.

Validation:

Documentation/runtime-package specification only. No fixture files, patch execution, verifier stdout, baseline scoring,
learner, or checking-cost comparison was created. `CMLP-002` remains `planned_skeleton_available`, not evidence.

Next:

Create the actual tiny fixture files and record verifier-backed labels. Keep `predicate_check_cost` and `rerun_code_agent_cost`
as required fields before attempting any learned-contract claim.

## mlp-20260524-023 - 2026-05-24

Type: planned package tightening

Files:

- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/cases.jsonl`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/predicate_matrix.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Bind every planned CMLP-002 case and predicate row to the concrete parser seed `code_fix_parser_missing_guard_v0` rather than
leaving the package as generic reasoning perturbations. Each row now references the planned fixture directory
`tests/preconditions/reasoning/fixtures/parser_missing_guard/`, keeps `planned_only_not_evidence`, and maps the perturbation to
`parser.py::parse_record`, `test_rejects_empty_record`, `test_parse_valid_record`, or the parser fixture's local edit state.

Reason:

The reasoning branch is only worth keeping if it becomes a concrete artifact-state validity experiment. Binding the planned
rows to one parser missing-guard seed prevents drift back into hidden-CoT reuse, broad reflection memory, or LLM applicability
judgment. It also makes the next implementation step unambiguous: create exactly one tiny fixture, run the verifier obligations,
then replace planned labels with checker-backed labels.

Validation:

Documentation/runtime-package specification only. This update creates no fixture files, no patch execution, no verifier stdout,
no baseline scoring, no learner, and no checking-cost comparison. `CMLP-002` remains `planned_skeleton_available`, not evidence.

Next:

Implement the tiny parser fixture and record verifier-backed labels, or keep the branch parked behind CMLP-001 if the GUI
evidence path remains higher priority.

## mlp-20260524-024 - 2026-05-24

Type: reasoning answer / no-promotion decision

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Re-answer the user's "what if it is reasoning" question without changing the experiment status: CMLP-002 remains a secondary
planned branch, and "reasoning" is acceptable only after it is reduced to artifact-state validity learning over a public
verifier-grounded skeleton.

Reason:

The current documents already contain the correct boundary. The important research distinction is negative: do not cache or
reuse hidden reasoning text, reflection, explanation memory, or an LLM's agreement with the old rationale. A model upgrade can
improve raw reasoning, but it does not remove the artifact-validity problem for code, SQL, proof, build/config, or schema tasks:
the current artifact state can violate anchors, verifier obligations, tool versions, patch slots, or failure signatures.

Validation:

Reasoning runtime JSON and JSONL still parse, `minimal-precondition-research` skill validation passes, targeted keyword search
finds the expected boundary/status records, and targeted `git diff --check` is clean. No fixture files, patch execution,
verifier stdout, baseline scoring, learner, or checking-cost comparison were created. This is a no-promotion decision that keeps
`CMLP-002` at `planned_skeleton_available`.

Next:

If this branch is prioritized, implement exactly one tiny parser fixture and replace planned labels with checker-backed labels.
Otherwise keep the GUI CMLP-001 evidence path ahead of reasoning.

## mlp-20260524-025 - 2026-05-24

Type: direct-admission diagnostic / no label promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Register `runtime/evals/preconditions/contacts_minimal_live_preconditions/live_attempts/c0_direct_admission_probe_20260524.json`
as diagnostic evidence only. Do not add a new promoted case, do not change the six-row deterministic baseline denominator,
and do not change `learned_contract.json.status=not_learned_insufficient_data`.

Reason:

The probe attempted `aw_function_515be7bf7e` from HOME/launcher with valid arguments and terminal verification enabled. It
failed before any device action with `EXEC_START_NODE_PRECONDITION_FAIL` (`aw_node_8225087450 -> aw_node_22653dacf0`) and
terminal verification found no target contact. Inspecting the seed store shows `aw_node_22653dacf0` is the recorded launcher
entry node whose first sequence action is `open_app`, while `aw_node_8225087450` is a Contacts form node in the same Function.
This reframes the missing positive as a recorded start-node / matcher instability gap rather than evidence that launcher direct
admission is valid.

Validation:

Manifest JSON and all CMLP-001 JSON files parse; `cases.jsonl` parses; structured assertions confirm the direct probe is listed
only as `manifest.input_evidence`, not as an outcome case or scored baseline case. Targeted keyword search finds the diagnostic
boundary, and targeted `git diff --check` is clean.

Next:

Collect a terminal-verified direct-admission positive row with `p_start_node_match=true`, or a terminal-verified route-repair
positive row with explicit route predicate semantics. Then score PageVectorSet threshold / LLM judge and measure checking cost
against replanning.

## mlp-20260524-026 - 2026-05-24

Type: reasoning execution protocol / acceptance tightening

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/manifest.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Tighten CMLP-002's next step into a first executable pass protocol without creating fixture files or promoting evidence. The
branch remains `planned_skeleton_available`, but the next pass now has fixed artifact classes and acceptable outcomes.

Reason:

The reasoning branch can easily drift into a broad reasoning-memory benchmark. The tightened protocol makes that invalid:
after explicit confirmation to create fixture files, the first pass must use exactly one parser fixture for
`code_fix_parser_missing_guard_v0` and produce fixture artifacts, predicate artifacts, verifier artifacts, and cost artifacts.
The only acceptable outcomes are `evidence_available`, `not_learned_insufficient_data`, or `negative_result`. A negative result
is useful because it tests whether verifier-grounded skeleton reuse is cost-positive against rerunning a strong code agent.

Validation:

Pending targeted validation after log updates: reasoning runtime JSON/JSONL parse, skill validation, targeted keyword search, and
targeted `git diff --check`. No fixture files, verifier output, baseline scoring, learner, or cost comparison were created in
this entry.

Next:

If CMLP-002 is prioritized, ask for explicit confirmation to create the parser fixture tree, then run exactly one executable
fixture pass and fill the required verifier and cost fields. Otherwise keep CMLP-001's direct-admission positive as the main
evidence priority.

## mlp-20260524-027 - 2026-05-24

Type: cost/risk accounting gate

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Add an explicit cost/risk accounting gate for both GUI and reasoning CMLP work. A minimal precondition contract is useful only
if its check/reuse path has lower expected cost and risk than rerunning or replanning; otherwise the correct outcome is
`negative_result`.

Reason:

The user's objection is correct: if detecting applicability costs more than rereading/reobserving/rerunning, the method is not
worth doing. The updated protocol therefore treats the formulas as accounting gates, not as claimed mathematical novelty. For
reasoning, this is especially important because a strong code agent plus tests is already a strong baseline.

Validation:

Pending targeted validation after log updates: JSON/JSONL parse for CMLP runtime packages, minimal-precondition skill validation,
targeted keyword search for the cost/risk gate, and targeted `git diff --check`. No parser fixture, verifier output, baseline
scoring, learner, or evidence promotion is introduced by this entry.

Next:

For CMLP-001, collect the missing terminal-verified direct-admission or route-repair positive and then measure checking cost
against replanning. For CMLP-002, only proceed to the parser fixture if its first pass reports verifier-backed labels and
`C_check + C_patch + C_verify < C_rerun_code_agent`; otherwise stop as a negative result.

## mlp-20260524-028 - 2026-05-24

Type: reasoning answer capsule

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Record the compact answer to "what if it is reasoning?" as a reusable CMLP-002 rule: yes, but only after private reasoning is
erased into a public verifier-grounded skeleton. The method is then artifact-state validity learning, not reasoning-text reuse.

Reason:

This prevents the branch from drifting into hidden-CoT cache, reflection memory, explanation retrieval, or LLM agreement. The
only reusable object is an artifact-bound dependency record with failure signatures, anchors, patch/query/proof slots, actions,
verifier obligations, and verifier results. It still must pass the cost gate against rerunning a strong code/tool agent.

Validation:

Pending targeted validation after log updates: reasoning runtime JSON/JSONL parse, minimal-precondition skill validation,
targeted keyword search for the answer capsule, and targeted `git diff --check`. No parser fixture, verifier output, baseline
scoring, learner, or evidence promotion is introduced by this entry.

Next:

Keep CMLP-002 at `planned_skeleton_available` until a single parser fixture exists with verifier-backed labels and measured
cost. Continue prioritizing CMLP-001's missing direct-admission or route-repair positive unless the user explicitly pulls the
reasoning branch forward.

## mlp-20260524-029 - 2026-05-24

Type: verifier-boundary offline-oracle promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/cases.jsonl`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/predicate_matrix.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/baseline_report.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Promote `c7_terminal_verifier_missing` from planned-only to an offline-oracle `abstain` row using the current
terminal-verifier contract in code, tests, CLI/API behavior, and README docs.

Reason:

The control/API/CLI contract is explicit: skipped, disabled, or unavailable terminal verification yields
`effect_verifier.verifiable=false` and `terminal_state.terminal_page_reached=null`. Tests assert the same behavior for
`skip_terminal_verify`, `terminal_verify.enabled=false`, disabled settings, and missing terminal matcher cases. Therefore
c7 can be promoted as a verifier-boundary row, but it must remain `abstain` rather than success or failure.

Validation:

Pending targeted validation after log updates: CMLP JSON/JSONL parse, structured promoted-row and baseline assertions,
`minimal-precondition-research` skill validation, targeted keyword search, targeted `git diff --check`, and trailing-whitespace
checks. This entry does not add live GUI execution, terminal success evidence, PageVectorSet scoring, LLM-judge scoring,
or checking-cost measurement.

Next:

Collect the missing terminal-verified direct-admission positive with `p_start_node_match=true`, or a terminal-verified
route-repair positive with explicit route predicate semantics. Then score PageVectorSet / LLM baselines and measure
checking cost against replanning before claiming a learned contract.

## mlp-20260524-030 - 2026-05-24

Type: reasoning domain triage

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Tighten the answer to "what if it is reasoning?" into a domain triage rule. CMLP-002 accepts only verifier-grounded artifact
domains: code repair as the first executable target, with SQL repair, proof/program synthesis, and build/config repair parked
behind it. Open-ended chat reasoning, reflection/explanation memory, hidden chain-of-thought reuse, and LLM-agreement labels
are rejected.

Reason:

The research object is not reasoning text. It is reuse admission for a public skeleton whose dependencies and obligations are
visible in the current artifact state. This keeps the branch testable and prevents it from becoming prompt memory or a
model-version-dependent reasoning benchmark.

Validation:

Pending targeted validation after log updates: CMLP JSON/JSONL parse, `minimal-precondition-research` skill validation,
targeted domain-triage grep, and targeted `git diff --check`. No parser fixture, verifier output, baseline scoring, learner,
or evidence promotion is introduced by this entry.

Next:

If the reasoning branch is pulled forward, create only the planned parser fixture after explicit confirmation and measure
checker plus verifier cost against rerunning a strong code agent. Otherwise keep CMLP-001's direct-admission positive as the
main evidence priority.

## mlp-20260524-031 - 2026-05-24

Type: machine-readable reasoning triage

Files:

- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/manifest.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`

Decision:

Mirror the CMLP-002 domain triage in the runtime manifest as `domain_triage`. The machine-readable contract says code repair is
the only current executable target; SQL repair, proof/program synthesis, and build/config repair are parked; open-ended chat
reasoning, reflection/explanation memory, hidden chain-of-thought reuse, and LLM-agreement labels are rejected.

Reason:

The previous rule was prose-only. A future agent or script that reads only the runtime package could still treat generic
reasoning as admissible. The manifest now carries the same accepted/parked/rejected split and keeps the parser seed marked
`planned_only_not_evidence`.

Validation:

Pending targeted validation after log updates: reasoning runtime JSON/JSONL parse, structured assertions for `domain_triage`,
`minimal-precondition-research` skill validation, targeted grep, and targeted `git diff --check`. No fixture, verifier,
baseline, learner, cost comparison, or evidence promotion is introduced.

Next:

Keep CMLP-002 parked unless explicitly pulled forward. If it is pulled forward, create only the parser fixture after
confirmation and measure verifier/checking cost before expanding any other reasoning domain.

## mlp-20260524-032 - 2026-05-24

Type: reasoning user-answer reaffirmation

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the user's reasoning question by reaffirming the existing CMLP-002 boundary: the branch is valid only as artifact-state
validity learning over a public verifier-grounded skeleton. It is not reasoning-text reuse, hidden chain-of-thought reuse,
reflection memory, or an LLM-judge applicability layer.

Reason:

Reasoning becomes researchable only when the reusable object has externally checkable dependencies: failure signatures, anchors,
patch/query/proof slots, actions, verifier obligations, and verifier results. The current live state is then the artifact state,
such as files, AST/schema, tests, tool versions, checker availability, and local diffs. A stronger model upgrade can improve raw
reasoning, but it does not remove the validity question for reusing a previously verified skeleton in a changed artifact state.

Validation:

Pending targeted validation after log updates: reasoning runtime JSON/JSONL parse, `minimal-precondition-research` skill
validation, targeted reasoning-boundary grep, and targeted `git diff --check`. No parser fixture, patch execution, verifier
stdout, baseline scoring, learner, checking-cost comparison, or evidence promotion is introduced.

Next:

If CMLP-002 is prioritized, create exactly one tiny parser fixture after explicit confirmation and measure whether
`C_check + C_patch + C_verify` is below rerunning a strong code/tool agent. Otherwise keep the reasoning branch secondary behind
CMLP-001's missing direct-admission or route-repair positive evidence.

## mlp-20260524-033 - 2026-05-24

Type: contacts c3/c5 evidence-gap audit

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Record the current XML audit as an evidence gap rather than a label promotion for `c3_missing_create_entry` or
`c5_missing_phone_field`.

Reason:

The targeted clean Contacts collection and broader XML roots contain many Contacts pages without a visible create-contact entry,
but those pages are routeable or page-local states such as detail, search, tab, menu, account, edit, or overlay views. They do
not prove a stable non-transient missing-entry condition. The broader audit also found no create/form-like Contacts page with
Save / First name / Last name that lacks Phone, so there is no promotable `c5_missing_phone_field` fixture in current artifacts.

Validation:

Pending targeted validation after log updates: Contacts runtime JSON/JSONL parse, structured assertions that c3/c5 remain
unlabeled/planned-only, `minimal-precondition-research` skill validation, targeted evidence-gap grep, and targeted
`git diff --check`. No outcomes, predicate values, baselines, learned contract, PageVectorSet baseline, LLM judge baseline,
live replay, or checking-cost measurement is changed by this entry.

Next:

Do not spend another pass trying to promote c3/c5 from the same XML collection. If those rows are needed, collect a curated or
live fixture that satisfies the `contacts_protocol.md` promotion rules. Otherwise prioritize the missing terminal-verified
direct-admission positive with `p_start_node_match=true`, or a terminal-verified route-repair positive with explicit route
predicate semantics.

## mlp-20260524-034 - 2026-05-24

Type: reasoning boundary reaffirmation / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the user's "if it is reasoning" question by keeping CMLP-002 as artifact-state validity learning over a public
verifier-grounded skeleton. The reusable unit is not reasoning text, hidden chain-of-thought, reflection memory, or LLM
agreement; it is a public dependency skeleton such as failure signature, source/schema/proof anchor, patch/query/proof slot,
action, verifier obligation, and verifier result.

Reason:

Reasoning becomes researchable only when its dependencies are externally checkable against the current artifact state. For code,
SQL, proof, build/config, and schema repair, the current state exposes files, AST/schema, tests, tool versions, checker
availability, local diffs, and verifier outcomes. This gives the same reuse-admission problem as GUI preconditions: learn the
minimal predicates under which reuse is safe and cheaper than rerunning the agent. Open-ended chat reasoning fails this test
because labels collapse to model agreement.

Validation:

Pending targeted validation after log updates: reasoning runtime JSON/JSONL parse, `minimal-precondition-research` skill
validation, targeted reasoning-boundary grep, targeted `git diff --check`, and trailing-whitespace check. No parser fixture,
patch execution, verifier stdout, baseline scoring, learner, checking-cost comparison, or evidence promotion is introduced.

Next:

If CMLP-002 is pulled forward, create exactly one tiny parser fixture only after explicit confirmation and measure whether
`C_check + C_patch + C_verify` is below rerunning a strong code/tool agent. Otherwise keep it secondary behind CMLP-001's
missing terminal-verified direct-admission or route-repair positive evidence.

## mlp-20260524-035 - 2026-05-24

Type: positive evidence protocol / no label promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Freeze the next CMLP-001 positive evidence acquisition protocol before another live run. The missing positive can be promoted
only as either a direct-admission positive or a route-repair positive.

Reason:

The current seven promoted rows are enough to show that always replay is unsafe, but they are not enough to learn a defensible
direct-allow contract. The only success row used a start-node-skip sidecar, while Contacts-home and overlay rows are repair-only
and the direct-admission probe is another pre-action failure. Without a stricter positive protocol, a future run could produce
stdout success, XML navigation success, or skipped-start success that still cannot fill the missing `p_start_node_match=true`
or route-repair success evidence.

Validation:

Pending targeted validation after log updates: CMLP JSON/JSONL parse, `minimal-precondition-research` skill validation,
targeted positive-protocol grep, targeted `git diff --check`, and trailing-whitespace check. This entry changes protocol and
handoff records only. It does not modify `outcomes.json`, `predicate_matrix.json`, `baseline_report.json`, or
`learned_contract.json`, and it does not promote any label.

Next:

Collect either a direct-admission positive with `label_source=execution`, no `__utg_skip_start_node_precondition__`,
`p_start_node_match=true`, valid arguments, and terminal provider verification, or a route-repair positive with
`p_start_node_match=false`, route-only bridge evidence before semantic form actions, and terminal provider verification after
bridge plus semantic execution.

## mlp-20260524-036 - 2026-05-24

Type: reasoning one-line definition / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the user's reasoning question by pinning CMLP-002 to one sentence: do not reuse thoughts; learn when a previous
verifier-backed artifact-edit skeleton is still valid in the current artifact state, and admit it only when checking plus
patching plus verification is cheaper and safer than rerunning the agent.

Reason:

This preserves the research core as reuse admission rather than reasoning cache. It also states the two hard gates that keep
the branch from becoming prompt memory: labels must come from public verifiers over artifacts, and the admission path must beat
rerunning a strong code/tool agent on cost and risk.

Validation:

Pending targeted validation after log updates: CMLP runtime JSON/JSONL parse, `minimal-precondition-research` skill validation,
targeted one-line-definition grep, targeted `git diff --check`, and trailing-whitespace check. No parser fixture, patch
execution, verifier stdout, baseline scoring, learner, checking-cost comparison, or evidence promotion is introduced.

Next:

Keep CMLP-002 secondary. If it is pulled forward, create exactly one tiny parser fixture only after explicit confirmation and
measure whether `C_check + C_patch + C_verify` is below rerunning a strong code/tool agent. Otherwise continue CMLP-001's
terminal-verified direct-admission or route-repair positive evidence path.

## mlp-20260524-037 - 2026-05-24

Type: contacts cost measurement protocol / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Freeze a row-level Cost Measurement Protocol for CMLP-001 before collecting more positives. Future execution-labeled positives
and selected contracts must record admission, route-repair, semantic replay, terminal-verifier, and replan/rerun costs instead
of treating predicate checking as free.

Reason:

The user's cost objection is decisive: a predicate contract that is slower or riskier than reobserving or replanning is not a
useful research result. The protocol now requires fields such as `observation_capture_ms`, `predicate_check_ms`,
`node_match_ms`, `route_bridge_execution_ms`, `semantic_replay_ms`, `terminal_verify_ms`, `total_admission_check_ms`,
`total_reuse_path_ms`, `replan_wall_ms`, `replan_model_tokens`, and `timing_source`. Direct-admission timing cannot come from
skipped-start rows, and route-repair timing must separate route check/execution from semantic replay.

Validation:

Pending targeted validation after log updates: CMLP JSON/JSONL parse, `minimal-precondition-research` skill validation,
targeted cost-protocol grep, targeted `git diff --check`, and trailing-whitespace check. This entry changes protocol/schema
and handoff records only. It does not modify `outcomes.json`, `predicate_matrix.json`, `baseline_report.json`, or
`learned_contract.json`, and it does not promote any label.

Next:

Collect either a direct-admission or route-repair positive under the positive protocol and fill the cost fields. If
`total_reuse_path_ms` plus false-accept risk is not lower than a measured or explicitly proxied replan/rerun path, mark
`learned_contract.json.status=negative_result`.

## mlp-20260524-038 - 2026-05-24

Type: runtime cost gate metadata / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Align the CMLP-001 runtime evidence package with the new Cost Measurement Protocol by marking the learned-contract cost gate as
machine-readable `unresolved` and listing the missing row-level timing/token fields in the runtime summary.

Reason:

The previous `checking_cost_below_replan=false` value was ambiguous: it could be read as a measured failure, but the correct
state is that no full row-level admission/reuse/terminal-verifier and replan/rerun cost comparison has been collected. The
runtime package now distinguishes unresolved cost evidence from a true `negative_result`.

Validation:

Pending targeted validation after log updates: Contacts runtime JSON/JSONL parse, structured cost-gate assertions,
`minimal-precondition-research` skill validation, targeted cost-gate grep, targeted `git diff --check`, and trailing-whitespace
check. No outcomes, predicate values, baseline decisions, learner result, or label promotion is changed by this entry.

Next:

Use the next direct-admission or route-repair positive run to fill the required cost fields. If the measured/proxied replan
path is cheaper or safer, convert the selector result to `negative_result` instead of broadening the method.

## mlp-20260524-039 - 2026-05-24

Type: reasoning branch answer / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the renewed "reasoning" question by keeping CMLP-002 narrow: it is acceptable only as artifact-state validity learning
over a public verifier-grounded skeleton. It is not a reasoning-text cache, hidden-CoT reuse, reflection memory, or an
LLM-agreement system.

Reason:

The same reuse-admission idea survives outside GUI only when the state and labels are externally checkable. In the code-agent
case, the reusable object must be a dependency skeleton such as failure signature, AST/source anchor, patch slot, patch action,
verifier obligation, and verifier result. The method remains researchable only if predicate checking plus patching and
verification is cheaper and safer than rerunning a strong code/tool agent.

Validation:

Pending targeted validation after log updates: reasoning runtime JSON/JSONL parse, `minimal-precondition-research` skill
validation, targeted reasoning-boundary grep, targeted `git diff --check`, and trailing-whitespace check. No parser fixture,
patch execution, verifier stdout, baseline scoring, learner, checking-cost comparison, or evidence promotion is introduced.

Next:

If the branch is pulled forward, create exactly one tiny parser fixture only after explicit confirmation and report either
`evidence_available`, `not_learned_insufficient_data`, or `negative_result`. Otherwise keep CMLP-001 as the main branch and
collect terminal-verified direct-admission or route-repair evidence first.

## mlp-20260524-040 - 2026-05-24

Type: contacts row-level cost schema alignment / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/outcomes.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add an explicit unresolved `cost_measurement` object to all nine CMLP-001 outcome rows. Each row now lists the admission,
route, semantic replay, terminal verification, replan, token, proxy, and timing-source fields required by the Cost Measurement
Protocol, with null values and `unresolved_reason=not_measured`.

Reason:

The runtime package should make missing cost evidence machine-readable at the row level. This prevents future scoring or
selector diagnostics from accidentally treating absent timing fields as zero cost or as a measured negative result.

Validation:

Pending targeted validation after log updates: Contacts JSON/JSONL parse, structured cost-measurement assertions,
`minimal-precondition-research` skill validation, targeted cost-field grep, targeted `git diff --check`, and
trailing-whitespace check. This entry changes cost metadata only. It does not change labels, predicate values, baseline
decisions, learned-contract status, or promotion state.

Next:

Collect a terminal-verified direct-admission or route-repair positive row and fill measured cost fields. Until then, keep the
cost gate unresolved and do not call any selected contract useful.

## mlp-20260524-041 - 2026-05-24

Type: baseline blocker metadata / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/baseline_report.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Make the unscored CMLP-001 baselines machine-readable blockers instead of prose-only caveats. `baseline_report.json` now records
`scoring_status`, `blocked_reason`, `required_evidence`, `next_unblock_action`, and `must_not_infer` for
`pagevectorset_threshold`, `llm_judge`, and `learned_minimal_preconditions`.

Reason:

The current package should distinguish missing evidence from measured failure. PageVectorSet cannot be scored without per-case
similarity scores and a frozen threshold rule; LLM judge cannot be scored without a fixed prompt, per-case decisions, and
token/cost records; learned minimal preconditions cannot be scored while the positive evidence and cost gate are unresolved.

Validation:

Completed targeted validation: Contacts JSON/JSONL parse, structured blocker assertions,
`minimal-precondition-research` skill validation, targeted blocker grep, targeted `git diff --check`, and
trailing-whitespace check. This entry changes blocker metadata only. It does not change labels, predicate values, deterministic
baseline metrics, learned-contract status, or promotion state.

Next:

Unblock baselines by collecting per-case PageVectorSet scores, running a frozen LLM judge prompt if useful, collecting a
terminal-verified direct-admission or route-repair positive row, and measuring checking/reuse cost versus replan/rerun cost.

## mlp-20260524-042 - 2026-05-24

Type: reasoning boundary answer / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the reasoning variant as a boundary decision, not as a new experiment: generic reasoning is not the object. CMLP-002
remains valid only as artifact-state validity learning over a public verifier-grounded skeleton. The reusable object is a
failure signature, source/schema/proof anchor, patch/query/proof slot, action, verifier obligation, and verifier result; it is
not hidden chain-of-thought, reflection memory, free-form explanation reuse, or an LLM agreement label.

Reason:

This keeps the direction researchable and robust against model upgrades. A stronger future code agent may reason better, but it
still cannot make a stale artifact skeleton valid when the current repo/schema/test/proof state violates required predicates.
The branch is useful only if public predicates and verifier-backed labels exist, and only if checking plus patching plus
verification is cheaper and safer than rerunning a strong code/tool agent.

Validation:

Completed targeted validation: reasoning runtime JSON/JSONL parse, `minimal-precondition-research` skill validation, targeted
reasoning-boundary grep, targeted `git diff --check`, and trailing-whitespace check. This entry changes records only. It does
not create a parser fixture, run patch/verifier/baseline/learner, change labels, or promote CMLP-002 beyond
`planned_skeleton_available`.

Next:

If CMLP-002 is pulled forward, create exactly one parser fixture only after explicit confirmation and require fixture,
predicate, verifier, and cost artifacts. Otherwise keep CMLP-001 as the main branch and collect terminal-verified
direct-admission or route-repair evidence first.

## mlp-20260524-043 - 2026-05-24

Type: PageVectorSet baseline blocker audit / no baseline scoring

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/baseline_report.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/learned_contract.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Keep `pagevectorset_threshold` blocked, but replace the vague blocker with a partial diagnostic audit. Existing artifacts can
produce diagnostic scores for only part of the promoted rows: `c1` has existing matcher top-1 score `0.810096263885498`,
`c2` has best-over-seed-node `xml_similarity=0.944586`, `c4` has `0.569046`, and `c8` has `0.907145`. These scores are
recorded as `partial_diagnostic_only_not_scored`.

Reason:

The audit is useful because it already warns that a similarity-only best-over-Function-nodes rule near the current
`BEST_THRESHOLD=0.87` would allow non-success rows: `c2_contacts_home_reachable` is repair-only, and
`c8_similar_non_contacts_page` is a cross-app reject. It still cannot score the baseline because `c0_correct_create_flow`
lacks a direct-admission or route-repair first/current XML score tied to terminal-verified positive evidence, `c6` is a
schema-level pre-GUI row, `c7` is a verifier-boundary abstain row, and no comparator or threshold rule has been frozen.

Validation:

Completed targeted validation after log updates: Contacts JSON/JSONL parse, structured PageVectorSet diagnostic assertions,
and `minimal-precondition-research` skill validation passed. This entry changes blocker metadata only. It does not change
labels, deterministic baseline metrics, learned-contract status, cost-gate status, or promotion state.

Next:

To score the PageVectorSet baseline, collect score provenance for every scored row, freeze the comparator rule
(`start_node` vs best-over-Function-nodes vs external page-family registry), freeze the threshold before scoring, and then
compute false accept / false reject against grounded labels.

## mlp-20260524-044 - 2026-05-24

Type: reasoning branch verdict / no promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the user's "if it is reasoning" question with the already frozen CMLP-002 boundary: reasoning is only viable after it is
erased into a public verifier-grounded skeleton. The reusable object is a failure signature, artifact anchor, patch or action
slot, verifier obligation, and verifier result; the research question is whether the current artifact state satisfies the
minimal validity predicates for that skeleton.

Reason:

This avoids the weak version of the idea: hidden chain-of-thought reuse, reflection memory, free-form explanation retrieval,
or an LLM agreeing with old rationale. The branch remains worth keeping only for code, SQL, proof, build, schema, or config
domains with external verifiers, and only if predicate checking plus patch/action plus verification is cheaper and safer than
rerunning a strong code/tool agent.

Validation:

This is a record-only answer. Contacts runtime JSON/JSONL parse, structured PageVectorSet diagnostic assertions, and
`minimal-precondition-research` skill validation passed in this turn. CMLP-002 remains `planned_skeleton_available`; no parser
fixture, patch execution, verifier labels, baseline scoring, learner, or cost comparison was created.

Next:

If CMLP-002 is pulled forward, require explicit confirmation before creating the single parser fixture and measure
`C_check + C_patch + C_verify` against rerunning a strong code/tool agent. Otherwise keep CMLP-001 as the main branch and collect
terminal-verified direct-admission or route-repair positive evidence.

## mlp-20260524-045 - 2026-05-24

Type: evidence acquisition queue / no label promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Freeze the next-step order as an evidence acquisition queue rather than another concept expansion: P0 is one terminal-verified
CMLP-001 positive row; P1 is PageVectorSet scoring only after complete score provenance plus frozen comparator and threshold;
P2 is LLM-judge only as a frozen baseline; P3 keeps CMLP-002 parked unless explicitly pulled forward.

Reason:

The current CMLP-001 package has enough negative/repair labels to show false-accept pressure, but it still lacks the specific
positive evidence needed to learn an admission contract. The current CMLP-002 package is a planned skeleton only. The queue
prevents future work from spending effort on unscored baselines, broad reasoning-memory variants, or post-hoc threshold choices
before the missing positive and cost evidence exist.

Validation:

Pending targeted validation after this record update: Contacts and reasoning JSON/JSONL parse, `minimal-precondition-research`
skill validation, targeted evidence-queue grep, targeted `git diff --check`, and trailing-whitespace check. This entry changes
planning and status documentation only; it does not promote labels, score new baselines, create a parser fixture, run a learner,
or resolve the cost gates.

Next:

Collect a terminal-verified direct-admission or route-repair positive row for CMLP-001 with cost fields. If the row lacks
terminal/business-effect evidence or required cost fields, keep it diagnostic. If reuse cost plus false-accept risk is not below
measured/proxied replan, record `negative_result`.

## mlp-20260524-046 - 2026-05-24

Type: P0 route-repair diagnostic / no label promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/live_attempts/c2_contacts_home_route_repair_probe_20260524.json`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Record the live Contacts-home route-repair attempt as diagnostic-only evidence, not a promoted `CMLP-001` row.

Reason:

The replay ran `aw_function_515be7bf7e` from Contacts home with valid arguments and terminal verification enabled, but current
runtime rejected before device actions with `EXEC_START_NODE_PRECONDITION_FAIL`: no route-only bridge exists from
`aw_node_51b4969e57` to required start node `aw_node_22653dacf0`. Terminal verification reported
`contacts_record_not_found`. This means the current seed Function cannot supply the missing P0 route-repair positive through
existing graph semantics; the next positive needs a different admissible start/create-flow Function or independently collected
route-only bridge evidence.

Validation:

Pending targeted validation after this record update: Contacts/reasoning JSON and JSONL parse, `minimal-precondition-research`
skill validation, targeted grep for the route-repair diagnostic, targeted `git diff --check`, and trailing-whitespace check.
No outcome label, baseline score, learner, parser fixture, or cost comparison is promoted in this entry.

Next:

Keep answering the reasoning branch as artifact-state validity learning only. For CMLP-001, collect either a terminal-verified
direct-admission positive row or a genuine route-repair positive row with route-only bridge evidence before semantic actions.

## mlp-20260524-047 - 2026-05-24

Type: CMLP-002 pre-fixture readiness audit / no evidence promotion

Files:

- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/manifest.json`
- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a machine-readable pre-fixture readiness audit for `CMLP-002` and keep the experiment at `planned_skeleton_available`.

Reason:

The reasoning branch now has a concrete planned parser seed, but no fixture files, verifier output, predicate extraction,
baseline scoring, learned selector, or cost comparison. The audit prevents planned seed bindings from being mistaken for
evidence. It marks materialization and predicate gates as `planned_only`, verifier and cost gates as `missing`, and records that
planned labels are not verifier-backed labels and the manual predicate hypothesis is not a learned contract.

Validation:

Pending targeted validation after this record update: reasoning/Contacts JSON and JSONL parse, structured readiness-audit
assertions, `minimal-precondition-research` skill validation, targeted grep, targeted `git diff --check`, and trailing-whitespace
check. No parser fixture, patch execution, verifier label, baseline score, learner, or cost comparison is created in this entry.

Next:

Keep `CMLP-002` parked unless explicitly pulled forward. If it is pulled forward, create exactly one parser fixture and collect
fixture, predicate, verifier, and cost artifacts before changing status.

## mlp-20260524-048 - 2026-05-24

Type: CMLP-002 reasoning user answer / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the user's "if it is reasoning" question with the current CMLP-002 boundary: generic reasoning is rejected, while
verifier-grounded artifact reasoning remains viable only as artifact-state validity learning.

Reason:

The reusable object cannot be hidden chain-of-thought, reflection memory, explanation text, or an LLM agreement label. The only
defensible object is a public skeleton such as failure signature, artifact anchor, patch/action slot, verifier obligation, and
verifier result. The research question is whether the current artifact state satisfies minimal validity predicates cheaply enough
to reuse the skeleton instead of rerunning a strong code/tool agent.

Validation:

This entry is record-only. It relies on the existing CMLP-002 protocol, runtime manifest readiness audit, and summary; no parser
fixture, verifier-backed label, baseline score, learned selector, or checking-cost comparison is created or promoted.

Next:

Keep CMLP-002 parked unless explicitly pulled forward. If pulled forward, create exactly one parser fixture for
`code_fix_parser_missing_guard_v0` and stop expansion if `C_check + C_patch + C_verify` is not lower than rerunning a strong
code/tool agent.

## mlp-20260524-049 - 2026-05-24

Type: CMLP-001 P0 seed-asset audit / no label promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/asset_audits/p0_existing_function_start_state_audit_20260524.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Record a read-only audit of the existing Contacts seed UTG as diagnostic-only evidence. Do not update `outcomes.json`,
baseline denominators, or learned-contract status from this artifact.

Reason:

The stored asset has only one top-level Function, `aw_function_515be7bf7e`, whose required start node is
`aw_node_22653dacf0`. Contacts home `aw_node_51b4969e57` has no path back to that start node in normal or route-only graph
search. The only non-route hop from Contacts home to a form node uses the semantic top-level Function itself, and node-scoped
functions are atomic replay steps rather than terminal-verified top-level Functions with the complete argument schema. Therefore
the current seed asset cannot itself supply the missing P0 direct-admission or route-repair positive row.

Validation:

Pending targeted validation after this record update: asset audit JSON parse, Contacts/reasoning JSON and JSONL parse,
structured no-promotion assertions, `minimal-precondition-research` skill validation, targeted grep, targeted `git diff --check`,
and trailing-whitespace check.

Next:

The P0 positive still requires a new or derived Function with an admissible start/create-flow state, or independently collected
route-only bridge evidence before semantic actions, plus terminal provider verification and row-level cost fields.

## mlp-20260524-050 - 2026-05-24

Type: CMLP-002 reasoning answer / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the user's "if it is reasoning" question by rejecting generic reasoning reuse and keeping only the CMLP-002 artifact-state
validity version.

Reason:

Reasoning text itself is not a stable, publishable, or externally checkable reuse object. The only defensible unit is a public
verifier-grounded skeleton: failure signature, source or schema anchor, patch/action slot, verifier obligation, and verifier
result. The research method is to learn minimal current-artifact predicates that admit that skeleton only when checking,
patching, and verification are cheaper and safer than rerunning a strong code/tool agent.

Validation:

Record-only update. No parser fixture, patch execution, verifier-backed label, baseline score, learned selector, or cost
comparison is created or promoted. CMLP-002 remains `planned_skeleton_available`.

Next:

Keep CMLP-002 parked unless explicitly pulled forward. If pulled forward, create exactly one parser fixture for
`code_fix_parser_missing_guard_v0`, collect fixture/predicate/verifier/cost artifacts, and stop the branch as
`negative_result` if `C_check + C_patch + C_verify` is not lower than rerunning a strong code/tool agent.

## mlp-20260524-051 - 2026-05-24

Type: CMLP-001 planned derived candidate / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Record `contacts_home_create_contact_candidate_20260524` as a planned-only derived Function hypothesis in the existing CMLP-001
runtime manifest.

Reason:

The P0 seed-asset audit showed that the stored Contacts Function cannot provide a direct-admission or route-repair positive
from Contacts home because no route-only bridge reaches the original required launcher-start node. The next concrete hypothesis
is therefore not to force the old start node, but to test a derived Function that removes the launcher/open-app prefix action and
starts at Contacts home `aw_node_51b4969e57`, keeping the create-entry, first-name, last-name, phone, and save steps. This turns
the gap into a live-testable candidate without mutating the seed UTG store.

Validation:

Pending targeted validation after this record update: Contacts JSON/JSONL parse, structured planned-candidate assertions,
no-promotion assertions for outcomes/baselines/learned contract, `minimal-precondition-research` skill validation, targeted grep,
targeted `git diff --check`, and trailing-whitespace check.

Next:

Materialize the candidate as a separate Function asset only after explicit follow-up. A future promoted positive still requires
live execution from Contacts home with no start-node skip, `p_start_node_match=true` for `aw_node_51b4969e57`, valid arguments,
terminal provider verification, and complete row-level cost fields.

## mlp-20260524-052 - 2026-05-24

Type: CMLP-001 dry-run materialization audit / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a dry-run materialization audit to `contacts_home_create_contact_candidate_20260524` without creating a new Function asset.

Reason:

The planned candidate needed one more evidence-hygiene step before asset creation: prove that it is mechanically derivable from
the stored seed Function while still keeping the seed asset immutable and the experiment below evidence status. The dry run
derives the candidate in memory, removes only `aw_seq_22653dacf0`, keeps the five semantic actions, preserves
`first_name` / `last_name` / `phone`, rewrites the start node to `aw_node_51b4969e57`, and preserves end node
`aw_node_dbf1a7e3cd`.

Validation:

Pending targeted validation after this record update: Contacts JSON/JSONL parse, structured dry-run assertions, no-promotion
assertions for outcomes/baselines/learned contract, seed-file unchanged check, `minimal-precondition-research` skill validation,
targeted grep, targeted `git diff --check`, and trailing-whitespace check.

Next:

The next concrete step is materializing a separate candidate asset and running it from live Contacts home with no start-node
skip. Promotion still requires terminal provider verification and complete row-level cost fields.

## mlp-20260524-053 - 2026-05-24

Type: CMLP-001 materialization readiness gate / no asset creation / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add `materialization_readiness` to the planned Contacts-home derived candidate without writing the candidate asset.

Reason:

The dry-run audit proved mechanical derivability but did not say whether the next step was safe to execute or promotable. The
new readiness gate separates those states: the candidate is ready for a future explicit asset materialization step, but the seed
asset remains immutable, the reserved candidate path is not written, file creation still requires explicit confirmation,
promotion is disallowed, and outcomes, baselines, and learned-contract status must not change.

Validation:

Completed targeted validation: Contacts/reasoning JSON and JSONL parse; structured materialization-readiness assertions;
no-promotion assertions for outcomes, baselines, and learned contract; seed-file unchanged check; local
`minimal-precondition-research` skill frontmatter/key-field validation; targeted grep for readiness fields; targeted
`git diff --check` for tracked logs; and trailing-whitespace check over the CMLP runtime, exploration, records, and skill files.
`quick_validate.py` was not present in this checkout, so the skill check used a local structural validator.

Next:

After explicit confirmation to create a new file, materialize the reserved candidate asset under
`runtime/evals/preconditions/contacts_minimal_live_preconditions/derived_candidates/` and then run a live Contacts-home replay
with terminal provider verification and complete row-level cost fields. Until then, the candidate remains planned-only.

## mlp-20260524-054 - 2026-05-24

Type: CMLP-001 candidate materialization protocol / no asset creation / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add `candidate_materialization_protocol` to the planned Contacts-home derived candidate and mirror the protocol in
`contacts_protocol.md`.

Reason:

The previous readiness gate established that the candidate is ready for a future explicit asset-writing step, but it did not
freeze the exact post-write checks or live-probe promotion boundary. The new protocol prevents the next step from drifting:
materialization may only write the reserved separate candidate asset after explicit confirmation, must not mutate the seed UTG,
must not update outcomes/baselines/learned-contract status from materialization alone, and must pass post-write schema checks
before any live probe. A live probe remains diagnostic unless it has no start-node skip, `p_start_node_match=true` for
`aw_node_51b4969e57`, terminal provider verification, and complete row-level cost fields.

Validation:

Completed targeted validation: Contacts/reasoning JSON and JSONL parse; structured candidate-materialization-protocol
assertions; no-promotion assertions for outcomes, baselines, and learned contract; reserved asset non-existence check; seed-file
unchanged check; local skill structure/key-field validation; targeted grep; targeted `git diff --check`; and trailing-whitespace
check.

Next:

The next write-capable step still requires explicit confirmation to create the reserved candidate asset. Until then, the
candidate remains planned-only and cannot affect labels, baselines, learned contracts, or paper claims.

## mlp-20260524-055 - 2026-05-24

Type: CMLP-002 reasoning boundary answer / no fixture creation / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the renewed "what if it is reasoning" question by keeping CMLP-002 only as public verifier-grounded artifact-state
validity learning.

Reason:

Generic reasoning text, hidden chain-of-thought, reflection memory, and LLM agreement are not stable research objects for this
direction. The reusable object must be a public dependency skeleton with failure signature, artifact anchor, patch/action slot,
verifier obligation, and verifier result. The admission question then becomes whether current files, AST/schema, tests, tool
versions, local diffs, and verifier availability still satisfy the minimal validity predicates.

Validation:

Record-only validation for this answer. No parser fixture was created, no patch was applied, no verifier was run, and no
baseline, learner, or checking-cost comparison was executed. CMLP-002 remains `planned_skeleton_available`.

Next:

Only pull CMLP-002 forward by explicitly creating the single planned `code_fix_parser_missing_guard_v0` fixture and measuring
`C_check + C_patch + C_verify` against rerunning a strong code/tool agent. If that cost gate fails, record `negative_result`
and stop expansion.

## mlp-20260524-056 - 2026-05-24

Type: CMLP-001 candidate blueprint digest / no asset creation / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Record a deterministic canonical blueprint and SHA-256 digest for
`contacts_home_create_contact_candidate_20260524` without writing the candidate asset.

Reason:

The planned Contacts-home candidate was materializable in memory, but future asset materialization still needed a stable identity
checkpoint. The blueprint digest freezes the source Function lookup, original start/end nodes, removed launcher/open-app prefix,
retained five semantic actions with argument bindings, derived start/end nodes, required arguments, and reserved candidate path.
This lets a future agent detect drift before writing the separate asset.

Validation:

Pending targeted validation after this record update: Contacts/reasoning JSON and JSONL parse, blueprint digest recomputation,
no-promotion assertions for outcomes/baselines/learned contract, reserved asset non-existence check, seed-file unchanged check,
local skill structural validation, targeted grep, targeted `git diff --check`, and trailing-whitespace check.

Next:

The next write-capable step still requires explicit confirmation to create the reserved candidate asset. Future materialization
must recompute the blueprint digest before writing and must keep outcomes, baselines, and learned-contract status unchanged
until a live probe satisfies terminal verifier and cost gates.

## mlp-20260524-057 - 2026-05-24

Type: CMLP-001 materialization preflight / blocked until explicit asset creation confirmation / no evidence promotion

Files:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/manifest.json`
- `runtime/evals/preconditions/contacts_minimal_live_preconditions/summary.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/contacts_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a no-write `materialization_preflight` block for the planned Contacts-home candidate and keep it blocked until explicit
confirmation to create the reserved candidate asset.

Reason:

The blueprint digest fixed the candidate identity, but the future write step still needed a machine-readable preflight state.
The new preflight records digest recomputation, seed Function lookup and action count, removed-prefix match, retained-action
sequence match, derived node and argument checks, reserved asset absence, and seed-file immutability. It deliberately remains
blocked because writing the reserved candidate asset is a new-file operation requiring explicit confirmation.

Validation:

Pending targeted validation after this record update: Contacts/reasoning JSON and JSONL parse, preflight assertions, digest
recompute, no-promotion assertions, reserved asset non-existence check, seed-file unchanged check, skill key-field validation,
targeted grep, targeted `git diff --check`, and trailing-whitespace check.

Next:

If explicit asset-creation confirmation is provided later, write only the reserved separate candidate asset, then run post-write
checks before any live probe. Until then, the preflight is a blocked checklist and cannot affect labels, baselines, learned
contracts, or paper claims.

## mlp-20260524-058 - 2026-05-24

Type: CMLP-002 reasoning answer / record-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the user's "if it is reasoning" question by keeping CMLP-002 as artifact-state validity learning over a public
verifier-grounded skeleton, not as reasoning-text reuse.

Reason:

Reasoning text is too model-version-dependent and has no stable external label. The viable object is a public dependency
skeleton whose assumptions can be checked against the current artifact state: failure signature, source/schema/proof anchor,
patch/query/proof slot, action, verifier obligation, and verifier result. The branch remains researchable only if checking,
patching, and verification are cheaper and safer than rerunning a strong code/tool agent.

Validation:

Record-only update. No parser fixture, patch execution, verifier-backed label, baseline score, learned selector, or cost
comparison is created or promoted. CMLP-002 remains `planned_skeleton_available`.

Next:

If CMLP-002 is pulled forward, create exactly one parser fixture for `code_fix_parser_missing_guard_v0` only after explicit
confirmation, then report `evidence_available`, `not_learned_insufficient_data`, or `negative_result` based on verifier-backed
labels and measured cost.

## mlp-20260524-059 - 2026-05-24

Type: CMLP / PABench comparison answer / record-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Answer the user's PABench comparison by keeping CMLP distinct from Project Control. PABench is project-level closed-loop
control over the next verifier-observed transition. CMLP is reuse admission for a past trace, Function, or verifier-backed
skeleton in the current state.

Reason:

The two lines share sensors and cost discipline, but their decision variables differ. PABench asks which control action to take
now: delegate, verify, rollback, stop, or escalate. CMLP asks whether one already-known reusable object is currently admissible:
allow reuse, reject, reobserve, or repair the precondition. The reasoning variant can become a PABench subroutine for cached
repair skeleton admission, but it is not the Project Control Layer itself.

Validation:

Record-only update. No PABench benchmark, CMLP runtime package, parser fixture, verifier-backed label, baseline score, learned
selector, or cost comparison is created or promoted.

Next:

If the two lines are combined later, define the interface narrowly: ProjectState plus VerifierObservation may call a CMLP
reuse-admission gate for a cached repair skeleton, and the gate must return allow/reject/reobserve/repair plus cost and
false-accept risk. It must not replace PABench's delegate/verify/rollback/stop/escalate policy.

## mlp-20260524-060 - 2026-05-24

Type: CMLP / PABench boundary formalization / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Move the PABench comparison from a chat-only answer into the CMLP exploration workspace. The README now has a dedicated
boundary section, the agenda has an explicit composition rule, and the local skill prevents future agents from treating CMLP as
the Project Control policy.

Reason:

The active goal requires maintaining a separate exploration directory and separate commit-like record. Since PABench and CMLP
are adjacent, the boundary must live in the exploration docs where future work starts. The key rule is decision-level: PABench
chooses the next project-control action, while CMLP admits or rejects one cached reusable object.

Validation:

CMLP JSON/JSONL parse passed; the reasoning runtime manifest still reports `planned_skeleton_available`; targeted boundary
grep found the README, agenda, skill, commit-log, and iteration-log records; targeted `git diff --check` and trailing-whitespace
checks passed.

Next:

Continue evidence-first work. For CMLP-001, prioritize one terminal-verified positive row with cost fields. For CMLP-002, keep
the parser fixture parked until explicitly pulled forward.

## mlp-20260524-061 - 2026-05-24

Type: CMLP-002 / PABench sub-gate interface draft / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a concrete PABench-compatible CMLP sub-gate draft for CMLP-002. The draft defines input as
`ProjectState + VerifierObservation + cached_skeleton` and output as a reuse-admission decision with selected predicates, cost,
risk, and controller obligations.

Reason:

The previous boundary note said CMLP can be a Project Control subroutine. This turn makes that statement operational without
running an experiment: the gate can decide whether one cached parser repair skeleton is admissible, but the outer Project
Control runtime still owns ProjectGraph updates and `delegate / verify / rollback / stop / escalate` policy.

Validation:

CMLP JSON/JSONL parse passed; the reasoning runtime manifest still reports `planned_skeleton_available`; targeted sub-gate
grep found the protocol, agenda, skill, commit-log, and iteration-log records; targeted `git diff --check` and
trailing-whitespace checks passed.

Next:

If the user explicitly pulls CMLP-002 forward, instantiate exactly one parser fixture and produce one row-level admission
decision using this interface. Until then, the interface remains planned-only documentation.

## mlp-20260524-062 - 2026-05-24

Type: CMLP-002 / first PABench sub-gate row plan / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Define the first planned PABench sub-gate row as `pabench_subgate_parser_missing_guard_original`. The row asks only whether a
cached parser missing-guard skeleton is admissible under the current `ProjectState + VerifierObservation`; it does not choose a
Project Control action.

Reason:

The previous interface draft proved that CMLP can be a subroutine of Project Control, but it still lacked a row-level test
shape. This entry fixes the first executable comparison before any fixture is created: positive conditions, negative
perturbations, baselines, metrics, and promotion/kill rules. The key kill rule is cost: if admission checking plus patching and
verification is not cheaper than rerunning a strong code agent, the correct result is `negative_result`.

Validation:

Pending targeted validation after this record update: CMLP JSON/JSONL parse, reasoning status unchanged, targeted sub-gate row
grep, targeted `git diff --check`, and trailing-whitespace check.

Next:

Keep CMLP-002 planned-only until explicit confirmation to create the parser fixture. If the fixture is created later, produce one
row-level admission result first and keep ProjectGraph update ownership in PABench.

## mlp-20260524-063 - 2026-05-24

Type: CMLP-002 / PABench sub-gate row evidence schema / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/reasoning_subgate_evidence_schema.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a planned row-level evidence schema for `pabench_subgate_parser_missing_guard_original`. The schema requires ProjectState,
VerifierObservation, cached skeleton, executable predicate values, admission decision, patch application evidence, verifier
results, baseline decisions, cost measurement, risk metrics, and a promotion gate.

Reason:

The first-row plan defined what should be compared, but future work still needed a concrete artifact contract to prevent weak
rows that only store a gate decision. The schema keeps CMLP-002 as a reuse-admission sub-gate and blocks promotion unless
verifier-backed labels, patch evidence, baselines, and admission-vs-rerun cost all exist.

Validation:

Pending targeted validation after this record update: CMLP JSON/JSONL parse, reasoning status unchanged, targeted schema grep,
targeted `git diff --check`, and trailing-whitespace check.

Next:

Keep the schema as planning metadata until explicit confirmation creates the parser fixture. If a future row is created, reject
or mark it diagnostic if any required evidence class is missing.

## mlp-20260524-064 - 2026-05-24

Type: CMLP-002 / sub-gate schema registration / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Register `reasoning_subgate_evidence_schema.md` as the planned CMLP-002 / PABench sub-gate evidence schema in the exploration
README, experiment register, and minimal-precondition skill read-first list.

Reason:

The schema existed after the previous entry, but future agents also need an explicit index entry that ties it to
`CMLP-002` and `pabench_subgate_parser_missing_guard_original`. This prevents the schema from being treated as an orphan
artifact or, worse, as already collected runtime evidence.

Validation:

Pending targeted validation after this record update: CMLP JSON/JSONL parse, reasoning status unchanged, schema registration
grep, targeted `git diff --check`, and trailing-whitespace check.

Next:

Keep CMLP-002 parked. A future executable pass must still wait for explicit parser-fixture confirmation and must follow the
registered schema before any row can be promoted.

## mlp-20260524-065 - 2026-05-24

Type: CMLP-002 / PABench sub-gate execution checklist / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a planned-only `PABench Sub-Gate Execution Checklist` that maps the first CMLP-002 / PABench row to exact future artifact
paths under
`runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/`.

Reason:

The row evidence schema defined the required fields, but future work still needed an execution checklist that separates fixture
source, public skeleton, ProjectState, VerifierObservation, predicates, patch evidence, verifier logs, baselines, cost, risk, and
promotion gate into concrete artifacts. This reduces ambiguity for the first executable pass while preserving the current
no-fixture/no-evidence boundary.

Validation:

Pending targeted validation after this record update: CMLP JSON/JSONL parse, reasoning status unchanged, targeted checklist
grep, absence checks for future row-root and parser-fixture paths, targeted `git diff --check`, and trailing-whitespace check.

Next:

Keep CMLP-002 planned-only. Do not create `subgate_rows/`, parser fixture files, patch diffs, verifier logs, or row JSON until
explicit confirmation creates the single parser fixture and executable evidence collection begins.

## mlp-20260524-066 - 2026-05-24

Type: CMLP-002 / PABench non-overlap audit / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a planned-only `PABench Non-Overlap Audit` for the first CMLP-002 / PABench sub-gate row. A row remains CMLP only when it
decides admission for one cached trace, Function, or verifier-backed skeleton; if it chooses `delegate`, `verify`, `rollback`,
`stop`, or `escalate`, updates ProjectGraph, marks a transition resolved, owns rollback/stop/escalation, or optimizes
transition-quality metrics as the primary controller objective, it must move back to PABench.

Reason:

The user asked whether this is the same as the PABench work. The answer is no under the current boundary, but the distinction
needed an operational audit so future sub-gate rows cannot drift into Project Control policy. This keeps CMLP as reuse
admission and PABench as project-level transition control.

Validation:

CMLP JSON/JSONL parse passed; the reasoning runtime manifest still reports `planned_skeleton_available`; planned row-root and
parser-fixture paths are absent; targeted non-overlap grep found the protocol, agenda, register, skill, commit-log, and
iteration-log records; tracked `git diff --check`, untracked no-index diff checks, and trailing-whitespace checks passed.

Next:

Keep CMLP-002 planned-only. If a future artifact needs project-level control actions or ProjectGraph ownership, record it under
PABench instead of expanding the CMLP action space.

## mlp-20260524-067 - 2026-05-24

Type: CMLP-002 / PABench subsumption test / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a `PABench Subsumption / Replacement Test` for the CMLP-002 sub-gate. A row remains CMLP only if, under the same
`ProjectState + VerifierObservation`, changing the cached skeleton can change the correct admission decision through
skeleton-specific predicates, verifier obligations, allowed-write binding, or admission cost/risk. If the cached skeleton is
irrelevant and the decision collapses to `delegate / verify / rollback / stop / escalate`, the row is PABench evidence.

Reason:

The non-overlap audit says what CMLP must not do, but it does not prove CMLP has a separate variable after being connected to
Project Control. The subsumption test adds the positive criterion: the cached object must be causally necessary for the
admission decision. This prevents a future row from being a disguised Project Control transition while still allowing CMLP to
serve as a narrow sub-gate for cached repair skeletons.

Validation:

Pending targeted validation after this record update: CMLP JSON/JSONL parse, reasoning status unchanged, future row-root and
parser-fixture absence checks, targeted subsumption grep, targeted `git diff --check`, untracked no-index diff checks, and
trailing-whitespace check.

Next:

Keep CMLP-002 planned-only. A future executable row should include at least two cached skeleton alternatives, or one cached
skeleton plus a predicate perturbation, under the same outer ProjectState and VerifierObservation. Without skeleton-specific
admission variation, move the result to PABench rather than promoting CMLP.

## mlp-20260524-068 - 2026-05-24

Type: CMLP decision snapshot / records governance / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/README.md`
- `docs/agent_context/INDEX.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a current decision snapshot to the independent CMLP records README instead of creating a new `decision_register.md`.
The snapshot records accepted, rejected, and parked decisions for reuse admission, the GUI/reasoning branch split, the
CMLP/PABench action-space boundary, the cached-object subsumption test, and the current partial status of CMLP-001 and
CMLP-002.

Reason:

The active goal requires maintaining exploration documents and a separate commit-like record. Future agents need a stable
handoff view that is shorter than the full commit log, but the repository rule discourages adding a new file without explicit
confirmation. Keeping the decision snapshot in the records README improves traceability while avoiding a new parallel record
surface.

Validation:

CMLP runtime JSON/JSONL parse passed for 12 files. Targeted decision-snapshot grep found the records README, agent index,
minimal-precondition skill, commit log, and repo iteration logs. Structured assertions confirmed `CMLP-D001` through
`CMLP-D007`, `mlp-20260524-068`, and the repo-wide log entries. Targeted `git diff --check` passed for tracked files, and a
targeted trailing-whitespace check passed for tracked and untracked files touched in this entry.

Next:

Continue evidence-first exploration. Do not promote CMLP into NSDI claims. For CMLP-001, the next material step remains a
terminal-verified positive row with cost fields. For CMLP-002, keep the parser fixture parked until explicit confirmation.

## mlp-20260524-069 - 2026-05-24

Type: CMLP research maturity gate / next-slice decision rule / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/README.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Add a `Research Maturity Gate` to the CMLP agenda and record it as decision `CMLP-D008`. Every future slice must be classified
as `explore`, `collect_evidence`, `park`, or `negative_result` before expansion. A slice can collect evidence only when it has a
named cached object, a label-acquisition rule, grounded labels, baselines, cost fields, and a PABench boundary statement when
applicable.

Reason:

The PABench boundary now says what CMLP is not, but the exploration still needed a forward decision rule. Without a maturity
gate, the next step could drift into generic reasoning memory, Project Control policy, or a guard layer whose checking cost is
higher than rerun/replan. The gate keeps CMLP evidence-first while still allowing a narrow PABench sub-gate when the cached
object remains causally necessary.

Validation:

Pending targeted validation after this record update: CMLP JSON/JSONL parse, maturity-gate grep, structured assertions for
`CMLP-D008`, targeted `git diff --check`, and trailing-whitespace checks.

Next:

Apply the gate to the next slice. For CMLP-001, continue only with a terminal-verified positive row with full cost fields or
explicit Contacts-home candidate materialization followed by a no-skip live probe. For CMLP-002, keep the parser fixture parked
until explicit confirmation; the first executable pass must end as `evidence_available`, `not_learned_insufficient_data`, or
`negative_result`.

## mlp-20260524-070 - 2026-05-24

Type: CMLP next-slice classification / research governance / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/README.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Apply the research maturity gate to the current candidate slices. The Contacts-home derived candidate, any alternative
CMLP-001 positive row from existing assets, the PageVectorSet threshold baseline, the LLM-judge baseline, the CMLP-002 parser
fixture, and the first PABench sub-gate row are all classified as `park`. Any CMLP proposal whose decision is
`delegate / verify / rollback / stop / escalate` is classified as a CMLP-scope `negative_result` and should move to PABench
records.

Reason:

The previous boundary answer said CMLP is distinct from PABench only when the cached trace, Function, or skeleton remains a
causal input to the reuse-admission decision. This entry turns that answer into an operational next-slice table so future work
does not quietly convert parked hypotheses into evidence or broaden CMLP into Project Control.

Validation:

CMLP runtime JSON/JSONL parse passed for 12 files. Targeted next-slice classification grep found the agenda, records README,
minimal-precondition skill, commit log, and repo iteration logs. Structured assertions passed for `CMLP-D009`,
`mlp-20260524-070`, the repo-wide logs, and the skill classification. Absence checks confirmed the reserved Contacts-home
candidate asset, parser fixture directory, and subgate row root are still absent. Targeted `git diff --check` passed for
tracked logs, and a targeted trailing-whitespace check passed for all touched CMLP/log files.

Next:

Do not collect new CMLP evidence until one parked item obtains its missing prerequisite. The two admissible unblockers are:
explicit confirmation to materialize the reserved Contacts-home candidate asset, or explicit confirmation to create the single
parser fixture for `code_fix_parser_missing_guard_v0`. If the next idea is a project-level control action, record it under
PABench instead.

## mlp-20260524-071 - 2026-05-24

Type: CMLP / PABench comparison verdict / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/README.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

After rereading PABench's own Project Control framing, record the sharper comparison verdict: GUI CMLP remains a distinct
reuse-admission problem because the cached Function is the object being admitted or rejected. CMLP-002 is not a separate
Project Control paper; it is only viable as a PABench sub-gate when the cached skeleton is a causal variable under fixed
`ProjectState + VerifierObservation`. If removing or swapping the cached skeleton does not change the correct admission
decision, PABench subsumes the row.

Reason:

The user's concern was correct: a reasoning version can easily become the same as PABench. PABench already owns the
controller input interface and the policy over `delegate / verify / rollback / stop / escalate` for verifier-observed project
transitions. CMLP only adds a separate research object when it answers a narrower question: whether this particular cached
trace, Function, or verifier-backed skeleton is still admissible. The parser-skeleton row therefore needs skeleton-specific
variation, not just another project-control transition.

Validation:

Pending targeted validation after this record update: PABench/CMLP comparison grep, `CMLP-D010` assertion, CMLP runtime
JSON/JSONL parse, reserved asset/parser fixture/subgate absence checks, targeted `git diff --check`, and trailing-whitespace
checks.

Next:

Keep CMLP-002 parked. If explicitly pulled forward, the first parser row must include either two cached skeleton choices or a
predicate perturbation that changes only skeleton validity while the outer `ProjectState + VerifierObservation` stays fixed.
Otherwise move the row to PABench records or record a CMLP negative result.

## mlp-20260524-072 - 2026-05-24

Type: CMLP-002 / PABench skeleton-causality contrast / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/reasoning_protocol.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/README.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Make the first CMLP-002 / PABench replacement test concrete by fixing a planned A/B cached-skeleton contrast. Under the same
parser `ProjectState + VerifierObservation`, `code_fix_parser_missing_guard_v0` is the only skeleton that may become
admissible after future fixture evidence, while `code_fix_parser_delimiter_rewrite_v0` is the planned contrasting skeleton to
reject for skeleton-specific anchor, patch-slot, write-set, verifier-obligation, or cost reasons.

Reason:

The previous verdict said reasoning CMLP is absorbed by PABench unless cached skeleton choice is causal. This entry turns that
verdict into a falsifiable contrast: if varying only the cached skeleton changes reuse admission while the outer project-control
state is fixed, the row can remain CMLP; if both skeletons collapse to the same `delegate / verify / rollback / stop / escalate`
choice, the row belongs to PABench.

Validation:

CMLP runtime JSON/JSONL parse passed. Skeleton-causality grep found the reasoning protocol, agenda, records README, skill,
commit log, and repo iteration logs. Structured assertions passed for `CMLP-D011`, `mlp-20260524-072`, the fixed A/B skeleton
contrast, and reasoning manifest status `planned_skeleton_available`. Absence checks confirmed the reserved Contacts candidate
asset, parser fixture directory, and subgate row root are still absent. Local skill validation passed. Targeted `git diff
--check` passed for tracked changes, and the separate touched-file trailing-whitespace check passed for the edited CMLP docs and
logs.

Next:

Keep the contrast planned-only. Do not create `code_fix_parser_delimiter_rewrite_v0` artifacts, parser fixture files, verifier
logs, baseline tables, or subgate rows without explicit parser-fixture confirmation. If future evidence cannot show different
admission decisions under fixed outer state, record the row under PABench or as a CMLP negative result.

## mlp-20260524-073 - 2026-05-24

Type: CMLP planned-only freeze / Go-No-Go gate / documentation-only / no evidence promotion

Files:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/README.md`
- `docs/agent_context/skills/minimal-precondition-research/SKILL.md`
- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/commit_log.md`
- `project_iterations/VERSION_LOG.md`
- `project_iterations/PART_ITERATION_LOG.md`

Decision:

Freeze planned-only expansion for the CMLP line. Future turns must not add more planned rows, schemas, skeleton families, or
contrasts unless they replace an existing item with stronger evidence. The only Go paths are explicit Contacts candidate
materialization, CMLP-001 terminal/cost positive evidence, explicit CMLP-002 parser fixture creation, verifier-backed
skeleton-causality evidence, or grounded baseline scoring.

Reason:

The exploration has enough protocol text to support the next evidence step. Continuing to add planned material would make the
thread look active without moving the research state. The Go/No-Go gate forces the next slice to either collect one concrete
evidence artifact, report the current parked status, or move project-control work back to PABench.

Validation:

CMLP runtime JSON/JSONL parse passed. Planned-only freeze grep found the agenda, records README, skill, commit log, and repo
iteration logs. Structured assertions passed for `CMLP-D012`, `mlp-20260524-073`, the Go/No-Go gate, and current runtime
statuses. Absence checks confirmed the reserved Contacts candidate asset, parser fixture directory, and subgate row root are
still absent. Local skill validation passed. Targeted `git diff --check` and touched-file trailing-whitespace checks passed.

Next:

Do not broaden CMLP further in planned-only form. If the next turn cannot satisfy a Go condition, keep the branch parked and
answer from the current status. If the user explicitly confirms a Go path, create exactly that bounded evidence artifact and
record verifier/cost results before changing status.
