# Reasoning Trace Minimal Validity Conditions Protocol

Status: Draft
Last Updated: 2026-05-24
Scope: Secondary protocol for the minimal live preconditions direction. This is not a current NSDI claim and must not reuse hidden chain-of-thought.

## Purpose

Turn the reasoning variant into a small, falsifiable reuse-admission problem:

> Given a public verifier-grounded reasoning skeleton from a previous successful code-agent task, decide whether the current repository state still satisfies the minimal validity conditions needed to reuse that skeleton safely.

This protocol is deliberately narrower than "reuse reasoning". It only applies when the reasoning trace is made of externally checkable objects: claims, premises, tool results, actions, verifier obligations, and verifier results.

One-line work definition:

> For reasoning, we do not reuse thoughts; we learn when a previous verifier-backed artifact-edit skeleton is still valid in the current artifact state, and only admit it when checking plus patching plus verification is cheaper and safer than rerunning the agent.

## Boundary Answer

The reasoning version is viable, but only after "reasoning" is converted into an artifact-state validity problem.

Allowed object:

```text
failure signature -> source anchor -> patch slot -> patch action -> verifier obligation -> verifier result
```

Rejected objects:

- hidden chain-of-thought;
- free-form reflection logs;
- "why the model thought this patch was right" prose;
- an LLM judge's applicability decision without verifier-backed outcomes.

The research question is therefore not "can we cache reasoning?" It is:

> Which minimal live validity predicates over the current artifact state make a previously successful public reasoning skeleton safe to reuse?

This keeps the method outside ordinary prompt engineering. The reusable unit must bind to files, symbols, schema fields, tests, tool versions, or proof/checker obligations that can be inspected before reuse and verified after reuse.

Four-gate litmus test:

1. **Materialization gate**: the private reasoning must be erased into a public skeleton with named artifacts, anchors, actions, and verifier obligations.
2. **Predicate gate**: every admission condition must be evaluable from the current artifact state before reuse, or from bounded verifier output after reuse.
3. **Verifier gate**: labels must come from tests, checkers, proof obligations, database constraints, or equivalent tool outcomes; model agreement is not a label.
4. **Cost gate**: checking the selected predicates plus running required verifiers must be cheaper than asking a strong code or tool agent to solve the task again.

If any gate fails, the reasoning branch is not a research target for this direction. It should be recorded as prompt memory, ordinary retrieval, or a negative result rather than CMLP-002 evidence.

Research score:

- `8/10` as a secondary branch.
- Keep it if the first fixture shows lower false accept than retrieval / LLM judge and lower checking cost than rerunning a strong code agent.
- Demote it if the learned contract becomes a long hand-written guard, if checker cost dominates, or if labels require model self-judgment.

Shortest defensible claim:

> A reasoning trace can be reused only after it is reduced to a public verified dependency skeleton, and reuse should be admitted by learned artifact-state validity predicates rather than by retrieval score, prompt memory, or a model's agreement with the old rationale.

This is the strongest version of the branch. The weakest version is "store a past explanation and ask an LLM whether it still applies"; reject that version because it has no externally checkable state, no pre-action admission oracle, and no stable label beyond model self-judgment.

Current user-facing answer:

- If the user asks "what if it is reasoning?", answer: yes, but only after reasoning is erased into a public verifier-grounded skeleton.
- The reusable object is not the model's rationale. It is an artifact-bound dependency record: failure signature, source/schema/proof anchor, patch/query/proof slot, action, verifier obligation, and verifier result.
- The state is not a GUI page. It is the current artifact state: files, AST/schema, tests, tool versions, checker availability, local diffs, or proof assumptions.
- The research method remains reuse admission: learn the minimal live predicates under which the skeleton is still valid.
- The first evidence step must be one tiny code-agent fixture with verifier-backed labels and measured cost. Without this, the branch is only a framing note.
- The kill condition is cost and verification: if predicate checking plus verifier probes costs more than rerunning a strong code/tool agent, record `negative_result`.

## Domain Triage For "Reasoning"

Use this branch only when "reasoning" can be turned into externally checkable artifact state. The first filter is the domain, not
the model or prompt:

| Domain | CMLP-002 decision | Required public skeleton | Verifier label source | Main risk |
|---|---|---|---|---|
| Code repair | accept as first target | failing test -> AST/source anchor -> patch slot -> patch action -> regression verifier | pytest, type checker, linter, build | checker cost may exceed rerunning a strong code agent |
| SQL repair | accept later | failing query/result invariant -> schema anchor -> query rewrite slot -> validation query | database constraints, expected result checks | schema predicates may be expensive or dataset-specific |
| Proof/program synthesis | accept only with checker | failed obligation -> theorem/program anchor -> proof/program slot -> checker command | proof checker, compiler, unit tests | skeleton may become too domain-specific |
| Build/config repair | accept later | failure log -> config anchor -> edit slot -> build/test verifier | build command, config parser, integration test | environment predicates may dominate cost |
| Open-ended chat reasoning | reject | none that is stable enough | none, unless converted into a tool-verifiable artifact | labels collapse to model agreement |
| Reflection / explanation memory | reject | free-form rationale only | none | becomes prompt memory, not reuse admission |
| Hidden chain-of-thought reuse | reject | private text is not allowed | none | not inspectable, not publishable, not verifiable |

The code-repair row is the only current executable target. Other accepted domains should not be added until the parser fixture
has produced verifier-backed labels and a measured cost result. A domain that needs an LLM to decide whether the old reasoning
still applies fails the verifier gate.

## Reasoning-Specific Eight Elements

1. Actual reproducible example: start with one tiny code-agent bug-fix fixture where a targeted test fails, a stable function/AST anchor exists, a patch slot is identifiable, and `pytest <target>` plus one nearby regression test can verify the patch.
2. Prior-work gap: retrieval, memory, and coding-agent systems can recall similar fixes, but they usually do not learn which current repo predicates are necessary for a recalled fix skeleton to remain valid.
3. Deep principle: textual similarity estimates whether two failures look related; reuse admission needs `Pr(verifier obligations pass | current validity predicates)`.
4. Relation to OmniFlow: this is the non-GUI analogue of Function reuse admission. GUI predicates become artifact predicates; terminal/business-effect verification becomes tests, checkers, or proof obligations.
5. Code outline: keep the first implementation test-only with fixture repo states, deterministic predicate extraction, patch application, verifier execution, baseline scoring, and a sparse/tree/MDL selector.
6. Detailed tests: include original failure, changed failure signature, missing anchor, changed anchor semantics, ambiguous patch slot, missing verifier, missing regression scope, tool version change, and conflicting local changes.
7. Scalability: new codebases add predicate extractors and verifier adapters, but the reusable contract remains a small validity predicate set rather than a full agent runtime.
8. Future results: collect false accept, false reject, unsafe patch count, predicate count, checker cost, verifier time, and cost versus rerunning a strong code agent before any claim promotion.

## Candidate Seed Skeleton

The first seed should be a small code-agent bug-fix trace with this public structure:

```text
failure signature -> source anchor -> patch slot -> patch action -> verifier command -> verifier result
```

Example seed shape:

```json
{
  "skeleton_id": "code_fix_missing_guard_v0",
  "task_family": "bug_fix",
  "failure_signature": "targeted test fails with an expected exception or assertion",
  "source_anchor": "file path + function/class name + AST node kind",
  "patch_slot": "stable insertion/replacement region",
  "patch_action": "insert guard / update branch / adjust parser case",
  "verifier_obligations": [
    "targeted failing test now passes",
    "nearby regression test still passes",
    "static parser or type checker remains clean"
  ]
}
```

Do not use private model thoughts, hidden chain-of-thought, or free-form natural language rationales as the reusable object.

## Concrete First Fixture Plan

Use one deliberately small parser bug as the first CMLP-002 fixture. This fixture is still planned; it is not evidence until the
files exist and verifier output is recorded.

Planned seed:

```json
{
  "skeleton_id": "code_fix_parser_missing_guard_v0",
  "domain": "python_parser_bug",
  "planned_fixture_dir": "tests/preconditions/reasoning/fixtures/parser_missing_guard/",
  "source_anchor": "parser.py::parse_record",
  "failure_signature": "pytest test_rejects_empty_record fails because empty input is parsed instead of rejected",
  "patch_slot": "before the first split/parse operation in parse_record",
  "patch_action": "insert an empty-input guard that raises ValueError before parsing",
  "verifier_obligations": [
    "pytest tests/preconditions/reasoning/fixtures/parser_missing_guard/test_parser.py::test_rejects_empty_record",
    "pytest tests/preconditions/reasoning/fixtures/parser_missing_guard/test_parser.py::test_parse_valid_record"
  ]
}
```

This fixture is intentionally boring. It is useful because each validity predicate can be perturbed independently:

- changed failure signature: the failing test now complains about malformed delimiters rather than empty input;
- anchor removed: `parse_record` is renamed or moved;
- anchor semantics changed: `parse_record` already has an empty-input guard or delegates parsing elsewhere;
- patch slot ambiguous: multiple parser functions contain similar split/parse sites;
- verifier missing: the targeted failing test is absent;
- regression scope missing: the valid-record regression test is absent;
- tool version changed: pytest or Python version no longer matches the seed environment;
- conflicting local changes: unrelated edits overlap `parser.py` or the parser tests.

The planned runtime package now binds all nine `cases.jsonl` rows and `predicate_matrix.json` rows to this seed. This is a
specification tightening only: every row still has `label_source=planned_expected_only`, and the package has no fixture files,
patch application, verifier stdout, baseline scoring, learned selector output, or cost comparison.

First fixture acceptance requires two measured cost fields:

- `predicate_check_cost`: static anchor/test/diff inspection plus targeted verifier probes before or after patch application;
- `rerun_code_agent_cost`: wall-clock and token/model-call cost for asking a strong code agent to solve the same tiny bug from scratch.

If `predicate_check_cost >= rerun_code_agent_cost`, record the fixture as a negative result even if the validity contract has low false accept.

The first fixture must also compute the expected admission inequality:

```text
admit_reuse iff
  E[C_check(P*) + C_patch + C_verify + L_FA * Pr(false_accept | P*)]
  <
  E[C_rerun_code_agent + L_rerun_failure * Pr(rerun_failure)]
```

This is a gate, not a theoretical novelty claim. The branch is only interesting if the public-skeleton admission path is both
safer than retrieval/LLM-judge reuse and cheaper than asking the code agent to solve the same bug again.

## PABench Sub-Gate Interface Draft

If this branch is connected to PABench / Project Control, it must be connected as a sub-gate for one cached repair skeleton,
not as the project controller. The outer Project Control policy still chooses `delegate`, `verify`, `rollback`, `stop`, or
`escalate`; this gate only decides whether one previous skeleton is admissible.

Draft input:

```json
{
  "project_state": {
    "failure_class": "missing_guard_or_parser_empty_input",
    "affected_artifact_refs": ["parser.py", "test_parser.py"],
    "allowed_write_set": ["parser.py"],
    "rollback_available": true,
    "budget": {"max_checker_ms": null, "max_model_tokens": null}
  },
  "verifier_observation": {
    "failing_verifier": "test_rejects_empty_record",
    "failure_signature": "empty input parsed instead of rejected",
    "regression_verifier": "test_parse_valid_record",
    "observation_trust": "executable_checker"
  },
  "cached_skeleton": {
    "skeleton_id": "code_fix_parser_missing_guard_v0",
    "source_anchor": "parser.py::parse_record",
    "patch_slot": "before first split/parse operation",
    "patch_action": "insert empty-input guard",
    "verifier_obligations": [
      "test_rejects_empty_record",
      "test_parse_valid_record"
    ]
  }
}
```

Draft output:

```json
{
  "gate": "cmlp_reuse_admission.v0",
  "decision": "allow_reuse | reject | reobserve | repair_precondition | abstain",
  "selected_predicates": [
    "failure_signature_matches",
    "source_anchor_exists",
    "patch_slot_unique",
    "allowed_write_set_contains_patch",
    "targeted_verifier_available",
    "regression_verifier_available"
  ],
  "missing_or_failed_predicates": [],
  "cost": {
    "predicate_static_check_ms": null,
    "verifier_probe_ms": null,
    "patch_apply_ms": null,
    "total_admission_check_ms": null,
    "rerun_code_agent_wall_ms": null,
    "rerun_code_agent_tokens": null
  },
  "risk": {
    "false_accept_risk": null,
    "false_reject_risk": null,
    "label_source": "planned_only_not_evidence"
  },
  "controller_contract": {
    "may_apply_cached_skeleton": false,
    "must_run_verifier_after_apply": true,
    "must_not_update_project_graph_without_verifier_pass": true
  }
}
```

Promotion rule:

- `decision=allow_reuse` is not allowed while `label_source=planned_only_not_evidence`.
- A promoted `allow_reuse` row requires executable predicate values, patch application evidence, targeted verifier pass,
  regression verifier pass, and measured cost below the rerun baseline.
- A `reject`, `reobserve`, or `repair_precondition` row can be useful negative evidence if it prevents a false accept under the
  same verifier observation.
- ProjectGraph update remains owned by PABench after current verifier pass. This gate must never mark a Project Control
  transition resolved by itself.

## First PABench Sub-Gate Row Plan

Status: planned only. This is the first row-level plan for testing whether CMLP-002 is a useful PABench sub-gate rather than a
duplicate Project Control policy.

Planned row id:

```text
pabench_subgate_parser_missing_guard_original
```

The row tests one question:

```text
Given a current ProjectState and VerifierObservation for the parser missing-guard failure, should the controller admit the
cached skeleton code_fix_parser_missing_guard_v0?
```

Positive row conditions:

- the parser fixture exists and `test_rejects_empty_record` fails before applying the skeleton;
- `parser.py::parse_record` exists and still has one unambiguous pre-parse insertion slot;
- `allowed_write_set=["parser.py"]` contains the patch target and no other file is required;
- targeted and regression verifier obligations are both executable;
- after applying the skeleton, both `test_rejects_empty_record` and `test_parse_valid_record` pass;
- `C_check + C_patch + C_verify` is measured and lower than the rerun-code-agent baseline for the same fixture.

Negative rows / perturbations:

- changed failure signature: the failing verifier now reports a delimiter or type error instead of empty input acceptance;
- missing source anchor: `parser.py::parse_record` is renamed, moved, or deleted;
- ambiguous patch slot: multiple parser functions or multiple split/parse sites match the skeleton;
- missing targeted verifier: `test_rejects_empty_record` is absent or cannot run;
- missing regression verifier: `test_parse_valid_record` is absent or cannot run;
- outside allowed write set: applying the skeleton requires a file outside `allowed_write_set`;
- conflicting local diff: current uncommitted or fixture-local edits overlap the anchor or verifier files.

Baselines for the same rows:

- `always_reuse`: apply the cached skeleton whenever it is retrieved;
- `retrieval_score_reuse`: allow by lexical or embedding similarity to the seed failure;
- `llm_judge`: ask a model whether the skeleton applies, then score only by verifier-backed labels;
- `always_rerun_code_agent`: never admit reuse and use the rerun cost as the cost baseline;
- `hand_written_guard`: manually require the obvious anchor, verifier, and write-set predicates.

Metrics:

- false accept: gate allows reuse but targeted or regression verifier fails after apply;
- false reject: gate rejects a row where skeleton reuse would pass both verifiers;
- abstain rate: gate cannot decide because verifier or predicate evidence is missing;
- verifier pass after apply: targeted and regression verifier pass rate among allowed rows;
- total admission cost: static predicate check, verifier probe, patch apply, and post-apply verification time;
- rerun code-agent cost: wall time, model calls, and tokens or a stated local-agent proxy.

Promotion / kill conditions:

- no `allow_reuse` row can be promoted while labels are planned-only;
- the CMLP gate must not update ProjectGraph or mark the transition resolved;
- if `total_admission_cost >= rerun_code_agent_cost`, record `negative_result` even if the gate reduces false accepts;
- if `llm_judge` or `hand_written_guard` matches the gate at equal or lower cost, do not claim a learned-admission advantage;
- if the row requires a broader `delegate / verify / rollback / stop / escalate` choice, move it back to PABench instead of
  expanding CMLP.

Evidence template:

- The planned row must follow
  `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/reasoning_subgate_evidence_schema.md` before any result is
  promoted.
- Required evidence classes are ProjectState, VerifierObservation, cached skeleton, executable predicate values, admission
  decision, patch application, verifier results, baseline decisions, cost measurement, risk metrics, and promotion gate.
- Do not create a row that only records a decision. Without verifier-backed labels, patch evidence, baseline decisions, and
  admission-vs-rerun cost, the row remains planned or diagnostic.

## PABench Sub-Gate Execution Checklist

Status: planned only. This checklist maps the schema to exact future artifacts. It does not create a fixture, row JSON, patch,
verifier output, baseline score, or evidence label.

Future row root:

```text
runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/
  pabench_subgate_parser_missing_guard_original/
```

Future artifacts, if explicitly created later:

| Evidence class | Future artifact path | Promotion boundary |
|---|---|---|
| fixture source | `tests/preconditions/reasoning/fixtures/parser_missing_guard/parser.py` | Requires explicit confirmation before creation |
| fixture tests | `tests/preconditions/reasoning/fixtures/parser_missing_guard/test_parser.py` | Must fail targeted verifier before patch and keep regression verifier |
| public skeleton | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/cached_skeleton.json` | Must contain no hidden chain-of-thought |
| row envelope | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/row.json` | Must use `omniflow.preconditions.reasoning_subgate_row.v1` |
| ProjectState | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/project_state.json` | Controller-owned state only; not prompt context |
| VerifierObservation | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/verifier_observation.json` | Must include failing-before checker evidence before promotion |
| predicate values | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/predicate_values.json` | Values must be executable/static-derived, not planned expectations |
| patch evidence | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/patch_application.json` and `patch.diff` | Patch must stay inside `allowed_write_set` |
| verifier logs | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/verifier_results.jsonl` | Targeted and regression verifiers must be recorded with commands, exit codes, digests, and wall time |
| baseline decisions | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/baseline_decisions.json` | Must compare always-reuse, retrieval-score, LLM judge, always-rerun, and hand-written guard under the same labels |
| cost measurement | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/cost_measurement.json` | Must include admission cost and rerun-code-agent cost/proxy |
| risk metrics | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/risk_metrics.json` | Must be computed from verifier-backed labels |
| promotion gate | `runtime/evals/preconditions/reasoning_minimal_validity_conditions/subgate_rows/pabench_subgate_parser_missing_guard_original/promotion_gate.json` | `promotion_allowed=true` only after verifier, baseline, cost, and ProjectGraph-ownership gates pass |

Execution order for the first row:

1. Create exactly the parser fixture and public skeleton after explicit confirmation.
2. Run the targeted verifier before applying the skeleton and record failing-before evidence.
3. Extract predicate values from source, test inventory, tool metadata, and local diff state.
4. Make a reuse-admission decision for the cached skeleton only; do not choose `delegate`, `verify`, `rollback`, `stop`, or
   `escalate`.
5. Apply the skeleton only if the gate allows reuse and `allowed_write_set` contains the patch target.
6. Run targeted and regression verifiers after patch application.
7. Score required baselines using the same verifier-backed labels.
8. Measure admission cost and rerun-code-agent cost or a stated local-agent proxy.
9. Set row status to `evidence_available`, `not_learned_insufficient_data`, or `negative_result`; if any evidence class is
   missing, keep the row `planned_only` or diagnostic.
10. Leave ProjectGraph update and transition resolution to PABench after verifier pass.

## PABench Non-Overlap Audit

Status: planned guardrail only. This audit prevents the CMLP-002 / PABench sub-gate row from drifting into the outer Project
Control policy.

Non-overlap question:

```text
Does this row decide whether one cached skeleton is admissible,
or does it decide the project controller's next transition action?
```

The row remains CMLP only if all of the following hold:

- the input includes exactly one cached trace, Function, or verifier-backed skeleton under a current ProjectState and
  VerifierObservation;
- the output decision is limited to `allow_reuse`, `reject`, `reobserve`, `repair_precondition`, or `abstain`;
- selected predicates explain why that cached object is or is not currently admissible;
- cost and risk are reported as admission accounting, not as a controller policy score;
- `controller_contract` only states obligations for the outer controller, such as run verifier after apply or do not update
  ProjectGraph without verifier pass.

The row must be moved back to PABench, not expanded as CMLP, if any of the following happen:

- it chooses `delegate`, `verify`, `rollback`, `stop`, or `escalate`;
- it marks a Project Control transition resolved;
- it updates or folds ProjectGraph state;
- it schedules global verifier policy rather than reporting verifier obligations for the cached skeleton;
- it owns rollback, stop, escalation, agent selection, or budget allocation;
- it optimizes transition-quality metrics such as unsafe bypass, invalid edit, regression, verifier waste, stop correctness, or
  oracle regret as its primary action objective.

For the planned row `pabench_subgate_parser_missing_guard_original`, the allowed interpretation is:

```text
ProjectState + VerifierObservation + cached parser skeleton
  -> CMLP admission decision for that skeleton
  -> PABench still decides the outer controller action and any ProjectGraph update
```

If a future artifact cannot pass this audit, record it as a PABench controller row or a negative CMLP result. Do not make the
CMLP action space larger to absorb it.

## PABench Subsumption / Replacement Test

Status: planned guardrail only. This test decides whether the CMLP-002 sub-gate still contributes a separate research object
after being connected to PABench.

Test question:

```text
With the same ProjectState and VerifierObservation,
does changing only the cached skeleton change the correct admission decision?
```

The row remains CMLP only if the answer can be "yes" for verifier-backed reasons. Examples:

- `code_fix_parser_missing_guard_v0` is admissible because `parser.py::parse_record` exists, the patch slot is unique, the
  targeted and regression verifiers are available, the patch stays inside `allowed_write_set`, and admission cost is below
  rerun cost.
- A different cached parser skeleton is rejected under the same `ProjectState + VerifierObservation` because its source anchor
  is absent, its patch slot is ambiguous, its regression verifier is missing, or it needs a file outside the allowed write set.

In that case CMLP is measuring skeleton-specific validity, which PABench's outer action space does not answer by itself.

PABench subsumes the row if all cached skeleton choices collapse to the same outer decision, such as:

```text
current observation unresolved -> delegate semantic repair
current verifier pass -> stop
regression observed -> rollback
missing checker binding -> verify or escalate
```

Those are Project Control decisions. They should be recorded in PABench because the cached skeleton is not the causal variable.

CMLP does not replace PABench in the opposite direction. A CMLP gate can at most return:

```text
allow_reuse / reject / reobserve / repair_precondition / abstain
```

The outer controller still owns `delegate / verify / rollback / stop / escalate`, ProjectGraph update, rollback policy,
stopping policy, AgentProfile choice, budget allocation, and transition regret.

Minimum evidence for a non-subsumed CMLP/PABench row:

- at least two cached skeletons or one cached skeleton plus one perturbation that changes skeleton-specific predicates under the
  same outer `ProjectState + VerifierObservation`;
- verifier-backed labels for allow/reject/abstain decisions;
- baseline decisions for always-reuse, retrieval-score reuse, LLM judge, always-rerun, and hand-written guard;
- measured admission cost and rerun-code-agent cost;
- an explicit statement that ProjectGraph update happens only after PABench receives verifier pass evidence.

If those conditions are not met, keep the artifact planned/diagnostic or move it back to PABench.

## First Skeleton-Causality Contrast Plan

Status: planned guardrail only. This contrast is not fixture evidence, not a PABench result, and not a CMLP-002 promotion. It
exists to make the replacement test concrete before any parser fixture or sub-gate row is created.

Hold the outer state fixed:

```json
{
  "project_state": {
    "failure_class": "missing_guard_or_parser_empty_input",
    "affected_artifact_refs": ["parser.py", "test_parser.py"],
    "allowed_write_set": ["parser.py"],
    "rollback_available": true
  },
  "verifier_observation": {
    "failing_verifier": "test_rejects_empty_record",
    "failure_signature": "empty input parsed instead of rejected",
    "regression_verifier": "test_parse_valid_record",
    "observation_trust": "executable_checker"
  }
}
```

Then vary only the cached skeleton:

| Skeleton | Planned admission | Skeleton-specific reason |
|---|---|---|
| `code_fix_parser_missing_guard_v0` | `allow_reuse`, but only after fixture evidence exists | The future fixture must prove `parser.py::parse_record` exists, the empty-input guard slot is unique, the patch writes only `parser.py`, targeted and regression verifiers are available, and admission cost is below rerun cost. |
| `code_fix_parser_delimiter_rewrite_v0` | `reject` under the same outer state | This contrasting skeleton targets a delimiter normalization or parse-loop rewrite rather than an empty-input guard. It should be rejected if its anchor is not `parser.py::parse_record`, its patch slot is ambiguous, it requires writing parser tests or schema files outside `allowed_write_set`, or it lacks `test_parse_valid_record` as a regression obligation. |

The row is non-subsumed only if this contrast changes the CMLP admission result while the outer PABench inputs stay unchanged.
If both skeletons collapse to the same project-control decision such as "delegate a semantic repair" or "run verifier", the row
is PABench, not CMLP.

Future evidence requirements for this contrast:

- one row envelope that records the fixed `ProjectState + VerifierObservation`;
- two cached skeleton records with no hidden chain-of-thought;
- executable predicate values for both skeletons;
- verifier-backed labels for the allow/reject decisions;
- cost measurements for admission plus patch plus verification versus rerunning the code agent;
- an explicit note that ProjectGraph update remains owned by PABench after verifier pass.

## Pre-Fixture Readiness Audit

Before creating the first fixture, CMLP-002 must stay in a pre-fixture readiness state. The runtime manifest records this as
`pre_fixture_readiness_audit`.

The audit exists to prevent four common mistakes:

- treating the planned parser seed as a fixture;
- treating expected labels as verifier labels;
- treating the manual predicate hypothesis as a learned contract;
- treating the branch as reasoning-text cache rather than artifact-state validity learning.

Required pre-fixture gate states:

| Gate | Required current status | Required missing evidence |
|---|---|---|
| Materialization | `planned_only` | fixture source files, public skeleton JSON, failing-before verifier output |
| Predicate | `planned_only` | static/test/tool/diff-derived predicate values |
| Verifier | `missing` | pytest output, patch-applied flag, exit codes, stdout/stderr digest |
| Cost | `missing` | predicate/check/verifier/patch/rerun timing and token or local-agent proxy |

The only allowed transition out of this audit is one explicit first executable pass for
`code_fix_parser_missing_guard_v0`. Any attempt to add another reasoning domain, use LLM agreement labels, or reuse hidden
reasoning text should leave the package at `planned_skeleton_available` or produce a `negative_result`.

## First Executable Pass Protocol

The next pass is allowed to create fixture files only after explicit confirmation, because this would add a new test fixture tree.
Until then, the protocol is locked as a handoff plan rather than evidence.

The first executable pass must produce four classes of artifacts:

1. **Fixture artifacts**
   `parser.py`, `test_parser.py`, and a public skeleton JSON for `code_fix_parser_missing_guard_v0`.
   The seed state must fail `test_rejects_empty_record` before applying the skeleton and must keep
   `test_parse_valid_record` as a nearby regression obligation.
2. **Predicate artifacts**
   A per-case predicate table whose values are derived from static source inspection, test inventory, tool metadata, and local
   diff state. Planned or guessed predicate values cannot promote a row.
3. **Verifier artifacts**
   For every promoted row, record command, exit code, stdout/stderr path or digest, wall time, and whether the patch/action was
   applied. A `success` row requires failing-before evidence, patch-applied evidence, targeted verifier pass, and regression
   verifier pass.
4. **Cost artifacts**
   Record `predicate_static_check_ms`, `verifier_probe_ms`, `patch_apply_ms`, `total_admission_check_ms`,
   `rerun_code_agent_wall_ms`, and rerun model tokens or a clearly stated local-agent cost proxy.

The planned verifier commands are:

```bash
pytest tests/preconditions/reasoning/fixtures/parser_missing_guard/test_parser.py::test_rejects_empty_record
pytest tests/preconditions/reasoning/fixtures/parser_missing_guard/test_parser.py::test_parse_valid_record
```

These commands are not evidence until the fixture files exist and the command output is captured.

The first executable pass has exactly three acceptable outcomes:

- `evidence_available`: at least one verifier-backed success row and four verifier-backed negative/repair rows exist, all required
  baselines are scored, and cost is measured.
- `not_learned_insufficient_data`: verifier-backed rows exist but no compact validity contract is defensible.
- `negative_result`: checking plus verifier cost is not lower than rerunning a strong code agent, the selected contract degenerates
  into a long hand-written checklist, or an LLM judge matches the selected contract at equal or lower cost.

Do not broaden CMLP-002 after a negative result. A negative first fixture is useful because it tests whether reasoning-skeleton
reuse has a cost-positive niche at all.

## Candidate Predicate Vocabulary

The first version should keep predicates deterministic and inspectable.

| Predicate id | Meaning | Source | Expected role |
|---|---|---|---|
| `p_failure_signature_match` | current failure message, stack, or assertion matches the seed skeleton's failure family | test output / verifier log | necessary |
| `p_anchor_exists` | referenced file, symbol, and AST node still exist | static source inspection | necessary |
| `p_anchor_semantics_compatible` | local code shape still has the branch/dataflow assumed by the skeleton | AST / lightweight static analysis | likely necessary |
| `p_patch_slot_unambiguous` | exactly one safe insertion or replacement site is identified | static source inspection | necessary |
| `p_tool_versions_compatible` | test runner, language version, or framework behavior is compatible with the seed skeleton | tool metadata | likely necessary |
| `p_verifier_available` | targeted verifier commands can run in the current environment | command availability / test inventory | necessary for labels |
| `p_regression_scope_available` | at least one nearby negative/regression verifier exists | test inventory | necessary for unsafe-effect detection |
| `p_no_conflicting_local_changes` | the current artifact does not contain unrelated edits that invalidate the skeleton assumptions | git diff / file state | likely necessary |
| `p_patch_effect_local` | proposed patch effect is limited to the intended function/module boundary | static diff analysis | sufficient evidence, not always necessary |

## Perturbation Table

The labels below are expected labels. Actual labels must come from test/checker execution or a stronger offline oracle.

| Case id | State perturbation | Predicate intervention | Expected outcome | Why it matters |
|---|---|---|---|---|
| `r0_original_failure` | original failing repository state | all key predicates true | `success` after patch and verifier | positive control |
| `r1_failure_signature_changed` | failure now comes from a different assertion or stack | `p_failure_signature_match=false` | `reject` | prevents irrelevant trace reuse |
| `r2_anchor_removed` | target function or AST anchor is renamed, moved, or deleted | `p_anchor_exists=false` | `precondition_failure` | checks static anchor necessity |
| `r3_anchor_semantics_changed` | anchor exists but control/dataflow shape changed | `p_anchor_semantics_compatible=false` | `unsafe_or_wrong_effect` or `reject` | separates exact anchor from semantic compatibility |
| `r4_patch_slot_ambiguous` | multiple candidate insertion sites exist | `p_patch_slot_unambiguous=false` | `reject_or_reobserve` | prevents arbitrary patch placement |
| `r5_verifier_missing` | targeted test or checker is unavailable | `p_verifier_available=false` | `abstain` | prevents weak labels |
| `r6_regression_scope_missing` | failing test passes but no regression checker exists | `p_regression_scope_available=false` | `abstain_or_reject` | tests safety evidence boundary |
| `r7_tool_version_changed` | framework or language version changes behavior | `p_tool_versions_compatible=false` | `reject_or_reobserve` | checks environment dependency |
| `r8_conflicting_local_changes` | unrelated local edits overlap affected module | `p_no_conflicting_local_changes=false` | `reject_or_reobserve` | avoids stale patch transfer |

## Labels

Use these labels consistently:

- `success`: patch/action is applied and all verifier obligations pass.
- `precondition_failure`: reuse is blocked before patch/action because required predicates are false.
- `unsafe_or_wrong_effect`: patch/action is applied but verifier or regression evidence fails.
- `reject`: admission policy refuses reuse before patch/action.
- `reject_or_reobserve`: admission policy requires a bounded static inspection or fresh verifier run before reuse.
- `abstain`: label cannot be trusted because verifier evidence is unavailable.

LLM agreement with the skeleton is not a success label.

## Baselines

The first reasoning experiment should compare:

1. **Always reuse skeleton**
   Apply the skeleton whenever retrieval finds it.
2. **Always rerun code agent**
   Never reuse; this is the cost upper bound and false-accept lower bound.
3. **Retrieval-score threshold**
   Allow reuse if embedding or lexical similarity exceeds a fixed threshold.
4. **Exact anchor match**
   Allow only if the file/symbol/AST anchor exactly matches.
5. **Hand-written guard**
   Use manually selected obvious predicates.
6. **LLM judge**
   Ask a model whether the skeleton applies, but score it only by verifier-backed labels.
7. **Learned minimal validity conditions**
   Learn a compact predicate set from perturbation-labeled outcomes.

## Metrics

Primary:

- false accept rate: policy allows reuse but verifier outcome is not `success`;
- false reject rate: policy rejects a state where reuse would pass verifier obligations;
- unsafe/wrong-effect count;
- success rate among allowed reuses.

Secondary:

- predicate count in the learned validity contract;
- serialized contract length;
- static/checker cost before patch/action;
- saved online model tokens versus rerunning a code agent;
- verifier time versus rerun time;
- generalization to unseen source perturbations.

## First Minimal Learners

Start with interpretable models:

1. Decision tree with max depth 2-4.
2. Sparse logistic regression over binary predicates.
3. MDL-style selector:

```text
score(P') =
  C_check(P')
  + C_patch * allow_rate(P')
  + C_rerun_code_agent * reject_rate(P')
  + L_FA * false_accept_rate(P')
  + L_FR * false_reject_rate(P')
  + lambda_complexity * |P'|
```

The expected contribution is not a new code repair algorithm. The contribution is learning an admission contract for when a previous verifier-grounded reasoning skeleton remains valid.

## Expected First Contract

A plausible manual hypothesis may look like:

```text
allow_reuse(code_fix_missing_guard_v0) iff
  p_failure_signature_match
  and p_anchor_exists
  and p_anchor_semantics_compatible
  and p_patch_slot_unambiguous
  and p_verifier_available
  and p_regression_scope_available
  and p_no_conflicting_local_changes
```

This is a hypothesis. Do not report it as learned until perturbation-labeled outcomes exist.

## Implementation Boundary

First implementation should be test-only:

```text
tests/preconditions/reasoning/
  fixtures/
  perturbation_cases.json
  predicate_table.json
  evaluate_reasoning_validity.py
```

Do not modify agent runtime or store hidden reasoning. If this later enters a product, it should store only public skeleton fields and verifier obligations.

## Minimum Evidence Package

The first evidence package should write:

```text
runtime/evals/preconditions/reasoning_minimal_validity_conditions/
  manifest.json
  cases.jsonl
  predicate_matrix.json
  outcomes.json
  baseline_report.json
  learned_contract.json
  summary.md
```

Expected labels must be marked as planned until tests/checkers produce outcomes.

## Promotion Gate

Do not promote this into any claim source until:

- at least one positive and four negative/repair perturbations are labeled;
- every `success` row has verifier command output;
- false accept improves over retrieval-score threshold and LLM judge, or false reject improves at fixed false accept;
- checking cost is lower than rerunning the code agent for the target case;
- the learned contract is smaller than a hand-written exhaustive guard;
- no hidden chain-of-thought is stored, consumed, or evaluated.

## First Evidence Step

The first useful next step is not a broad reasoning-memory benchmark. It is one tiny verifier-backed fixture:

```text
repo snapshot + failing pytest output
  -> public skeleton fields
  -> predicate table
  -> patch/action application
  -> targeted verifier and regression verifier output
  -> false accept / false reject scoring
```

If this fixture cannot make checker cost lower than rerunning a code agent, the reasoning branch should be recorded as a negative result rather than expanded.

The first fixture should be a small missing-guard or parser/schema bug, not a broad refactor. It is acceptable only if it provides:

- a seed failing test output and nearby regression verifier;
- a public skeleton JSON with no hidden chain-of-thought;
- at least five repo-state perturbations that independently alter failure signature, anchor existence, anchor semantics, patch-slot ambiguity, verifier availability, regression scope, tool version, or conflicting local changes;
- actual patch/action execution and verifier output for every promoted label;
- baseline decisions for always-reuse, always-rerun, retrieval threshold, exact-anchor, hand-written guard, LLM judge, and learned validity conditions;
- static/checker cost compared against rerunning a strong code agent.
