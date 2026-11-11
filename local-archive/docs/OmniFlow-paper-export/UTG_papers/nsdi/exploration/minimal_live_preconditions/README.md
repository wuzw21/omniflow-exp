# Minimal Live Preconditions Research Workbench

Status: Draft
Last Updated: 2026-05-24
Scope: Future research exploration only. This directory is not a claim source for the current NSDI paper until a claim is promoted through `CORE_CONCEPTS.md`, `UTG_papers/nsdi/core_claims.md`, and `UTG_papers/nsdi/eval_matrix.md`.

## One Sentence

We study when an agent can safely reuse a past successful trace by learning the minimal live preconditions that make the trace still valid in the current state.

Chinese working version:

> 我们研究 agent 经验复用里的核心判定问题：一段历史成功轨迹在当前状态下什么时候仍然适用。

## Why This Is Narrow Enough

This is not a new all-agent system. It extracts one hard problem from OmniFlow:

- OmniFlow already has `Function`, `PageVectorSet`, `run_log -> function asset`, cache-first execution, and control-aware execution.
- The unresolved research question is reuse admission: a retrieved Function can look similar to the current state, but similarity does not prove that replay is admissible.
- The proposed work focuses only on learning the smallest state predicate set that separates safe reuse from unsafe reuse.

The first scope should remain GUI/OmniFlow because the repo already has real traces, Functions, page matching, control verification, and live Android evidence. A reasoning-trace extension is useful, but should stay secondary until the GUI version is experimentally sharp.

## Relation To PABench / Project Control

This exploration is adjacent to PABench, but it is not the same work.

Current verdict after reading PABench's own Project Control framing:

- GUI CMLP remains a separate reuse-admission problem because its input is a retrieved GUI trace/Function and its primary
  failure is stale or unsafe replay.
- Reasoning CMLP is not a separate project-control paper. It is at most a sub-gate inside PABench for one cached repair
  skeleton.
- If a reasoning row asks "what should the project controller do next?", it is PABench.
- If a reasoning row asks "is this specific cached skeleton still admissible under the same ProjectState and
  VerifierObservation?", it can remain CMLP.
- The decisive variable is the cached object. If removing or swapping the cached trace/Function/skeleton does not change the
  correct decision, PABench subsumes the row.

PABench / Project Control asks:

```text
Given ProjectState + VerifierObservation,
which control action should the runtime take next?
```

Its action space is project-level control:

```text
delegate / verify / rollback / stop / escalate
```

CMLP asks a narrower admission question:

```text
Given one past trace, Function, or verifier-backed skeleton,
is that specific reusable object still valid in the current state?
```

Its action space is reuse admission:

```text
allow_reuse / reject / reobserve / repair_precondition
```

The two lines share verifier-backed labels, cost accounting, and false-accept discipline, but they optimize different
decisions. A future Project Control runtime may call a CMLP gate before using a cached repair skeleton. In that interface,
PABench still owns the outer policy over `delegate / verify / rollback / stop / escalate`, while CMLP only returns whether a
specific cached skeleton is admissible under current predicates, cost, and false-accept risk.

Do not broaden CMLP into a project-level controller. If the research object becomes "what should the project controller do
next?", it belongs to PABench. If the object remains "when is this old trace or skeleton safe to reuse?", it remains CMLP.

### Subsumption Test Against PABench

Before adding any new CMLP/PABench row, run this conceptual replacement test:

```text
Same ProjectState + same VerifierObservation + different cached skeletons
  -> different admission decisions because skeleton-specific predicates differ
```

CMLP remains distinct only if the cached object is a necessary variable. A good CMLP row should show that two candidate
traces or skeletons under the same outer project-control state can have different outcomes because one still satisfies its
anchor, action slot, verifier obligations, write-set binding, and cost gate while the other does not.

PABench subsumes the row if the cached object can be removed from the input and the correct decision is still just a
project-level `delegate / verify / rollback / stop / escalate` action. In that case, the row should be recorded as PABench
Project Control evidence, not expanded into CMLP.

CMLP cannot replace PABench in the other direction. Even a perfect CMLP gate only answers whether one reusable object is
admissible. It does not own outer action choice, ProjectGraph update, rollback, stopping, escalation, or controller regret.

Implication for CMLP-002: the parser-skeleton plan is useful only if it proves skeleton-specific admission under fixed
`ProjectState + VerifierObservation`. Without an alternative cached skeleton or a perturbation that changes skeleton-specific
predicates, the result is just a PABench transition-control row or a parked protocol note.

## Core Claim

> A reusable agent trace should be guarded by learned minimal live preconditions rather than fixed similarity thresholds, hand-written checks, or LLM applicability judgments.

The claim is intentionally about the admission layer, not about making a stronger planner.

## Formal Object

Given:

- a successful trace or Function `tau`;
- a current live state `s`;
- candidate predicates `P = {p1, ..., pn}` extracted from observations, action targets, parameters, permissions, app state, tool schema, and verifier availability;
- perturbations `do(p_i = false)` or state edits that invalidate candidate predicates;
- an execution outcome `Y in {success, failure, unsafe, abstain}`.

Learn a compact predicate subset `P*` such that:

```text
allow_reuse(tau, s) iff all predicates in P* hold in s
```

The optimization target is high reuse success and low unsafe false acceptance under a complexity penalty:

```text
P* = argmin_P'  Loss(Y, reuse(tau, s | P')) + lambda * Complexity(P')
```

The useful version is not "more checks". It is the minimum sufficient condition set that avoids both:

- false accept: execute a stale or incompatible trace;
- false reject: discard a reusable trace and pay full replanning cost.

The decision rule must be cost-aware. A predicate contract is useful only when its admission path has lower expected cost and
risk than rerunning the host agent:

```text
admit_reuse iff
  E[C_check(P*) + C_reuse + L_FA * Pr(false_accept | P*)]
  <
  E[C_replan + L_replan_failure * Pr(replan_failure)]
```

For selector diagnostics, score candidate contracts with:

```text
score(P') =
  C_check(P')
  + C_reuse * allow_rate(P')
  + C_replan * reject_rate(P')
  + L_FA * false_accept_rate(P')
  + L_FR * false_reject_rate(P')
  + lambda_complexity * |P'|
```

This is an accounting rule, not a claimed new mathematical contribution. It prevents the method from turning into an expensive
guard layer whose checks cost more than simply observing again or rerunning the agent.

## Method Sketch

1. **Predicate extraction**
   Extract candidate predicates from trace and current state:
   page family, app package, route node, target element presence, actionability, overlay absence, parameter binding, account/permission state, terminal verifier availability, and tool/schema fields.

2. **Counterfactual perturbation**
   Generate bounded perturbations:
   remove a button, rename a label, insert a dialog, move a target, change list ordering, clear a field, switch app route, change tool schema, hide a permission, or remove a terminal verifier.

3. **Grounded execution**
   Run the historical Function or trace against original and perturbed states. Label success using terminal/business-effect evidence, not just low-level action dispatch.

4. **Minimal precondition learning**
   Learn `P*` using sparse logistic models, decision trees, MDL-style complexity penalties, or causal feature selection over intervention outcomes.

5. **Admission runtime**
   At recall time, evaluate `P*` on the current state and choose:
   `allow_reuse`, `reject_and_replan`, `reobserve`, or `repair_precondition`.

## Required Eight Elements For The Main Claim

1. **Actual reproducible example**
   Use `ContactsAddContact` as the initial case. A Function can create a contact only if the Contacts app is reachable, the create-contact UI is available, required fields can be bound, overlays do not intercept input, and the Contacts provider can verify the created row. The current repo already has bounded Contacts replay evidence and start-node precondition failures that show why "just replay" is insufficient.

2. **What prior work misses**
   Existing GUI-agent reuse, memory, and skill systems often focus on retrieving or generating a reusable procedure. They usually do not isolate the admission question: which live state predicates are necessary for the retrieved procedure to remain executable now?

3. **Deep mechanism**
   Page similarity estimates `Pr(state looks compatible)`. Reuse admission needs `Pr(trace succeeds | predicates hold, current state)`. These are different variables. Counterfactual perturbations let us estimate which predicates causally affect reuse success instead of treating all matched UI features as equally important context.

4. **Relation to existing OmniFlow components**
   `run_log -> Function` creates the reusable trace. `PageVectorSet` provides candidate page and element evidence. Control-aware execution provides effect and terminal evidence. This research adds a learned admission contract between recall and execution.

5. **Code outline**
   Future implementation should be isolated first:
   - `tests/preconditions/fixtures/`: trace/state/perturbation cases.
   - `tests/preconditions/perturbations.py`: bounded UI/tool-state perturbation generation.
   - `tests/preconditions/predicate_extractors.py`: deterministic candidate predicate extraction.
   - `tests/preconditions/learn_minimal_preconditions.py`: sparse or tree-based learner.
   - `tests/preconditions/evaluate_admission.py`: baselines, false accept, false reject, and cost metrics.
   Only after the method works should a small runtime adapter be considered for `src/utg/execution/control/`.

6. **Detailed tests**
   Minimum tests should include correct page, wrong page, similar page, overlay, missing field, moved target, changed label, missing verifier, and stale account/permission cases. Each case needs an executable trace, a predicate vector, an outcome label, and at least one baseline comparison.

7. **Scalability**
   New GUI apps add predicate extractors and perturbation templates. Tool/MCP/code-agent extensions can reuse the same formulation if they provide observable predicates and executable oracles. The method scales by keeping the contract small and host-specific observation adapters separate.

8. **Future test results to collect**
   Track reuse success rate, unsafe false accept rate, false reject rate, number of predicates in `P*`, perturbation coverage, execution cost versus replanning, and robustness under unseen UI perturbations. These results must be added to `eval_matrix.md` before any paper claim is promoted.

## Baselines

- Always replay.
- Always replan.
- PageVectorSet or embedding threshold only.
- Exact UI tree match.
- Hand-written guard rules.
- LLM judge: "is this trace reusable in this state?"
- Control-only postcondition verification without learned precondition admission.

## Metrics

- Reuse success rate.
- Unsafe false accept rate.
- False reject rate.
- Predicate count and contract complexity.
- Cost saved versus full replanning.
- Extra observe/check cost.
- Terminal/effect verification pass rate.
- Generalization to unseen perturbation families.

## Reasoning Trace Variant

The reasoning version is viable, but only if the reusable object is not hidden chain-of-thought. The durable object must be a verifier-grounded reasoning skeleton:

```text
claim / premise / tool result / action / verifier obligation / verifier result
```

Working title:

> Learning Minimal Validity Conditions for Reusable Code-Agent Reasoning Traces

One sentence:

> We study when a past verifier-grounded reasoning skeleton remains valid in a new artifact state by learning the minimal live validity conditions that make its claims, tool calls, and verifier obligations still sound.

The right first domain is code, not open-ended chat reasoning. A reusable code-agent reasoning skeleton might say: a failing test exposes a missing branch; the relevant function has a stable AST anchor; the patch inserts a guard before a specific return; the required verifier is `pytest <case>` plus a targeted static check. The reusable object is the public dependency graph of evidence and obligations, not the model's private thought process.

Formal mapping:

```text
tau_r = verifier-grounded reasoning skeleton
s     = current artifact state, e.g. repo, schema, tests, tool versions
P     = candidate validity predicates over s
Y     = verified outcome: pass, fail, unsafe, abstain

allow_reuse(tau_r, s) iff all predicates in P* hold in s
```

Candidate predicates should be externally checkable:

- failure signature still matches;
- AST/schema anchor still exists;
- tool/API version is compatible;
- referenced file/function/table/proof obligation is present;
- patch or query slot can be rebound without ambiguity;
- verifier command is available and still exercises the claimed behavior;
- negative/regression verifier does not fail.

The research question is the same as GUI reuse admission: which small subset of live predicates is enough to allow reuse without false acceptance? The method is useful only if checking `P*` is cheaper than re-solving the task and if verifier-backed outcomes show lower false accept or false reject than similarity, retrieval score, or LLM judgment.

Current verdict after the reasoning variant question:

- Generic reasoning is not a stable research object here.
- Code / SQL / proof / build / schema reasoning can be a stable object only after the private reasoning is erased into a public skeleton with verifier obligations.
- The reusable unit is not "the model's rationale"; it is a state-bound dependency skeleton plus a learned admission contract.
- The first experiment should therefore be a tiny code-agent repair trace, not a benchmark of open-ended reasoning reuse.

Research positioning:

- Score: 8/10 as a secondary branch, not the first paper-sized target.
- It is defensible when the domain has public verifiers: code, SQL, proof, build systems, schema migration, or config repair.
- It is weak for open-ended chat reasoning because the state predicates and success labels are not externally checkable.
- It is harder than the GUI branch to make cost-positive because a strong code agent plus tests is already a strong baseline; the method must prove that predicate checking plus skeleton reuse is cheaper than rerunning that agent.
- The contribution is not a better reasoner. It is a learned admission contract for when a previously verified reasoning skeleton still applies.
- If the measured first fixture shows `C_check + C_verifier + C_patch >= C_rerun_code_agent`, the correct result is
  `negative_result`, even if the selected predicates reduce false accepts.

Kill condition for reasoning:

- If the trace is private CoT, discard it.
- If labels come from LLM self-judgment rather than tests/checkers/tool outcomes, discard it.
- If the validity predicates cannot be evaluated before or after reuse, discard it.
- If the verifier cost is higher than simply asking a strong code agent to solve the task, record a negative result.

## Kill Conditions

Stop or demote the direction if:

- learned predicates collapse to page-family similarity only;
- perturbations are synthetic noise and do not change real reuse outcomes;
- an LLM judge baseline matches the method under the same oracle and cost;
- checking `P*` costs more than replanning for most target tasks;
- terminal/effect oracles are too weak to label success and failure;
- the method cannot reduce false accept without exploding false reject.

## Current Decision

Recommended first paper-sized target:

> Learning Minimal Live Preconditions for Reusable GUI Agent Traces

Score: 9/10 as a small, concrete research problem. It is narrower and more defensible than an all-agent self-evolving system, and it is less likely to be invalidated by a model upgrade because the core object is host-state validity, not model reasoning quality.

Secondary target:

> Learning Minimal Validity Conditions for Reusable Code-Agent Reasoning Traces

Score: 8/10. Keep it as the non-GUI extension only after it is grounded in a tiny failing-test fixture. It should not displace the GUI protocol until it has verifier-backed labels, baseline scoring, and a measured checking-cost advantage over rerunning a code agent.

## Current Protocol

The first concrete protocol is:

- `contacts_protocol.md`: ContactsAddContact predicate vocabulary, perturbation table, labels, baselines, metrics, and promotion gate.

This protocol is the next bridge from idea to evidence. It must produce predicate/outcome data before any runtime integration or paper-claim promotion.

The secondary reasoning protocol is:

- `reasoning_protocol.md`: verifier-grounded code-agent reasoning skeleton vocabulary, perturbation table, labels, baselines, metrics, and promotion gate.

This protocol is parked behind the GUI protocol. It is useful only if it produces verifier-backed outcomes and stores no hidden chain-of-thought.

The first registered experiments are:

- `CMLP-001 - Contacts Minimal Live Preconditions`, tracked in `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`.
- `CMLP-002 - Reasoning Trace Minimal Validity Conditions`, tracked in `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/experiment_register.md`.

The first runtime evidence package schema is frozen in:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/evidence_schema.md`.

The planned CMLP-002 / PABench sub-gate row evidence schema is frozen in:

- `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/reasoning_subgate_evidence_schema.md`.

It applies only to the planned row `pabench_subgate_parser_missing_guard_original`. It is not runtime evidence and does not
promote CMLP-002 beyond `planned_skeleton_available`.

The first runtime evidence package now exists at:

- `runtime/evals/preconditions/contacts_minimal_live_preconditions/`

It is marked `partial`. Seven rows are promoted:
`c1_launcher_wrong_start` is execution-backed, while `c0_correct_create_flow`, `c2_contacts_home_reachable`,
`c4_overlay_blocks_target`, `c6_args_missing_or_invalid`, `c7_terminal_verifier_missing`, and
`c8_similar_non_contacts_page` are offline-oracle labeled.
This is still not `evidence_available`: two labels remain `planned_expected_only`,
only deterministic baselines are partially scored, the c0 positive row proves bounded create-flow terminal success rather than direct start-node
admission because the successful sidecar rows skipped the start-node precondition, c2 proves route/form reachability rather than
terminal contact creation, c4 proves an overlay-present reobserve/repair state rather than a terminal outcome, c6 is only a
schema-level admission label, c7 is a verifier-boundary abstain label rather than success/failure evidence, and c8 is a cross-app XML reject label rather than a live replay result. The c1 execution row proves wrong-start precondition failure from the live launcher: replay stopped with
`EXEC_START_NODE_PRECONDITION_FAIL`, no semantic actions ran, and terminal verification reported the target contact absent.
The separate `c0_direct_admission_probe_20260524.json` artifact is diagnostic only: it attempted direct admission from
HOME/launcher with valid arguments, failed before device actions with `EXEC_START_NODE_PRECONDITION_FAIL`
(`aw_node_8225087450 -> aw_node_22653dacf0`), and found no target contact. The required start node
`aw_node_22653dacf0` is the recorded launcher entry node whose first sequence action is `open_app`, while
`aw_node_8225087450` is a Contacts form node in the Function. This is node/matcher instability evidence, not a
terminal-verified positive row.
An explicit selector diagnostic has been run on these seven promoted rows and marks
`learned_contract.json.status=not_learned_insufficient_data`: apparent perfect predicate subsets either exploit unknown values
or depend on `p_start_node_match` while the only positive row lacks direct start-node evidence. The c1 execution upgrade,
c4 overlay promotion, and c7 verifier boundary do not supply the missing positive evidence; the c0 direct-admission diagnostic
also does not fill it because it is another pre-action precondition failure.

The secondary reasoning planned-only runtime skeleton now exists at:

- `runtime/evals/preconditions/reasoning_minimal_validity_conditions/`

It is also marked `planned_skeleton_available`. It is not evidence yet: it has no code-agent fixture, no patch execution, no verifier command output, no learner, and no scored baselines. Its next useful step is a tiny verifier-backed bug-fix fixture, not a broader reasoning-memory system.
