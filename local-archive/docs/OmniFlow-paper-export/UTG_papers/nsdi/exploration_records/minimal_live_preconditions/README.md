# Minimal Live Preconditions Exploration Records

Status: Active
Last Updated: 2026-05-24
Scope: Independent exploration record for the minimal live preconditions research direction.

## Purpose

This directory keeps a separate, commit-like research log for the exploration thread around:

- learned minimal live preconditions for reusable GUI traces;
- minimal validity conditions for verifier-grounded reasoning traces;
- future variants of reuse admission that stay narrow and testable.

This is not a replacement for `project_iterations/VERSION_LOG.md` or `project_iterations/PART_ITERATION_LOG.md`. Those repo-wide logs still must be updated for every delivery.

## Record Files

- `commit_log.md`: chronological commit-like entries for each exploration step.
- `experiment_register.md`: planned and completed perturbation suites.
- `evidence_schema.md`: required output schema for the first Contacts evidence package.
- This README: current accepted / rejected / parked research decisions for the exploration thread.
- Future optional files:
  - `decision_register.md`: accepted/rejected research decisions.
  - `reading_queue.md`: papers to verify before citation.

## Current Decision Snapshot

These decisions summarize the current research boundary. The detailed history remains in `commit_log.md`; this section is the
stable handoff view for future agents.

| ID | Status | Decision | Consequence | Revisit Trigger |
|---|---|---|---|---|
| `CMLP-D001` | accepted | The research object is reuse admission, not all-agent self-evolution. | Keep proposals centered on whether one historical trace, Function, or skeleton is currently admissible. | Revisit only if there is execution evidence for a broader controller-level method that is not already PABench. |
| `CMLP-D002` | accepted | GUI Contacts remains the primary branch; reasoning remains secondary. | Prioritize CMLP-001 terminal-verified positive evidence and cost fields before broadening CMLP-002. | Revisit if the parser fixture produces verifier-backed labels, baselines, and a cost-positive result. |
| `CMLP-D003` | accepted | Reasoning means public verifier-grounded artifact skeletons, not hidden chain-of-thought. | Reject hidden CoT, reflection memory, explanation reuse, and LLM-agreement labels as research objects. | Revisit only for domains with public skeletons and executable verifier labels. |
| `CMLP-D004` | accepted | CMLP and PABench are separated by action space. | CMLP returns `allow_reuse / reject / reobserve / repair_precondition / abstain`; PABench owns `delegate / verify / rollback / stop / escalate`. | Revisit if a row demonstrably needs both decisions; then record it as PABench calling a CMLP sub-gate. |
| `CMLP-D005` | accepted | A CMLP/PABench row must pass the cached-object subsumption test. | With fixed `ProjectState + VerifierObservation`, changing only the cached skeleton must change admission for skeleton-specific reasons. | Move the row to PABench if cached skeleton choice is irrelevant. |
| `CMLP-D006` | accepted | CMLP-001 is partial and has no learned contract yet. | Do not call selector diagnostics learned; continue seeking a terminal-verified positive with cost fields. | Revisit after a direct-admission or route-repair positive row is promoted with terminal provider verification and timing. |
| `CMLP-D007` | parked | CMLP-002 has a planned parser skeleton but no fixture evidence. | Do not create parser fixture, subgate rows, patch evidence, verifier logs, or costs without explicit confirmation. | Revisit when the user explicitly pulls CMLP-002 forward as one parser fixture. |
| `CMLP-D008` | accepted | Every next slice must pass the research maturity gate before expansion. | Classify each step as `explore`, `collect_evidence`, `park`, or `negative_result`; do not continue if labels are ungrounded, the cached object is not causal, or checking plus reuse is cost-dominated. | Revisit only if the gate blocks a slice that has stronger terminal/verifier evidence and measured rerun/replan cost. |
| `CMLP-D009` | accepted | The current next-slice classification is mostly parked. | Contacts-home candidate, alternative positive row, PageVectorSet, LLM judge, CMLP-002 fixture, and PABench sub-gate row all stay `park`; controller-action rows are `negative_result` for CMLP and should move to PABench. | Revisit only when explicit candidate-asset or parser-fixture creation is confirmed, or when a named executable cached object has grounded labels and cost fields. |
| `CMLP-D010` | accepted | PABench subsumes CMLP-002 unless the cached skeleton is a causal variable. | GUI CMLP stays separate; reasoning CMLP can only be a PABench sub-gate that admits or rejects one cached skeleton under fixed `ProjectState + VerifierObservation`. | Revisit only if the first parser row shows skeleton-specific admission variation with verifier-backed labels and lower admission cost than rerun. |
| `CMLP-D011` | accepted | The first CMLP-002/PABench contrast must vary cached skeleton only. | Use fixed parser `ProjectState + VerifierObservation`, compare `code_fix_parser_missing_guard_v0` against `code_fix_parser_delimiter_rewrite_v0`, and require different admission for skeleton-specific anchor, patch-slot, write-set, verifier-obligation, or cost reasons. | Revisit only after parser fixture evidence can populate predicate values, verifier labels, baselines, and cost fields for both skeleton choices. |
| `CMLP-D012` | accepted | Planned-only expansion is frozen. | Do not add more CMLP planned rows, schemas, skeleton families, or contrasts unless they replace an existing item with stronger evidence. The only Go paths are explicit Contacts candidate materialization, CMLP-001 terminal/cost positive evidence, explicit CMLP-002 parser fixture creation, verifier-backed skeleton-causality evidence, or grounded baseline scoring. | Revisit after one Go path produces evidence, or if the direction is closed as `negative_result`. |

## Entry Format

Use this structure for each exploration entry:

```text
## <exploration_id> - <date>

Type:
Files:
Decision:
Reason:
Validation:
Next:
```

Rules:

- `exploration_id` is not a git commit hash. Use a stable local id such as `mlp-20260524-001`.
- Keep claims separate from evidence. If an experiment has not run, label it as a plan.
- Do not cite papers from memory. Add them to a reading queue until verified.
- Do not promote anything from this directory into NSDI text without updating the official claim/eval sources first.

## Linked Research Workbench

Main exploration documents:

- `UTG_papers/nsdi/exploration/minimal_live_preconditions/README.md`
- `UTG_papers/nsdi/exploration/minimal_live_preconditions/agenda.md`
