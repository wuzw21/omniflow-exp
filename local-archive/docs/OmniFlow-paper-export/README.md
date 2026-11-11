# OmniFlow Paper Workspace

This repository is the standalone paper workspace for OmniFlow.

Current mainline:

> OmniFlow is a GUI Agent cache system with co-evolution through trajectory reuse.

The active paper source is the NSDI line under `UTG_papers/nsdi/`. Legacy
MobiCom material is kept only as read-only archive context.

## Start Here

1. Read `UTG_papers/README.md`.
2. Read `UTG_papers/paperagent.md`.
3. Read `CORE_CONCEPTS.md`.
4. Read `docs/reference/core_concepts/PAPER_NSDI27.md`.
5. Read `UTG_papers/nsdi/core_claims.md`.
6. Read `UTG_papers/nsdi/eval_matrix.md`.
7. For reviewer debate or wording decisions, update
   `UTG_papers/nsdi/story_outline_cn.md` first.
8. Edit only the matching source under `UTG_papers/nsdi/overleaf/` or
   `UTG_papers/nsdi/figures/`.

## Layout

| Path | Purpose |
|---|---|
| `CORE_CONCEPTS.md` | Repository-level concept vocabulary and current term map. |
| `docs/reference/core_concepts/` | Concept boards that freeze paper-facing system semantics. |
| `docs/agent_context/` | Paper context, file database, and related agent skills. |
| `UTG_papers/README.md` | Paper workspace source-of-truth declaration. |
| `UTG_papers/paperagent.md` | Rules for writing and editing the paper. |
| `UTG_papers/nsdi/core_claims.md` | Claim source of truth. |
| `UTG_papers/nsdi/eval_matrix.md` | Evaluation source of truth. |
| `UTG_papers/nsdi/overleaf/` | LaTeX source for the current English paper. |
| `UTG_papers/nsdi/figures/` | Figure sources, prompts, and rendered image assets. |
| `UTG_papers/nsdi/story_outline_cn.md` | Chinese story outline and reviewer debate ledger; update this before editing paper text. |
| `UTG_papers/nsdi/story_outline_debate_digest_cn.md` | Compact 001-008 feature/system debate digest synced from the OmniFlow workspace. |
| `UTG_papers/nsdi/reviewer_concerns_analysis.md` | Reviewer-risk and open-issue ledger. |
| `UTG_papers/nsdi/MAINLINE_FILE_CONSOLIDATION.md` | File governance and cleanup ledger. |
| `UTG_papers/nsdi/exploration/` | Independent research exploration notes; not claim or result truth. |
| `UTG_papers/nsdi/exploration_records/` | Commit-like records for exploration-only work. |
| `UTG_papers/mobicom/` | Read-only legacy draft archive, not a current source of truth. |
| `EXPORT_MANIFEST.md` | What was exported from the OmniFlow monorepo and what was excluded. |

## Local Sync Policy

This repository is maintained through one local staging clone:

```text
/Users/wuzewen/Projects/Omni/omniflow-paper-release
```

The OmniFlow development workspace remains at:

```text
/Users/wuzewen/Projects/Omni/OmniFlow
```

For future updates, sync paper-only material into this staging clone, commit
here, and push this repository. Do not create a new copy for every update.

## Current Truth Rules

- Claims must be frozen in `UTG_papers/nsdi/core_claims.md` before they appear
  in the paper text or figures.
- Evaluation results must be registered in `UTG_papers/nsdi/eval_matrix.md`
  before they appear as paper numbers.
- `Function`, `UDEG node`, `run_log -> function asset`, `cache-first`, and
  `control-aware execution` are the current paper objects and semantics.
- Historical terms such as `Path`, `trajectory cache`, `executable memory`, and
  `Vector Automaton` must not override the current vocabulary.
- Legacy `UTG_papers/mobicom/` files are archive-only and should not be copied
  back into the NSDI mainline without first translating them to the current
  claim and evaluation contracts.

## Build

Use the root build wrapper:

```bash
bash build_pdf.sh
```

It delegates to `UTG_papers/nsdi/overleaf/build.sh --pdf` and writes
`UTG_papers/nsdi/omniflow.pdf`.

The Overleaf packaging script is still available directly:

```bash
cd UTG_papers/nsdi/overleaf
bash build.sh
```
