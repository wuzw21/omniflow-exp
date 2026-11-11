# Export Manifest

Export date: 2026-05-24

Source workspace:

```text
/Users/wuzewen/Projects/Omni/OmniFlow
```

Target repository:

```text
https://github.com/wuzw21/omniflow
```

## Included

The export preserves the current OmniFlow paper source paths so existing
cross-references remain readable.

| Source path | Included as | Notes |
|---|---|---|
| root wrapper | `build_pdf.sh` | Stable repository-root command for compiling `UTG_papers/nsdi/omniflow.pdf`. |
| `CORE_CONCEPTS.md` | `CORE_CONCEPTS.md` | Current vocabulary and concept-board index. |
| `docs/reference/core_concepts/` | `docs/reference/core_concepts/` | Paper-facing concept boards. |
| `docs/agent_context/UTG_PAPERS_NSDI_CONTEXT.md` | same path | NSDI paper context for agents. |
| `docs/agent_context/paper_mainline_file_database.json` | same path | Machine-readable active/deleted paper file ledger. |
| `docs/agent_context/AGENT_PROMPT_SKILL_MATRIX.md` | same path | Agent, prompt, and skill context matrix. |
| `docs/agent_context/skills/paper-mainline-consolidation/` | same path | Paper file governance skill. |
| `docs/agent_context/skills/paper-svg-figure/` | same path | Figure editing and SVG workflow skill. |
| `docs/agent_context/skills/nsdi-paper-eval/` | same path | NSDI evaluation workflow skill. |
| `docs/agent_context/skills/minimal-precondition-research/` | same path | Exploration-only minimal live precondition / reuse-admission workflow skill. |
| `UTG_papers/README.md` | same path | Paper workspace entry. |
| `UTG_papers/paperagent.md` | same path | Paper writing rules. |
| `UTG_papers/nsdi/` | same path | Current NSDI source, figures, PDF, zip, details, exploration, and reference index. |
| `UTG_papers/nsdi/story_outline_cn.md` | same path | Chinese story outline and reviewer-debate ledger; update before editing paper text. |
| `UTG_papers/nsdi/story_outline_debate_digest_cn.md` | same path | Compact feature/system debate digest synced from the OmniFlow workspace. |
| `UTG_papers/nsdi/exploration/minimal_live_preconditions/` | same path | Exploration-only notes for minimal live preconditions / reuse admission. |
| `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/` | same path | Commit-like record log and schemas for the exploration-only line. |
| `UTG_papers/mobicom/` | same path | Legacy OmniFlow draft archive, excluding third-party ACM template files. |

## Excluded

The export intentionally leaves out files that are not OmniFlow-authored paper
source material or would make the new repository noisy.

| Excluded path | Reason |
|---|---|
| `UTG_papers/nsdi/papers/*.pdf` | Third-party reference PDFs; `REFERENCE_INDEX.md` and `refs.bib` are included instead. |
| `UTG_papers/mobicom/acmart-primary/` | Third-party ACM template bundle, not OmniFlow paper content. |
| `.DS_Store` | Local macOS metadata. |
| runtime outputs outside `UTG_papers/nsdi/` | Evaluation artifacts belong in the original OmniFlow runtime workspace unless explicitly promoted to paper source. |
| source code, tests, scripts, and device assets | This repository is paper-only. |

## Mainline Boundary

Current source of truth:

- `UTG_papers/nsdi/core_claims.md`
- `UTG_papers/nsdi/eval_matrix.md`
- `UTG_papers/nsdi/overleaf/`
- `UTG_papers/nsdi/figures/`
- `docs/reference/core_concepts/*.md`
- `docs/agent_context/paper_mainline_file_database.json`
- `UTG_papers/nsdi/story_outline_cn.md` for reviewer/story debate before
  paper text promotion

Archive-only:

- `UTG_papers/mobicom/`
- `UTG_papers/nsdi/exploration/**`
- `UTG_papers/nsdi/exploration_records/**`
