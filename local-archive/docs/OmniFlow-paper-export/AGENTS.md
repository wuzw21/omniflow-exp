# Agent Instructions

This repository contains only OmniFlow paper material.

Before editing, read:

1. `README.md`
2. `UTG_papers/README.md`
3. `UTG_papers/paperagent.md`
4. `CORE_CONCEPTS.md`
5. `docs/reference/core_concepts/PAPER_NSDI27.md`
6. `UTG_papers/nsdi/core_claims.md`
7. `UTG_papers/nsdi/eval_matrix.md`
8. The target file being changed

Rules:

- Do not treat `UTG_papers/mobicom/` as current source. It is archive-only.
- Do not introduce claims that are absent from `UTG_papers/nsdi/core_claims.md`.
- Do not introduce result numbers that are absent from `UTG_papers/nsdi/eval_matrix.md`.
- Do not restore excluded third-party PDF or template bundles unless the user explicitly asks.
- For reviewer debate, story framing, or claim wording discussions, update
  `UTG_papers/nsdi/story_outline_cn.md` first. Do not edit LaTeX paper text
  unless the user explicitly asks to promote an outline item into the paper.
- Keep the current vocabulary: `Function`, `PageVectorSet`,
  `run_log -> function asset`, `cache-first`, and `control-aware execution`.
- If a concept changes, update the relevant concept board before editing paper
  text or figures.
- Long-term writing style: load `PaperSpine3` for paper prose.  Each section
  should state the core object or claim, explain the mechanism or conversion,
  and state the advantage over prior work.  Avoid contrastive filler such as
  "not X but Y" / "不是……而是……"; write direct affirmative paper statements.
