# Paper Workspace

当前论文主线已经重置为 **NSDI**，系统定位统一为：

> **OmniFlow is a GUI Agent cache system.**

## NSDI Mainline

当前只认以下文件为论文主线入口：

1. `CORE_CONCEPTS.md`
   - 仓库级概念总索引与最新版词表
2. `docs/reference/core_concepts/*.md`
   - 按主题冻结的概念板块真源
3. `UTG_papers/nsdi/core_claims.md`
   - 论文级 claim 真源
4. `UTG_papers/nsdi/eval_matrix.md`
   - 实验内容、指标、数据源、脚本、输出文件真源
5. `UTG_papers/nsdi/overleaf/`
   - 英文真源
6. `UTG_papers/nsdi/figures/`
   - 最终图与图规格
7. `UTG_papers/nsdi/MAINLINE_FILE_CONSOLIDATION.md`
   - 当前 paper 文件压缩与删除账本，不是 claim 真源
8. `docs/agent_context/paper_mainline_file_database.json`
   - 给其他 agent 查询的 paper 文件数据库
9. `UTG_papers/nsdi/exploration/2026_paper_roadmap.html`
   - 独立探索章节，只做 2026 论文路线规划；不是 claim / eval / overleaf 真源
10. `UTG_papers/nsdi/story_outline_cn.md`
   - 中文故事大纲与 reviewer debate 工作文档；先改这里，不直接改正文；不是 claim / eval / overleaf 真源
11. `UTG_papers/nsdi/story_outline_debate_digest_cn.md`
   - 从主 OmniFlow workspace 同步的 001-008 功能 / 整体 debate 摘要；不是 claim / eval / overleaf 真源
12. `UTG_papers/nsdi/exploration/minimal_live_preconditions/`
    - 独立研究探索工作区，记录 "minimal live preconditions / reuse admission" 方向；不是 claim / eval / overleaf 真源
13. `UTG_papers/nsdi/exploration_records/minimal_live_preconditions/`
    - 独立探索提交记录目录，记录 minimal live preconditions 方向的 commit-like research log

固定规则：

- 代码是真相，论文只能跟随当前代码实现。
- 系统定位、cache-first、control、vector、writeback、OOB、AndroidWorld、多用户口径先冻结在相关 concept board。
- 术语变化时，必须先更新 `CORE_CONCEPTS.md` 里的最新版词表，再改 claim / 正文 / 图。
- claim 只能写进 `core_claims.md` 里已经冻结的内容。
- 实验只能写进 `eval_matrix.md` 已经登记的数据链和输出链。
- 探索章节可以记录 future paper directions，但其中任何新主张或结果进入正文前，必须回写到 `core_claims.md` / `eval_matrix.md`。
- review / debate / 说法回复先维护 `story_outline_cn.md`，只有用户明确要求进入正文时再改 `overleaf/`。
- 草稿、中文迭代稿、旧 research 笔记、NeurIPS 旧稿、storybook 和重复图已清理出活跃 workspace；不要再作为 source of truth 恢复。

## Archive Rules

`2026-05-14` aggressive cleanup 后，旧 root controls、旧 archive、旧 reference、旧 research notes、旧中文协同稿、旧 NeurIPS 稿和重复 figure 已从 `UTG_papers/` 删除。

当前规则：

- 不再保留 pre-NSDI 的第二套论文真源。
- 任何新的写作、图、表、实验登记，都只允许落到 `UTG_papers/nsdi/`。
- 若后续需要补研究背景，必须先写入当前 `overleaf/`、`core_claims.md`、`eval_matrix.md` 或 `figures/` 的具体需求；不要重建平行 `research/` 草稿库。
- 新 research exploration 可以落在 `UTG_papers/nsdi/exploration/<topic>/`，对应独立探索记录可以落在 `UTG_papers/nsdi/exploration_records/<topic>/`；两者默认只是 future direction，进入正文前必须先回写 concept board / `core_claims.md` / `eval_matrix.md`。
