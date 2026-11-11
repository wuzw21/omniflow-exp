# PaperAgent Rules (NSDI)

本文档只定义 **NSDI 主线** 的写作规则。

## 适用范围

- 当前论文主线：`UTG_papers/nsdi/`
- 当前系统口径：**GUI Agent cache system**
- `2026-04-18` 之前的旧论文根控制文件、旧 archive、旧 reference 已删除

## 必须先读

1. `agents.md`
2. `UTG_papers/README.md`
3. `CORE_CONCEPTS.md`
4. 相关 `docs/reference/core_concepts/*.md`
5. `UTG_papers/nsdi/core_claims.md`
6. `UTG_papers/nsdi/eval_matrix.md`
7. 当前要写的目标文件
8. 对应代码真源

## Source Of Truth

- 仓库级概念总索引与最新版词表：`CORE_CONCEPTS.md`
- 仓库级概念板块真源：`docs/reference/core_concepts/*.md`
- claim 真源：`UTG_papers/nsdi/core_claims.md`
- 评测真源：`UTG_papers/nsdi/eval_matrix.md`
- 英文真源：`UTG_papers/nsdi/overleaf/`
- 最终图：`UTG_papers/nsdi/figures/`
- 文件治理账本：`UTG_papers/nsdi/MAINLINE_FILE_CONSOLIDATION.md`
- 中文故事大纲与 reviewer debate：`UTG_papers/nsdi/story_outline_cn.md`
- Agent 可查询数据库：`docs/agent_context/paper_mainline_file_database.json`

固定规则：

- 不再恢复 `UTG_papers/core.md`、`soul.md`、`VERSION.md` 这类旧 root control 文件。
- 不再恢复旧的 `mobicom/overleaf_upload` 或旧 archive 口径。
- 旧中文迭代稿、research 笔记、NeurIPS 稿、storybook、重复图和 PPT workflow 已清理，不再作为当前写作输入。
- 论文术语变更时，必须先改 `CORE_CONCEPTS.md` 的最新版词表。

## 写作规则

1. 先冻结相关 concept board，再冻结 claim。
2. 先冻结 claim，再写正文。
3. 先冻结 eval matrix，再写结果段落。
4. Review / debate / 说法回复先改 `UTG_papers/nsdi/story_outline_cn.md`，不要直接改 LaTeX 正文。
5. 先写 `Design / Implementation / Evaluation`，再写 `Introduction / Abstract`。
6. 若代码与旧文档冲突，以代码为准。
7. 若数据还没冻结，正文只能写边界和计划，不能伪造数字。
8. 不要默认引入最新版词表之外的新论文术语；若必须引入，先向用户说明必要性，并先更新 `CORE_CONCEPTS.md` 后再写正文。
9. 论文方向评估必须包含 GPT/Codex 版本更新替代性判断：凡是会被一次普通 GPT/Codex 能力升级完全取代的方向，都必须明确告知用户并淘汰，不得包装成主线论文方向。
10. 论文方向必须先收束为具体问题，而不是停留在大方向名称；每个候选至少说明 failure mode、系统边界、长期性、可验证指标和与现有基础的差异，否则不进入主线讨论。
11. 当前新增优先探索方向是 **agent efficiency / token-cost reduction**，但必须从 method 开始，不做 benchmark-first，不把 prompt/skill 质量当核心变量，不依赖单个 API 或某次模型效果。具体方法默认围绕“结构化 agent runtime state 到 tokenized context 的调度”：token-class ledger、runtime state materialization policy、tool/observation/evidence/context token 的边际效用，以及 correctness / verification 保持约束。
12. 论文正文默认加载 `PaperSpine3` 写法：先给出核心对象或核心主张，再解释机制 / 转换过程，最后明确相对已有工作的优势。避免 “not X but Y” / “不是……而是……” 这类对比填充句，改用直接的论文化陈述句。

## 固定表述

- 系统定位：`GUI Agent cache system`
- 读者主概念只保留 3 个：`Function` / `function asset`、`UDEG node`、`run_log -> function asset`
- 命中语义：`cache hit skips online reasoning for the matched segment`
- 其它正式术语一律以 `CORE_CONCEPTS.md` 的最新版词表为准

术语压缩规则：
- `cache-first`、`cache hit/miss`、`Function recall` 只作为 `Function` 的运行状态。
- `control-aware execution`、`terminal verify`、`route-only bridge` 只作为执行安全约束。
- `co-evolution` 只作为 `run_log -> function asset` 闭环产生的系统效果。
- OOB / AndroidWorld / MCP / provider 只作为接入或评测环境。
- C1-C5 是证据组织方式，不是正文里并列抛给读者的五个新概念。

## System Design 分节边界

- 第二节 `Background and Motivation` 写静态 UI transition graph / UTG 的问题定义：
  node = UI tree snapshot，edge = action transition；动态环境中的两个核心变量是当前位置
  `Pr(v_t | O_t)` 和下一步效果 `Pr(Delta_t | O_t, a, v_t)`。
- 第三节 `System Design` 只写 OmniFlow 自己的设计目标和系统结构，不展开其他工作的缺陷。
- 第三节开头 `Core Design Goals` 固定按三条线写：
  表示层（UDEG node / page vector）、执行层（`Function`）、数据流（`run_log -> function asset`）。
  route-only bridge、control-aware execution、terminal verify 只能在执行层内部展开，不新增为平行主概念。

## 禁止事项

- 不把系统写成新的 foundation planner
- 不把旧 `Path` 口径重新写回主叙事
- 不把任何 pre-`2026-04-18` 旧稿当成当前 source of truth
- 不恢复已删除的草稿、研究笔记、storybook 或重复 figure 来作为新入口
- 不写“全程零成本”“覆盖所有 GUI 任务”这类不可防守表述
- 不在未说明必要性的情况下引入词表外的新概念或新术语
- 不在没有冻结数据前先写最终摘要和结果结论
- 不把 prompt 调参、上下文压缩、通用 tool-use 选择、模型本身 GUI grounding 能力提升后会自然消失的问题，包装成需要长期系统论文投入的方向
- 不把“agent infra”“safety”“efficiency”“verification”这类大标签直接当论文方向；必须改写成具体、可复现、可评测的问题后再讨论
- 不把 agent efficiency 写成单纯 benchmark、prompt compression、skill 写法优化或 API 成本统计；必须提出可迁移 method，并证明不依赖某个模型版本或 provider 行为。

## 最小检查

- `rg -n "TODO|XX|\\?" UTG_papers/nsdi/overleaf/*.tex`
- `rg -n "mobicom|Trajectory Cache|soul.md|core.md" UTG_papers/nsdi`

## 交付格式

- 先说明改了哪个 NSDI 主线文件
- 再说明这些改动依赖的代码或数据
- 最后说明哪些结果仍未冻结
