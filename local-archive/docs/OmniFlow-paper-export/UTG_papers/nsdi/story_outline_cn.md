# OmniFlow NSDI Story Outline

本文件维护当前 NSDI 主线的故事大纲与 reviewer debate 决策。历史 debate ledger 保留在 Git 历史中；当前文件从活跃证据开始继续维护。

## 2026-07-14：AITW 中相似 instruction 是否能支撑 Function 复用动机

### Reviewer 可能追问

- AITW 约 30K 个 instruction 是否真的代表 30K 个互不相关的任务？
- instruction 相似是否等于两条 trace 可以对应，甚至等于 Function 可以直接重放？
- 相似阈值是否人为选择，结论对阈值是否敏感？
- AITW 的五个 subset 如何合并，跨 subset 重复字符串是否会改变分母？

### 冻结回答

- AITW 的官方五个 subset 共报告 30,378 个 subset-level unique prompt entries；合并后有 30,165 个全局不同字符串，差值来自 213 个额外的跨 subset occurrence。
- 参数模板归一化后，11,052/30,378（36.4%）的 prompt 至少有一个模板相同的对应项。
- 使用 `BAAI/bge-small-en-v1.5`，并把“模板相同或 cosine similarity >= 0.98”作为严格标准，同时过滤相反意图，22,834/30,378（75.2%）的 prompt 至少有一个相似对应项。
- 将语义阈值放宽到 0.95 时，比例为 26,386/30,378（86.9%）。正文采用 0.98 的严格结果，0.95 只作为附录 sensitivity。
- 75.2% 是 prompt-level similarity estimate，也是 Function-reuse candidate 的 workload upper bound。它说明 recurring intent 在 AITW 中普遍存在，不能证明 paired traces 可执行等价、页面状态兼容或端到端重放成功。

### 正文与附录分工

- `overleaf/background.tex`：在 AITW 30K instruction 事实之后，用一个段落报告 36.4%、75.2% 和 86.9% sensitivity，并立即限定为候选上界。
- `overleaf/appendix.tex`：说明五个 subset 的分母、参数 masking、embedding、contradiction filter、统计公式、阈值 sensitivity 和各 subset 的严格结果。
- `UTG_papers/nsdi/evidence/aitw_prompt_similarity_20260714/paper_values.json`：保存论文数字和口径快照。

### AITW trace-pair 泛化实验边界

AITW 适合继续构造跨 trace 匹配实验：给定一条 source trace，系统从候选集中召回同一任务模板的 target trace，并对齐可复用片段。正例应来自同 app、不同 episode、相同参数化任务模板；hard negative 应覆盖同 app 近邻任务、相同动词不同对象和显式相反意图。数据划分应至少包含 leave-one-template-out，并可增加 leave-one-app-out。指标报告 Function Recall@K、trace/segment alignment accuracy 和 abstain rate。

AITW 缺少可恢复的当前 app/account/server state，因此该协议只支撑离线 `instruction -> Function candidate` 与 trace correspondence。AndroidWorld 或真机上的 control-aware execution 与 terminal/effect verifier 继续承担可执行重放成功的证据。

### 禁止升级的说法

- 不把 75.2% 写成 AITW replay success、Function cache hit rate 或 production workload coverage。
- 不把 prompt 相似写成相同 action sequence、相同 PageVectorSet、相同 terminal effect 或跨 app Function 等价。
- 不把 0.95 sensitivity 写成 headline；正文使用 0.98 的严格口径。
- trace-pair 实验未运行和冻结前，不在正文报告 Recall@K、alignment accuracy 或执行成功数字。
