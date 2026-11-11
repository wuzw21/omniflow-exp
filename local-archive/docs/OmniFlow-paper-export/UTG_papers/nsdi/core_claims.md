# OmniFlow NSDI Core Claims

系统的一句话定义：

> **OmniFlow is a GUI Agent cache system with co-evolution through trajectory reuse.**

本文档是当前论文的唯一 claim 真源。凡是正文、图、表、摘要中的核心主张，都必须先在这里冻结。

在修改本文件前，必须先读 `CORE_CONCEPTS.md` 和相关 `docs/reference/core_concepts/*.md`。如果系统定位、主对象、cache-first/control/vector/OOB/AndroidWorld、多用户或写回语义发生变化，先改对应 board，再回到这里。

## In Scope

- GUI agent goal 的 cache-first cache 命中
- `run_log -> function asset` 的资产化链路
- `UDEG node` 驱动的隐私安全状态表示与共享
- control-aware deterministic execution
- 统一 run event stream 与 case-study observability
- **共同进化**：一个用户的轨迹可被其他用户复用，降低集体探索成本

## Out Of Scope

- 新的 foundation planner
- 全任务全流程都零成本
- 尚未冻结数据支撑的大规模跨设备结论
- 持久化共享原始 XML / screenshot

## Claim C1: Cache-First Function Execution

**主张**

OmniFlow 会先把自然语言 goal 映射到可复用 `Function`；命中后直接执行函数资产，而不是每一步都重新让上层 agent 规划。

**代码真源**

- `src/utg/runtime/utg_runtime.py`
- `src/utg/execution/recall.py`
- `src/utg/core/models.py`

**当前允许写进论文的表述**

- cache hit 对应的是 `Function` 命中
- 命中段可以跳过在线 reasoning
- 内部小模型 JSON 路径只做两件事：replay 前的公开参数填充，以及 replay 失败时的当前步普通 UI action 修复；它不是模型可见 Function 工具
- miss 进入现有 online UI-action 路径；模型不能自己拼 `function_id` 或隐藏 Function tool
- 从宿主 agent 视角看，`Function` 可以描述为 OmniFlow 为黑盒 app 合成的、可重放的 tool-like API surface；它来自成功 GUI 执行，不依赖开发者提供的 app API
- `Function` 是当前论文的核心缓存资产；Function store / Function Store 可以写成由成功 GUI 执行合成的 API-like replay primitive 集合
- UDEG 可以写成 Function store 自然诱导出的状态索引 / 稀疏执行结构：UDEG node 记录 Function 的可调用状态簇，Function edge 记录可执行的任务效果和转移证据
- 与传统 UTG 的关系可以写成结构升级：OmniFlow 保留 replay 所需的 UTG 职能（状态定位、action-compatible state family 合并、转移证据组织、route-only bridge / start-state guard），但核心不是完整 app graph 或全局图规划，而是把历史 transition 升级成带参数、source context、action transfer 和 terminal/effect verify 的可调用 Function
- 关键语义 `Function` 只有在 `start_node_id` 前置条件满足后才能执行；当前 node
  不一致时，只允许先用 route-only `goto` / navigation 边最短路桥接到起点，再执行语义
  `Function` 并做 terminal / effect verify
- UTG 中的边需要区分定位边和语义边：`open_app`、back/home、tab switch、dialog dismiss
  是 goto/navigation 边；表单输入、create/save 等改变任务状态的边不是路由捷径

**当前不能写的表述**

- `recall(...)` 一次返回整条多 function 计划
- OmniFlow 取代了宿主 agent
- 参数填充和失败步补齐是两个模型可见工具
- replay 失败后把剩余目标重新交给 agent 重新规划整条任务
- 只要找到关键语义 `Function`，就能从任意页面直接执行成功
- 在全图最短路中允许 semantic/input/save 边作为 `goto` bridge
- OmniFlow 构建或覆盖完整 mobile app UTG 的所有功能
- UDEG 是主要算法贡献，而 Function 只是普通图边

**当前证据入口**

- `UTG_papers/nsdi/eval_matrix.md`
- `tests/compile/README.md`
- `runtime/evals/key_function_ablation/key_function_goto_bridge_report.md`
- `tests/integrations/test_utg_api.py`

### AITW Prompt Recurrence Evidence (Workload Characterization)

**允许写进论文的表述**

- AITW 官方五个 subset 共报告 30,378 个 subset-level unique prompt entries；参数模板归一化后，11,052/30,378（36.4%）至少有一个模板相同的对应项。
- 采用 `BAAI/bge-small-en-v1.5`，以参数模板相同或 cosine similarity $\geq 0.98$ 为严格标准，并过滤相反意图后，22,834/30,378（75.2%）至少有一个相似 prompt；阈值 $\geq 0.95$ 的 sensitivity 为 26,386/30,378（86.9%）。
- 该结果只能写成 prompt-level similarity estimate 和 Function-reuse candidate workload upper bound，用于说明 recurring mobile intents 的规模。
- AITW 可以进一步用于离线 `instruction -> Function candidate` 与 trace correspondence 泛化评测；执行成功仍需 AndroidWorld 或真机上的 control-aware execution 和 terminal/effect evidence。

**当前不能写的表述**

- 75.2% 的 AITW traces 可以直接重放或已经完成 Function 对应。
- prompt 相似等于 action sequence、PageVectorSet、current state、argument binding 或 terminal effect 等价。
- 该统计是 OmniFlow cache hit rate、端到端成功率或 production workload coverage。

**当前证据入口**

- `UTG_papers/nsdi/evidence/aitw_prompt_similarity_20260714/paper_values.json`
- `UTG_papers/nsdi/eval_matrix.md`
- `overleaf/appendix.tex` 的 AITW prompt-similarity protocol

### Claim C1a: Key Function With Route-Only Goto Bridge

**主张**

一个任务路径可以由少量关键语义 `Function` 承担主要 task effect，但它们不是“从任意状态可执行”的脚本。
执行算法必须先匹配当前 UTG node；若当前 node 与关键 `Function.start_node_id` 不一致，则用
route-only shortest path over goto/navigation edges 桥接到起点，再执行关键语义 `Function`，最后用
terminal / business-effect 证据验收。

**代码真源**

- `src/utg/core/graph_ops.py`
- `src/integrations/utg_api/_bundle_graph.py`
- `src/utg/execution/executor.py`
- `src/utg/runtime/utg_runtime.py`

**八项要素**

1. 实际例子：ContactsAddContact 在 `emulator-5554` 上，完整函数插入 `Key Path` 成功；语义尾段从
   Contacts home 插入 `Tail Only` 成功；同一语义尾段从 launcher 且关闭 controller 时 terminal verify
   失败，Contacts provider 未出现 `No Bridge`。
2. 之前工作未注意到的点：底层 action execution 返回 true 不等于任务成功；wrong-start semantic replay
   可以执行完动作但没有业务效果。
3. 深刻原理：goto 边只满足页面/应用位置前置条件，语义边改变任务状态；把语义边放进路由最短路会混淆
   “到达可执行页面”和“完成任务”。
4. 与论文组件关联：page-vector / node match 识别当前节点；UTG graph search 生成 goto bridge；
   control-aware executor 执行桥接和语义函数；terminal verifier 验收业务效果。
5. 代码大纲：`find_shortest_path_edges(..., route_only=True)` 搜索桥接边；
   `is_route_bridge_function(...)` 过滤可路由 Function；provider `shortest_path_between_nodes(...)`
   只返回 route-only path。
6. 详尽测试：Contacts live 消融报告、provider shortest-path 单测、semantic edge 不能当 bridge 的回归测试、
   graph/goto/offline replay 门禁。
7. 可扩展性：新 app 只需给 Function 写清 edge role / metadata，并补 terminal/effect verifier；
   关键语义 Function 仍可复用，只是执行前必须满足起点。
8. 未来添加的测试结果：跨 app route-edge classifier accuracy、shortest-path bridge success rate、
   wrong-start semantic-only failure rate、goto+semantic terminal success rate。

## Claim C2: Privacy-Safe UDEG Node Representation

**主张**

OmniFlow 用 `UDEG node` 表示可共享的 GUI 状态；每个 node 的内部编码是 page vector。向量化链路会哈希文本、剥离 raw text，只保留可用于匹配和共享的匿名特征。

**代码真源**

- `src/utg/assets/embedding/vector_set.py`
- `src/integrations/utg_api.py`

**当前允许写进论文的表述**

- XML 可以被转成 privacy-safe `UDEG node`，内部编码为 page vector
- shared state store 的主对象应是 UDEG node，而不是 raw XML
- UI element vector 的用途是把元素相似度计算变成紧凑的 pairwise matrix scoring，支持低开销 action transfer
- anchor transfer 的观察基础：目标元素本身可能移动或缺少属性，但页面中常存在相似且稳定的局部结构和全局标志，例如 repeated card templates、search bars、tab bars、section headers、persistent navigation controls
- `UDEG node` 主要负责候选状态 / 候选簇召回；目标元素迁移由 element / anchor mapping 与 control-aware verify 收口
- action transfer 可以写成 coarse-to-fine：`UDEG node` 先召回候选 state family，element-similarity matrix 产生一对一 source-target anchor 对；同一 source 行或 target 列不能被重复用作多个 anchor；anchor 对沿 `source action -> source anchor -> target anchor -> candidate target` 路径投票；投票分数结合 action 相似度、anchor 相似度，以及树距离和几何距离的一致性；聚合投票后用 support 和 margin gate 选择 click 或 fallback
- page family / UTG node 的合并标准应是 action-compatible，而不是单纯视觉相似或 exact XML 相同
- fixed replay 和 selector replay 可写成单点证据 baseline：fixed replay 迁移历史坐标，selector replay 迁移历史目标元素自身属性；它们不显式建模目标元素与上下文 anchor 的关系。
- `UI tree exact replay` 与 `UI tree + anchor replay` 是强 baseline，必须在 RQ3 中显式对比

**当前不能写的表述**

- 系统当前长期共享 raw XML
- 向量化已经被大规模跨设备实验完全验证
- 视觉相似页面必然属于同一个可执行簇
- 单个已知元素的重放一定优于保留 raw UI tree 和 target element 的 baseline
- page vector 的 top-1 exact page id 命中率本身就是最终执行复用收益

**当前证据入口**

- `UTG_papers/nsdi/figures/F03_xml_to_pagevectorset_prompt.md`
- `src/utg/assets/embedding/README.md`
- `src/integrations/utg_api.py`
- `UTG_papers/nsdi/eval_matrix.md` RQ3

## Claim C3: Run Logs Become Reusable Cache Assets

**主张**

OmniFlow 能把 canonical `run_log` 写回成可复用的图资产和 `Function` 资产，使缓存来自真实执行，而不是单独维护一套 prompt memory。

**代码真源**

- `src/integrations/utg_api.py`
- `src/utg/core/models.py`
- `src/utg/runtime/utg_runtime.py`

**当前允许写进论文的表述**

- canonical run log 可以进入 UTG asset store
- 资产是 cache-backed store，不是静态脚本库
- 资产化链路是当前系统主线的一部分

**当前不能写的表述**

- 所有导入 run 都自动变成 ready-to-ship 大规模共享资产
- 当前 repo 已经有成熟的线上 eviction / promotion 结果

**当前证据入口**

- `tests/fixtures/offline_replay/`
- `runtime/demo/multi_function_demo/`
- `src/integrations/utg_api.py`

## Claim C4: Control-Aware Execution And Observability

**主张**

命中后的执行并不是裸 replay；它仍受 shared control plane 和统一 run event stream 约束，并能导出 case-study 级 timeline 结果。

**代码真源**

- `src/utg/runtime/utg_runtime.py`
- `src/utg/execution/executor.py`
- `src/utg/log/run_logger.py`
- `src/integrations/utg_api.py`

**当前允许写进论文的表述**

- 命中执行仍有 control-aware verification / recovery
- 统一事件流是论文时序图和运行统计的真源
- provider 可以直接导出 timeline payload

**当前不能写的表述**

- 当前仓库已经冻结了完整的大规模 event-stream benchmark
- timeline page 本身就是论文主贡献

**当前证据入口**

- `runtime/paper_figures/runlog_timeline/`
- `scripts/nsdi_eval.py --rq rq6`
- `runtime/evals/nsdi/rq6_case_study/summary.json`
- `src/utg/log/run_logger.py`
- `src/utg/log/run_log_view.py`

## Claim C5: Co-evolution through Trajectory Reuse (描述性概念)

**定位说明**

这是一个**叙事概念**，用于与 Hermes 等 self-evolution 系统区分。不需要专门的评测（RQ），现有的 RQ2（资产压缩）和 RQ5（多用户共享）已经证明了这个能力。

**主张**

OmniFlow 支持共同进化：一个用户执行并贡献的轨迹，可以被其他用户复用。这降低了集体探索成本——用户越多，缓存越好，人均成本越低。

**核心概念**

```
传统方式（独立探索）:
  User A: Goal "登录应用" → 探索 → 5 LLM calls
  User B: 相同 Goal → 重新探索 → 5 LLM calls
  User C: 相同 Goal → 重新探索 → 5 LLM calls
  总成本：15 LLM calls

OmniFlow（共同进化）:
  User A: Goal "登录应用" → 探索 → 5 LLM calls → 贡献轨迹到缓存
  User B: 相同 Goal → 缓存命中 → 0 LLM calls (复用 A 的轨迹)
  User C: 相同 Goal → 缓存命中 → 0 LLM calls (复用 A 的轨迹)
  总成本：5 LLM calls

集体探索降低率 = (15 - 5) / 15 = 67%
```

**代码真源**

- `src/integrations/utg_api.py` (multi-user import/export)
- `tests/multi_user/` (跨用户共享验证)
- `src/utg/cache/` (缓存管理)

**当前允许写进论文的表述**

✅ 一个用户的轨迹可以被其他用户复用
✅ 多用户共享降低集体探索成本
✅ UDEG node 支持隐私安全的跨用户共享
✅ 缓存质量（命中率、覆盖度）随用户数增长而提升
✅ 存在网络效应：用户越多，人均成本越低

**当前不能写的表述**

❌ 使用复杂的动力学方程建模（dU/dt = ...）
❌ 声称系统完全自动进化（仍需要基本的质量控制）
❌ 已有百万级用户验证（实验规模未达到）

**当前证据入口**

- `eval_matrix.md` RQ2: Asset Compression（证明轨迹去重）
- `eval_matrix.md` RQ5: Multi-User Sharing（证明跨用户共享）
- `tests/multi_user/`（多用户测试验证）

**如何从现有数据证明共同进化**：
- RQ2 的 `compression_ratio` 显示：N 个 runs 压缩成 M 个 functions（M < N），说明存在复用
- RQ5 的 `unique_users` 和 `shared_functions` 显示：多个用户贡献到同一个缓存池
- 这两个指标组合起来就证明了"共同进化"

**关键观察**（通过现有指标自然展现）

从 RQ2 + RQ5 的组合可以看出共同进化：

**示例**（基于现有测试数据）：
```
RQ5 结果：
- uploaded_runs: 10（10次执行）
- unique_users: 3（3个用户：alice, bob, charlie）
- shared_functions: 3（压缩后只有3个唯一轨迹）

推论：
- 10次执行 → 3个唯一轨迹 = 70%去重率
- 这意味着：7次执行复用了已有轨迹
- 如果独立探索：10次 × 5 LLM calls = 50 calls
- 共享缓存：3次 × 5 LLM calls = 15 calls
- 集体降低：(50-15)/50 = 70%
```

这就是"共同进化"的自然体现，不需要专门的新实验。

## 写作边界

全文必须保持以下固定口径：

1. OmniFlow 是 **GUI Agent cache system**，不是新的通用 agent。
2. 论文主对象是 `Function` / `function asset`，不是旧的 `Path` 名义。
3. 论文主文的 state 表示写成 `UDEG node`；page vector 只作为内部编码出现，不写成“共享 raw XML”。
4. 只有在 `eval_matrix.md` 里登记过的数据和输出，才允许写成实验结果。
