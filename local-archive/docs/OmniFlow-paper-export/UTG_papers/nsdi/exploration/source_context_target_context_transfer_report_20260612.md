# Source-Context to Target-Context Action Transfer Research Report

Status: Draft research note
Last Updated: 2026-06-13
Scope: 前期调研与方法方案。本文不是当前 NSDI claim / eval / overleaf 真源；任何结论进入正文前，必须先回写 `CORE_CONCEPTS.md`、`UTG_papers/nsdi/core_claims.md` 和 `UTG_papers/nsdi/eval_matrix.md`。

## One-Sentence Problem

OmniFlow 需要在端侧快速判断一个历史 `source action` 在当前 `target screen` 上应该迁移到哪个 target UI element，或者在证据不足时拒识。

工作定义：

```text
source_context + target_context candidates + verified historical support pairs
  -> rank target candidates
  -> emit target_action with calibrated confidence, or abstain
```

这不是 0-shot natural-language GUI grounding。OmniFlow 的关键优势是已经有历史 verified source-target pair，可以把每次迁移转化为 memory-conditioned / few-shot UI correspondence。

## Core Observation

我们发现 source-action 到 target-action 的迁移不是纯粹的零样本界面定位问题。相同功能动作往往在历史执行中形成稳定的局部 UI 上下文对应关系，而错误主要来自目标页面中多个候选元素的语义和布局歧义。因此，少量已验证的历史 source-target pair 可以作为 few-shot support，把迁移问题转化为带记忆的上下文匹配问题。

论文方案上的第一原则应更严格地写成 external-public-data-first：先找公开已有数据集和已有 test migration / GUI grounding / web-agent ranking 工作，确认哪些能直接提供 source action、target candidate、gold widget/action 或 action trace。OmniFlow 自己的 frozen/runtime/review artifact 只能做实现 sanity check、系统集成验证和补充 case，不作为论文主监督数据来源。

更具体地说：

1. Functional correspondence is recurrent: 同一类 source action 经常映射到一类稳定 target local context。
2. Zero-shot similarity is under-specified: text、icon、layout 单独看会产生多个 plausible candidates。
3. Verified history resolves ambiguity cheaply: 历史中已验证的 source-target context 对齐关系可以在端侧用结构化特征快速复用。

## What This Is Useful For

这个模块的用处不是“让 agent 更聪明地看屏幕”，而是解决 OmniFlow cache-hit 执行里的一个关键断点：

```text
Function 已经召回了，
source run 里的下一步 action 也知道了，
但当前 target screen 不是当年的 exact screen。
现在到底应该点 / 输入 / 滑哪个 target element？
```

如果没有这层 source-context -> target-context transfer，cache 命中只能退化成三种不稳的做法：

1. replay old coordinate：快，但 UI 一变就容易点错。
2. replay old selector：id 稳定时有效，但遇到重复列表、无 id、动态 text、跨设备布局会失败。
3. hand off to VLM/LLM：更通用，但慢、贵、不可控，且会把 cache-hit path 又拉回在线 reasoning。

因此，这个模块的直接价值是：

| Use | Why it matters |
|---|---|
| 让 `Function` 从 exact replay 变成可迁移执行 | 同一个历史 Function 可以适配当前 UI variant，而不是只能在完全相同页面上复用 |
| 保住 cache-hit 的低成本路径 | 大多数命中段不需要再调用 VLM/LLM，看当前 observation 后本地 rank target candidate |
| 降低 wrong click 风险 | ranker 带 NULL / abstain / margin / conflict gate，不确定时 fail closed |
| 把历史 run_log 变成可学习资产 | 每次 verifier-backed 成功/失败都能沉淀为下一次匹配的 support / hard negative |
| 给论文一个可测的中间机制 | 不只报告端到端 success，还能解释 cache 为什么能跨 UI drift 安全复用 |
| 支持隐私安全共享 | shared asset 可以是 hashed/structured context 和 verified support，不必共享 raw XML / screenshot |

一句话：

> 它是 `Function recall` 和 `control-aware execution` 之间的 action projection layer，用历史验证样本把“当前屏幕该操作哪个元素”变成一个端侧可解释、可拒识、可训练的 ranking 问题。

## Non-Goals

本报告刻意不把下面问题放进第一阶段核心：

- 不做 repair search。缺失目标、页面错误、需要额外导航时，第一阶段返回 `abstain / fallback / reobserve`。
- 不把 RL 作为单步匹配主方法。RL 可用于探索和长期 memory evolution，但当前问题是 supervised / selective ranking。
- 不把 0-shot VLM grounding 作为主路径。轻量 VLM 可作为 fallback 或 visual prior，但不是端侧快路径。
- 不把 SuperGlue / LightGlue 式全量 attention matching 直接搬到端侧主链。它们的方法论有启发，但在线计算和数据形态不匹配当前约束。

## Input And Output Contract

### Input

```text
source_action:
  action_type: click | input | long_click | swipe | ...
  point_or_args: source coordinate / text / gesture args
  source_element: attrs + bbox + actionability
  source_local_context: parent / siblings / children / row / nearby OCR
  source_step_context: optional prev/next action, effect summary

target_observation:
  target_screen_id / page vector / UDEG node evidence
  target_candidates: visible actionable/editable/scrollable candidates
  target_screen_context: tree structure, OCR, global regions

history H:
  verified pairs:
    (source_context_i, target_context_i, source_action_i, target_action_i, outcome_i)
  reliability:
    verifier status, timestamp/version, user/app/device scope, failure count
```

### Output

```text
result:
  decision: execute | abstain | fallback | reobserve
  target_candidate_id: optional
  target_action: optional
  confidence:
    calibrated probability / margin / support count
  evidence:
    matched features, memory support ids, rejection reason
```

执行安全边界：

```text
execute only if:
  action_type_compatible
  top1 confidence >= threshold
  top1 - top2 margin >= threshold
  null / absent score below threshold
  no high-risk conflict in support memory
```

## Why Few-Shot Matching Fits OmniFlow

OmniFlow 的资产不是 prompt memory，而是 `run_log -> Function asset` 与可验证执行证据。历史成功执行天然形成 support set：

```text
H = {
  h_i = (C_s_i, C_t_i, a_s_i, a_t_i, verifier_i)
}
```

当前迁移不是问“这张图里哪个按钮符合一句自然语言指令”，而是问：

```text
历史上与当前 source_context 相似的动作，
曾经在 target UI 中落到什么 target_context？
当前 target screen 上哪个 candidate 最像那些 target_context？
```

这让问题从开放式视觉语言推理变成端侧可控的 retrieval + ranking。

## Recommended Main Method

第一版建议采用：

```text
memory-augmented learning-to-rank
  = base contextual UI pair features
  + few-shot memory support features
  + calibrated abstention
```

主模型：

```text
LightGBM / XGBoost LambdaMART ranker
query = one source action under current target screen
documents = target UI candidates + null candidate
label = gold target candidate or null
```

理由：

- GUI matching 是典型 query-document ranking。
- 特征是 text/action/layout/context/visual/memory/reliability 的异构混合，GBDT 比端到端小模型更数据高效。
- 端侧推理快，适合几十到几百个 candidate。
- 可解释，便于论文和 debug 分析。
- 可以自然加入 hard negatives、presence mask、null candidate 和 confidence margin。

XGBoost 官方学习排序文档说明 LambdaMART 按 query group 做 pairwise ranking，使用 `qid` 指定同一 query 下的候选集合；LightGBM 也提供 `lambdarank` 和 `rank_xendcg` 等排序 objective。这和本问题的数据组织完全一致。

## Mathematical Formulation

### Candidate Ranking Problem

令当前 source action 的上下文为：

```text
C_s = context(a_s, O_s)
```

当前 target observation 中的候选元素集合为：

```text
E_t = {e_1, ..., e_m}
C_t(e_j) = context(e_j, O_t)
```

历史 verified support memory 为：

```text
H = {h_i}_{i=1..n}
h_i = (C_s^i, C_t^i, a_s^i, a_t^i, y_i, r_i)
```

其中 `y_i` 是历史执行/verifier outcome，`r_i` 是 reliability weight，来自 verifier、时间、app/device scope、失败次数等。

目标是学习一个 scorer：

```text
f_theta(C_s, C_t(e_j), H) -> R
```

并在 `E_t` 加入显式 null candidate：

```text
E'_t = E_t union {NULL}
```

最终选择：

```text
e* = argmax_{e in E'_t} f_theta(C_s, C_t(e), H)
```

如果 `e* = NULL` 或置信度不足，则输出 `abstain / fallback`。

### Feature Decomposition

把 scorer 拆成三部分：

```text
f_theta(C_s, C_t(e), H)
  = g_theta(phi_base(C_s, C_t(e)),
            phi_memory(C_s, C_t(e), H),
            phi_conflict(C_s, C_t(e), H))
```

其中：

- `phi_base`: text/action/layout/context/visual/presence mask 特征。
- `phi_memory`: 历史 support 与当前 source/candidate 的一致性。
- `phi_conflict`: 多个 support 互相冲突、target absent、重复 label、top1/top2 接近等风险特征。

GBDT/LambdaMART 学的是 `g_theta`，而不是端到端从截图直接学坐标。

### Memory Attention

对每个历史 support：

```text
s_i = sim_s(C_s, C_s^i)
t_i(e) = sim_t(C_t(e), C_t^i)
w_i = reliability(h_i)
```

用 source 相似度做 support attention：

```text
alpha_i = exp(s_i / tau) / sum_k exp(s_k / tau)
```

候选 `e` 的 memory evidence：

```text
M(e) = sum_i alpha_i * t_i(e) * w_i
```

同时保留最大证据和冲突证据：

```text
M_max(e) = max_i s_i * t_i(e) * w_i
Conflict(e) = count_i[alpha_i high and t_i(e) low or historical outcome failed]
```

这给出一个清晰解释：当前 source 先检索相似历史 source，再看候选 target 是否像这些历史 target。

### Listwise Probability With Null

对于一个 query `q = (C_s, O_t, H)`，给每个候选打分：

```text
z_j = f_theta(C_s, C_t(e_j), H)
z_null = f_theta(C_s, NULL, H)
```

可以定义候选概率：

```text
p(e_j | q) = exp(z_j) / (exp(z_null) + sum_k exp(z_k))
p(NULL | q) = exp(z_null) / (exp(z_null) + sum_k exp(z_k))
```

训练时若有 gold target `e_g`：

```text
L_ce = -log p(e_g | q)
```

若目标缺失、语义漂移或应该拒识：

```text
L_null = -log p(NULL | q)
```

这比 pairwise binary classifier 更适合，因为同一 screen 上的候选之间存在竞争关系。

### Pairwise / LambdaMART View

LambdaMART 视角下，每个 source action 是 query，同屏候选是 documents：

```text
q_j = source action + target screen + support memory
D_j = target candidates + NULL
label(e_g)=1, label(others)=0
```

训练优化的是排序质量，例如 NDCG 或 pairwise order：

```text
z_g > z_k, for every non-gold candidate e_k
z_null > z_k, for every candidate e_k when target is absent
```

XGBoost 的 `rank:ndcg` / LambdaMART 和 LightGBM 的 `lambdarank` / `rank_xendcg` 都能直接表达这种 query-group ranking。

### Selective Decision Rule

执行不是 `argmax` 后直接点，而是风险约束决策：

```text
execute(e*) iff
  e* != NULL
  p(e*) >= theta_p
  p(e*) - p(e_2) >= theta_margin
  p(NULL) <= theta_null
  Conflict(e*) <= theta_conflict
```

也可以写成期望风险最小化：

```text
Execute if:
  E[U_execute(e*)] > E[U_abstain]
```

其中：

```text
E[U_execute(e)]
  = p_correct(e) * R_success
  - p_wrong(e) * L_wrong_click
  - p_absent * L_invalid_action
  - C_match
  - C_execute

E[U_abstain]
  = - C_match
  - C_fallback
```

由于 `L_wrong_click` 通常远大于 `C_fallback`，阈值策略应该偏 fail-closed。

### What The Math Explains

这个形式化解释了四个设计选择：

1. **为什么是 ranking**：同一 target screen 上的 candidates 互斥竞争。
2. **为什么要 null**：目标不存在或不安全时，正确输出不是低分候选，而是拒识。
3. **为什么 history 有用**：history 提供 `p(target context | source context)` 的经验估计。
4. **为什么不是 RL 主线**：当前监督信号是候选对应关系和 verifier outcome，不需要探索环境才能定义 reward。

## Optimization Strategy

如果去掉 repair，这个问题的核心就是定位 / 匹配：

```text
Given source action context and a current target screen,
find the matching target candidate, or reject.
```

最优路线不是先训练端到端模型，而是把它拆成可控的统计学习流水线：

```text
candidate generation
  -> deterministic action-compatible pruning
  -> feature extraction
  -> supervised ranking
  -> calibrated reject
  -> verifier-backed writeback
```

### Stage A: Candidate Pruning

先把明显不可能的元素剔除：

```text
click source -> clickable / long-clickable candidates
input source -> editable candidates
swipe source -> scrollable region candidates
terminal / navigation action -> host-specific safe candidates
```

这一步是规则，不交给模型学。它保证负样本集合仍然 hard，但不会把不可执行对象混进 ranker。

### Stage B: Strong Feature Baseline

先实现一个无训练或弱训练 scorer：

```text
score(e) =
  w_text * text_similarity
  + w_id * resource_tail_similarity
  + w_action * action_compatibility
  + w_layout * region_consistency
  + w_context * parent_sibling_similarity
  + w_visual * crop_similarity
```

这就是 B2 baseline。它的作用是暴露 feature 是否有信号，并生成后续 hard negatives。

### Stage C: Supervised Ranker

第一版主模型用 GBDT / LambdaMART：

```text
query = source action + target screen + memory scope
documents = target candidates + NULL
label = gold target or NULL
```

优化目标：

```text
maximize rank(gold)
maximize margin(gold, nearest confusing candidate)
maximize rank(NULL) when target absent
```

推荐训练顺序：

```text
1. train base ranker with base features
2. inspect false positives
3. mine hard negatives by same text/id/class/row/list item
4. add memory features
5. add null candidate
6. calibrate score to execute / abstain
```

### Stage D: Memory As Few-Shot Support

历史 support 不作为 prompt 拼进去，而是变成统计特征：

```text
support_i = verified(source_context_i -> target_context_i)
source attention = sim(current_source, source_context_i)
target evidence = sim(candidate_target, target_context_i)
memory feature = attention-weighted target evidence
```

这个设计把 few-shot 从 LLM 上下文学习变成端侧可复算的条件概率估计。

### Stage E: Calibration And Active Data Growth

Ranker score 不能直接当执行概率。需要：

```text
Platt / isotonic / temperature calibration
precision-coverage threshold search
false-accept audit
abstain reason taxonomy
```

每次错误执行或人工修正都回写成：

```text
positive if verifier success
negative if wrong target / no effect
null if target absent / unsafe / semantic drift
```

这样优化闭环是 supervised data flywheel，不是 RL-first。

## Statistical Learning Options

可用统计学习方法按优先级分成四层。

### Tier 1: Learning-To-Rank

推荐第一选择：

```text
LambdaMART / GBDT ranker
```

适用原因：

- 每个 source action 对应一个 target candidate list，本质是 query-document ranking。
- text/action/layout/context/visual/memory 特征是异构结构化特征，GBDT 数据效率高。
- 可以直接训练 hard negative 的相对顺序。
- 输出 margin、feature importance 和 per-candidate evidence，便于 debug。
- 端侧 CPU 推理快，适合 cache-hit path。

目标函数：

```text
rank:ndcg / lambdarank / rank_xendcg
```

适合指标：

```text
NDCG@1, MRR, top1 accuracy, precision-coverage
```

### Tier 2: Calibrated Pairwise / Listwise Classifiers

如果 ranking framework 接入成本高，可以先做：

```text
pairwise logistic regression
pairwise GBDT classifier
multinomial / conditional logit over candidates + NULL
```

适用原因：

- 实现更简单。
- 便于输出 calibrated probability。
- 可以显式建模 `p(NULL | q)`。

缺点：

- pairwise classifier 容易被 easy negatives 淹没。
- 如果不按 query group 归一化，会忽略同屏候选竞争。

推荐只作为 B5/B6 的实现备选，不作为论文叙事主方法。

### Tier 3: Metric Learning / Prototypical Matching

可训练：

```text
encoder(C_s), encoder(C_t)
score = cosine / bilinear / MLP(abs diff, product)
```

也可做 prototypical memory：

```text
prototype_target(source_cluster) = weighted average historical target contexts
candidate_score = sim(candidate, prototype_target)
```

适用场景：

- 历史样本多，想做 ANN 检索。
- 需要把 support memory 压成小向量。
- 未来想把 cross-encoder 蒸馏成端侧 scorer。

第一阶段不作为主方法，因为：

- 小数据下 embedding 容易学到 selector shortcut。
- 校准和解释性弱于 GBDT。
- 需要更严格的 leakage control。

### Tier 4: Structured Prediction

多步 Function 迁移时，可以把单步 ranker 分数放进全局解码：

```text
Hungarian matching
min-cost flow
Viterbi / linear-chain CRF
factor graph
```

适用场景：

- 一个 Function 内多个 source actions 同时迁移。
- 候选之间有互斥关系，比如同一个 target candidate 不能被多个不同 source semantic action 消耗。
- prev/next action 能提供 sequence consistency。

第一版只做 optional layer。单步 target selection 先用 ranker 解决。

### Safety And Data Methods

这些不是主 scorer，但应该进入第一版工程设计：

```text
hard-negative mining
PU learning for silver labels
isotonic / Platt calibration
conformal prediction for candidate-set abstain
Bayesian reliability weighting for historical support
active learning / human review queue from low margin cases
```

其中最重要的是：

```text
hard negatives + calibration + null candidate
```

因为真实失败通常不是“完全找不到相似元素”，而是“多个元素都看起来合理，但只有一个能安全执行”。

## Feature Design

### Base Pair Features

每个 `(source_context, target_candidate_context)` 提取：

```text
text:
  exact token overlap
  normalized edit distance
  BM25 / idf-weighted overlap
  text/content-desc/resource-id tail match
  semantic embedding cosine
  missing-text / missing-id presence bits

action:
  action_type compatible
  clickable/editable/scrollable flags
  enabled/visible/focusable state
  input field constraints if available

layout:
  normalized bbox center, size, aspect ratio
  screen region / toolbar / bottom nav / list row
  source relative offset inside element
  target region compatibility
  distance to screen landmarks

context:
  parent class/text/id
  sibling texts and sibling actionability
  list item / card container fingerprint
  nearby OCR tokens
  local subtree hash or masked structural signature
  previous / next source action type and effect if available

visual:
  icon/crop embedding cosine
  shape/color histogram distance
  crop size and aspect consistency
  optional lightweight grounding prior distance

confidence:
  candidate count
  duplicate label count
  top1/top2 preliminary margin
  number of missing critical fields
```

### Presence Mask

缺失字段不应直接当负信号。模型输入应显式包含 presence mask：

```text
feature_value:
  text_overlap = 0.0
presence:
  source_text_present = false
  target_text_present = true
```

这可以区分：

- 两边都有文本但不匹配。
- source 本来没有文本。
- target 缺 accessibility text，但 visual/context 仍可能匹配。

### Memory Support Features

对历史支持样本 `h_i = (C_s_i, C_t_i, outcome_i)`：

```text
source_sim_i = sim(C_s_current, C_s_i)
target_sim_i = sim(C_t_candidate, C_t_i)
support_score_i = source_sim_i * target_sim_i * reliability_i
```

候选级 memory features：

```text
best_history_source_sim
best_history_target_sim
best_history_pair_score
topk_history_pair_score_mean
topk_history_pair_score_max
history_agreement_count
history_conflict_count
history_target_selector_hit
history_target_text_alias_hit
history_geometry_residual
history_same_function_count
history_same_app_pair_count
history_recent_success_count
history_recent_failure_count
```

推荐聚合：

```text
memory_score(e_t) =
  sum_i softmax(source_sim_i / tau) * target_sim_i * reliability_i
```

同时保留 max 聚合：

```text
max_support_score(e_t) = max_i support_score_i
```

max 对少量强 support 敏感，softmax top-k 对噪声更稳。

## Training Data Construction

本节的“数据构建”不是从零造 benchmark，而是把公开已有 artifact 转成 learning-to-rank 所需的 query/candidate/label/support 表。所有 row 必须保留 provenance，优先能回到公开 test-migration benchmark、released action trace、grounding annotation、DOM/action oracle 或论文 artifact；OmniFlow 自己的 frozen/review/live evidence 只能作为接入 sanity 和补充 case，不作为主监督数据来源。

## External-Dataset-First Protocol

这条线如果要成为好 paper，核心不是新造一个 OmniFlow 私有数据集，也不是先使用我们自己收集的 runtime/review 数据，而是把公开已有数据集和已有工作整理成统一的 candidate-level matching protocol。新代码只做 normalization、feature export、leakage audit、ranking evaluation 和 error review queue，不创造新的任务分布，也不把 heuristic proposal 写成 gold。

数据优先级应固定如下：

| Tier | Source | Allowed use | Not allowed |
|---|---|---|---|
| T1 | GUI test migration / test reuse datasets and artifacts: CraftDroid, FrUITeR, SemFinder, TEMdroid-style widget matching | 最接近 source-action -> target-action transfer；直接评估 source event/widget 到 target event/widget 的 matching | 只引用旧结果，不复用公开 artifact 或不复现协议 |
| T2 | Public mobile action-trace datasets: AITW, MobileViews complete traces, AndroidControl, GUIOdyssey, AMEX stepwise chains | 派生 action-to-candidate rows、hard negatives、history/context features、cross-app/task splits | 把 task success 或 raw coordinate 当 per-candidate gold，除非能匹配到 UI annotation / bbox |
| T3 | Public grounding datasets/corpora: ScreenSpot, ScreenSpot-v2, ScreenSpot-Pro, OS-Atlas grounding data, AMEX element grounding | 训练/评测 visual/text/layout grounding feature，不作为 transfer 主任务 | 声称验证了 source-target transfer |
| T4 | Public web-agent datasets with DOM/action history: Mind2Web, WebLINX, WebChain | 借鉴 candidate pruning、element ranking、history-conditioned matching；可做跨平台 analog | 混写成 Android/mobile 结果 |
| T5 | OmniFlow internal frozen/runtime/review artifacts | 系统接入 sanity check、pipeline debug、claim consistency、small live illustration | 作为 paper headline supervised dataset |
| T6 | synthetic / controlled cases | 单元测试、泄漏测试、calibration sanity check、ablation only | headline paper evidence 或主实验分母 |

因此第一版 supervised ranker 的训练/评测顺序应是：

```text
public dataset / public artifact -> provenance-preserving candidate dump
published baseline / existing method -> feature columns and baseline scores
public gold / benchmark oracle / released action annotation -> gold/null labels
GBDT/LambdaMART -> re-ranker/calibrator over existing candidates
```

这也定义了 method reuse first 的边界：`get_new_coordinates(...)`、`PageVectorSet`、UI-tree exact、UI-tree+anchor、`text_feature`、anchor vote 和现有 review template 都是第一批方法和数据入口。GBDT/LambdaMART 不是独立新系统，而是把这些已有信号合成可解释、可校准的端侧 ranking layer。

这里需要再收紧一句：OmniFlow 的现有方法可以作为要比较/替换的 system method，但训练和主要评测数据优先来自公开数据集。否则 reviewer 会把贡献理解成“我们自己收集了一个小私有数据集并在上面调参”，这不是强 paper。

## External Existing Dataset And Work Inventory

下面按“是否能直接支撑 source-action -> target-action transfer”排序。

### A. Directly Matching The Problem

| Dataset / work | Public evidence | Why useful | How to use |
|---|---|---|---|
| CraftDroid | [project / GitHub](https://github.com/seal-hub/CraftDroid), [paper site](https://seal.ics.uci.edu/projects/craftdroid/index.html) | Android test transfer across apps through semantic mapping；任务结构就是 source test event -> target event/widget | 第一优先级 baseline：coordinate/selector/text/semantic matching、event transfer precision/recall |
| FrUITeR | [artifact page](https://felicitia.github.io/FrUITeR/), [arXiv](https://arxiv.org/abs/2008.03427) | 提供 UI test reuse 的 reproducible evaluation framework 和 released final datasets；字段含 source/target/method/src_events/trans_events/gt_events/correct/incorrect/missed/nonExist | 直接转成 query=source event、documents=target events、label=gt/correct/nonExist |
| Semantic matching / SemFinder | [Empirical Software Engineering paper](https://link.springer.com/article/10.1007/s10664-023-10406-8) | 明确把 test reuse 的核心拆成 semantic matching；报告 8,099 GUI events、337 configurations、MRR/Top1/F1 | 作为统计学习和 IR baseline：Jaccard/edit distance/embedding/event descriptor extractor |
| TEMdroid | [GitHub](https://github.com/YakZhang/TEMdroid), [ICSE paper PDF](https://yakzhang.github.io/2024-icse-temdroid.pdf) | learning-based widget matching for GUI test migration；仓库公开 Dataset/Match | 如果 artifact 可跑，作为 supervised widget matching baseline；若 public implementation incomplete，则只作为 related work / protocol guide |
| Vision-based widget mapping for test migration | [ASE PDF](https://seg.nju.edu.cn/uploadPublication/copyright/123-629490201.pdf) | 构造 6,730 bi-directional mapped widget pairs across iOS/Android；直接评测 visual widget matching | 适合 visual feature / crop embedding / LightGlue-style baseline，不适合作为 Android-only 主结果 |
| ITeM | [ISSTA 2025 page](https://conf.researchr.org/details/issta-2025/issta-2025-papers/100/Intention-based-GUI-Test-Migration-for-Mobile-Apps-using-Large-Language-Models), [GitHub](https://github.com/Vallxy/ITeM) | 2025 LLM-based intention generation + dynamic fulfillment；显式指出传统 widget-matching 在 source/target interaction logic 不一致时会受限 | 作为最新 related work / end-to-end migration baseline；不作为端侧快速 matcher，除非只抽取其 generated intention / matched step artifact |
| LLMigrate | [arXiv](https://arxiv.org/abs/2411.17933), [GitHub](https://github.com/seal-hub/llmigrate) | multimodal LLM test transfer，不要求 source code；面向 usage-based E2E UI test transfer | 作为 2025-era LLM baseline / upper-bound reference；成本、online calls 和 privacy profile 与端侧 ranker 不同 |
| MACdroid / MigratePro | [MACdroid site](https://sites.google.com/view/macdroid-test?usp=sharing), [MACdroid arXiv](https://arxiv.org/html/2409.05028v2), [MigratePro PDF](https://yakzhang.github.io/2024-issta-migratepro.pdf) | abstraction/concretization 或 synthesis-enhanced migration；说明 migration 不只是局部 widget 匹配，还涉及 test logic 与 incomplete migrated test repair | 作为 boundary related work：我们的第一版只解决定位/匹配层，不声称替代 full migration / repair |

### B. Public Mobile Action Traces For Candidate-Level Rows

| Dataset | Public evidence | Useful fields | Fit |
|---|---|---|---|
| AITW | [Google Research README](https://github.com/google-research/google-research/tree/master/android_in_the_wild), [NeurIPS abstract](https://proceedings.neurips.cc/paper_files/paper/2023/hash/bbbb6308b402fe909c39dd29950c32e0-Abstract-Datasets_and_Benchmarks.html) | 715k episodes, screenshots, UI annotation bboxes/text/types, action type, touch/lift coordinates, Android version/device splits | Good for action-to-candidate grounding and OOD split; not direct source-target transfer unless paired by instruction/context |
| MobileViews | [HuggingFace dataset](https://huggingface.co/datasets/mllmTeam/MobileViews), [arXiv](https://arxiv.org/html/2409.14337v1) | 600K+ screenshot-VH pairs; complete traces with states/actions/transitions/views/action.csv | Good for large-scale candidate enumeration, trace context, hard negatives; gold transfer pairs absent |
| AndroidControl | [OpenReview](https://openreview.net/forum?id=yUEBXN3cvX) | 15,283 demos / 14,548 unique tasks / 833 apps; high- and low-level instructions | Good for supervised action prediction and in/out-domain split; candidate ids need dataset adapter |
| GUIOdyssey | [arXiv](https://arxiv.org/html/2406.08451v2), [HuggingFace](https://huggingface.co/datasets/OpenGVLab/GUI-Odyssey) | 8,334 episodes, avg 15.3 steps, 6 devices, 212 apps, 1,357 app combos, reasoning annotations | Good for history-conditioned / cross-app context transfer; not direct widget-pair gold |
| AMEX | [arXiv](https://arxiv.org/abs/2407.17490), [ACL Anthology](https://aclanthology.org/2025.findings-acl.110/) | 104K high-res screenshots, interactive element grounding, element/screen function descriptions, stepwise GUI-action chains | Good for element/function grounding and action-chain supervision; may support target-candidate labels if annotations expose bboxes/actions |
| Rico | [dataset site](https://www.interactionmining.org/archive/rico) | 66k+ screens, 3M elements, screenshots, view hierarchies, interaction traces, semantic annotations | Good for feature/pretraining and UI similarity; older apps and weaker task/action semantics |

### C. Public Grounding And Web-Agent Corpora

| Dataset / work | Public evidence | Use |
|---|---|---|
| ScreenSpot / ScreenSpot-v2 / ScreenSpot-Pro | [SeeClick GitHub](https://github.com/njucckevin/SeeClick), [ScreenSpot-Pro arXiv](https://arxiv.org/html/2504.07981v1), [OS-Atlas data](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) | visual/text/icon grounding baseline; useful for crop embedding and abstain calibration, not transfer main task |
| OS-Atlas grounding data | [HuggingFace](https://huggingface.co/datasets/OS-Copilot/OS-Atlas-data) | large cross-platform GUI grounding pretraining data with instruction/bbox fields |
| Mind2Web | [project page](https://osu-nlp-group.github.io/Mind2Web/) | web element ranking / DOM candidate pruning; method analog for ranker design |
| WebLINX | [ICML/PMLR](https://proceedings.mlr.press/v235/lu24e.html) | 100K interactions over 2,300 demonstrations; explicitly ranks relevant HTML elements with screenshots/action history |
| WebChain | [arXiv 2026](https://arxiv.org/html/2603.05295v3) | 31,725 trajectories / 318k steps with visual-structural-action alignment; strongest recent public web analog |

### Public Artifact Suitability Matrix

下面这个矩阵是本问题的核心取舍：不要问“哪个数据集最大”，而要问它能不能给出 `query=source action/event`、`documents=target candidates`、`label=gold target or NULL`。能直接给这三件事的，才适合训练/评测 B5/B6；只能给截图、单屏点击或 task success 的，只能做 feature/pretraining/hard negative。

| Dataset / work | Public artifact status | Source action/event | Target candidates | Gold target | NULL / absent | UI tree / bbox / text | Direct transfer fit | Recommended role | Main limitation |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| FrUITeR | artifact page + GitHub + final datasets | yes | yes, transferred/GT events | yes | yes, `nonExist` / missed classes | event descriptors; UI evidence depends on artifact field | high | P0 adapter for query/candidate/label/null | older benchmark; event abstraction may not expose all visual/layout features |
| CraftDroid | GitHub/project artifact | yes | yes, through target test/event mapping | partial/yes via transferred test success and mapping outputs | weak | dynamic UI state can be regenerated if env works | high | P0 baseline and protocol compatibility check | app versions and rerun environment may be brittle |
| TEMdroid | GitHub with Dataset/Match dirs | yes | yes, widget matching candidates | yes if released match labels are complete | weak | likely widget text/layout/metadata; verify actual files | high | P0 supervised widget-matching baseline | need artifact audit before claiming full gold/null coverage |
| SemFinder / semantic GUI event matching | paper + accessible article; raw data availability must be verified | yes | yes conceptually | yes in paper protocol | unclear | semantic event descriptors | medium-high | IR/statistical baseline if data is obtainable | do not rely on it as primary dataset until raw artifact is confirmed |
| Vision-based widget mapping / MAPIT-style | paper artifacts need audit | yes, widget/action pairs | yes, paired screens/widgets | yes, mapped widget pairs | weak | bbox/crops/visual pairs | medium | visual/crop feature ablation | cross-platform sibling apps; not Android-only action transfer |
| ITeM / LLMigrate | GitHub/paper artifacts | yes | generated by live reasoning | success-level, not always candidate-level | weak | depends on runtime traces | medium | latest LLM migration baseline / upper bound | online LLM and dynamic fulfillment conflict with 端侧 fast path |
| MACdroid / MigratePro | paper/site/PDF artifacts need audit | yes | generated concrete tests | success/incomplete-test repair outcomes | weak | depends on migration traces | medium | related work for full migration and repair boundary | solves broader test synthesis, not just candidate selection |
| AITW | Google Research dataset | action trace yes | via UI annotation / bbox matching | same-screen action target after coordinate-to-element matching | no | screenshot + UI annotation | low for transfer; high for grounding | feature pretraining, hard negatives, OOD app/device splits | no source-target pair; raw coordinate is not transfer gold |
| MobileViews | HuggingFace dataset | traces/action logs in releases | view hierarchy candidates | coordinate/action matching possible | no | screenshot + VH | low for transfer | candidate enumeration, layout/text feature coverage | no gold source-target transfer labels |
| AndroidControl | Google Research/OpenReview dataset | demonstrations yes | accessibility/screen candidates | same-step action target possible | no | Android screenshots/actions/instructions | low for transfer | supervised action prediction analog; in/out-domain split | candidate ids need adapter; no paired source/target app migration |
| GUIOdyssey | HuggingFace/arXiv | long traces yes | screen candidates | step action target possible | no | screenshot + metadata/reasoning | low-medium | history-conditioned matching stress data | not widget-pair transfer gold |
| AMEX | ACL/arXiv/project | stepwise chains yes | interactive element boxes | grounding labels yes | no | bbox + element/screen function descriptions | medium for grounding, low for transfer | element/function grounding features | no source-target migration pair |
| Rico | public archive | weak traces | view hierarchy candidates | weak/derived | no | large screenshot + VH + semantics | low | UI representation pretraining / retrieval ablation | older apps; weak action semantics |
| ScreenSpot / OS-Atlas | public grounding corpora | no source action | bbox candidates | grounding bbox yes | no | bbox + instruction/screenshot | low | visual/text grounding prior | validates grounding, not source-target transfer |
| Mind2Web / WebLINX / WebChain | public web trajectories | web actions yes | DOM candidates | DOM target/action yes | sometimes | DOM + screenshot + action history | analog only | method transfer: ranking, pruning, history features | web DOM != Android UI tree; report separately |

Practical implication:

```text
Main supervised/eval data:
  FrUITeR + CraftDroid + TEMdroid first

Latest method/baseline context:
  ITeM + LLMigrate + MACdroid/MigratePro

Feature/pretraining/hard-negative data:
  AITW + MobileViews + AndroidControl + GUIOdyssey + AMEX + Rico

Grounding/model prior:
  ScreenSpot + OS-Atlas + AMEX grounding

2026 public analog:
  WebChain, but reported as web-side analog only
```

### Recommended External-First Experiment Plan

不要第一步接 OmniFlow 数据。第一步应该做一个 public artifact adapter：

```text
Phase E0:
  FrUITeR / CraftDroid / SemFinder / TEMdroid adapter
  -> source_event, target_event_candidates, gold/correct/nonExist
  -> B0/B1/B2/B5/B6 ranking protocol

Phase E1:
  AITW + MobileViews adapter
  -> action coordinate matched to UI annotation / VH element
  -> candidate-level rows and hard negatives

Phase E2:
  AMEX + ScreenSpot/OS-Atlas adapter
  -> visual/text grounding feature pretraining and out-of-domain grounding check

Phase E3:
  WebLINX / Mind2Web / WebChain adapter
  -> DOM candidate-ranking analog; use only as method evidence, not Android result

Phase E4:
  OmniFlow internal data
  -> smoke test: trained scorer can plug into Function execution path
```

主论文证据优先使用 E0/E1/E2。OmniFlow 自己的数据只回答“这个外部验证过的 matcher 能否接到我们的 system fast path”，不能回答“ranker 是否有效”。

### Training Unit

一个训练 query 是：

```text
q_j = (current source action, current target screen, optional history support H_j)
```

候选集合是：

```text
D_j = {all action-compatible target candidates} union {null}
```

label 是：

```text
y_jk = 1 if candidate k is gold target
y_j_null = 1 if target absent / unsafe / semantic drift
```

### Positive Labels

可来自：

- verified source-target action pairs；
- raw-step-aligned source/target traces；
- human-reviewed action mapping；
- live terminal/effect verified replay；
- benchmark oracle with clear target element。

银标可以用于训练和诊断，但论文 headline 需要 gold 或 verifier-backed label。

### Hard Negatives

必须显式采样 hard negatives，否则模型会在真实页面里误点：

```text
same_screen_all_non_gold
same_action_type_non_gold
same_text_or_alias_non_gold
same_resource_tail_non_gold
same_class_non_gold
same_row_or_neighbor_non_gold
duplicate_list_item_non_gold
model_top_false_positive
historical_conflict_candidate
null_target_near_miss
```

### Split Policy

为避免过拟合和数据泄漏，至少做四类 split：

```text
same-app held-out screen
same-app held-out task/function
cross-device held-out target
cross-app or app-family held-out
```

特别要防止：

- 同一 target screenshot 既出现在 train 又出现在 test。
- 同一个 raw trace 的相邻 step 同时进 train/test。
- gold target selector 被泄露进 target candidate id。
- memory support H 中包含当前 query 的同一 pair。

## Candidate-Level Export Specification

第一阶段最重要的产物不是模型，也不是一个新 benchmark，而是对已有 artifact 的可审计 candidate-level export。建议把派生结果拆成四类 JSONL，避免后续把 raw trace、候选集合、特征和模型输出混在一起。

下文如果使用 `dataset` 一词，只表示 learning-to-rank 工具链需要的派生表，不表示新收集或新定义的论文数据集。每条 query / candidate / label / support 都必须带 provenance 字段，优先指向 FrUITeR、CraftDroid、TEMdroid、SemFinder、AITW、MobileViews、AndroidControl、GUIOdyssey、AMEX、Rico、ScreenSpot/OS-Atlas、Mind2Web/WebLINX/WebChain 等公开 artifact 或 benchmark oracle。OmniFlow 内部 run_log / review queue 只能以 `omniflow_sanity` artifact type 出现。

### Query Record

一个 query 对应“一次 source action 在当前 target screen 上找目标”：

```json
{
  "query_id": "app/task/run/step/target_screen",
  "source_action_id": "source_run:step_003",
  "function_id": "optional_function_asset_id",
  "source_screen_hash": "hash",
  "target_screen_hash": "hash",
  "target_page_family": "optional_family_id",
  "action_type": "click",
  "split": "train|valid|test",
  "scope": {
    "app": "contacts",
    "device_class": "phone",
    "locale": "en_US",
    "user_scope": "local|shared-anonymous"
  },
  "history_cutoff_ts": "2026-06-12T00:00:00Z",
  "label_status": "gold|silver|null|unsafe|unlabeled"
}
```

关键点：

- `query_id` 是排序 group id。
- `history_cutoff_ts` 用来防止把未来 support 泄漏进当前 query。
- `target_page_family` 只用于评测和分桶，不作为模型输入，除非明确登记为 oracle feature。

### Candidate Record

一个 candidate 是当前 target screen 上一个可执行 UI entity：

```json
{
  "query_id": "same_as_query",
  "candidate_id": "target_screen_hash:element_017",
  "candidate_kind": "element|region|null",
  "action_compatible": true,
  "bbox_norm": [0.71, 0.08, 0.94, 0.15],
  "element_attrs": {
    "class": "android.widget.Button",
    "clickable": true,
    "editable": false,
    "scrollable": false,
    "text_hash": "optional_hash",
    "content_desc_hash": "optional_hash",
    "resource_tail_hash": "optional_hash"
  },
  "local_context": {
    "parent_class": "android.widget.Toolbar",
    "sibling_text_hashes": ["..."],
    "nearby_ocr_hashes": ["..."],
    "region": "top_toolbar"
  }
}
```

共享 / paper-facing 版本默认不保存 raw text、raw resource-id、raw XML 或截图 crop，只保存 hash、bucket、presence bit 和可复算的结构特征。raw audit 版本只留在 local evidence 包。

### Label Record

label 要区分 gold、silver、null 和 unsafe：

```json
{
  "query_id": "same_as_query",
  "gold_candidate_id": "target_screen_hash:element_017",
  "label_type": "gold|silver|null|unsafe",
  "label_source": "live_verifier|human_review|benchmark_oracle|trace_alignment",
  "verifier": {
    "verifiable": true,
    "success": true,
    "evidence_ref": "runtime/evals/.../case.json"
  },
  "negative_reason": null
}
```

推荐规则：

| Label type | 进入训练 | 进入 headline eval | 说明 |
|---|---:|---:|---|
| `gold` | yes | yes | live verifier、human review 或 benchmark oracle 支撑 |
| `silver` | yes | diagnostic only | trace alignment / heuristic proposal，可降权 |
| `null` | yes | yes if verifier-backed | 目标缺失、语义漂移或正确行为应拒识 |
| `unsafe` | no or safety training only | no | 需要 host policy / confirmation，不用于普通 accuracy |
| `unlabeled` | no | no | 可进入 active review queue |

### Support Record

support memory 必须显式记录可用范围：

```json
{
  "support_id": "support_run:step_002->target_step_004",
  "source_context_hash": "hash",
  "target_context_hash": "hash",
  "source_action_type": "click",
  "target_action_type": "click",
  "outcome": "success|failure",
  "reliability": 0.92,
  "scope": {
    "same_app": true,
    "same_function": true,
    "same_device_class": false,
    "same_locale": true
  },
  "timestamp": "2026-06-01T00:00:00Z"
}
```

每个 test query 的 support set 应按下面规则构造：

```text
H_test(q) =
  verified support before history_cutoff_ts
  - same raw trace as q
  - same target screenshot as q
  - manually linked duplicate of q
```

这样 B6 的提升才能解释为历史 support 泛化，而不是数据泄漏。

### Feature Row

最终给 ranker 的每一行是 `(query, candidate)`：

```json
{
  "query_id": "q",
  "candidate_id": "c",
  "label": 1,
  "group_size": 57,
  "features": {
    "text_token_jaccard": 0.5,
    "resource_tail_match": 1.0,
    "action_type_compatible": 1.0,
    "bbox_center_dx": 0.08,
    "same_screen_region": 1.0,
    "parent_class_match": 0.0,
    "nearby_ocr_bm25": 2.4,
    "memory_score": 0.73,
    "memory_conflict_count": 0,
    "duplicate_label_count": 2,
    "source_text_present": 1.0,
    "target_text_present": 1.0
  }
}
```

强制约束：

- 同一个 `query_id` 下必须有完整 candidate list 和一个 `NULL` candidate。
- 所有 feature 必须可从 Query/Candidate/Support record 复算。
- presence mask 与 feature value 分开。
- 模型输出不能回写覆盖原始 feature dump，只能写到 prediction file。

### Export Versioning

每次派生 export 都要生成 manifest：

```json
{
  "export_id": "action_transfer_existing_artifacts_20260613",
  "source_artifacts": [
    "runtime/evals/nsdi/page_replay_baselines/headline/summary.json",
    "runtime/evals/nsdi/page_replay_baselines/headline/live_verify.json",
    "runtime/evals/action_projection_representative/..."
  ],
  "candidate_builder_version": "v0",
  "feature_builder_version": "v0",
  "redaction_policy": "hash_raw_text_local_audit_only",
  "num_queries": 0,
  "num_candidates": 0,
  "num_gold": 0,
  "num_null": 0,
  "num_silver": 0,
  "split_policy": "leave-trace-and-screen-out",
  "known_limitations": ["exploration placeholder until runner exists"]
}
```

这份 manifest 是进入 `eval_matrix.md` 的前置条件之一。

## Model Variants

### M0: Deterministic Baselines

```text
coordinate replay
selector / resource-id replay
text-feature replay
UI tree exact replay
UI tree + anchor replay
```

这些 baseline 必须保留。它们回答“如果保留更强的 raw UI tree / selector 信息，问题有多简单”。

### M1: Base Ranker Without Memory

```text
features = base pair features + presence mask
model = LightGBM/XGBoost ranker
```

作用：证明统计学习能替代手调权重。

### M2: Memory-Augmented Ranker

```text
features = base features + memory support features
model = same ranker
```

作用：证明 history/few-shot support 不是 prompt 装饰，而是实际提高 disambiguation / selective precision 的变量。

### M3: Metric Retrieval + Ranker

```text
encoder_source(C_s) -> vector
encoder_target(C_t) -> vector
ANN retrieve top-K historical support
ranker consumes retrieved support features
```

第一版 encoder 可以是手工向量 + small MLP；后续可蒸馏 cross-encoder。

### M4: Tiny MLP Pair Scorer

输入：

```text
concat([
  v_s * v_t,
  abs(v_s - v_t),
  v_s * mask_t,
  v_t * mask_s,
  pair_features,
  memory_features
])
```

优点是端侧很快；缺点是校准和可解释性通常不如 GBDT。建议作为 ablation，不是第一主线。

### M5: Graph / Assignment Model

如果一次 Function 中多个 source actions 同时迁移，可以在单步 ranker 外加全局一致性：

```text
source action nodes <-> target candidate nodes
edge compatibility = local ranker score + sequence/context consistency
solver = Hungarian / min-cost flow / CRF / Viterbi
```

第一阶段不建议上 SuperGlue-style transformer。可把它作为 future direction 或 offline teacher。

## Baselines

Baseline 必须覆盖从最弱 replay 到最强本地 oracle-like replay 的完整梯度。否则模型提升会被 reviewer 质疑为只打败了 strawman。

最短回答：

```text
Primary paper-facing baselines:
  B0 coordinate replay
  B1 UI tree / selector exact replay
  B2 UI tree + anchor replay

Research-method baselines:
  B3 text/resource/class rule scorer
  B4 base GBDT ranker without memory
  B5 memory-augmented GBDT ranker

Diagnostic baselines:
  lightweight grounding prior
  VLM/LLM grounding
  contextual bandit / RL
```

每一层回答一个不同问题：

| Question | Baseline | If it wins, what it means |
|---|---|---|
| 旧坐标够不够？ | coordinate replay | 问题主要是 layout-stable replay，不需要复杂 matching |
| 旧 selector/raw tree 够不够？ | selector / UI tree exact replay | 在允许保留 raw UI evidence 的本地场景很强 |
| anchor/context 规则够不够？ | UI tree + anchor replay | non-learning context mapping 已经解决大部分 case |
| 手写特征够不够？ | text/resource/class scorer | 学习模型收益有限，主要是 feature engineering |
| 统计学习有没有用？ | base GBDT ranker | supervised ranker 能学到组合特征和 hard negatives |
| 历史 support 有没有用？ | memory-augmented GBDT | OmniFlow 的 verified history 是核心增益来源 |
| 2026 grounding 模型能不能替代？ | GoClick / VLM diagnostic | 若替代成立，主线应改成 grounding prior；若不成立，它只是补充视觉特征 |
| RL 是否必要？ | contextual bandit / RL diagnostic | 若样本效率低或 wrong-click 高，说明第一版不应 RL-first |

论文里最公平的 baseline 梯度是：

```text
coordinate replay
  < selector / exact UI tree replay
  < UI tree + anchor replay
  < base learned ranker
  < memory-augmented learned ranker
  + VLM/grounding diagnostic for cost/latency comparison
```

注意：如果 `UI tree + anchor replay` 在单设备、本地 raw-tree 设置中很强，这是预期结果。OmniFlow 要证明的不是“任何情况下都比 raw tree replay 准”，而是：

```text
在 privacy-safe / assetized / current-screen / fail-closed 的约束下，
verified history + ranker 能提高 action transfer 的 precision-coverage，
并保持端侧低延迟和低在线模型成本。
```

### B0: Coordinate Replay

输入：

```text
source action recorded coordinate or normalized bbox center
target screen size / density
```

方法：

```text
scale source coordinate to target screen
execute same action at mapped coordinate
```

输出：

```text
target coordinate
```

证明的问题：

- 裸坐标 replay 在 layout drift、density 变化、列表滚动、弹窗和同屏元素重排时会失败。
- 它是最低保真 baseline，但必须保留，因为很多 GUI replay 系统的隐含 fallback 就是坐标。

公平边界：

- 不给它 XML、selector、OCR 或历史 support。
- 允许按分辨率归一化，不允许事后找最近元素。

预期失败：

```text
wrong region
old coordinate hits sibling element
target is offscreen
keyboard/dialog shifts layout
```

### B1: Selector / Resource-Id Replay

输入：

```text
source target element attributes:
  resource-id, text, content-desc, class, index path, clickable/editable flags
target UI tree
```

方法：

```text
find target element by exact resource-id
fallback to resource-id tail + class + text/content-desc
fallback to index path if stable
```

输出：

```text
matched target element or null
```

证明的问题：

- 如果 app 保留稳定 accessibility id，selector replay 很强。
- 它暴露我们方法是否只是重新实现 selector matching。

公平边界：

- 它可以访问 raw target UI tree 和 recorded target attributes。
- 它不能访问 verified cross-screen support memory 或 learned ranker。

预期失败：

```text
resource-id absent or obfuscated
same id repeated in list rows
text localized / dynamic
class/index path changes
```

### B2: Text / Resource / Class Feature Scorer

输入：

```text
source context
target candidates
hand-designed feature weights
```

方法：

```text
score candidates by token overlap, edit distance,
resource tail match, content-desc match, class/action compatibility,
basic region consistency
```

输出：

```text
top target candidate or null by threshold
```

证明的问题：

- 简单结构化特征是否已经足够。
- 训练模型的收益是否来自学习特征组合，而不是 feature 本身。

公平边界：

- 不使用训练标签调参到 test set。
- 可以用 validation split 选少量固定权重或阈值。

预期失败：

```text
duplicate labels
icon-only controls
same text with different semantic role
context-dependent list item identity
```

### B3: UI Tree Exact Replay

输入：

```text
recorded source UI tree
recorded target element
current target UI tree
```

方法：

```text
match local subtree / index path / sibling order / parent chain
then verify element attributes are compatible
```

输出：

```text
exact or near-exact replay target element
```

证明的问题：

- 在允许保留 raw UI tree 和目标元素证据时，本地 replay 可以非常强。
- OmniFlow 的主张不能是“比 raw tree exact replay 永远准”，而是 privacy-safe sharing、assetization、cross-state reuse 和 fail-closed。

公平边界：

- 这是强 baseline，应允许访问比 OmniFlow 共享层更多的 raw local evidence。
- 但它只能用于本地或诊断，不代表可共享隐私安全资产。

预期失败：

```text
tree shape changes
dynamic inserted rows
accessibility hierarchy rewritten
same subtree repeated across list/card items
```

### B4: UI Tree + Anchor Replay

输入：

```text
recorded UI tree and target element
current target UI tree
stable anchors:
  sibling labels, parent/card fingerprints, toolbar/tab markers, row anchors
```

方法：

```text
first match page/local anchors
then map source target relative to anchors
rerank by geometry and tree-distance consistency
```

输出：

```text
anchor-consistent target element or null
```

证明的问题：

- 这是 strongest non-learning local baseline。
- 它测试“anchor/context mapping”本身是否足够，不依赖 GBDT。

公平边界：

- 可以使用 raw UI tree anchors。
- 不使用 learned memory across verified source-target pairs。
- 对隐私共享不友好，因此不应被包装成我们的系统替代品。

预期失败：

```text
anchor itself drifts
multiple repeated cards share same anchors
target absent but anchors still present
visual icon changes while text anchors remain ambiguous
```

### B5: Base GBDT Ranker Without Memory

输入：

```text
source context
target candidates
base features + presence mask
```

方法：

```text
LightGBM / XGBoost LambdaMART ranker
no historical support features
NULL candidate included in calibrated variant
```

输出：

```text
ranked candidate list + confidence
```

证明的问题：

- 统计学习是否能稳定组合 text/action/layout/context/visual features。
- 和 B2 对比，衡量 learning-to-rank 本身收益。

公平边界：

- 使用同样 candidate set 和 split。
- 不能让 train/test 泄露同一 screenshot 或同一 trace pair。

预期失败：

```text
insufficient feature signal
unseen app-specific alias
multiple semantically plausible candidates
```

### B6: Memory-Augmented GBDT Ranker

输入：

```text
source context
target candidates
base features
top-K verified historical support pairs
support reliability metadata
```

方法：

```text
ranker consumes memory_score, support agreement,
history conflict, recent success/failure, same-function support features
```

输出：

```text
execute / abstain decision with support evidence
```

证明的问题：

- 我们最核心的假设：verified history 是 useful statistical evidence。
- few-shot support 能否在 duplicate label、icon-only、layout drift、target absent 中提高 precision-coverage。

公平边界：

- 对每个 test query，support set 必须排除同一 query、同一 screenshot、同一 raw trace 的泄漏样本。
- 如果跨用户共享，需要 scope / privacy / reliability 字段进入报告。

预期失败：

```text
stale support
conflicting support
new UI variant with no close memory
history from incompatible user/app/device scope
```

### B7: Lightweight Grounding Prior + Ranker

输入：

```text
natural-language action label or source-context summary
target screenshot
target candidates
base + memory ranker features
```

方法：

```text
run lightweight grounding model only as prior
features:
  predicted point inside candidate
  center distance
  grounding rank
  overlap / IoU if bbox available
ranker makes final decision
```

输出：

```text
ranker decision with visual prior features
```

证明的问题：

- 2026 轻量 GUI grounding 是否能补 textless/icon-only cases。
- 它是 feature augmentation，不是替代 verified history。

公平边界：

- 单独报告 additional latency、model size、device/backend。
- 若无法端侧运行，只能作为 diagnostic 或 offline teacher。

预期失败：

```text
grounding follows natural-language prior but ignores source-specific context
ambiguous repeated controls
target absent but model still points somewhere
```

### B8: VLM / LLM Grounding Diagnostic

输入：

```text
source action textual description
target screenshot or XML text
optional history examples in prompt
```

方法：

```text
ask VLM/LLM to output target coordinate / element id / abstain
```

输出：

```text
model-proposed target action
```

证明的问题：

- 当前 strongest general model 的 zero/few-shot grounding 能到什么水平。
- 作为 upper diagnostic，不作为端侧主 baseline。

公平边界：

- 必须报告 token、latency、provider/model/version、prompt、image/XML payload。
- 如果使用 history examples，必须与 B6 使用相同 support split。

预期失败：

```text
high latency
provider/token cost
overconfident hallucinated target
privacy boundary mismatch
model update changes results
```

### B9: Contextual Bandit / RL Diagnostic

输入：

```text
candidate features
execution reward or verifier outcome
```

方法：

```text
select candidate under exploration policy
update policy from success/failure reward
```

输出：

```text
candidate action policy
```

证明的问题：

- 长期在线 memory evolution 是否有收益。
- 不适合作为第一版主方法，因为 wrong-click exploration 成本高，且 gold labels / verifier outcomes 已经能直接监督 ranker。

公平边界：

- 只能在 safe sandbox 或 replay environment 中探索。
- 必须把 exploration cost、wrong-click risk 和 sample efficiency 单独报告。

预期失败：

```text
sparse reward
unsafe exploration
slow convergence compared with supervised labels
environment non-stationarity
```

### Baseline Summary Table

| ID | Method | Uses raw UI tree | Uses verified history | Uses learned ranker | End-side fast path | Main role |
|---|---:|---:|---:|---:|---:|---|
| B0 | Coordinate replay | no | no | no | yes | weakest replay floor |
| B1 | Selector/resource-id replay | yes | no | no | yes | stable-id baseline |
| B2 | Text/resource/class scorer | optional | no | no | yes | feature sanity check |
| B3 | UI tree exact replay | yes | no | no | yes | strong local replay |
| B4 | UI tree + anchor replay | yes | no | no | yes | strongest non-learning local baseline |
| B5 | Base GBDT ranker | no/shared-safe fields | no | yes | yes | supervised feature ranker |
| B6 | Memory-augmented GBDT | no/shared-safe fields | yes | yes | yes | proposed first method |
| B7 | Grounding prior + ranker | screenshot/crop | optional | yes | maybe | optional visual prior |
| B8 | VLM/LLM grounding | screenshot/XML text | optional prompt | model-internal | no | diagnostic comparison |
| B9 | Bandit/RL | features/env | learns over time | yes | no initially | future online evolution |

## Calibration And Reject Option

生产路径不能只优化 top-1 accuracy。需要 selective prediction：

```text
execute if:
  p_top1 >= theta_p
  margin(top1, top2) >= theta_m
  p_null < theta_null
  support_conflict_count <= theta_conflict
```

校准方法：

- Platt scaling / isotonic regression on validation split。
- Temperature scaling for softmax ranker。
- Conformal prediction: 输出 candidate set；只有 set size = 1 且不含 null 时执行。

核心指标：

```text
selective_accuracy@coverage
mapped_precision
wrong_click_rate
abstain_rate
false_accept_rate
false_reject_rate
expected_cost = C_match + C_execute + C_fallback * abstain + L_wrong * wrong_click
```

## On-Device Runtime Design

推荐运行时：

```text
offline:
  build support memory from verified pairs
  precompute source/target context embeddings
  store reliability and scope metadata

online:
  observe target screen
  enumerate target candidates
  filter by action compatibility
  retrieve top-K support pairs by source_context similarity
  compute base + memory features for each candidate
  rank candidates with GBDT/LambdaMART
  calibrate confidence
  execute or abstain
```

复杂度：

```text
num_candidates = 50-200
support_topK = 5-20
feature cost = O(num_candidates * support_topK)
ranker cost = O(num_candidates * num_trees * tree_depth)
```

目标：

```text
P50 < 5 ms on host CPU
P95 < 20 ms for 200 candidates
0 online LLM calls on cache-hit path
```

实际阈值需要用当前设备和实现测量后再冻结。

## Related Work Map

这个问题跨了多个研究传统。最容易误判的是把它归入 0-shot GUI grounding；更准确的定位是：

```text
mobile GUI test migration / widget matching
+ robust element relocalization
+ entity matching / learning-to-rank
+ memory-conditioned GUI agents
```

### Mobile GUI Test Migration And Widget Matching

CraftDroid、AppTestMigrator、TEMdroid 等工作都把“source test 在 target app / target UI 中找到对应 widget”作为核心问题。它们和我们的相似点很强：

```text
source event/widget -> target widget
with text, resource-id, class, GUI structure, semantic similarity
```

关键差异：

| Line | Their setting | Our setting |
|---|---|---|
| Task granularity | whole GUI test case migration | one recalled Function step / source action transfer |
| Target scope | often cross-app same-function tests | same app / app-family / cross-device / current runtime state |
| Output | migrated test event/oracle | executable target action or abstain |
| Learning signal | migrated test labels / widget matches | verifier-backed source-target action pairs |
| Runtime goal | test generation / maintenance | low-latency cache-hit execution |

TEMdroid 尤其相关，因为它把 widget matching 建成 learning problem，并显式处理 positive/negative imbalance 与 hard negatives。这支持我们的判断：第一版不应只写规则，也不应直接 RL，而应先把已有 evidence 派生成候选级 supervised matching table。

不过这些工作通常默认 test artifact 可以保留更多原始 GUI/test 信息。OmniFlow 的边界更接近：

```text
privacy-safe asset + current observation + verified support -> rank / reject
```

所以它们更适合作为 B1-B4/B5 的历史 baseline 和 motivation，而不是直接替代方法。

### Web Element Relocalization And Test Repair

WATER、Vista、Similo、VON Similo / HybridSimilo 一类 web test repair 工作处理的是：

```text
old web element locator breaks after UI evolves
-> find the corresponding element in the new version
```

这与我们的子问题非常接近，尤其是 Similo 的思想：不要依赖单一 locator，而是组合多种 element attributes 做 similarity scoring。它们给我们的直接启发是：

- 多属性相似度比 exact id / XPath 更鲁棒。
- visual、DOM/tree、text、geometry 都应作为弱信号组合。
- relocalization 需要 false-positive 控制，而不是只最大化找到一个元素。
- LLM 可以作为 top-K reranker，但成本和稳定性使它更适合 diagnostic / fallback。

差异也很重要：

| Line | Web relocalization | OmniFlow transfer |
|---|---|---|
| State substrate | DOM / web versions | Android/UI tree/OCR/screenshot/UDEG node |
| Historical unit | old locator / old DOM element | verified source-target action context |
| Online objective | repair broken test locator | execute cache-hit Function safely |
| Safety output | repaired selector | execute / abstain / fallback |

因此，web relocalization 是最直接的 prior-work family；我们的 novelty 不应声称“首次做 element relocalization”，而应写在 memory-conditioned action transfer、privacy-safe assetization、端侧 latency 和 verifier-backed reject 上。

### General Correspondence: SuperGlue / LightGlue

SuperGlue 和 LightGlue 解决的是图像局部特征集合之间的 correspondence。它们的方法论与本问题有对应关系：

```text
source keypoints <-> target keypoints
source UI elements <-> target UI candidates
non-matchable keypoints <-> NULL / abstain
attention / optimal assignment <-> global candidate consistency
```

可以借鉴的思想：

- 候选之间不是独立二分类，而是竞争式 matching。
- 应显式支持 unmatched / null。
- 对多步 Function，可用 assignment / min-cost flow / CRF 保持全局一致性。
- LightGlue 的 adaptive computation 思路适合 future：easy cases 用 cheap scorer，hard cases 才升级。

不建议第一版直接实现：

- SuperGlue/LightGlue 假设输入是稠密或稀疏视觉局部特征，不是带 actionability、resource-id、tree relation、history reliability 的异构 UI context。
- 端侧 cache-hit path 需要毫秒级排序和可解释 evidence。
- 我们当前 gold labels 更适合 LambdaMART/GBDT 数据效率，而不是训练 graph transformer。

结论：SuperGlue/LightGlue 是 future graph/assignment teacher，不是第一版主线。

### Entity Matching / Record Linkage

Entity matching 研究的问题是判断两个结构化记录是否指向同一个实体。DeepMatcher / Ditto 等工作证明，matching 可以从手工特征、传统模型、深度 cross-encoder 不同层次推进。

对应到我们的任务：

```text
record A = source action context
record B = target candidate context
match label = same functional target
```

这个类比有三个价值：

1. 说明 text/id/class/layout/context 这类异构字段可以用 structured matching 学习。
2. 说明 blocking / candidate generation 很关键；先枚举 action-compatible candidates，再排序。
3. 说明深度模型不是永远优先；在结构化字段强、数据有限、需要解释时，GBDT 或 learning-to-rank 经常是更稳的第一步。

与普通 entity matching 的区别：

- UI matching 是 query-group ranking，不是独立 pair classification。
- 同屏候选互斥，且存在 target absent / null。
- 历史 support 是条件变量，不只是训练集。
- wrong positive 会触发真实 GUI action，成本高于数据库误匹配。

### Learning-To-Rank

XGBoost/LightGBM 的 LambdaMART 直接提供本问题需要的训练接口：

```text
qid = source action + target screen
documents = candidates + NULL
relevance = gold target / null label
```

相比 binary classifier：

- 排序目标天然建模同一 screen 上的 candidate competition。
- 可以使用 NDCG / pairwise loss 优化 top candidate。
- 可以加入 hard negatives，而不是被大量 easy non-target candidates 淹没。
- score margin 可以直接进入 execute/abstain gate。

这是为什么第一版建议 GBDT / LambdaMART，而不是“embedding cosine + threshold”或“每对 candidate 二分类”。

## Relation To 2026 Work

### MAGNET

[MAGNET: Towards Adaptive GUI Agents with Memory-Driven Knowledge Evolution](https://arxiv.org/abs/2601.19199) 明确提出 GUI app 会发生 appearance drift 和 workflow drift，但 functional semantics 和 task intents 具有稳定性。它用 dual-level memory：stationary memory 把多样 visual features 连接到稳定功能语义，procedural memory 捕捉跨 workflow 的 task intent。

对我们的启发：

- 支持“历史 UI element/function 对齐可以帮助应对 UI drift”这个方向。
- stationary memory 与我们的 verified source-target context memory 类似，但 MAGNET 面向 agent grounding / workflow memory，我们面向端侧 candidate ranking。
- MAGNET 仍主要把 memory 注入 planner/actor/VLM；我们应把 memory 转成结构化 ranking features，避免在线长上下文推理。

### UI-Mem

[UI-Mem: Self-Evolving Experience Memory for Online Reinforcement Learning in Mobile GUI Agents](https://arxiv.org/abs/2602.05832) 用 Hierarchical Experience Memory 累积 workflows、subtask skills 和 failure patterns，并用 online RL 内化这些经验。

对我们的启发：

- 支持“经验迁移能减少重复错误”的方向。
- failure patterns 可以成为 hard negative 和 abstain 训练数据。
- 但当前单步匹配不应先做 RL。RL 适合探索、长程 credit assignment 和 memory evolution；目标元素匹配先用 supervised ranking 更直接、更可控。

### MemGUI-Bench

[MemGUI-Bench: Benchmarking Memory of Mobile GUI Agents in Dynamic Environments](https://arxiv.org/abs/2602.06075) 指出现有 mobile GUI benchmarks 很少系统评测 memory，提出 128 tasks / 26 apps 的 memory-centric benchmark，并强调 cross-temporal / cross-spatial retention。

对我们的启发：

- 可作为外部 memory 评测背景，说明 GUI agent memory 已成为 2026 明确问题。
- 我们的评测应避免只报告 single-screen grounding；需要报告 cross-session support、history coverage、stale memory 和 memory conflict。
- 它评测 agent memory，不直接评测 source-context to target-context element correspondence。我们可以把它作为外部 motivation，而不是直接 baseline。

### STAMP

[STAMP: Training Explicit Memory for Mobile GUI Agents in Controllable and Scalable Virtual Environments](https://arxiv.org/abs/2605.29324) 用可控虚拟环境注入 deterministic memory variables，训练 agent 何时记忆、何时检索，并用 Memory-World 做验证。

对我们的启发：

- 合成/可控环境只适合做单元测试、泄漏测试、calibration sanity check 和 ablation，不应作为本文核心证据。
- hard negatives、null targets、duplicate labels 和 stale-memory cases 应优先从已有 frozen/review/live artifacts 中挖掘； synthetic 只补覆盖不到的边界。
- 它关注显式记忆训练；我们可以借鉴可控变量设计，但目标是 element correspondence ranker。

### Executable Agentic Memory

[Executable Agentic Memory for GUI Agent](https://arxiv.org/abs/2605.12294) 把 GUI planning 从 step-wise free-form generation 转向 structured KG retrieval-and-execution，并通过 graph search 执行已构建 memory。

对我们的启发：

- 强化了“可执行 memory / retrieval-and-execution”是 2026 GUI agent 的重要方向。
- 它偏长程 planning 与 graph recovery；我们的子问题更窄：在执行已召回 Function 时，如何把 source action 快速定位到当前 target candidate。

### ActionEngine

[ActionEngine: From Reactive to Programmatic GUI Agents via State Machine Memory](https://www.microsoft.com/en-us/research/publication/actionengine-from-reactive-to-programmatic-gui-agents-via-state-machine-memory/) 把 GUI agent 从 reactive execution 推向 programmatic planning：offline crawler 构造 state-machine memory，online execution agent 基于 memory 合成可执行程序。

对我们的启发：

- 2026 趋势不是让 LLM 每步重新看图，而是把历史探索沉淀成可执行结构。
- 它支持 state-machine / executable memory 方向，但目标是 program synthesis / planning。
- 我们应把它作为系统级 related work；本文子问题仍是 Function 内部 action projection / candidate matching。

### LearnAct

[LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark](https://arxiv.org/abs/2504.13805) 用 demonstration-based learning 改善 mobile GUI agent 在 unseen scenarios 的表现；其公开摘要报告单个 demonstration 能显著提高 offline accuracy，online task success 也提升。

对我们的启发：

- 支持“few-shot demonstration 比追求 universal 0-shot 更贴近真实长尾 GUI”的判断。
- 我们不需要把 demonstration 交给 agent prompt，而是把 demonstration 压缩成 verified source-target support features。

### GoClick

[GoClick: Lightweight Element Grounding Model for Autonomous GUI Interaction](https://arxiv.org/abs/2604.23941) 是 2026 的轻量 GUI grounding model，关注在资源受限设备上做 natural-language-to-element grounding。

对我们的判断：

- 它不直接 match 我们的问题，因为输入是自然语言指令和截图，不是 source-context / target-context correspondence。
- 可作为可选 visual prior：输出点是否落在 candidate bbox、离 candidate center 距离、top-K overlap 等。
- 不应作为主方法，否则会把 OmniFlow 的 verified history 优势退化成 0-shot grounding。

### GUI-Actor / Coordinate-Free Grounding

[GUI-Actor](https://www.microsoft.com/en-us/research/project/gui-actor-coordinate-free-visual-grounding-for-gui-agents/) 代表另一条 2025/2026 grounding 技术线：避免直接生成坐标，改为提出 action regions / grounding verifier。

对我们的启发：

- 对 B7 visual prior 有价值，尤其是 textless controls 和 icon-only controls。
- 它仍主要解决 instruction-to-region，不解决 verified source-context to target-context transfer。
- 如果接入，应输出 candidate-level prior features，而不是替代 ranker 和 reject gate。

### MGA

[MGA: Memory-Driven GUI Agent for Observation-Centric Interaction](https://arxiv.org/abs/2510.24168) 强调 observe-first 和 structured memory，把每个交互 step 压缩成 verified state deltas。

对我们的启发：

- 支持先做 task-agnostic observation/context extraction，再做 action decision。
- 我们的 `source_context` / `target_context` 应该是结构化、可检查、可缓存的状态表示，而不是 VLM 临时描述。

## Positioning Against Recent Work

可写成：

> Recent GUI memory systems show that historical interactions are useful for adapting to UI drift and long-horizon tasks. OmniFlow focuses on a narrower execution-time problem: after a Function is recalled, how to transfer one source action to the current target UI. We formulate this as memory-conditioned contextual UI matching, where verified source-target pairs serve as few-shot support examples for a calibrated ranker over target candidates.

差异点：

| Work type | Typical input | Typical output | Our difference |
|---|---|---|---|
| Lightweight grounding | natural language + screenshot | coordinate / bbox | We use source UI context + target candidate context + history support. |
| GUI memory agent | task instruction + memory | next action / workflow | We solve the narrower candidate correspondence inside recalled Function execution. |
| Online RL memory | rollouts + reward | improved policy | We use verifier-backed supervised ranking for fast deterministic cache-hit execution. |
| Executable memory graph | task + KG | path / plan | We solve source action projection within current screen. |
| Demonstration agent | demonstrations in prompt/memory | agent action | We convert demonstrations into structured support features and calibrated abstain. |

## Evaluation Plan

### Metrics

候选匹配指标：

```text
top1_element_accuracy
MRR
target_action_accuracy@r pixels
mapped_precision
mapped_recall / coverage
abstain_rate
null_precision
wrong_click_rate
ambiguous_case_accuracy
duplicate_label_case_accuracy
```

系统指标：

```text
P50/P95 match latency
candidate enumeration latency
memory retrieval latency
serialized support memory size
online model calls
online tokens
fallback rate
terminal/effect verified success after execute
```

选择性执行指标：

```text
precision-coverage curve
false_accept_rate at fixed coverage
coverage at fixed false_accept_rate
calibration ECE / Brier score
```

### Baseline Matrix

```text
B0 coordinate replay
B1 selector/resource-id replay
B2 text/resource/class feature scorer
B3 UI tree exact replay
B4 UI tree + anchor replay
B5 base GBDT ranker without memory
B6 memory-augmented GBDT ranker
B7 optional lightweight grounding prior + ranker
B8 VLM / LLM grounding baseline, diagnostic only
```

B3/B4 很重要，因为它们代表保留 raw UI tree / target element 的强本地 replay baseline。OmniFlow 的优势不应写成“单点已知元素一定更准”，而应写成隐私安全共享、端侧低成本、run_log asset 化、跨状态复用和拒识边界。

注意：上面的 B0-B9 是本探索稿的细粒度 taxonomy。若进入当前 NSDI RQ3 正文，paper-facing baseline id 应以 `UTG_papers/nsdi/eval_matrix.md` 为准：

| Exploration taxonomy | Paper-facing RQ3 mapping | Meaning |
|---|---|---|
| B0 coordinate replay | `B0 coordinate replay` | 直接迁移历史坐标 |
| B1 selector/resource-id replay + B3 UI tree exact replay | `B1 UI tree exact replay` | 保留 raw UI tree / recorded target element 的 exact/near-exact replay |
| B4 UI tree + anchor replay | `B2 UI tree + anchor replay` | raw UI tree + anchor / geometry consistency |
| B2 text/resource/class scorer | Baseline-Failure Recovery Set 的 `text_feature` proposer | 非 gold oracle，只作自动 proposal / hard sample construction |
| B5/B6 rankers | Future method / ablation, not current frozen RQ3 claim | 需要先登记数据和 runner |
| B7/B8/B9 | Diagnostic / future only | 不能写成当前主结果 |

### Ablations

```text
no_text_features
no_layout_features
no_context_features
no_visual_features
no_memory_features
no_presence_mask
no_hard_negative_mining
no_null_candidate
no_calibration
topK_support = 0/1/3/5/10/20
history_age_decay on/off
history_failure_pattern on/off
```

### Stress Sets

```text
duplicate labels:
  multiple OK / Add / Done / same-name list rows

textless controls:
  icon-only buttons, obfuscated ids

layout drift:
  toolbar moved, bottom nav changed, density/font/theme changes

semantic drift:
  same text but different function, stale workflow, app version change

target absent:
  source action no longer available on target screen

occlusion:
  keyboard, popup, dialog, ad overlay

list identity:
  same text repeated across rows, offscreen target
```

## Platform Support In OmniFlow

当前仓库已经具备支撑这条研究线的若干入口。下面只列平台能力，不把已有 diagnostic 数字写成最终 claim。

可复用对象：

- `Function` / `function asset`
- `PageVectorSet` / page vector
- `run_log -> function asset`
- `source_context` in action params
- `get_new_coordinates(...)` / action projection

可评测入口：

```text
tests/vector/benchmarks/action_accuracy.py
tests/vector/README.md
src/utg/assets/embedding/README.md
scripts/nsdi_eval.py --experiment function-transfer
```

可扩展为新 runner 的字段：

```text
case_id
source_context
target_context
target_candidates
history_support_ids
candidate_features
ranker_score
calibrated_confidence
decision
verifier_result
latency_ms
failure_reason
```

推荐派生的中间产物：

```text
runtime/evals/action_transfer_ranker/
  feature_dump.jsonl
  train_split.json
  valid_split.json
  test_split.json
  model_config.json
  ranker_metrics.json
  calibration_metrics.json
  precision_coverage.tsv
  hard_negative_audit.json
```

注意：这些产物进入论文前必须登记到 `eval_matrix.md`。

## Public Data, Gaps, And Gold Policy

当前结论应改成：先不要使用 OmniFlow 自己收集的数据来证明方法有效。第一版 ranker 研究必须从公开数据/公开 artifact 做起，然后再把模型接回 OmniFlow 做系统 sanity。这样 reviewer 看到的是“已有 GUI migration / grounding 工作上的统一 candidate-level matching protocol”，而不是一个私有小数据集。

### Public Dataset Priority

第一批应该按可用性分三层推进：

| Priority | Public evidence | Use in this line | Gate before use |
|---|---|---|---|
| P0 | FrUITeR final datasets, CraftDroid artifact, TEMdroid Dataset/Match | 直接构造 source event -> target candidate -> gold/null 的主监督/eval表 | 下载/运行 artifact，确认 license、字段、gold/null 定义和可复算候选集合 |
| P1 | SemFinder, vision-based widget mapping, MAPIT-style mapping datasets if obtainable | 补 semantic/visual widget matching baseline 和 feature ablation | 先确认 raw data 是否公开，不能只引用论文数字 |
| P2 | ITeM, LLMigrate, MACdroid, MigratePro | 最新 LLM/abstraction/full-migration related work 和 end-to-end baseline 对照 | 只比较迁移成功、LLM calls、latency/cost 等，不把它们混作端侧 ranker |
| P3 | AITW, MobileViews, AndroidControl, GUIOdyssey, AMEX, Rico | 候选枚举、feature coverage、same-screen action grounding、hard negative、pretraining | 只在能把 action 坐标/annotation 匹配到 candidate 时生成 label |
| P4 | ScreenSpot, ScreenSpot-v2/Pro, OS-Atlas, AMEX grounding | visual/text/layout grounding prior 和 crop/icon feature评估 | 报为 grounding，不报为 transfer |
| P5 | Mind2Web, WebLINX, WebChain | web/DOM candidate-ranking analog，尤其是 history-conditioned ranking | 单独报告为 web analog，不和 Android 数字合并 |
| P6 | OmniFlow internal frozen/runtime/review artifacts | 系统接入 sanity、pipeline debug、claim consistency、small live illustration | artifact type 必须标 `omniflow_sanity`，不进入 headline supervised/eval 分母 |

这也意味着：现有 RQ3 `function-transfer` frozen/live outputs、Rico review queue、AndroidWorld/run_log evidence 都不应再被写成“第一版训练数据”。它们最多回答“外部验证过的 matcher 能不能接到 OmniFlow fast path”。

### Public Adapter Contract

ranker 需要补齐的不是模型，而是公开 artifact 到统一监督表的转换：

```text
query:
  source action/event/widget from a public artifact

documents:
  target UI/event/widget candidates + NULL if the artifact exposes absent/nonExist

label:
  public gold target candidate id
  or public NULL / absent / nonExist / missed label

support:
  few-shot history pairs drawn only from allowed training/support split
```

每个 adapter 都要先输出 `adapter_feasibility.json`：

```json
{
  "artifact_type": "fruiter",
  "has_source_event": true,
  "has_target_candidates": true,
  "has_gold_candidate": true,
  "has_null_label": true,
  "has_ui_tree_or_bbox": "partial",
  "can_recompute_candidate_set": true,
  "license_status": "verified|needs_review",
  "leakage_risk": ["same_app_duplicate", "same_trace_neighbor"]
}
```

### Missing For Supervised Ranker

公共数据仍会有缺口，但缺口要按 artifact 透明记录，而不是用内部数据补成“看起来完整”：

| Gap | Why it blocks B5/B6 | Public-first treatment |
|---|---|---|
| per-candidate rows | 不能训练 query-document ranker | 从 FrUITeR/TEMdroid/CraftDroid 先做候选展开；不行则标 `adapter_incomplete` |
| gold target candidate id | 不能计算 top1/MRR/NDCG | 只接受 released gold/mapping/oracle；heuristic match 只能是 silver |
| null / absent labels | 没有 NULL 就无法评估 fail-closed | 优先使用 FrUITeR `nonExist`；其他数据集单独报告 no-null limitation |
| target candidate recomputation | 不能复现负样本集合 | 记录候选枚举规则、候选数量、screen/app版本 |
| support cutoff | few-shot memory 可能泄漏 test answer | split 必须按 app/trace/screen/time 控制，并记录 blocked support edge |
| hard negative type | 只学 easy cases | 从重复文本、同类 widget、neighbor/list item、same action type 采样 |
| feature missingness | 缺字段本身是信号 | 输出 presence bits；不能用默认值掩盖 |
| calibration split | threshold 不能在 test 上调 | public artifact 内部切 train/valid/test，或跨 artifact transfer |
| prediction audit | false accept 必须能追溯 | 每条 prediction 回到 public artifact row 和 candidate evidence |

### Gold Policy

第一版 label set 可以允许四种等级，但主结果只用 public gold：

| Label source | Status | Allowed use |
|---|---|---|
| released mapping / GT event / benchmark oracle | public_gold | 主监督和主评测 |
| released absent / nonExist / missed target | public_null_gold | NULL/abstain 主评测 |
| coordinate-to-element match from public trace | public_derived_gold | grounding/action prediction auxiliary，不能声称 transfer gold |
| heuristic proposal / current algorithm output / internal review | silver_or_sanity | pretraining、hard-negative mining、sanity check，不进 headline |

`text_feature`、`auto_label_target`、existing algorithm proposal、OmniFlow review label 都只能当 proposal 或 sanity source。没有 public gold target 的 query 只可用于 feature coverage、candidate coverage、pretraining 或 review queue，不进入 transfer 主结果。

## Offline Runner Concrete Design

第一版 runner 应该完全离线，不连接设备，不执行 action。它不是新 benchmark generator，也不是 OmniFlow data collector，而是 public artifact adapter：把已经公开的 test-migration、action-trace、grounding、web-agent artifact 规范化成可审计、可复算的 learning-to-rank 表。

### Proposed Entry

不建议为单个数据集新造一次性脚本。若实现，应先做一个评测层统一入口，数据集 adapter 挂在它下面：

```text
python scripts/nsdi_eval.py --experiment public-action-transfer-ranker --offline \
  --artifact-manifest runtime/evals/action_transfer_ranker/public_artifacts.json
```

实现细节可以先放在 benchmark/eval utility 层，但命名和 manifest 必须外部数据优先：

```text
tests/vector/benchmarks/action_transfer_public_adapters.py
tests/vector/benchmarks/action_transfer_feature_dump.py
```

最终输出仍落到统一 runtime eval 目录：

```text
runtime/evals/action_transfer_ranker/
```

若未来晋升成 NSDI paper-facing experiment，再把该目录登记到 `eval_matrix.md`。

### Inputs

```text
artifact_manifest.json:
  source artifact path / URL / checksum
  artifact type:
    fruiter | craftdroid | semfinder | temdroid | vision_widget_mapping
    | item | llmigrate | macdroid | migratepro
    | aitw | mobileviews | androidcontrol | guiodyssey | amex | rico
    | screenspot | os_atlas
    | mind2web | weblinx | webchain
    | omniflow_sanity
  public artifact version / commit / download date
  license / redistribution status
  evidence status:
    public_gold | public_null_gold | public_derived_gold | silver | sanity
  allowed use:
    train | valid | test | baseline | pretraining | hard_negative | diagnostic | sanity

case_manifest.json:
  case_id
  dataset / artifact type
  app / package / task family
  source event/widget/action ref
  target screen/event/widget ref
  target candidate set ref
  gold label ref
  null label ref
  evidence refs
  source artifact refs

support_manifest.json:
  support_id
  public artifact ref
  source case id
  target case id
  label status
  timestamp / version / scope if available
  allowed split

split_policy.json:
  split unit: artifact / app / trace / screen / event / time
  leakage exclusions
  random seed
```

source/target raw XML、DOM、截图和 screenshot crop 可以存在本地 evidence 包中，但共享 feature dump 默认不保存 raw text、raw content-desc、raw screenshot crop。公开 artifact 本身若允许再分发，也要在 manifest 里明确 `redistribution_status`。

### Outputs

```text
runtime/evals/action_transfer_ranker/
  manifest.json
  artifact_manifest.json
  query_manifest.jsonl
  candidate_dump.jsonl
  feature_dump.jsonl
  support_manifest.jsonl
  leakage_audit.json
  train_split.json
  valid_split.json
  test_split.json
  b2_rule_scores.jsonl
  b5_base_ranker_predictions.jsonl
  b6_memory_ranker_predictions.jsonl
  ranker_metrics.json
  calibration_metrics.json
  precision_coverage.tsv
  prediction_audit.jsonl
  hard_negative_audit.json
  latency_breakdown.json
```

最关键的是 `candidate_dump.jsonl` 和 `feature_dump.jsonl`，它们必须保留
`source_artifact_ref` / `evidence_ref`，否则不能进入 paper-facing manifest。

`candidate_dump.jsonl` 一行一个 query：

```json
{
  "query_id": "q_001",
  "source_action_id": "s_001",
  "target_observation_id": "t_001",
  "candidate_ids": ["c1", "c2", "NULL"],
  "gold_candidate_id": "c2",
  "label_source": "gold_reviewed",
  "source_artifact_ref": "runtime/evals/action_projection_representative/...",
  "evidence_ref": "review_payload:pairs[17]",
  "support_ids": ["h1", "h2"],
  "split": "train"
}
```

`feature_dump.jsonl` 一行一个 candidate：

```json
{
  "query_id": "q_001",
  "candidate_id": "c1",
  "label": 0,
  "is_null": false,
  "features": {
    "action_type_compatible": 1,
    "text_token_jaccard": 0.5,
    "bbox_region_same": 1,
    "memory_top1_target_similarity": 0.72,
    "support_count": 3
  },
  "presence": {
    "text": 1,
    "resource_tail": 0,
    "crop_embedding": 0
  }
}
```

### Determinism And Reproducibility

runner 必须记录：

```text
code git sha
feature schema version
artifact manifest hash
case manifest hash
support manifest hash
split seed
model config
library versions
raw evidence local path / redacted artifact path
```

同一个 manifest + seed 应该生成 byte-stable split 和 candidate order。候选顺序稳定很重要，
因为 LambdaMART 的 group construction 依赖 query group 边界。

### Leakage Audit

每个 test query 都要检查：

```text
support trace id != query trace id
support target screen id != query target screen id unless explicitly allowed
support gold candidate id is not copied from query
support timestamp <= query timestamp for temporal split
same raw XML / screenshot hash excluded from target support
same human annotation row not reused as both support and label
```

输出：

```json
{
  "queries_checked": 1000,
  "blocked_support_edges": 37,
  "violations": [],
  "policy": "trace_screen_time_cutoff_v1"
}
```

如果有 violation，B6 只能标 diagnostic，不能晋升主结果。

## Minimal Code Plan

这一节只是实现计划，不代表当前代码已经具备这些模块。

### New Or Extended Components

推荐最小改动面：

| Component | Location | Responsibility |
|---|---|---|
| public artifact inventory | evaluation-only manifest utility | 下载/登记 FrUITeR、CraftDroid、TEMdroid、SemFinder、AITW、MobileViews、AndroidControl、GUIOdyssey、AMEX、Rico、ScreenSpot/OS-Atlas、Mind2Web/WebLINX/WebChain 的版本、license、checksum 和字段覆盖 |
| adapter feasibility checker | evaluation-only utility | 对每个 artifact 输出 has_source_event / has_target_candidates / has_gold / has_null / has_bbox_or_tree / can_recompute_candidate_set |
| direct migration adapters | benchmark/eval utility | 把 FrUITeR、CraftDroid、TEMdroid、可获得的 SemFinder/MAPIT-style artifact 转成 query/candidate/gold/null |
| action trace adapters | benchmark/eval utility | 把 AITW、MobileViews、AndroidControl、GUIOdyssey、AMEX/Rico 转成 grounding/action-candidate auxiliary rows 和 hard negatives |
| web analog adapters | benchmark/eval utility | 把 Mind2Web、WebLINX、WebChain 转成 DOM candidate-ranking analog，单独报告 |
| candidate extractor | shared eval helper | 在公开 artifact 的 target screen/event pool 上枚举 action-compatible candidates + NULL |
| feature builder | `tests/vector/benchmarks/` first; promote later only if reused by runtime | 生成 text/action/layout/context/visual/memory/presence/confidence features；不依赖私有 run_log |
| ranker trainer | evaluation-only utility | 训练 B5/B6，不进入 runtime 主链 |
| calibration evaluator | evaluation-only utility | valid split 上拟合 threshold，test 上报 precision-coverage |
| prediction audit renderer | evaluation-only renderer | 把 false accept / false reject 回链到 public artifact row；必要时输出人工 review queue |
| OmniFlow sanity adapter | future, only after offline pass | 把 trained scorer 接到 Function execution 的 action projection step 做 smoke，不进主监督/eval |

不要第一步就改 `src/utg/execution/executor.py`，也不要先造新的任务集合。先把公开 artifact 转成同一协议，并让已有方法、公开 baseline、GBDT/LambdaMART 在同一 candidate set 上可比。

### Feature Schema Version

第一版 schema 建议命名：

```text
action_transfer_features_v1
```

分组：

```text
base_text.*
base_action.*
base_layout.*
base_context.*
base_visual.*
memory_source.*
memory_target.*
memory_reliability.*
confidence.*
presence.*
```

每个 feature 必须有：

```text
name
type
range / normalization
missing policy
raw-sensitive status
shared-safe status
```

例如：

| Feature | Type | Shared-safe | Missing policy |
|---|---|---|---|
| `base_text.token_jaccard_hash` | float [0,1] | yes | 0 + presence bit |
| `base_action.type_compatible` | bool | yes | false |
| `base_layout.center_dx_norm` | float | yes | 0 + bbox presence bit |
| `memory_target.topk_max_similarity` | float [0,1] | yes | 0 if no support |
| `memory_reliability.positive_support_count` | int | yes | 0 |
| `confidence.top1_top2_margin_rule` | float | yes | 0 |

### Model Artifacts

如果训练模型，产物不要直接写成 Function asset。应单独存：

```text
runtime/evals/action_transfer_ranker/models/
  b5_base_ranker.model
  b6_memory_ranker.model
  calibration.json
  feature_schema.json
  threshold_table.json
```

只有离线结果通过后，再决定是否复制到 runtime model registry。否则它仍是
Diagnostic / Experiment。

## Concrete Paper RQs If Promoted

若后续实验支持，这条线可以压缩成一个 RQ，而不是新增整套论文主概念。

### Primary RQ

```text
RQ-Transfer:
Does verified historical support improve source-action to target-candidate
matching under strict leakage control and selective execution?
```

对应假设：

```text
H1: B6 memory-augmented ranker has higher coverage than B5 base ranker
    at the same false-accept target.

H2: B6 reduces duplicate-label and stale-selector false accepts compared with
    B2 text/resource/class rule scorer.

H3: Calibration + NULL candidate turns accuracy gain into safer execution by
    improving precision-coverage, not merely top-1 accuracy.
```

### Secondary RQs

```text
RQ-Latency:
Can the ranker run in the cache-hit fast path without online model calls?

RQ-Memory:
Which memory features matter: positive support, failure support, reliability,
or source/target similarity?

RQ-Robustness:
Where does memory hurt: stale UI, user-specific state, or scope mismatch?
```

### Minimum Table Shape

主表不要只放 top-1：

| Method | Top1 | Mapped precision | Coverage | False accept | Null precision | P95 latency | Online calls |
|---|---:|---:|---:|---:|---:|---:|---:|
| B2 rule scorer | TBD | TBD | TBD | TBD | TBD | TBD | 0 |
| B5 base ranker | TBD | TBD | TBD | TBD | TBD | TBD | 0 |
| B6 memory ranker | TBD | TBD | TBD | TBD | TBD | TBD | 0 |
| B6 + calibration | TBD | TBD | TBD | TBD | TBD | TBD | 0 |
| VLM grounding diagnostic | TBD | TBD | TBD | TBD | TBD | TBD | >0 |

`TBD` 只能由 runner 输出填充；不要手写估计值。

### Promotion Gate

只有满足下面条件才值得晋升正文：

```text
B6 > B5 on precision-coverage under strict split
B6 + calibration lowers false accept or raises coverage at fixed false accept
null-target and duplicate-label buckets are separately reported
P95 latency remains inside cache-hit budget
prediction audit explains at least the main false accepts
```

若只提升 raw top-1 accuracy，但 false accept 不降、coverage 不可控，则不建议作为
paper method 主张。

## Integration Design In OmniFlow

这个 ranker 不应该成为新的 planner，也不应该替代 control plane。它在 OmniFlow 里的位置应非常窄：

```text
Function recall
  -> start-node / page-family compatibility check
  -> action transfer ranker
  -> control-aware execution
  -> terminal/effect verify
  -> run_log writeback
```

### Runtime Boundary

输入来自三侧：

```text
Function asset:
  source action
  source_context
  expected action_type
  optional source before/after evidence

Current observation:
  PageVectorSet / UDEG node evidence
  target UI candidates
  OCR / local context / screenshot crop if local-only

Memory store:
  verified support pairs
  failure support
  reliability / scope metadata
```

输出只允许是：

```text
TransferDecision:
  decision = execute | abstain | fallback | reobserve
  target_candidate_id
  target_action
  confidence
  evidence_refs
  rejection_reason
```

它不负责：

- 路由到 Function 起点。
- 额外探索或 repair search。
- 执行设备动作。
- 判定最终 task success。
- 覆盖 host policy 或高风险确认。

这些仍由 cache-first runtime、control-aware execution、host adapter 和 verifier 承担。

### Proposed Minimal API

可以先设计一个纯函数接口，便于离线 runner 和 runtime 共用：

```python
def transfer_action(
    source_action: SourceActionRecord,
    target_observation: TargetObservationRecord,
    support_memory: Sequence[SupportRecord],
    *,
    model: ActionTransferRanker,
    policy: TransferPolicy,
) -> TransferDecision:
    ...
```

必要约束：

- `source_action` 不能只带坐标，必须带 source local context。
- `target_observation` 必须提供候选集合；候选枚举失败时返回 `reobserve/fallback`。
- `support_memory` 必须经过 scope 和 leakage 过滤。
- `model` 只打分，不执行。
- `policy` 负责 threshold、risk、host permission 和 null/abstain gate。

第一版 runtime 可以把 `model` 实现为：

```text
rule scorer -> GBDT ranker -> calibrated gate
```

这样在模型未训练前也能跑 B2/B5/B6 同协议对比。

### Where It Hooks Into Function Execution

在 cache-hit Function 执行时，每个 source step 可以按下面流程处理：

```text
for source_step in function.steps:
  observe current target state
  if source_step requires page/action transfer:
    decision = transfer_action(...)
    if decision.execute:
      executor applies target_action
      verifier / guard checks local effect
    else:
      hand off to fallback / reobserve / abort by policy
  else:
    execute literal host-safe operation
```

重点是每一步都要重新观察或至少确认 observation freshness。不能把一开始的 target candidate list 用完整个 Function，除非有明确的 cached screen freshness guarantee。

### Writeback Loop

每次执行结果都应该反向沉淀：

| Runtime outcome | Writeback object | Future use |
|---|---|---|
| verifier success | positive support pair | 提升相似 source/context 下的 candidate score |
| verifier failure / wrong effect | hard negative support | 降低相似 candidate，提升 conflict features |
| abstain but fallback succeeds | reviewed positive pair | 补 memory coverage |
| abstain and fallback also fails | null / unsafe example | 训练 target absent / unsafe gate |
| human correction | gold label + support | 进入 high-quality training set |

这让优化闭环保持在 `run_log -> function asset` 的主线内，而不是新增 prompt memory 或独立 RL policy。

### Interaction With PageVectorSet

PageVectorSet / UDEG node 的职责是 coarse state recall 和候选屏幕 / page-family evidence，不负责直接证明 target action 正确。

推荐分层：

```text
PageVectorSet:
  which current state / page family are we in?
  is this Function start/precondition compatible?

Action transfer ranker:
  within this current observation, which candidate matches the source action?

Control-aware execution:
  did the action actually produce the expected local/terminal effect?
```

这样可以避免把 page recall top-1 accuracy 误写成 action transfer success。

### Failure Handling

ranker 的失败输出应结构化：

```text
rejection_reason:
  no_action_compatible_candidate
  low_confidence
  low_margin
  null_candidate_wins
  memory_conflict
  high_risk_requires_confirmation
  target_observation_stale
  support_scope_mismatch
```

这些 reason 后续可直接汇总成：

- fallback trigger 统计；
- human review queue；
- hard-negative mining 输入；
- paper appendix failure taxonomy。

### Privacy Boundary

共享层默认只能存：

```text
hashed text / resource tails
presence bits
bucketed geometry
actionability flags
local structure fingerprints
support ids and verifier outcomes
```

本地 raw audit 包可以保留：

```text
raw XML
screenshot
OCR text
element crop
human review annotation
```

论文或多用户共享口径必须继续写成：

```text
raw evidence local-only; shared memory uses structured / hashed support.
```

否则会和 OmniFlow 的 privacy-safe UDEG node 主线冲突。

## Promotion Criteria

这条研究线要从 exploration 晋升到 NSDI 主线，至少满足下面条件。

### Minimal Promotion To Eval Matrix

可以登记到 `eval_matrix.md` 的最低门槛：

```text
existing-artifact export manifest exists
feature dump exists
split policy documented
B0/B1/B2/Ours mapping clarified
gold/null labels have evidence refs
runner emits summary.json
no raw-sensitive sharing in paper-facing artifact
```

如果只是 silver trace alignment 或人工未裁决 queue，只能标 `diagnostic` 或 `review_required`。

### Minimal Claim Boundary

可写进 claim /正文的最小表述：

```text
OmniFlow treats action transfer as a guarded candidate-ranking problem.
Verified historical source-target pairs provide support features for
selecting or rejecting target UI candidates during cache-hit execution.
```

不能写：

```text
the ranker solves all UI drift
the model guarantees safe execution
history always improves matching
VLMs are unnecessary in all cases
raw UI tree replay is weaker in every setting
```

### Minimal Experiment To Defend The Method

最小实验不是端到端大表，而是：

```text
Does B6 memory-augmented ranker improve precision-coverage
over B5 base ranker and B2 rule scorer
under strict train/test leakage control?
```

必要结果形态：

```text
precision-coverage curve
wrong-click / false-accept audit
null-target performance
duplicate-label stress set
latency breakdown
feature importance or SHAP-style explanation
```

### Expected Reviewer Questions

| Reviewer question | Required answer |
|---|---|
| Why not selector replay? | Compare against UI tree exact replay and explain privacy/current-state boundary |
| Why not VLM grounding? | Report diagnostic latency/token/cost and show history-conditioned ranker solves a narrower cache-hit problem |
| Why not RL? | Labels and verifier outcomes directly supervise ranking; exploration is unsafe and sample-inefficient for first-line matching |
| Is this just entity matching? | Yes, partly; novelty is query-group UI action transfer with null/reject, verifier-backed support, and cache-hit execution constraints |
| Does history leak test answers? | Enforce support cutoff, trace/screen exclusion, and manifest support audit |
| What if history is stale? | Reliability decay, conflict features, null gate, failure writeback |

### Decision

推荐推进顺序：

```text
1. Keep current report as exploration source.
2. Build offline feature dump runner.
3. Train B2/B5/B6 on the same candidate groups.
4. Add calibration and null candidate.
5. Only if B6 improves precision-coverage under strict split,
   register the ranker experiment in eval_matrix.md.
6. Then decide whether to promote a concise method paragraph into Design/Evaluation.
```

## Paper Framing If Promoted

如果后续实验支持 B6，这条线进入论文时不应写成一个新的大系统概念，而应作为 `Function` 执行的一段机制：

```text
When a recalled Function is executed on a current screen, OmniFlow must map
the Function's recorded source action to a target UI candidate. OmniFlow treats
this step as a guarded candidate-ranking problem. A local ranker scores
action-compatible target candidates using source-context, target-context, and
verified historical support features, and executes only when the calibrated
margin and null-gate pass.
```

中文版口径：

```text
Function 命中后，OmniFlow 不直接重放历史坐标或 selector，而是把每个历史
source action 投影到当前 observation 中的 target candidates。该投影是一个带
拒识的排序问题：模型使用 source context、target candidate context 和历史
verified support features 打分；只有置信度、margin、NULL gate 和 control policy
同时通过时，executor 才执行 target action。
```

### Contribution Wording

可防守贡献写法：

- `action transfer as guarded ranking`: 把 Function 内部 action projection 表达成 candidate ranking + NULL/reject。
- `verified support features`: 把历史成功/失败 run 变成结构化 support 和 hard negative。
- `cache-hit fast path`: 在 cache-hit 段保持本地、低延迟、可解释，而不是回到 VLM/LLM step-wise reasoning。
- `precision-coverage rather than raw top-1`: 用 selective execution 评估安全边界。

不建议贡献写法：

- “新的 GUI grounding foundation model”。
- “解决 UI drift”。
- “完全替代 VLM / RL”。
- “跨所有 app/device 的通用 action transfer”。
- “比 raw UI tree replay 永远更强”。

### Figure/Table Sketch

如果晋升到正文，可用一个小图和一张表：

```text
Figure:
  source Function step
    -> current target candidates
    -> support memory retrieval
    -> rank + null gate
    -> execute/abstain

Table:
  B0 coordinate
  B1 UI tree exact
  B2 UI tree + anchor
  B5 base ranker
  B6 memory ranker
  B6 + calibration
```

表格列不应该只放 accuracy：

```text
top1
mapped_precision
coverage
wrong_click / false_accept
null_precision
P95 latency
online model calls / tokens
```

### What Would Falsify The Method

需要提前承认失败条件：

| Observation | Interpretation |
|---|---|
| B6 不优于 B5 | verified history support 暂时不是有效特征，回到 base ranker / feature engineering |
| B5 不优于 B2 | 数据太少或特征设计不足，学习排序暂不成立 |
| B4 强过所有 ranker | raw UI tree + anchor 在本地强，但仍可保留 privacy/shared/cache-hit 边界作为系统论点 |
| calibration 覆盖率很低 | 可安全执行的 transfer subset 小，应作为 selective cache hit，而非全面 transfer |
| VLM diagnostic 更准但慢很多 | VLM 可作为 fallback / teacher，不改变端侧 fast path 的定位 |

这些不是坏结果；它们能帮助论文把主张收窄到真实可证范围。

## First Implementation Checklist

第一版实现应避免先接入 executor。先做离线 feature dump 和 runner。

### Step 1: Candidate Dump

目标：

```text
for each source action transfer case:
  enumerate target candidates
  emit query/candidate/label skeleton
```

验收：

- 每个 query 有完整 candidate list。
- 每个 query 有 `NULL` candidate。
- 每个 candidate 有 actionability、bbox、class、text/id presence bit。
- 共享 dump 不含 raw sensitive text。

### Step 2: Rule Scorer B2

目标：

```text
implement deterministic text/resource/class/action/layout scorer
```

验收：

- 输出 top1、top2、margin、null decision。
- 错误样本能进入 hard-negative audit。
- 与 current RQ3 `text_feature` proposer 口径不冲突。

### Step 3: Feature Builder

目标：

```text
build base features + presence mask + optional memory features
```

验收：

- 每个 feature 可复算。
- feature missingness 有 summary。
- hard negative 类型有统计。
- support leakage audit 通过。

### Step 4: Ranker Train/Eval

目标：

```text
train B5 base ranker
train B6 memory ranker
evaluate with identical query groups
```

验收：

- train/valid/test group 文件固定。
- 输出 `ranker_metrics.json`、`prediction_audit.jsonl`。
- 报告 top1、MRR、precision-coverage、wrong-click、null precision、latency。

### Step 5: Calibration

目标：

```text
fit gate on validation split
choose thresholds for false_accept target
```

验收：

- 输出 `calibration_metrics.json`。
- 输出 threshold table。
- 输出 abstain reason distribution。

### Step 6: Runtime Dry Run

目标：

```text
replay decisions without touching device
```

验收：

- 对每个 Function step 输出 `execute/abstain/fallback`。
- 不执行真实 device action。
- 与 offline labels 对齐。

只有这些通过后，再讨论接入 `control-aware execution` 做 live verify。

## Training And Evaluation Protocol

### Training Pipeline

推荐训练 pipeline：

```text
raw runs / reviewed pairs
  -> query/candidate/label/support records
  -> feature rows
  -> train/valid/test group files
  -> ranker model
  -> calibration model
  -> threshold table
  -> prediction audit
```

LightGBM / XGBoost 需要 query group：

```text
X: one row per candidate
y: relevance label, gold=1, non-gold=0, NULL=1 for null query
group/qid: number of candidates per query or qid column
```

训练时不能 shuffle candidate rows across query groups after group construction。

### Suggested Model Config

第一版保守配置：

```text
objective:
  LightGBM: lambdarank or rank_xendcg
  XGBoost: rank:ndcg

metrics:
  ndcg@1, ndcg@3, map, mrr

tree params:
  num_leaves / max_depth small enough for edge inference
  learning_rate: 0.03-0.10
  n_estimators: 100-500
  min_data_in_leaf: tune against query count
  feature_fraction / colsample_bytree: 0.7-1.0
```

不建议第一版做大搜索。更重要的是：

```text
strict split > hard negatives > null labels > calibration > feature importance audit
```

### Calibration Protocol

Ranker score 可以通过 validation set 转成执行 gate：

```text
score_top1
margin_top1_top2
score_null
memory_conflict_count
support_count
```

校准候选：

```text
Platt scaling:
  logistic(correct_top1 ~ score_top1, margin, score_null, conflict)

isotonic regression:
  monotonic map from score or margin to empirical precision

conformal set:
  output candidate set; execute only if set size=1 and NULL absent
```

第一版建议：

```text
train ranker on train
fit calibration on valid
choose thresholds on valid for target false_accept_rate
report test precision-coverage
```

### Threshold Selection

不要只报 top-1 accuracy。每个模型都要输出：

```text
theta_p
theta_margin
theta_null
theta_conflict
coverage
mapped_precision
false_accept_rate
wrong_click_rate
abstain_rate
```

一个可防守的 gate：

```text
execute if:
  p_top1 >= theta_p
  margin >= theta_margin
  p_null <= theta_null
  support_count >= theta_support_min
  conflict_count <= theta_conflict
else:
  abstain / fallback
```

阈值可以按风险分层：

| Action risk | Example | Gate |
|---|---|---|
| low | open tab, expand menu | lower confidence threshold |
| medium | edit field, navigate form | default threshold |
| high | submit, delete, send, pay | require confirmation or verifier precondition |

高风险动作即使 ranker 置信度高，也不应绕过 host policy。

### Metrics By Split

每个 split 至少报告：

```text
all
same-app held-out screen
same-app held-out function
cross-device target
duplicate-label cases
textless/icon cases
target-absent/null cases
stale-memory cases
```

每个分桶报告：

```text
num_queries
top1_accuracy
MRR
mapped_precision@chosen_gate
coverage@chosen_gate
wrong_click_rate
null_precision
P50/P95_latency_ms
```

### Error Taxonomy

每个 false accept / false reject 都要归因：

| Error class | Definition | Likely fix |
|---|---|---|
| `duplicate_label` | same text/id repeated, wrong row/card selected | row/list context, neighbor features, sequence consistency |
| `selector_shortcut` | model overuses resource-id or class | held-out selector split, feature dropout |
| `textless_visual` | icon-only control lacks text/id | crop embedding, lightweight grounding prior |
| `target_absent_false_accept` | correct action should be NULL | null candidate, absent support, stronger gate |
| `stale_memory` | old support conflicts with current UI | recency/reliability decay, failure writeback |
| `scope_mismatch` | user/device/locale/app variant incompatible | scope compatibility features |
| `layout_shift` | bbox/region changed | anchor/context features |
| `verifier_gap` | predicted target seems right but effect not checked | terminal/effect verifier |

这个 taxonomy 也能直接转成 human review queue。

### Ablation Protocol

Ablation 不要只移除 feature group 后报一个总 accuracy。建议固定：

```text
B5 base ranker:
  all base features
  - text
  - resource-id
  - layout
  - context
  - visual
  - presence mask

B6 memory ranker:
  B5 + memory
  - source attention
  - target support similarity
  - reliability
  - failure/conflict features
  topK = 1/3/5/10/20

Safety:
  - null candidate
  - calibration
  - hard negatives
  - conflict gate
```

每个 ablation 必须同时报告：

```text
top1 accuracy
precision at fixed coverage
coverage at fixed false_accept_rate
wrong_click_rate
latency
```

### Runtime Microbenchmark

端侧路径至少分解：

```text
candidate enumeration
feature extraction
support retrieval
ranker inference
calibration/gating
total match latency
```

建议目标：

```text
P50 < 5 ms
P95 < 20 ms for <=200 candidates and topK<=20 support
0 online LLM calls
```

这些目标进入论文前必须由 runner 输出真实设备/host 上的测量值。

## Implementation Roadmap

### Phase 0: Public Artifact Inventory And Feasibility

目标：不训练复杂模型，先确认公开 artifact 到底能不能支撑 query/candidate/gold/null。输出不是结果数字，而是可审计的 data feasibility map。

产物：

```text
public_artifacts.json:
  artifact_type
  URL / local path / checksum / commit
  license status
  has_source_event
  has_target_candidates
  has_gold_candidate
  has_null_label
  has_ui_tree_or_bbox
  can_recompute_candidate_set
  recommended_use
```

验收：

- FrUITeR、CraftDroid、TEMdroid 至少完成 artifact feasibility audit。
- AITW、MobileViews、AndroidControl、GUIOdyssey、AMEX、Rico 明确只能做 auxiliary / hard-negative / feature coverage，除非 adapter 能生成 public-derived candidate labels。
- WebLINX、Mind2Web、WebChain 标为 web analog，不与 Android 主结果混合。
- OmniFlow internal artifact 只注册为 `omniflow_sanity`。

### Phase 1: Direct Migration Adapter

目标：把最贴合本问题的 GUI test migration / widget matching artifact 转成 candidate-level export。

优先顺序：

```text
FrUITeR:
  source event -> target candidate events -> gt/correct/nonExist

TEMdroid:
  source widget/event -> target widget candidates -> match label

CraftDroid:
  source test event -> target event/widget mapping -> transfer oracle where available

SemFinder / vision-based mapping:
  only after raw artifact is confirmed
```

验收：

- 每个 query 有稳定 candidate set。
- `public_gold` / `public_null_gold` 与 `silver` 分开。
- candidate dump 和 feature dump 能回链到 public artifact row。
- NULL 不存在的数据集单独报告 no-null limitation。

### Phase 2: Public Action Trace And Grounding Adapter

目标：不把 action trace 冒充 transfer gold，而是提取端侧 matcher 需要的 feature、hard negative 和 grounding auxiliary data。

产物：

```text
AITW / AndroidControl / GUIOdyssey:
  action coordinate or annotation -> same-screen UI candidate label

MobileViews / Rico:
  screenshot + VH -> candidate enumeration and feature missingness

AMEX / ScreenSpot / OS-Atlas:
  text/visual/element grounding features

Mind2Web / WebLINX / WebChain:
  DOM/action-history ranking analog
```

验收：

- 所有 rows 标注 `public_derived_gold`、`pretraining`、`hard_negative` 或 `web_analog`。
- 不把 task success、raw coordinate 或 DOM target 写成 Android transfer gold。
- feature coverage、candidate count、hard-negative bucket 有统计。

### Phase 3: GBDT / LambdaMART Ranker

目标：在公开 direct-migration data 上训练 base ranker 与 memory/few-shot ranker。

配置：

```text
model: LightGBM LGBMRanker or XGBoost ranker
objective: lambdarank / rank:ndcg
group: query_id
features: text + action + layout + context + visual + presence + support
```

验收：

- B6 明显优于 B5 或在相同 false-accept 下覆盖更高。
- direct migration data 与 action-trace auxiliary data 分开报告。
- hard-negative stress set 有单独提升。
- model importance 能解释哪些 source/target/support features 有效。

### Phase 4: Calibration, Reject, And Latency

目标：把 ranker score 变成端侧执行 gate，而不是只追求 top-1。

产物：

```text
calibration model
precision-coverage curve
threshold table
false accept audit
abstain reason distribution
latency_breakdown.json
```

验收：

- 能给出固定 false accept 目标下的 coverage。
- NULL / absent / duplicate-label bucket 单独报告。
- P50/P95 match latency 满足端侧 cache-hit budget。
- 错误样本有 top evidence 和 support conflict。

### Phase 5: OmniFlow Integration Sanity

目标：接入 cache-hit Function execution 的 action projection path，只做系统 smoke，不作为主 supervised/eval 证据。

边界：

- ranker 只做 target candidate selection。
- repair、route、recovery 仍由 control-aware execution / fallback 处理。
- target absent 或低置信度时 fail-closed。
- sanity case 必须标 `omniflow_sanity`，不能回写成 public-data result。

### Phase 6: Optional Visual Prior

目标：只在低 margin 或 text/context 弱时调用轻量 grounding prior。

特征：

```text
goclick_inside_candidate
goclick_center_distance
goclick_rank
goclick_iou
```

它是 feature，不是主决策器。

## Recommended First Experiment

最小可防守实验：

```text
Question:
  On public GUI migration/widget-matching artifacts, does split-safe few-shot
  support improve source-context -> target-context action matching beyond
  selector/text/layout features?

Models:
  B2 text/resource/class scorer
  B5 base GBDT ranker
  B6 memory-augmented GBDT ranker
  B6 + calibration

Splits:
  held-out query/source action
  held-out target screen
  held-out app/task family if data allows
  cross-artifact transfer if multiple public artifacts are usable

Metrics:
  top1 accuracy
  mapped precision
  wrong click rate
  abstain rate
  precision-coverage
  P95 latency
```

第一篇结果不需要证明端到端 agent 全面领先，只需要证明：

```text
split-safe few-shot support from public artifacts is a useful feature for
target candidate disambiguation under strict reject and latency constraints
```

## Risk Register

| Risk | Why it matters | Mitigation |
|---|---|---|
| Silver labels are noisy | raw-step alignment may not be gold | separate diagnostic silver from gold/verifier labels |
| History leakage | support may contain test query | strict support exclusion by trace/screen/task |
| Selector-dominant shortcut | model may learn exact id only | split by app/version/device and ablate selector fields |
| False accept | wrong click is costly | calibrated abstain and null candidate |
| Stale memory | old support may become invalid | timestamp, version, recent failure features |
| User-specific state | history may not apply across users | scope metadata and compatibility gates |
| Privacy | raw XML/screenshot/text may leak | hashed/normalized fields and local-only raw audit |
| Latency creep | visual or graph model too expensive | two-stage cheap ranker, visual prior only on low margin |
| Method swallowed by new VLMs | 0-shot grounding may improve | keep contribution on verified memory, structured support, calibration, and cache-hit execution |

## Concrete Recommendation

第一版不要做 RL，不要做端到端 VLM fine-tuning，也不要直接实现 SuperGlue-like attention graph。建议路线：

```text
1. build candidate-level feature dump
2. train base GBDT / LambdaMART ranker
3. add few-shot memory support features
4. mine hard negatives from ranker false positives
5. add null candidate and calibration
6. report precision-coverage and latency
7. only then consider tiny MLP / graph consistency / lightweight grounding prior
```

论文叙事应聚焦：

> OmniFlow turns action transfer from zero-shot grounding into memory-conditioned contextual matching. Verified source-target pairs provide few-shot support for a fast, calibrated ranker over target UI candidates, enabling cache-hit execution to stay local, explainable, and fail-closed.

## References

- [TEMdroid: A Testing-Based Approach to Android GUI Test Migration](https://yakzhang.github.io/2024-icse-temdroid.pdf), ICSE 2024 PDF.
- [SuperGlue: Learning Feature Matching with Graph Neural Networks](https://arxiv.org/abs/1911.11763), arXiv 2019.
- [LightGlue: Local Feature Matching at Light Speed](https://arxiv.org/abs/2306.13643), arXiv 2023.
- [DeepMatcher: A Deep Learning Approach to Entity Matching](https://pages.cs.wisc.edu/~anhai/papers1/deepmatcher-sigmod18.pdf), SIGMOD 2018 PDF.
- [MAGNET: Towards Adaptive GUI Agents with Memory-Driven Knowledge Evolution](https://arxiv.org/abs/2601.19199), arXiv 2026.
- [UI-Mem: Self-Evolving Experience Memory for Online Reinforcement Learning in Mobile GUI Agents](https://arxiv.org/abs/2602.05832), arXiv 2026.
- [MemGUI-Bench: Benchmarking Memory of Mobile GUI Agents in Dynamic Environments](https://arxiv.org/abs/2602.06075), arXiv 2026.
- [STAMP: Training Explicit Memory for Mobile GUI Agents in Controllable and Scalable Virtual Environments](https://arxiv.org/abs/2605.29324), arXiv 2026.
- [Executable Agentic Memory for GUI Agent](https://arxiv.org/abs/2605.12294), arXiv 2026.
- [ActionEngine: From Reactive to Programmatic GUI Agents via State Machine Memory](https://www.microsoft.com/en-us/research/publication/actionengine-from-reactive-to-programmatic-gui-agents-via-state-machine-memory/), Microsoft Research, 2026.
- [GoClick: Lightweight Element Grounding Model for Autonomous GUI Interaction](https://arxiv.org/abs/2604.23941), arXiv 2026.
- [GUI-Actor: Coordinate-Free Visual Grounding for GUI Agents](https://www.microsoft.com/en-us/research/project/gui-actor-coordinate-free-visual-grounding-for-gui-agents/), Microsoft Research project page.
- [LearnAct: Few-Shot Mobile GUI Agent with a Unified Demonstration Benchmark](https://arxiv.org/abs/2504.13805), arXiv 2025.
- [MGA: Memory-Driven GUI Agent for Observation-Centric Interaction](https://arxiv.org/abs/2510.24168), arXiv 2025.
- [XGBoost Learning to Rank documentation](https://xgboost.readthedocs.io/en/latest/tutorials/learning_to_rank.html).
- [LightGBM Parameters documentation](https://lightgbm.readthedocs.io/en/latest/Parameters.html).
