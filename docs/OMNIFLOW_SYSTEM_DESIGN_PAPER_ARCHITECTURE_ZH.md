# OmniFlow System Design 论文架构建议稿

> **原稿保留：** 本文档不覆盖
> `docs/OMNIFLOW_SYSTEM_DESIGN_ZH_REVIEW.md`。原稿继续作为三段文字的完整材料库。
>
> **本文目标：** 将原稿从“按三次讨论堆叠”改为“按论文读者的理解顺序组织”，形成约三页 LaTeX 的 System Design 架构。

## 一、先给结论：论文不能直接沿用当前排布

当前审阅稿完整保留了三段讨论，适合作为材料库，但还不是论文结构，原因有四点：

1. `Architecture Overview` 直接从 OmniTransfer 开始，读者尚未知道完整系统输入、输出和三个组件的关系。
2. `Offline Function Augmentation` 同时讨论 Function、Checker、执行正确性和 plugin 接入，离线与在线职责交叉。
3. `OmniTransfer` 同时包含 Motivation、相关方法讨论、模型设计、实现参数和在线流程，单节承担的任务过多。
4. `Online Function Recall and Execution` 再次解释 Recall 和 OmniTransfer，导致同一机制重复出现。

论文版应采用一个清楚的总分结构：

```text
Architecture Overview
        ↓
Offline：将成功 RunLog 组织成 Function Memory
        ↓
Shared Component：OmniTransfer 连接 source UI 与 current UI
        ↓
Online：Recall、执行、Checker、fallback 和经验回流
```

这里不是简单地把三个大段依次放入三个 subsection，而是让每个 subsection 回答一个独立问题：

| Subsection | 唯一需要回答的问题 |
|---|---|
| Architecture Overview | 整个系统如何从 RunLog 得到并复用 Function？ |
| Offline Function Augmentation | Function Memory 是什么，如何从 RunLog 得到？ |
| OmniTransfer | 历史 Function 如何连接到变化后的 current UI？ |
| Online Function Recall and Execution | Function 如何进入现有 agent loop，并安全地执行或 fallback？ |

## 二、推荐的最终 LaTeX 层级

```latex
\section{System Design}
\label{sec:design}

\subsection{Architecture Overview}

\subsection{Offline Function Augmentation}
\paragraph{Function Memory Representation.}
\paragraph{Function Augmentation.}

\subsection{OmniTransfer}
\paragraph{Shared UI Representation.}
\paragraph{Local-context-enhanced action Transfer.}

\subsection{Online Function Recall and Execution}
\paragraph{Function Recall and Tool Exposure.}
\paragraph{State-bounded Function Execution.}
\paragraph{Shared Checkers and Planner Fallback.}
\paragraph{Experience Feedback.}
```

建议不要在 System Design 中设置过深的三级或四级标题。OmniTransfer V10 的“第一层、第二层、第三层”是模型实现结构，不应成为 System Design 的主层级。

## 三、三页篇幅分配

| 部分 | 建议篇幅 | 段落数 | 主要内容 |
|---|---:|---:|---|
| Architecture Overview | 0.30--0.40 页 | 2 | 输入输出、三个组件、整体数据流 |
| Offline Function Augmentation | 0.65--0.75 页 | 3 | Function 表示、状态绑定、RunLog augmentation |
| OmniTransfer | 0.95--1.10 页 | 4 | shared UI Representation、局部信息、Recall readout、action Transfer |
| Online Function Recall and Execution | 0.75--0.90 页 | 4 | tool exposure、执行、Checker、fallback、反馈 |
| **合计** | **2.65--3.15 页** | **13** | 不含大图或只含一张跨栏架构图 |

如果架构图占据约半页，正文必须控制在 2.5 页左右。Implementation 中再解释模型维度、层数、参数量和部署配置。

# 四、逐节设计

## 4.1 Architecture Overview

> **本节目的：** 让第一次阅读论文的人在两段内理解系统做什么、三个组件如何配合。
>
> **核心设计：** 成功 RunLog 经过离线 augmentation 形成 Functions；OmniTransfer 提供统一 UI Representation；在线 runtime Recall 并执行 Function，失败时返回 planner。

### 第一段：系统输入、输出和三个组件

这一段只定义整体系统，不讨论 UTG、模型层数或具体执行循环。

应包含：

- 输入是 agent 或用户成功执行产生的 RunLog；
- 输出是可以加入 agent tool list 的 Functions；
- 系统包含 Offline Function Augmentation、OmniTransfer、Online Function Recall and Execution；
- Function 保存可复用 workflow，OmniTransfer 连接 source UI 和 current UI。

建议叙述顺序：

```text
RunLog 容易从真实执行中获得
→ 离线提取稳定段落并形成 Function
→ OmniTransfer 编码 Function 中的 source UI states
→ 在线根据 current UI 和 task goal Recall Function
→ 直接执行或返回 VLM/planner
```

### 第二段：Offline 与 Online 数据流

这一段配合架构图，从左到右走一遍系统：

```text
Offline:
successful RunLog
→ stable segments
→ Function interface + ordered actions + source UI states
→ Function Store

Online:
current UI
→ UI Representation
→ Function Recall
→ planner selects a Function
→ action Transfer
→ direct execution / fallback
```

这一节必须明确：`OmniTransfer` 不是独立放在 pipeline 末尾的 element matching 模块，而是 Offline 和 Online 共享的 UI component。

## 4.2 Offline Function Augmentation

> **本节目的：** 解释为什么 Function 不是普通完整脚本，以及 Function Memory 保存什么。
>
> **核心设计：** 从成功 RunLog 中提取稳定连续段落，将每段保存为可调用、可组合且与 UI state 绑定的 Function。

### Paragraph 1：Function Memory Representation

这一段先定义 Function，再解释它为何可以复用。Function 应包含：

1. semantic interface 和 parameters；
2. ordered actions；
3. entry UI state；
4. 每个 action 对应的 source UI state；
5. 由 OmniTransfer 为这些状态生成的 UI Representation。

这里必须第一次引入 `UI Representation`，否则 Function 会被读者理解为普通脚本。正确关系是：

```text
Function Memory
├── semantic interface
├── parameters
├── ordered actions
└── recorded UI states
    └── encoded as UI Representations by OmniTransfer
```

这一段要形成一个清楚的定义：Function 不只记录“做什么”，还记录“这些 actions 在什么 UI state 下成功执行”。

### Paragraph 2：为什么使用稳定段落而不是完整 workflow

使用原稿中的核心洞察：RunLog 容易收集，脚本难以由用户编辑；因此系统从成功 RunLog 中识别语义稳定的连续段落。

这一段需要强调三点：

- Function 是“一段稳定的执行”，不是完整任务的强制重放；
- Function 可以单独使用，也可以由 planner 组合；
- Function execution 可以与 VLM/planner reasoning 交替进行。

不应在这里声称 Function 自身拥有广泛语义泛化能力。Function 保持 workflow 的确定性，跨设备适用性由 recorded UI states 和 OmniTransfer 提供，复杂任务适应性由 Function composition 与 fallback 提供。

### Paragraph 3：Function Augmentation

这一段解释 RunLog 如何成为 Function，但只保留设计级步骤：

```text
successful RunLog
→ identify stable continuous segments
→ preserve action order
→ expose variable values as parameters
→ associate each action with its source UI state
→ register Functions in the Function Store
```

不需要在 Design 中写 compiler schema、`source_state_id` 文件格式、JSON 字段或生成 prompt；这些属于 Implementation。

### 本节不应出现的内容

- Checker 的触发阶段；
- fallback prompt；
- OmniTransfer 的具体网络结构；
- `128D`、`1024D`、`K=16`；
- “别人没有做状态绑定”这类需要 Related Work 和证据支持的绝对判断。

## 4.3 OmniTransfer

> **本节目的：** 解释同一 Function 为什么能连接到不同设备上的 current UI。
>
> **核心设计：** 一个共享的 UI Representation 同时支持页面级 Function Recall 和细粒度 action Transfer；局部 UI 信息提高相似 element 之间的区分能力。

### Opening paragraph：为什么需要 OmniTransfer

Function 保存 workflow 和 source UI states，但 source coordinate、Resource ID、XML path 和 node index 不能直接作为跨设备 identity。因此，需要一个组件回答两个不同粒度的问题：

1. current UI 支持哪些历史 Functions？
2. 被选中 Function 的 source action 应当 Transfer 到哪个 current UI element？

这两项任务应使用共享的 UI element representations，避免页面级 Recall 与 action Transfer 依赖两套不一致的 UI 编码。

UTG 的完整批评不放在这里。Design 中最多保留一句设计选择：系统不维护覆盖所有设备和动态页面的全局状态图，而是保存离散 Functions，并在当前 UI 上执行 Recall 和 Transfer。

### Paragraph 1：Shared UI Representation

这一段合并原稿“第一层”和“第二层”的设计思想，而不是逐层介绍模型实现。

每个 UI state 由同步的 screenshot 和 UI hierarchy 表示。OmniTransfer 为每个 UI element 编码：

- text 和 content description；
- widget type 和 interaction state；
- normalized geometry；
- element region 和 nearby visual region。

随后，模型使用 bounded local information 更新 UI element representation，包括 parent--child、sibling、proximity、relative direction、alignment 和 containment。

本段的关键不是列特征，而是解释局部信息为什么必要：单个 element 可能缺少 text、Resource ID，或者与多个 element 具有相似外观；它与附近 text、input field、list item 或 action icon 的关系更能说明其在当前 UI 中的作用。

### Paragraph 2：为什么使用 bounded local information

这一段专门说明设计取舍：

- 全页面稠密关系会带来二次计算开销；
- 只使用单个 element 属性无法区分相似控件；
- bounded local information 在计算成本和 Transfer 准确性之间取得平衡。

使用“给 Alice 打电话”的例子即可，不需要再列搜索框、联系人、卡片等多个例子。一个完整例子比四个碎片例子更适合论文。

### Paragraph 3：支持 Function Recall 的页面级读出

OmniTransfer 从 contextualized UI element representations 得到页面级 UI Representation。Function entry UI states 可以离线编码并加入 Function Store，current UI 在运行时使用同一模型编码。

这一段只回答：

> 当前 UI 支持哪些历史 Functions？

页面相似度用于筛选 UI-compatible Functions，task goal 随后用于对这些 Functions 进行语义排序。具体向量维度和 attention slots 放入 Implementation。

### Paragraph 4：Local-context-enhanced action Transfer

这一段解释 source UI 与 current UI 的联合处理：

- 输入是 source UI、source action point 和 current UI；
- 两个 UI 分别经过共享编码器；
- cross-UI association 比较 source action target 与 current UI candidates；
- local information 用于更新 Transfer scores；
- 达到 threshold 才执行，否则返回 Transfer failure。

结尾必须明确：系统不会在失败后使用 source-device coordinate。

### 本节不应出现的内容

- `V10` 名称反复出现；论文正文可统一称 `OmniTransfer`；
- 128 维 node、1024 维 page embedding、两层 local graph、八个 slots；
- 训练 loss、数据集、checkpoint；
- 九步在线执行流程，因为在线 workflow 属于下一节；
- “判断当前任务是否仍然能够完成”，因为 OmniTransfer 只负责 UI compatibility 和 action Transfer。

## 4.4 Online Function Recall and Execution

> **本节目的：** 解释 Function 如何进入现有 GUI agent，并在执行失败时自然返回 planner。
>
> **核心设计：** OmniFlow 是插入 `observe--plan--act` loop 的 executable cache layer，只动态暴露相关 Functions，并通过 state-bounded execution、shared Checkers 和 context-preserving fallback 保持执行连续性。

### Paragraph 1：Function Recall and Tool Exposure

在每次 planning 前，runtime 观察 current UI，并使用 OmniTransfer 的 UI Representation Recall UI-compatible Functions；task goal 再对候选 Functions 进行排序。

系统只把少量候选 Functions 加入当前 tool list，而不是把所有 Functions 或完整 RunLogs 放入 prompt。Planner 可以选择候选 Function，也可以继续使用原有 GUI actions。

这里应强调兼容接口，而不是宣称已经证明 token 大幅下降。Design 只说该设计使 planner 输入不必随 Function Store 线性扩张；具体 token reduction 由 Evaluation 证明。

### Paragraph 2：State-bounded Function Execution

Planner 选择 Function 并提供参数后，runtime 完成参数绑定，再按 ordered actions 执行：

```text
observe
→ check
→ action Transfer
→ act
→ observe resulting UI
```

每个 action 都使用自己的 source UI state 和 current UI 完成 Transfer。达到 threshold 才执行；失败立即停止当前 Function。

这一段把“控制流稳定”和“每一步状态绑定”连接起来：ordered actions 保持不变，但每一步是否可执行由 current UI 决定。

### Paragraph 3：Shared Checkers and Planner Fallback

Checker 应放在 Online，而不是 Offline Function Design。准确表述是：

- interruption-handling actions 可以从 RunLog 中注册为 shared checkers；
- checkers 存在于独立共享库，不固定属于某个 Function；
- runtime 在适用阶段根据 current UI 判断 checker 条件；
- checker action 仍由 OmniTransfer 完成 action Transfer。

Function action Transfer 或 Checker 失败后，runtime 返回 planner，并保留：

- Function 名称和描述；
- 已绑定参数；
- 已成功 actions；
- 失败 step 和错误；
- current UI；
- OmniTransfer 返回的候选 UI elements 和 Transfer scores。

Planner 从当前状态继续，而不是从任务起点重新开始；低置信度候选只能作为 hint，不能绕过 threshold 执行。

### Paragraph 4：Experience Feedback

成功任务无论由 Function、planner actions，还是两者交替完成，都使用同一记录接口形成 RunLog。经过验证的成功 RunLog 可以再次进入 Offline Function Augmentation；经过检查的 source/current UI pairs 可以用于后续 OmniTransfer training。

这里必须写成离线反馈：在线执行只记录数据，不在当前任务中立即更新 Function 或模型。

### Plugin 能力应如何表述

System Design 中保留一句高层接口要求即可：现有 GUI agent 需要提供 `observe`、`act` 和 tool-calling planner。MCP、具体 hook、tool schema、代码接口和部署方式放入 Implementation 的 `Plugin Integration`。

# 五、原稿内容如何迁移

| 原稿内容 | 论文位置 | 处理方式 |
|---|---|---|
| RunLog 廉价、脚本难收集 | Offline Function Augmentation 开头 | 保留并压缩为动机句 |
| 语义稳定段落 | Function Memory / Function Augmentation | 作为核心洞察保留 |
| Function 是 API 或宏 | Function Memory | 统一写为 callable、API-like operation |
| 起始状态与每一步状态绑定 | Function Memory + Online Execution | 两处各承担定义和执行含义 |
| Function 组合提供灵活性 | Function Memory | 保留，并说明与 planner 交替执行 |
| Checker 挂载每个节点 | Online Shared Checkers | 改为符合实现的 shared checker library |
| UTG 长篇讨论 | Challenges and Motivations / Related Work | 从 Design 移出，只保留一句设计选择 |
| 美团动态页面例子 | Motivation 或删除 | 没有系统性数据时不要作为实证结论 |
| 多模态 element 特征 | OmniTransfer / Shared UI Representation | 保留设计原因，压缩特征列表 |
| 128D、1024D、K、layers、slots | Implementation | 全部移出 Design |
| 两个任务输出 | OmniTransfer | 改写为两个用途，而非两个并列普通 head |
| 九步 OmniTransfer 流程 | Online Execution | 与完整系统 workflow 合并，避免重复 |
| 动态 tool search | Online Recall and Tool Exposure | 保留 |
| “大幅减少 token” | Evaluation | Design 中改为可扩展性机制 |
| Observe-Check-Act | Online State-bounded Execution | 扩展为 observe--check--transfer--act--observe |
| fallback 携带 Function 信息 | Online Fallback | 保留并具体化 |
| OmniTransfer 候选操作 | Online Fallback | 保留为 hint，不绕过 threshold |
| 完成后打包保存 | Experience Feedback | 明确保存为 RunLog，离线 augmentation |
| 与特定 Prompt 强绑定的问题 | Motivation / Related Work | Design 只说明我们的窄接口 |

# 六、架构图应该画什么

架构图建议只有三列，并用 OmniTransfer 贯穿 Offline 和 Online：

```text
┌──────────────────────┐
│ Offline Augmentation │
│ successful RunLog    │
│ → stable segments    │
│ → Functions          │
└──────────┬───────────┘
           │ source UI states
           ▼
┌──────────────────────────────────────┐
│ OmniTransfer                         │
│ shared UI Representation             │
│ page-level Recall + action Transfer  │
└──────────┬───────────────────────────┘
           │ Function Store / candidates
           ▼
┌──────────────────────────────────────┐
│ Online Recall and Execution          │
│ observe → recall → tool selection    │
│ → check → transfer → act             │
│ → fallback or success                │
└──────────┬───────────────────────────┘
           │ successful RunLog
           └──────────────→ Offline
```

图中必须让读者看到三个关系：

1. Function 来自成功 RunLog，而不是人工脚本；
2. OmniTransfer 同时服务 Function Store 构建和在线执行；
3. Function execution 与 planner fallback 可以交替，成功结果重新进入离线流程。

# 七、最终的论文叙述主线

完整 System Design 应让读者按以下顺序形成认识：

1. **先理解系统对象：** OmniFlow 将成功 RunLog 中的稳定段落保存为 Functions。
2. **再理解 Memory 表示：** Function 保存 semantic interface、ordered actions 和每一步 source UI state，因此不是普通脚本。
3. **再理解跨设备能力：** OmniTransfer 用共享 UI Representation 连接 source UI 和 current UI，同时支持 Function Recall 与 action Transfer。
4. **最后理解执行方式：** 候选 Function 作为 tools 进入现有 planner；runtime 逐步 check、Transfer 和 act，失败时带着上下文返回 planner。
5. **形成闭环：** 成功的 Function/planner 混合执行再次生成 RunLog，用于扩展 Function Store 和后续 OmniTransfer training。

用一句话概括整个 Design：

> OmniFlow 将真实执行产生的成功 RunLog 组织为与 UI state 绑定的 callable Functions，使用 OmniTransfer 将这些 Functions 连接到变化后的 current UI，并通过可插入现有 GUI agent 的在线 runtime 完成 Recall、直接执行与 planner fallback。
