# OmniFlow System Design：Challenge-Driven 中文架构 V2

> **版本关系：** 完整原始文字稿和上一版论文架构均保持不变。本版本在它们的基础上，显式恢复 `Challenge → Key Insight → Design` 的对应关系。
>
> **组织原则：** 正向说明已有材料如何形成论文结构，不否定原稿。原稿已经包含完整的动机、洞察和技术内容；V2 只加强这些内容之间的论文逻辑。

> **叙述优先级：** System Design 的核心是三项从真实 RunLog 得到的观察、由这些观察导出的系统抽象，以及每项抽象解决的问题。复杂执行顺序、模型维度、规则字段和存储格式不是本节主线，只保留证明抽象能够成立所需的最短说明，其余移至 Implementation。

## 一、整篇 System Design 的总逻辑

System Design 首先列出三个具体设计挑战，然后用三个系统部分逐一回答：

```text
C1：RunLog 如何成为细粒度、可执行的 Memory？
    ↓
Key Insight 1：RunLog 中存在语义稳定的连续段落；
               可复用 workflow 必须与 entry UI state 和每一步 source UI state 绑定。
    ↓
Design 1：Offline Function Augmentation

C2：Function 如何跨设备连接到变化后的 current UI？
    ↓
Key Insight 2：operation 和 action order 可以复用，
               但 action target 必须根据 current UI 重新 Transfer；
               Function 的跨设备适用性应由共享 UI Representation 和
               local UI information 提供，而不是要求 Function 自身覆盖 UI 变化。
    ↓
Design 2：OmniTransfer

C3：在线如何选取、执行 Function，并与现有 agent 稳定切换？
    ↓
Key Insight 3：只向 planner 暴露当前 UI 支持的候选 Functions；
               Function 通过现有 tool interface 接入，只要求 agent 提供
               observe/act loop 和支持 tool calling 的 planner。
    ↓
Design 3：Online Function Recall and Execution
```

三个挑战共同服务于论文的核心目标：

> 将真实设备上自然产生的成功 RunLog 转换为可以跨设备共享、直接执行，并能与现有 GUI agent 交替使用的 Function Memory。

## 二、三个 Challenge、Key Insight 与 Design 的对应表

| 编号 | Challenge | Key Insight | 对应设计 |
|---|---|---|---|
| C1 | 如何将带噪、重复且面向一次具体执行的 RunLog，组织为细粒度、可调用、可组合的 executable Memory？ | RunLog 中存在语义稳定的连续段落；稳定 workflow 只有与 entry UI state 和每一步 source UI state 共同保存时，才具备可控执行条件。 | Offline Function Augmentation：Function Memory Representation + Function Augmentation |
| C2 | 如何让同一份 Function Memory 在设备、布局、内容和 session 变化后仍能连接到 current UI？ | workflow 和 action order 可以复用，但 action target 需要重新 Transfer；Function 保持稳定的控制流，跨设备适用性由共享 UI Representation 和 local information 提供。 | OmniTransfer：Shared UI Representation + Local-context-enhanced action Transfer |
| C3 | 如何在在线执行中快速 Recall 合适的 Function，并在不替换原有 agent loop 的前提下完成复用与切换？ | 将 Function 作为动态提供的 tools，通过 observe/act 和 tool calling 两类已有接口接入；Function execution 与 planner reasoning 因而可以交替进行。 | Online Function Recall and Execution：Function Recall + Tool Exposure + State-bounded Execution + Fallback |

## 三、推荐的 LaTeX 层级

```latex
\section{System Design}
\label{sec:design}

\subsection{Design Overview}
% C1--C3 + architecture workflow

\subsection{Offline Function Augmentation}
\paragraph{Challenge and key insight.}
\paragraph{Function Memory Representation.}
\paragraph{Function Augmentation.}

\subsection{OmniTransfer}
\paragraph{Challenge and key insight.}
\paragraph{Shared UI Representation.}
\paragraph{Function Recall and action Transfer.}

\subsection{Online Function Recall and Execution}
\paragraph{Challenge and key insight.}
\paragraph{Function Recall and Tool Exposure.}
\paragraph{State-bounded Function Execution.}
\paragraph{Checker and Planner Fallback.}
```

这种层级保留了原稿中的三个主要部分，同时让每个部分都从自己的 Challenge 和 Key Insight 开始，再进入具体设计。

# 四、逐节叙述路径

## 4.1 Design Overview

> **本节回答：** OmniFlow 为跨设备 executable GUI experience sharing 解决哪些设计挑战，三个组件如何逐一回答？
>
> **核心设计：** 先建立 C1--C3 的设计合同，再通过一张图串联 Offline、OmniTransfer 和 Online。

### 第一段：列出三个设计挑战

System Design 可以直接采用 MobileGPT 式的提问方法，但问题必须围绕 OmniFlow 自己的目标：

```text
To enable cross-device sharing of executable GUI experience,
OmniFlow addresses three design challenges:

C1. How can noisy RunLogs be organized into fine-grained,
    callable, and state-bounded Function Memory?

C2. How can recorded Functions remain executable when UI states
    change across devices, users, and sessions?

C3. How can an agent Recall and execute the appropriate Functions
    through its existing observe/act loop and tool-calling planner?
```

这三个问题与前面的 `Challenges and Motivations` 保持一致，但这里不需要再次展开全部案例和数据。前文负责证明问题真实存在，本节负责把问题转换成系统设计任务。

### 第二段：给出三个组件及其对应关系

紧接着用一段话回答 C1--C3：

```text
OmniFlow contains three corresponding components.
Offline Function Augmentation extracts stable segments from successful
RunLogs and stores them as state-bounded Functions. OmniTransfer builds
a shared UI Representation for Function Recall and transfers each recorded
action to the current UI. Online Function Recall and Execution connects these
Functions to an existing agent through its tool interface, allowing Function
execution to alternate with planner reasoning.
```

### 第三段：配合架构图走完数据流

```text
successful RunLog
→ stable semantic segments
→ Functions in the Function Store
→ current UI Representation
→ Function Recall
→ planner selects a Function
→ check and action Transfer
→ direct execution or planner fallback
→ successful RunLog
```

这一段点明 OmniTransfer 是 Offline 和 Online 共享的 UI component：Offline 编码 Function 的 source UI states，Online 编码 current UI 并完成 Recall 和 action Transfer。

## 4.2 Offline Function Augmentation

> **对应 C1：** 如何将带噪、重复的 RunLog 组织为细粒度 executable Memory？
>
> **Key Insight 1：** RunLog 中存在语义稳定的连续段落；稳定 action order 需要与成功执行时的 UI states 一起保存。
>
> **Design 1：** 将稳定段落表示为 state-bounded、callable、composable Functions，并加入 Function Store。

### Paragraph 1：Challenge and Key Insight

这一段保留原稿最重要的动机和洞察：

- 用户或 engine 在真实执行中可以自然产生 RunLog；
- RunLog 的收集比人工编辑脚本容易；
- RunLog 包含噪声、重复内容和面向一次执行的具体操作；
- 成功 RunLog 中存在可以独立复用的语义稳定段落。

建议形成如下递进：

```text
RunLog 容易收集，但不能直接等价为可复用 Memory
→ 完整 RunLog 中只有部分连续段落在其他任务中仍然稳定
→ 将这些稳定段落作为独立 Function
```

这里必须给出清楚的抽象边界：Function 是从成功 RunLog 中提取、保持原始连续 action order 的可调用段，而不是对任务过程生成的一段语义总结。语义接口使它能够作为 tool 使用，连续 actions 和逐步 UI states 使它能够直接执行并接受状态检查。

### Paragraph 2：Function Memory Representation

Function 是一个可调用的 GUI operation，包含：

1. semantic interface 和 parameters；
2. ordered actions；
3. entry UI state；
4. 每个 action 关联的 source UI state。

这里首次引入 `UI Representation`：OmniTransfer 为 Function 保存的 source UI states 构建 UI Representation。Entry UI Representation 用于后续 Function Recall，每一步 source UI Representation 用于 action Transfer。

因此，Function 不只说明“执行哪些 actions”，还保存“这些 actions 在什么 UI state 下成功执行”。这就是 `state-bounded Function` 的定义，也是它与脱离状态的固定 action sequence 之间最清楚的区别。

更具体地说，逐步 source UI state 绑定将 Function Memory 与只保存 action sequence、selector 或 coordinate 的 action-only script 区分开来。后者记录了“做什么”，而 Function 还记录了“每一步在什么状态下成立”，从而为在线状态检查、action Transfer 和失败停止提供共同依据。这里应直接写出这项差异，而不能只把它隐藏在 `state-bounded Function` 的定义中。

### Paragraph 3：Function 的组合与交替执行

这一段必须完整保留原稿的设计哲学，而不能拆散到 Function 和 OmniTransfer 两节中分别说明：脚本自身难以覆盖所有任务和 UI 变化，因此 OmniFlow 不要求单个 Function 具备广泛的语义泛化能力。Function 只保存一段语义稳定、执行顺序确定的 workflow，并保持 API-like operation 的原子性。复杂任务的灵活性来自多个 Functions 的自由组合，以及 Function execution 与 VLM/planner reasoning 的交替；设备、布局和内容变化下的适用性则由 UI Representation 和 OmniTransfer 提供。

因此，Function 的职责不是“自己适应所有变化”，而是提供可控、可组合的稳定控制流；OmniTransfer 的职责是把这段控制流连接到 current UI。两者共同构成完整抽象，不能把跨设备适用性归因于 Function 本身。

### Paragraph 4：Function Augmentation

离线 augmentation 采用以下设计流程：

```text
successful RunLog
→ identify stable continuous segments
→ preserve the action order of each segment
→ expose reusable values as Function parameters
→ associate each action with its source UI state
→ register Functions in the Function Store
```

这一段讲清设计流程即可；compiler schema、文件格式和具体 prompt 由 Implementation 继续说明。

## 4.3 OmniTransfer

> **对应 C2：** 如何让 Function 在设备、用户、布局、内容和 session 变化后仍能连接到 current UI？
>
> **Key Insight 2：** Function 的 operation 和 action order 可以保留，但 action target 必须根据 current UI 重新 Transfer；UI element 的 local information 可以在单一属性变化或缺失时提供稳定的作用信息。
>
> **Design 2：** OmniTransfer 使用共享 UI Representation 同时支持页面级 Function Recall 和细粒度 action Transfer。

### Paragraph 1：Challenge and Key Insight

原稿中的跨设备问题在这里正面展开：同一 GUI operation 在不同设备上可能具有不同屏幕尺寸、布局、动态内容、UI hierarchy 和 element attributes。历史 RunLog 可以告诉系统“过去做了什么”，但不能保证 source coordinate、Resource ID、Index 或 XML path 在 current UI 中保持不变。

核心洞察是将两类信息分开：

```text
可复用：operation + ordered actions
需要重新确定：每个 action 在 current UI 中的 target element
```

OmniTransfer 因而承担两个粒度的任务：先判断 current UI 支持哪些历史 Functions，再将选中 Function 的 action Transfer 到 current UI。

### Paragraph 2：Shared UI Representation

OmniTransfer 从同步的 screenshot 和 UI hierarchy 中构建共享 UI Representation。它同时表示 UI element 自身的多模态信息和所在页面，使 Function Store 中的 source UI 与在线的 current UI 进入同一表示空间。该表示同时服务页面级 Function Recall 和细粒度 action Transfer，避免为两个任务维护彼此独立的 UI 表示。

### Paragraph 3：Local UI Information

单个 UI element 的 text、Resource ID、appearance 或位置可能缺失或变化，但它在局部组件中与附近 UI elements 的作用关系仍能为 action Transfer 提供信息。因此，OmniTransfer 使用有界 local UI information 校准 UI element 表示，使模型判断 element 在当前局部结构中“承担什么作用”，而不只比较“它自身看起来是什么”。有界局部范围也使该表示适合轻量、端侧友好的执行。

正文保留一个“给 Alice 打电话”的例子即可：当页面存在多个相似电话图标时，Alice 姓名与附近图标的局部关系帮助模型选择正确 target。具体输入属性、关系类型、候选范围和更新层数移至 Implementation。

### Paragraph 4：Function Recall and Action Transfer

共享 UI Representation 提供两个明确能力：页面级结果回答“current UI 支持哪些历史 Functions”，细粒度结果回答“选中 Function 的 action 应当作用于哪个 current UI element”。这两个能力共同把 Function 保存的稳定控制流连接到 current UI。

只有 Transfer score 达到 execution threshold 时才执行 action。否则返回明确的 Transfer failure，并进入正常 planner fallback，不使用 source-device coordinate。

### Implementation 承接的内容

V10 的 node dimension、page dimension、local candidate limit、模型层数、attention slots、训练 objective、数据集和 checkpoint 都继续保留，但由 Implementation 的 `OmniTransfer Model Architecture` 与 `Training` 承接。System Design 重点解释这些结构为什么存在。

## 4.4 Online Function Recall and Execution

> **对应 C3：** 在线如何选取适用的 Function，稳定执行，并在不适用时与现有 planner 切换？
>
> **Key Insight 3：** Function reuse 可以作为 agent tool use 的一种选择。OmniFlow 不替换 agent 原有的 Prompt、planning loop 或 action decoder，而是通过 observe/act 和 tool calling 两类已有接口动态提供 Functions，使 Function execution 与 planner reasoning 能够交替进行。
>
> **Design 3：** OmniFlow 作为 plug-in cache layer 插入 `observe--plan--act`，提供 Function Recall、tool exposure、state-bounded execution 和 planner fallback。

### Paragraph 1：Challenge and Key Insight

当 Memory retrieval 依赖特定 Prompt、planning loop 或 decoding format 时，Memory reuse 会与该 agent 的执行流绑定。OmniFlow 的观察是，GUI tool agent 已经具备完成接入所需的两个边界：获取 current UI 的 observe/act loop，以及从动态 tool list 中选择操作的 planner。

因此，OmniFlow 不替换原有 planner，而是让 Function 成为现有 tool list 中的一类可调用操作。系统先利用 current UI 和 task goal Recall 少量候选 Functions，再让 planner 在候选 Functions 与原有 GUI actions 之间选择。论文应把“窄接口接入”作为本节的核心设计，而不是把篇幅集中在具体执行器的内部顺序上。该兼容性应限定为支持 observe/act 和 tool calling 的 GUI agents，而不写成对任意 agent 的无条件兼容。

### Paragraph 2：Function Recall and Tool Exposure

OmniFlow 根据 current UI 和 task goal Recall 少量候选 Functions，并只把它们的 semantic interfaces 和 parameters 加入当前 tool list。Planner 可以选择 Function，也可以继续使用原有 GUI actions。该动态 tool exposure 使 Function Store 可以独立增长，而 agent 只接收当前步骤相关的 Functions；具体 Recall 顺序和 tool schema 移至 Implementation。

### Paragraph 3：State-Bounded Execution and Fallback

在线层只需说明这个抽象如何维持：Function 保留稳定的 ordered actions，但每一步都与自己的 source UI state 绑定；action Transfer 失败时停止当前 Function 并把控制权交还 planner。由此，Function execution 可以与 planner reasoning 交替，而不是以固定方式重放完整 RunLog。

Checker 作为辅助机制用一句话交代即可：Checker 保存在 shared library 中以支持跨 Function 复用；每条规则可限定 Function ID、step index、action type 和 package，并单独指定执行 phase。这里的“可插拔”是 Function/step-level 的可配置作用点，不是 UI node-level hook。Checker rule schema、phase 名称、触发预算和内部执行顺序属于 Implementation。

成功执行继续形成 RunLog，并可用于扩展 Function Store 和后续 OmniTransfer training，从而闭合数据使用链路。正文无需在此枚举 fallback context 的每个字段；只需说明 planner 从 current UI 和未完成 Function 的执行上下文继续任务。

## 五、三页篇幅安排

| 部分 | 建议篇幅 | 叙述重点 |
|---|---:|---|
| Design Overview | 0.35--0.45 页 | C1--C3、三个组件、完整数据流 |
| Offline Function Augmentation | 0.65--0.75 页 | C1、Key Insight 1、Function Memory、augmentation |
| OmniTransfer | 0.95--1.10 页 | C2、Key Insight 2、shared representation、local information、两个用途 |
| Online Function Recall and Execution | 0.60--0.75 页 | C3、窄接口接入、动态 tool exposure、交替执行与 fallback |

如果包含约半页架构图，正文应进一步压缩例子和特征枚举，但继续保留每一组 `Challenge → Key Insight → Design` 的首段。

## 六、与 Challenges and Motivations 的衔接

前面的 `Challenges and Motivations` 和这里的 `System Design` 承担不同任务：

```text
Challenges and Motivations：
证明为什么 reusable workflow 和 cross-device action Transfer 是真实问题。

System Design：
将这些问题收敛为 C1--C3，并说明每个 Key Insight 如何导出一个系统组件。
```

因此，System Design 可以简短回顾 Challenge，但不重复前文的数据、案例和完整相关方法分析。读者能够看到明确对应，而不会感觉同一段 Motivation 被写了两次。

## 七、最终读者路径

第一次阅读论文的人应当依次获得以下认识：

1. RunLog 很容易从真实执行中得到，但需要组织成稳定、细粒度的 Function Memory；
2. Function 保存 workflow、entry UI state 和每一步 source UI state，因此是 state-bounded executable Memory；
3. OmniTransfer 用共享 UI Representation 和 local information 支持跨设备 Function Recall 与 action Transfer；
4. OmniFlow 只通过 observe/act 和 tool calling 接口将候选 Functions 提供给现有 planner；
5. Function execution 与 planner reasoning 可以交替；成功结果重新形成 RunLog，扩展后续 Memory 和训练数据。

用一句话概括这一版的 System Design：

> OmniFlow 通过 Function Augmentation、OmniTransfer 和 Online Function Execution 依次解决 RunLog 组织、跨设备 UI Transfer 和在线 Memory reuse 三个挑战，将成功 GUI experience 转换为可调用、可组合、可跨设备执行并能与现有 planner 交替使用的 Function Memory。
