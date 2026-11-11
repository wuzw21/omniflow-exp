# OmniFlow 论文写作与术语规范

本文档统一 OmniFlow 论文的故事线、术语、章节边界和英文表达。论文的
核心不是普通 replay，也不是让 VLM 在新环境中重新 generalize，而是：

> 将成功 GUI 执行中的可复用操作保存为 Function，并通过 OmniTransfer
> 将历史 action Transfer 到当前 UI，从而实现跨设备的经验共享。

## 1. 统一故事线

```text
成功的 RunLog
    ↓
稳定的语义段
    ↓
可调用的 Function
    ↓
Function Recall
    ↓
OmniTransfer action Transfer
    ↓
直接执行，或回到 VLM / planner
```

VLM GUI agent 通常通过 `observe--reason--act` 循环完成任务。这个循环同时
承担两种能力：

1. `workflow/planning`：决定做什么，以及如何组合多个操作；
2. `action Transfer`：将 action Transfer 到当前 UI 中的实际目标。

OmniFlow 将两者分开。Function 保存可以复用的操作和执行顺序，
OmniTransfer 负责在设备、布局、内容或 session 变化后将 action Transfer 到
当前 UI 中可以执行的目标。

## 2. 固定术语

| 术语 | 统一含义 |
|---|---|
| `OmniFlow` / `\\sysname` | 完整系统 |
| `Function` | 可调用、state-bounded 的 GUI 操作 |
| `Function Store` | 保存 Function 及其 source UI state 的存储 |
| `RunLog` | 成功的 GUI 执行记录 |
| `UI` | GUI 界面 |
| `UI state` | 某一时刻的一次 UI observation |
| `source UI` | action 被记录时的 UI |
| `current UI` | action 需要执行时的 UI |
| `UI element` | UI 中的可见或可交互元素 |
| `UI Representation` | 从 screenshot 和 UI hierarchy 得到的表示 |
| `Function Recall` | 根据当前 UI 和 task goal 召回 Function |
| `action Transfer` | 将 source UI 中的 action Transfer 到 current UI |
| `OmniTransfer` | 提供 UI Representation 和 action Transfer 的学习组件 |
| `VLM` | 原有的 vision-language model |
| `planner` | 决定操作并组合 Function 的 agent 部件 |
| `checker` | 处理重复 UI 条件的可复用规则 |
| `observe` / `act` | 现有 GUI agent 的物理交互接口 |
| `MCP tool` / `tool list` | 向 agent 提供 Function 的接口 |

## 3. 推荐表达与禁用词

优先使用：

- `cross-device Function reuse`
- `executable GUI experience`
- `stable semantic segment`
- `state-bounded Function`
- `shared UI Representation`
- `local information`
- `page-level Function Recall`
- `fine-grained action Transfer`
- `Transfer score`
- `execution threshold`
- `direct execution`
- `VLM or planner fallback`
- `alternating Function execution and VLM reasoning`
- `device-specific UI variation`
- `callable tool`
- `API-like interface`

以下词汇不要使用，除非它们是相关工作的专有名词：

- `UDEG`、`Universal-Dynamic Execution Graph`、`state index`；
- `online encoder`、`functional subspaces`；
- `primitive`、`invoked`、`corresponding`；
- `trace`，统一使用 `RunLog`；
- `evidence`，统一使用 `UI information` 或 `recorded UI state`；
- `operational representation`、`grounding decoder`、`action relocation`；
- `source anchor`、`action endpoint`、`affinity logit`；
- 将 OmniFlow 称为 `replay system`；
- 把 `generalization` 作为论文的中心贡献。

## 4. 章节写作边界

### Introduction

按以下顺序写：

1. VLM 每一步通过 `observe--reason--act` 完成任务；
2. 重复 routine 会造成高延迟和高计算开销；
3. 语义 memory 仍然需要 VLM 读取、解释和规划；
4. 固定脚本和简单 selector 可以直接执行，但难以适应跨设备 UI 变化；
5. operation 和 action order 可以复用，而 UI target 需要重新 Transfer；
6. OmniFlow 用 Function 表示稳定段，用 OmniTransfer 完成 action Transfer；
7. 成功步骤直接执行，不确定步骤回到 VLM 或 planner。

### Challenges and Motivations

只保留两个 challenge：

1. `Reusable workflow units`：完整脚本过于固定，语义 memory 又需要重新
   思考，因此需要可独立调用、可组合、可与 VLM 交替执行的 Function；
2. `Cross-device action Transfer`：操作可以复用，但 UI target 会随设备、
   布局、内容和 session 改变，因此需要共享 UI Representation 和局部信息。

### System Design

推荐结构：

```text
Architecture and Workflow
Offline Function Augmentation
OmniTransfer
    Local UI Representation
    Function Recall and action Transfer
Online Function Recall and Execution
    observe--check--transfer--act
    checker
    VLM fallback
```

`Function Store` 放在 Offline，`Function Recall` 和 action Transfer 放在
Online。OmniTransfer 是 Offline 和 Online 共用的 UI Representation 与
action Transfer 学习组件，不要再引入 UDEG 或另一套 encoder。

### Implementation

Implementation 分为四层：

1. plugin integration；
2. OmniTransfer model architecture；
3. training data and objective；
4. deployment。

模型层数、输入维度、参数量、loss、训练数据、optimizer 和训练 epoch 放在
Implementation 或 Appendix。Evaluation Setup 只需说明使用哪个 frozen
checkpoint 以及训练数据规模。

### Evaluation

区分不同层次的结论：

- end-to-end success：完整 OmniFlow 系统；
- conditional Function reuse：成功 Function 是否能在 target device 执行；
- Function Recall：页面级 Function 召回；
- action Transfer：单步 UI Transfer；
- latency and model usage：执行效率和 VLM 开销。

不要仅用 task success 宣称 Transfer accuracy，也不要仅用 action Transfer
accuracy 宣称 workflow planning 能力。

## 5. OmniTransfer 的固定表述

OmniTransfer 是一个统一的 learned model，承担两个作用：

1. 生成共享的 `UI Representation`，用于 page-level Function Recall；
2. 利用 UI element 及其 local information，完成 action Transfer。

推荐英文表述：

```latex
\\textsc{OmniTransfer} provides the shared UI Representation used by both
Function Recall and action Transfer. It encodes screenshots and UI hierarchies
into a compact representation, while local information from nearby UI
elements helps transfer a recorded action to the current UI.
```

推荐中文含义：

```text
OmniTransfer 提供同时用于 Function Recall 和 action Transfer 的共享
UI Representation。它将 screenshot 和 UI hierarchy 编码为紧凑表示，
并利用附近 UI element 的局部信息，将历史 action Transfer 到当前 UI。
```

## 6. 执行与 fallback

执行契约必须明确：

```text
Transfer score 足够高 → 执行 transferred action
Transfer score 不足   → Transfer failure → 返回 VLM / planner
```

不能写成 Transfer 失败后继续执行 source coordinate。也不能把整个 Function
视为一次不可分割的 script。正确表述是：

> 一个 Function 可以直接执行多个稳定步骤；当某一步需要新的 UI 判断时，
> 系统返回 VLM 或 planner；之后 agent 可以继续选择并执行另一个 Function。

## 7. 结果与论断

只有在结果真实存在时才使用：

```text
Our experiments show that ...
```

尚未完成的实验使用：

```text
We evaluate whether ...
```

没有结果时使用 `TBD`，不要补写推测数字。不要声称“完全鲁棒”“完全隐私安全”
或“解决所有 UI 变化”，除非论文实验明确支持这些结论。

## 8. 最终校对清单

- 是否先解释 why，再解释 how？
- 是否明确跨设备 GUI experience sharing 是核心问题？
- 是否明确 Transfer 而不是 generalization？
- 是否区分 workflow/planning 与 action Transfer？
- 是否把 Function 写成 callable、stable、state-bounded operation？
- 是否说明 action 保留 source UI state？
- 是否只有 OmniTransfer 负责 UI Representation 和 action Transfer？
- 是否删除 UDEG 和单独的 encoder？
- 是否区分 Function Recall 与 action Transfer？
- 是否明确 direct execution、VLM fallback 和交替执行？
- 是否避免 `invoked`、`corresponding`、`primitive`？
- 是否避免无依据的 accuracy、robustness、privacy 和 scalability 结论？
- 是否检查模型参数和训练数据与最终 checkpoint 一致？
