# Agent 迁移：模型与 Harness 变化时的语义保持部署

状态：研究规格草案，尚未声称方法成立。

## 问题定义

云端 Agent 迁移到个人设备时，变化的通常不只是模型。模型、工具接口、上下文格式、控制循环、失败处理和推理后端可能一起变化。直接复制 prompt、替换模型或重写 tool schema，无法在部署前判断任务是否仍然可完成，也无法保证工具调用后的设备状态保持一致。

本项目研究：

> 给定一个源 Agent 及其执行轨迹，以及一个模型、Harness 和工具后端都可能不同的目标设备，能否生成一个任务语义保持的目标 Agent，并在真实任务上验证迁移是否成立？

这里的“语义保持”不是 token 相同，而是：

- 工具调用的输入含义相同；
- 工具调用前置条件和后置状态成立；
- 验证、重试、重规划和停止条件保持；
- 数据流满足本地/云端边界；
- 设备资源约束得到满足。

## 编译对象

编译器的输入不是一个模型文件，而是四类对象：

```text
SourceAgentSpec
  planner/model invocation
  observation channels
  tool contracts
  control loop and verification rules

SourceTrace
  observation -> decision -> tool call -> result -> verify/retry/stop

TargetProfile
  model and tokenizer
  inference engine and hardware
  local tools and permissions
  memory/latency/privacy budgets

TargetHarness
  ReAct, CodeAct, typed native tools, or another supported loop
```

输出是一个可执行部署包和一份验证报告：

```text
TargetAgentSpec
  target model/template
  target tool schemas and adapters
  observation serialization
  control-flow lowering
  local/cloud boundary
  resource configuration

ConformanceReport
  supported/unsupported tasks
  tool-schema validity
  state-transition correctness
  latency/memory/cloud exposure
  unresolved risks
```

## 最小中间表示

第一版只保留与迁移语义直接相关的事件，不引入新的记忆系统：

```text
Observe(source, modality, provenance)
Decide(goal, constraints)
Act(tool, typed_arguments, side_effect_class)
Verify(preconditions, postconditions)
Retry(condition, limit)
Replan(reason)
Stop(condition)
```

工具定义至少需要：输入类型、前置条件、副作用、返回类型、成功条件、失败条件和是否幂等。模型特定的 JSON、XML 或 function-call 格式只是这个接口的后端表示。

例如，源 Harness 可能用：

```text
bash("mv invoice.pdf Bills/")
```

目标手机 Harness 可能用：

```text
file.move(source="invoice.pdf", destination="Bills/invoice.pdf")
```

迁移成立的依据不是两个字符串相似，而是执行后源文件消失、目标文件出现等后置条件成立。

## Harness 变化的处理

Harness 变化分成四种可测的 lowering：

| 源/目标差异 | 需要迁移的语义 |
|---|---|
| ReAct ↔ CodeAct | Action 是 typed tool 还是代码/命令，以及如何解析结果 |
| function call ↔ native device tool | 工具 schema、权限和副作用边界 |
| full history ↔ compaction | 哪些观察和验证结果必须保留 |
| retry loop ↔ verifier-driven loop | 何时重试、何时停止、何时重新规划 |

第一版不实现所有组合。为避免把安装差异误当成研究结果，先在同一个 OpenJarvis 发行版内使用两个真实 Harness：`NativeOpenHandsAgent`（CodeAct）和 `NativeReActAgent`（Thought-Action-Observation）。官方文档明确提供这两个入口及其不同控制循环。模型、工具实现和设备先固定，以隔离 Harness 迁移；外部 OpenHands SDK、PalmClaw typed tools 和模型替换放入后续因素。

## 最小区分实验

采用 (2\times2) 因素设计：

| | 模型不变 | 模型改变 |
|---|---|---|
| Harness 不变 | 基准 | 模型适配 |
| Harness 改变 | Harness 迁移 | 完整迁移 |

每个格子使用相同的文件、代码和浏览器任务，并设置明确的最终设备状态。比较：

1. 直接复制 prompt、工具和控制循环；
2. 人工重写目标 Harness；
3. 中间表示编译器自动生成目标部署。

实验必须使用适配集和隐藏任务：适配集只能用于生成 adapter，隐藏任务只用于验收，防止按任务手工调参后冒充迁移能力。

记录：

- 最终设备状态正确率；
- 工具 schema 合法率和参数语义正确率；
- 前置/后置条件满足率；
- 重试、重规划和停止是否正确；
- 隐藏任务成功率；
- 人工修改的 prompt/tool/harness 行数；
- 迁移耗时、端到端延迟、内存、能耗和云端数据暴露。

## 与已有工作的边界

- OpenJarvis 已经提供可组合的 Intelligence、Engine、Agent、Tools/Memory 和 Learning 原语；本项目若只是在配置文件里换组件，就是重复工作。必须证明输入是一个已有源 Agent，输出是跨 Harness 的语义保持部署，并有隐藏任务验收。
- Agent-X 优化固定端侧 Agent 的 prefill/decode；它可以作为目标 Engine 的优化后端，不解决源 Agent 到目标 Harness 的迁移。
- Hera 做设备/云端的 step-level 路由；它假定 Agent 和工具语义已经存在，不负责 Harness lowering 和状态契约验证。
- PalmClaw 提供手机原生 Harness；它可以作为目标 Harness，但不能自动证明任意云端 Agent 已经迁移成功。

## 证伪条件

以下任一情况出现，就不能把本方向写成新贡献：

1. OpenJarvis 等现有系统在相同的零人工修改约束下已经完成源 Harness 到目标 Harness 的迁移；
2. 只需更换模型和一条 prompt 就能达到相同隐藏任务成功率；
3. 编译器只改善字符串格式合法率，不能改善最终设备状态正确率；
4. Harness 改变后不需要验证、重试或控制流适配；
5. 方法只在调参任务上有效，换到新的工具或新的设备就失效。

## 近期执行顺序

1. 固定 OpenJarvis 的 `NativeOpenHandsAgent`、`NativeReActAgent` 和一组本地 typed tools；
2. 先完成“模型不变、Harness 改变”的迁移矩阵；
3. 用状态级验证区分直接复制、人工移植和自动编译；
4. 再加入同 Harness 换模型；
5. 最后才加入外部 OpenHands SDK、PalmClaw、Engine 和设备变化。

在第 2 步没有观察到明确的 Harness 迁移损失前，不引入模型训练、KV 优化或端云路由。
