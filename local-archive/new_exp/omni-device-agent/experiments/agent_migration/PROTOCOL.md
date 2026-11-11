# Harness 迁移最小实验协议

状态：协议草案，尚未运行。

## 目的

隔离一个问题：在模型、设备和工具实现都不变时，源 Harness 改成目标 Harness 是否会改变最终任务状态；中间表示编译是否能减少这种损失。

## 固定条件

- 运行时：同一个 Python 环境和同一个本地 inference engine；
- 模型：一个固定的 tool-use 模型；
- 设备：单台机器，先不测跨硬件；
- 工具：同一组 typed local tools；
- 工作目录：每次任务使用新建、哈希固定的 fixture workspace；
- 随机性：固定 seed、temperature 和 max turns；
- 副作用：只允许 fixture workspace 内的读写，禁止网络和真实用户目录。

## Harness

同一 OpenJarvis 版本内：

```text
Source: NativeOpenHandsAgent
Target: NativeReActAgent
```

两者都使用同一模型和同一工具实现。Source trace 记录每个 `Observe`、`Act`、`ToolResult`、`Verify`、`Retry` 和 `Stop` 事件。

## 工具任务

先使用不依赖外部网络的三类任务：

1. `organize_files`：根据文件内容和日期移动文件；
2. `patch_repo`：读取指定文件、修改一处代码、运行测试并报告结果；
3. `recover_failure`：第一次工具调用故意失败，要求识别错误、修正参数并验证最终状态。

每个任务定义：

```text
initial fixture hash
allowed tools
preconditions
postconditions
failure conditions
maximum turns
```

任务成功由 fixture 的最终状态和测试结果判定，不由 Agent 的自然语言自述判定。

## 对照

每个任务运行三种条件：

1. `direct-copy`：把 source prompt、tool 描述和控制逻辑直接用于 target；
2. `manual-port`：为 target Harness 手工改写一次 schema、prompt 和循环配置；
3. `compiled`：由 canonical event IR 生成 target prompt、tool adapter、验证和重试配置。

适配集用于生成 `manual-port` 和 `compiled` 的配置；隐藏任务只用于最终验收。所有条件使用相同的模型和 fixture。

## 记录指标

- 最终 fixture 状态正确率；
- tool schema 合法率；
- 工具参数语义正确率；
- postcondition 满足率；
- 错误恢复成功率；
- 重复调用和无进展循环次数；
- 隐藏任务成功率；
- prompt、schema、控制逻辑的人工修改量；
- 端到端延迟和模型调用 token 数。

## 必须保存的证据

每次运行保存：

```text
source/target harness version
model revision and engine version
fixture hash
source trace
compiled artifact
target trace
final fixture hash
failure classification
```

## 判定标准

只有同时观察到以下事实，才能说明 Harness 迁移是独立问题：

1. `direct-copy` 在模型不变时仍出现工具或最终状态失败；
2. 失败来自控制循环、观察格式或验证语义，而不是模型随机性；
3. `compiled` 在隐藏任务上减少失败或人工修改；
4. 如果 `manual-port` 已经达到相同结果，必须报告为负结果；
5. 若 source 与 target 的最终状态本来完全一致，则不进入模型替换实验。

## 后续因素

只有上述 Harness-only 对照显示真实迁移损失后，才依次加入：

```text
同 Harness 换模型
跨模型 + 跨 Harness
外部 OpenHands SDK
PalmClaw typed device tools
不同 inference engine / hardware
```

不得在第一轮同时改变模型、Harness、工具和设备，否则无法判断方法解决了什么。
