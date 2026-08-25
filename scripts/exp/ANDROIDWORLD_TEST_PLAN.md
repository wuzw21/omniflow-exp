# AndroidWorld 冷启动、转换与热执行测试计划

所有真实执行仍通过唯一入口 `scripts/exp/run_androidworld.sh`。配方脚本
`scripts/exp/plans/run_condition.sh` 只负责固定共同参数并选择一个阶段；它不会扫描结果、
寻找最新 attempt、校验 seed 或保存额外 manifest。

## 共同条件

- task、device、source seed、evaluation seed、步数、fallback、deadline 和模型在各组间相同。
- 默认值为 `SystemBluetoothTurnOn`、`standard45562`、111、113、20、5、600 秒和
  `Qwen3.6-Plus`。
- 各方法使用相同 AndroidWorld task goal 和 task parameters。只改变方法原生的 Memory
  输入；不增加任务专用 Prompt、预启动动作或隐藏 plan。
- MobileGPT 聊天/规划、OmniFlow Planner、Function 转换和 T3A 均显式使用
  `Qwen3.6-Plus`。MobileGPT 向量检索继续使用专用 embedding 模型，不把聊天模型当
  embedding 模型。
- 每个阶段单独启动，RunLog 和 Memory 路径由实验者显式传给下一阶段。

## 实验组

| 编号 | 阶段 | 输入 | 输出/目的 |
|---|---|---|---|
| 1 | MobileGPT 冷启动 | 空 Memory | 官方自主探索 RunLog |
| 1.5 | MobileGPT 转换 | 成功冷启动 RunLog | 官方 MobileGPT Memory bundle |
| 2 | MobileGPT 热执行 | bundle 中的 `memory/` | 测量原生 Memory 复用 |
| 3 | OmniFlow 冷启动 | 空 Function Store | Qwen Planner 自主执行 RunLog |
| 3.5 | OmniFlow 增强转换 | 成功冷启动 RunLog | `store.json`、`compile_report.json`、`transfer_states.json` |
| 4 | OmniFlow 热执行 | `store.json` | 测量 Function/OmniTransfer 复用 |
| 5 | Script replay | 成功 source RunLog | `fixed_replay` 对照 |
| 6 | T3A + hint | 成功 source RunLog | `t3a_hint` 对照 |

## 执行

共同参数可以直接用默认值，也可以在每条命令前以环境变量覆盖。以下示例故意把阶段拆开，
避免任何自动结果发现。

```bash
# 1. MobileGPT 冷启动
bash scripts/exp/plans/run_condition.sh mobilegpt-cold

# 1.5. 把成功冷启动 RunLog 转成 MobileGPT 原生 Memory
SOURCE_RUN_LOG=/explicit/cold/run_log.json \
MEMORY=/explicit/mobilegpt_bundle \
bash scripts/exp/plans/run_condition.sh mobilegpt-convert

# 2. MobileGPT 热执行
MEMORY=/explicit/mobilegpt_bundle/memory \
bash scripts/exp/plans/run_condition.sh mobilegpt-hot

# 3. OmniFlow 冷启动
bash scripts/exp/plans/run_condition.sh omniflow-cold

# 3.5. RunLog -> 增强 Function
SOURCE_RUN_LOG=/explicit/cold/run_log.json \
MEMORY=/explicit/omniflow_memory \
bash scripts/exp/plans/run_condition.sh omniflow-convert

# 4. OmniFlow 热执行
MEMORY=/explicit/omniflow_memory/store.json \
bash scripts/exp/plans/run_condition.sh omniflow-hot

# 5. Script replay
SOURCE_RUN_LOG=/explicit/source/run_log.json \
bash scripts/exp/plans/run_condition.sh fixed-replay

# 6. T3A + hint
SOURCE_RUN_LOG=/explicit/source/run_log.json \
bash scripts/exp/plans/run_condition.sh t3a-hint
```

完整设备矩阵分别设置 `DEVICE=standard45562`、`DEVICE=fold45564` 和
`DEVICE=tablet45554` 后重复同一阶段。不要一次并发跑同一个设备上的多个方法。
临时设备版本可直接传显式 `LABEL:SERIAL:PORT`，例如
`DEVICE=OmniFlowTargetPixel6Pro:emulator-45566:45566`；这不会修改默认正式设备表。

## 统计口径

转换阶段与执行阶段分开统计。冷/热执行时间使用 AndroidWorld episode 结果中的任务时间，
不使用外层 shell wall time，因此不包含 emulator/app setup。每组至少报告：

1. official validator success 和 reward；
2. 成功发生的物理动作数，排除 Answer/结束标记和失败请求；
3. 前后页面 state 确实变化的物理动作数及占比；若旧方法没有保存前后 state，明确记为
   `unavailable`，不能用请求数代替；
4. chat、embedding、总模型调用，以及 prompt/completion/total tokens；
5. fallback steps；
6. Memory 命中/召回，或 Function 复用物理动作数及占比；
7. 转换阶段的模型调用、Token、wall time、Function 数和 transfer state 数。

主对比为 MobileGPT 冷/热与 OmniFlow 冷/热；转换开销单列，不摊入热执行。Script replay
和 T3A + hint 是补充基线，使用完全相同的 task、device、seed、步数和 deadline。
