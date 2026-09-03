# OmniFlow-exp

AndroidWorld 和 B-MoCA 实验仓库。AndroidWorld 只有一个公开入口：

论文评测冻结边界、可变项、证据要求和失败归因见
[`docs/PAPER_FREEZE.md`](docs/PAPER_FREEZE.md)。

```bash
bash scripts/exp/run_androidworld.sh
```

入口直接调用统一 runner；runner 只使用调用者明确传入的 task、method、device 和
Memory，然后启动一次 AndroidWorld task。设备 lifecycle、task setup 和最终 validator
由 AndroidWorld episode 负责。

常用参数都是可选的：

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562 \
  --memory data/androidworld/CameraTakePhoto/omniflow/OmniFlowSourceSmall_seed111/memory/current/store.json
```

论文 target 为 Pixel 6 Pro、7.6 英寸 Fold 和 10.1 英寸 WXGA Tablet。一次选择
全部三台设备时，入口会复用已在线的 AVD、启动缺失的 AVD，并保持每台设备一个并发
worker：

实验平台统一为：Source 是 Android 13、`720x1280` 的 small-phone emulator；
Standard 是 Android 13、`1440x3120` 的 Pixel 6 Pro；Fold 是 Android 14、
`1768x2208` 的 7.6-inch foldable；Tablet 是 Android 13、`1280x800` 的
10.1-inch WXGA tablet。

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device all \
  --memory data/androidworld/CameraTakePhoto/omniflow/OmniFlowSourceSmall_seed111/memory/current/store.json
```

正式方法固定为 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。
其中 AppAgent 使用同一份 source RunLog 生成一次官方 demo Memory，再进入统一的
AndroidWorld/OOB/validator 流程；当前执行批次只启动前三种方法，AutoDroid 保留为待运行 cell。
设备和默认值来自 `config/paper_androidworld.json`。正式运行和 Memory 转换固定使用
`Qwen3.6-Plus`；显式传入其他模型会在入口处拒绝。

```bash
bash scripts/exp/run_androidworld.sh --help
```

Memory 保存和实验执行是两个独立协议，但都使用同一个入口，不读取历史结果索引。
仓库内配置和 Memory manifest 使用相对路径；外部依赖只在进程启动边界解析为本机路径。
所有 Function 共用一个仓库级 Checker Store：`omniflow/checkers/default.json`。
Function Memory 不再携带 `checker_store.json` 副本，运行时直接加载该共享库。
`convert-memory` 只产生稳定的 Memory 地址；如果这个明确地址已经存在且校验通过，
入口直接复用，不再次调用模型；如果地址不完整或 source/model 不匹配，则报错并停止，
不会自动重跑或选择历史结果。

保存 Memory：

```bash
bash scripts/exp/run_androidworld.sh convert-memory \
  --task CameraTakePhoto \
  --method omniflow \
  --source-run-log data/androidworld/CameraTakePhoto/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json \
  --memory data/androidworld/CameraTakePhoto/omniflow/OmniFlowSourceSmall_seed111/memory/current
```

OmniFlow 的 Function 转换由 A 端 authoring Agent 和无语义判定的 Compiler 完成。
Agent 先从成功 RunLog 中找到可复用 Function，再直接决定参数名、每个 occurrence 的
source value，以及完整的 action/render binding（包括 `arg_name` 或
`node_id`/`attribute`/`recorded_value`）。Compiler 不生成候选、不推断参数、不选择节点，
也不补写遗漏的 binding；它只把 Agent 给出的 source step 索引映射为不可变 action 和
source state，并完成 schema/index 校验与 Store 序列化。三次 proposal 都不满足结构
合同时会保留 `authoring_failure.json`，随后注册不含任何 binding 的原始 RunLog replay；
这个 fallback 只转换已记录动作，不含 compiler-derived 参数。
同一语义 Function 在 RunLog 中重复出现时，Store 只注册一份定义，
`compile_report.json` 的 `source_calls` 按 source 顺序保存多次引用及各自参数。
局部 Function 不必覆盖完整 RunLog；重复 occurrence 的稳定性和绑定均由 Agent 负责，
Compiler 按 Agent 给出的第一个 occurrence 注册定义并保存其余 invocation 引用。

直接执行：

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562 \
  --source-run-log data/androidworld/CameraTakePhoto/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json \
  --memory data/androidworld/CameraTakePhoto/omniflow/OmniFlowSourceSmall_seed111/memory/current/store.json
```

如果需要从一份 source RunLog 一次性生成三个需要 Memory 的方法，可使用固定的
目录布局；`fixed_replay` 和 `t3a_hint` 直接使用同一份 source RunLog：

```bash
bash scripts/exp/run_androidworld.sh convert-memory \
  --task CameraTakePhoto --method all \
  --source-run-log data/androidworld/CameraTakePhoto/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json \
  --memory data/androidworld/CameraTakePhoto/omniflow/OmniFlowSourceSmall_seed111/memory/current

bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto --method all --device all \
  --source-run-log data/androidworld/CameraTakePhoto/source/OmniFlowSourceSmall_seed111/runlog/current/run_log.json \
  --memory data/androidworld/CameraTakePhoto/omniflow/OmniFlowSourceSmall_seed111/memory/current
```

该布局只包含 `omniflow/store.json`、`mobilegpt/memory/` 和 `appagent/` 三份派生
Memory，不复制 source RunLog，也不扫描历史结果。manifest 内的 source、demo、日志和
校验文件均相对于 Memory 根目录记录，因而可以随仓库一起搬迁。

OmniTransfer 使用 canonical checkout `~/Projects/Omni/OmniTransfer`，页面检索统一调用
V10 `omnitransfer_point_conditioned_sparse_graph_v10` 模型的归一化 1024D
page-attention readout；不维护第二套页面编码器或旧 64D/512D 表示。

Online Planner 仅通过 canonical 工具 Schema 输出一个动作；`finished`
必须带非空 `content`，作为已完成目标的审计说明。在 AndroidWorld 边界，
`finished` 只转换为官方 `JSONAction(action_type="status", goal_status="complete")`；
不发送 `answer` 动作，任务成功仍只由 AndroidWorld 官方 validator 判定。

Function 是一种普通 Action；它的 action list 和返回结果写入统一 action history，作为
Planner 的判断证据。Function replay 成功或失败都回到同一条 Planner 主线。Planner
模型输出带非空内容的 `finished` 后，Engine 立即结束 OmniFlow 生命周期并把终止结果
交给 AndroidWorld；任务是否成功只由 AndroidWorld 官方 validator 判定。`observation`
只表示当前状态，`finished` 只表示终止结果，二者不与同一轮设备动作混合。

架构和文件 owner 见 `docs/ARCHITECTURE.md` 与 `docs/FILE_EDIT_GUIDE.md`。
