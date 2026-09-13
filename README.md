# OmniFlow-exp

Version **1.0.0** is the preserved baseline before the host-harness refactor.
Release scope and validation are recorded in [release notes](docs/RELEASE_NOTES.md).

当前开发版将内核与宿主控制分开：内置 Planner 调用 `OmniFlow.arun`，外部 Agent
只接入两个服务：`omniflow_recall` 召回 Function，`omniflow_execute` 执行 Function。
Python 内核对应 `OmniFlow.arecall` / `aexecute_function`；它们与内置循环共用原来的
Recall 和 Function 闭环。外部失败返回当前状态和失败
位置，交还宿主决策。协议与迁移说明见 [Harness 协议](docs/HARNESS_PROTOCOL.md)。
AndroidWorld 正式入口不变。

内置 Harness 在取消、超时和 `effect_unknown` 退出时也保留退出前的模型用量与 Function
调用证据；终止原因和动作效果仍由同一个 execution control 决定，诊断快照在调用结束后清除。
内置 Harness 的 `diagnostics.function_resume` 保存实际 Function 调用、失败位置、
恢复后重新调用及断点续执行的证据；`success_count` 只表示续执行完成，不代表官方任务
成功。旧 RunLog 缺失该记录时，汇总保留 null，不能解释成零次恢复。新增回归入口为
`tests/test_completion_verification.py`；恢复行为仍须真机验收。

`run --method omniflow --function-reentry off` 是内置 Harness 的恢复后复用消融：
首个 Function 执行失败后，当前任务不再向 Router/Planner 暴露 Function，继续使用同一
Planner、OOB 和官方完成检查。默认 on。它不关闭动作映射，也不重放源坐标；结果作为
`reentry_ablation` 独立归档，不晋升为默认论文结果。此开关不适用于外部宿主。

严格的 Memory ON/OFF 配对使用同一个显式 `--memory STORE`，仅切换
`--function-memory on|off`。off 保留 Store 和相同模型、超时、Host、Planner 创建路径，
关闭内置循环的 Function 暴露、召回和调用，结果独立存为 `memory_ablation`。
不传 Memory 的原 baseline 仍可运行，但其配置路径不同，不能冒充这一单因素消融。

补充实验场景与执行条件见 [场景覆盖](docs/EXPERIMENT_SCENARIOS.md)。配对分析复用
`src.experiment.performance_metrics.summarize_paired_experiments`，显式传入冻结 pair
列表与样本；输出各自 success set、双方成功交集、快路径与恢复成功分层，以及 task
cluster bootstrap 区间。缺失运行、缺失耗时和不一致配置均不能伪装成可比较样本。
`paired_sample_from_runlog` 只读取一个显式 RunLog 和 sibling 原子结果，并结合独立的
环境/资产 receipt 构造样本；缺失 usage 保留未知，组件只使用已对账的 exclusive wall
time，较早 Function 失败不会被最后一次成功掩盖。

Android 验收与对外 Harness 接入使用同一条运行链，只替换决策宿主：

```bash
bash scripts/exp/run_androidworld.sh run --task SystemBluetoothTurnOn \
  --method omniflow --device standard45562 --memory /absolute/memory/store.json \
  --harness codex
```

`--harness builtin`（默认）、`codex`、`claude` 或 `package.module:factory` 共用
AndroidWorld setup/reset、同一 OOB Host、Function 闭环、官方 validator 和 RunLog。
外部 CLI 通过 MCP 连接当前 episode 的运行时，不再启动第二份设备运行时。
不同宿主的接入结果作为外部 Harness 证据保存，不晋升为冻结模型的论文结果。
RunLog 的 `diagnostics.harness` 保存宿主实际报告的模型与用量；宿主不提供 API
请求次数时，结果 `model_calls` / `vlm_calls` 为 null，表示未知，不计作零次调用。
恢复策略可用 `--checker on|off` 配置，默认 on；off 不关闭官方完成验证，
结果标记为 Checker 消融并单独归档。SDK 使用 `RuntimeSettings(checker_enabled=False)`。
RunLog 的 `diagnostics.wall_accounting` 保存公共执行边界内的非重叠耗时分账；映射
失败池保存紧凑 pair/hash 和有限候选。字段与当前覆盖限制见
[执行诊断](docs/HARNESS_PROTOCOL.md#执行诊断开发中)，不得将未归因时间当成模型耗时。
失败输入可按显式 manifest 调用 `replay_failure_inputs` 在原 Transfer 边界回放；只保存
失败 pair 的压缩页面和去重截图，不复制整个任务历史，也不把失败候选当作正确目标标签。
工厂 Harness 可声明 `requires_builtin_planner=True`，复用原 Planner/Router 做局部迭代。

对外提供的是可复用的跨设备 Function Memory 服务：宿主决定做什么，OmniFlow
负责召回已有操作片段，并在当前设备上用共享闭环执行、返回事实。交付物包含
Function Store、执行内核、Python SDK、MCP 服务和使用 Skill。Skill 是宿主使用说明，
不是独立执行引擎；接上 MCP 也不自动获得宿主的 Android 原始动作或任务恢复能力。
各接入路径的实际验收层级见 [验证范围](docs/HARNESS_MCP.md#验证范围)。

Codex/其他 MCP 宿主通过 [OmniFlow GUI Skill](skills/omniflow-gui/SKILL.md) 接入，
服务启动和配置见 [MCP 接入](docs/HARNESS_MCP.md)。它复用 OOB 和同一 Function 内核。
两工具使用标准 MCP tools/call，声明输入/输出 schema，同时返回 structuredContent 与
兼容文本结果；协议适配和幂等标识见 [标准 tool call 合同](docs/HARNESS_MCP.md#标准-tool-call-合同)。
宿主负责原始动作、任务完成判断和任务级循环；两工具服务不再对外注册每个 Function
或提供另一套原始动作/任务控制工具。已有 GUI-agent 的 OOB 原始动作适配属于 harness。

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
Agent 必须先逐步区分 `stable`、`task_parameter` 和 `online_observation`：稳定值可按
Source 默认值复用；任务参数必须声明参数名和完整的 action/render binding；依赖当前
页面读取、计算或条件判断的动作必须保留给在线 Planner。完成分类后，Agent 再从成功
RunLog 中找到可复用 Function，并直接决定参数名和完整的 action/render binding（包括 `arg_name` 或
`node_id`/`attribute`/`recorded_value`）。Compiler 不生成候选、不推断参数、不选择节点，
也不补写遗漏的 binding；它只把 Agent 给出的 source step 索引映射为不可变 action 和
source state，并从 Agent 已选定的 source binding 原样复制默认值，完成 schema/index
校验与 Store 序列化。只有 Agent 明确分类为稳定的未声明值才保留 RunLog 默认值。
只要存在 `online_observation`，完整 Source workflow 就以不可执行证据保存，运行时只暴露
安全的局部 Function 并交还在线 Planner。三次 proposal 都不满足结构合同时会保留
`authoring_failure.json`，原始 RunLog 也只作为不可执行证据注册，禁止以无参数静态 replay
进入正式 Function Recall。
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
Planner 的判断证据。完整 Function 通过当前 task 的官方完成检查后立即终止；未通过
或 replay 失败时回到同一条 Planner 主线。Planner
模型输出带非空内容的 `finished` 后，Engine 立即结束 OmniFlow 生命周期并把终止结果
交给 AndroidWorld；任务是否成功只由 AndroidWorld 官方 validator 判定。`observation`
只表示当前状态，`finished` 只表示终止结果，二者不与同一轮设备动作混合。

架构和文件 owner 见 `docs/ARCHITECTURE.md` 与 `docs/FILE_EDIT_GUIDE.md`。

官方 validator 与方法执行状态独立记录：MobileGPT 的
`official_validator_success` 只取官方 reward 判定；`task_finished`、进程退出码和
错误原因保留实际方法结果。不得用方法异常覆盖官方成功，也不得用官方成功掩盖进程
异常。字段合同见 [Schema 说明](schemas/README.md)，修正待真机验证。
