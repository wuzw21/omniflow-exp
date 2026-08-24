# OmniFlow-exp 顶层设计与精简边界

本文回答三个问题：仓库真正的运行路径是什么、哪些文件拥有哪种语义、如何
保证 AndroidWorld 没有测试专用执行旁路。文件级编辑位置见
[`FILE_EDIT_GUIDE.md`](FILE_EDIT_GUIDE.md)；本文件不替代它。

## 1. 仓库职责

本仓库只承载论文 AndroidWorld 实验和 B-MoCA 验证。它不是通用产品仓库，
也不保存 RunLog、截图、模型、APK、模拟器镜像或正式结果数据。运行时资产
统一位于 `data/`，代码只读取 `data/current.json` 这一入口。

顶层可以压缩成四个概念：

| 概念 | 唯一拥有者 | 责任 |
| --- | --- | --- |
| 任务执行 | `scripts/exp/run_androidworld.sh` → `src/experiment/run_tasks.py` → `src/experiment/run_task.py` | 选择任务、方法、设备，并运行一个原子结果 |
| Native episode | `src/integrations/android_world/run_episode.py` | 创建 AndroidWorld/B-MoCA 环境、调用官方 validator、封存 RunLog |
| Function 生命周期 | `omniflow/functions/compiler.py::compile_runlog_to_store` | 从成功 RunLog 编译 v2 Function、`store.json` 和外置 transfer-state catalog |
| 本地证据索引 | `src/experiment/data_index.py` | 物化并读取 `data/current.json`；运行时不扫描替代索引 |

## 2. 唯一运行路径

```text
scripts/exp/run_androidworld.sh       # 唯一公开入口
  ├─ source / development / formal
  ├─ B-MoCA campaign preparation
  └─ src/experiment/run_tasks.py  # 唯一 task + method + device 调度器
       └─ src/experiment/run_task.py  # 一个原子 AndroidWorld 结果
            └─ src/integrations/android_world/run_episode.py  # 一个 native episode
                 └─ src/integrations/android_world/methods.py
                      ├─ fixed_replay
                      ├─ omniflow
                      ├─ mobilegpt
                      ├─ appagent
                      └─ t3a_hint
```

MobileGPT and AppAgent leave this native episode path after
`run_task.py` and are launched from their pinned external checkouts. They are
formal experiment labels, but not AndroidWorld agent adapters.

The boundaries inside this path are intentionally different:

- `run_tasks.py` schedules one task-major attempt and interprets the
  result; it does not call the public shell as an internal API, own
  AndroidWorld lifecycle, or create a second replay runner.
- `run_task.py` translates protocol records into command specifications and
  collects result evidence; it does not own child-process lifecycle.
- `paths.py` owns repository-relative resolution, index-relative evidence
  references, and safe artifact components. This unifies path rules without
  moving external AndroidWorld, OmniTransfer, or B-MoCA roots into `data/`.
- `run_process.py` is the single process-group, timeout, and immutable-log
  seam shared by AndroidWorld and B-MoCA experiment commands, including
  background server/emulator cleanup.
- `run_episode.py` owns one native AndroidWorld episode, including setup,
  observation/action recording, official validation, and teardown.
- `methods.py` resolves one method adapter. AndroidWorld OmniFlow always enters
  `agent.py` with a goal and Function Store; no launcher layer selects a
  Function or supplies Function arguments.

普通 `omniflow` 的运行核心继续是：

```text
OmniFlow.run()
  -> Planner / Function selection
  -> shared checker-library evaluation
  -> execute_function / execute_robust_action
  -> OmniTransfer candidate mapping and target execution
  -> align_function_resume after fallback progress
  -> mapping failure => normal VLM fallback
```

Formal Function 动作绝不能回放 source 坐标。OmniTransfer 映射失败就是一次
Function 步骤失败，并回到 Planner 的正常 fallback；任何隐藏的 resource-id、
node-id、坐标直通分支都违反这个接口。

Function 工具可见性使用同一条两阶段路径：先按 goal 与 Function 的名称、描述
粗筛，OmniTransfer v10 learned state-attention 的 1024D Page Embedding 只辅助
候选排序；再用当前 observation 与首步 source observation 调用 canonical
OmniTransfer。只有映射返回非 NULL target、absolute
contextual confidence 不低于 `0.8`、目标对该动作可执行，且 Planner 选择工具后
页面仍与映射时一致，Function 才能执行。上述任一条件失败时，该 Function 不进入
当轮 Planner 工具列表或在执行前失效，并回到正常原子动作规划。

## 3. 唯一 AndroidWorld E2E 运行形态

正式 AndroidWorld 目标端没有 direct Function、Function 序列、source
qualification replay 或任务专用脚本。调度层只提供 goal、注册的 Function Store
和设备环境；每轮 Planner 同时看到原子动作与召回 Function 的 `input_schema`，
自行选择工具并填写参数。runtime 校验参数后通过 Function `bindings` 写入动作，
执行并重新观察，直到 Planner 返回 `finished`，最后交给官方 validator。

探索、RunLog 采集和 Function 增强属于目标运行之前的资产生成阶段。
`source_calls` 只记录 compiler provenance，不参与目标端选择或执行。
`src/integrations/script_replay.py` 仅实现 B-MoCA 的外部 benchmark replay selector；
它复用 canonical runtime，但不是 AndroidWorld 方法、入口或资格测试。

Page Embedding 与 OmniTransfer 的离线回归属于运行前的数据质量检查，不是第二条
AndroidWorld 执行路径。`src/experiment/offline_transfer_regression.py` 只读取并封存
source/target Observation pair，调用唯一 `PageEncoder` 和生产
`default_transfer`，不创建 Host、Planner、episode、ADB 连接或 validator。成功与失败
RunLog pair 都进入同一去重数据集；失败样本缺少可信目标时保持待标注。每轮全量评测
刷新一个派生 `errors.json` 视图，修复后仍必须重跑主数据集，不能只跑当前错误子集。

## 4. 索引、ledger 和历史输入的区别

| 对象 | 是否运行时入口 | 语义 |
| --- | --- | --- |
| `data/current.json` | 是，唯一 | source RunLog、Function Store、transfer states、method memory、result rows 的内联索引 |
| `result_registry` ledger | 否 | 不可变注册证据；由 `current.json` 引用，不能成为第二个选择器 |
| `batch_outcomes.py` 的汇总 | 否 | 从一次尝试和注册结果生成报告/compact row，不拥有调度 |
| MobileGPT/AppAgent manifest | 否 | 外部 baseline 的不可变证据，不是 Function catalog |
| 旧 source/index/manifest | 否 | 只可在迁移或兼容读路径中使用；不得写入新的运行副本 |

因此“只保留一套索引”不是删除所有 manifest 或 ledger，而是保证只有
`current.json` 参与运行时解析；其他文件各自只承担证据或统计语义。

### Function v2

运行时只读取 `omniflow.function.v2`。Function 步骤按成功 source RunLog 的动作
顺序保存，每步以唯一 `source_state_id` 引用同目录 `transfer_states.json` 中的
source observation。Authoring Agent 先生成零个或多个语义 Function，再生成一个
最大的安全组合 Function。RunLog 动作无需全部进入：重复、重试、`origin=checker`
以及必须让 Planner 读取中间 observation 的动作都可留在组合 Function 之外。
完整 Function 自己合并所选动作的语义、参数 schema 和 bindings；Compiler 校验
schema、source 顺序、原子观察边界和所选动作的参数提升。最终数量大于等于一，
不引入 Function 嵌套或 parent/child schema。Checker 使用独立
`omniflow.checker_store.v1`，规则可跨
Function 共享，触发预算在同一 Function 序列内共享。Store 本身不内联
observation，也不接受 v3 `transfer_state_ids`。

## 5. 当前精简候选与保留决定

### 五个大文件的直白职责

这五个文件不是五个重复的 runner，而是 Source RunLog 从输入证据变成可运行
baseline 记录时经过的五个位置：

| 文件 | 只应该回答的问题 | 不应该知道的事情 |
| --- | --- | --- |
| `source_evidence.py` | 这份 Source RunLog 的截图、XML、动作和 revision 是否可信？ | AppAgent/MobileGPT 的转换规则 |
| `appagent.py` | 如何把可信 source 转成 AppAgent 的 Prepared Memory？ | task 调度、Local Index、AndroidWorld episode 生命周期；执行由官方 AppAgent 进程完成 |
| `mobilegpt.py` | 如何把可信 source 转成 MobileGPT 的 Prepared Memory？ | AppAgent 规则、Local Index 选择策略 |
| `mobilegpt_memory.py` | MobileGPT Prepared Memory 的统计、图检查和完整校验 | task 调度、AndroidWorld episode 生命周期 |
| `official_forward.py` | 如何在统一 task 生命周期中启动官方 MobileGPT/AppAgent/AutoDroid？ | provider memory 转换、官方 agent/action loop |
| `checks.py` | 这次运行的依赖、root 设备、已安装 Accessibility 服务和 Prepared Memory 是否 ready？ | 具体 provider 的转换实现 |
| `data_index.py` | 如何物化和读取唯一 Local Index？ | AndroidWorld runner 和 provider 内部校验细节 |
| `source_records.py` | Source RunLog 的共享数据模型是什么？ | 读取、执行或转换 RunLog |

### AndroidWorld observation persistence

新采集和新转换的 RunLog observation 统一只保存两项：`screenshot`（直接路径、尺寸和 MIME）与 `xml`（完整 UI XML）。reader/schema 同时接受旧采集器的 `pixels+xml+auxiliaries` 和四字段 native AndroidWorld snapshot；旧截图 SHA-256 可读取但在 canonical 输出中移除。新 writer 不再输出这些旧字段。动作、结果、validator 和 reasoning 仍属于 RunLog 证据合同。

AndroidWorld 的正式采集、手工采集和 fixed replay capture 共用 `build_androidworld_run_log` 与 `persist_androidworld_run_log`。一个 attempt 只有一份 `run_log.json`；截图按 `screenshots/screenshot_NNNNNN.png` 直接保存。attempt 使用递增编号，不使用时间戳，也不创建 `object_store` 或哈希命名截图。

这里的 schema `$defs.state` 是这份 compact observation 的复用定义，不是要求 RunLog 额外嵌套一个 JSON `state` 字段。AndroidWorld SDK 的 `get_state()` 仍返回原生 `state.pixels` 和 accessibility tree；只有进入 OmniFlow 持久化边界时，才投影为 `screenshot + xml`。

`source_evidence.py` 曾经暴露 `convert_runlog_memory(method=...)`，让共享 source
模块根据字符串选择 provider。这是已经删除的浅 seam：AppAgent 直接调用
`convert_runlog_to_appagent_memory`，MobileGPT 直接调用
`convert_runlog_to_mobilegpt_bundle`。新增 provider 时应新增自己的 adapter，
而不是把分派字符串重新塞回 source 层。

### AndroidWorld public result row

`src/experiment/result_schema.py::RESULT_FIELDS` 是唯一公开 cell 表的字段
合同；`run_task.py` 用它写 `result_summary.md`、`result_summary.json`，注册器
用同一行写 immutable result ledger，批量汇总层只补充报告字段，不另建一张
口径不同的表。每个 cell 至少记录：

- episode 与 prep 的模型调用数、chat/embedding 调用数、prompt/completion/total
  tokens、两阶段合计调用数/Token、动作数、VLM calls、延迟和能耗；
- Function 命中、覆盖步数/总步数/覆盖率，以及 memory 的
  `used/prepared/not_applicable/unavailable` 状态、来源和命中率；
- fallback 步数、配置上限、是否耗尽预算和 measurement status；
- validator 结论、method outcome、failure stage/reason、environment failure、
  attempt/device/provenance。

统计没有上报时写 `null` 并在对应的 `*_measurement_status` 或状态字段中说明，
不能把官方 baseline 未提供的 fallback 或 memory 命中伪装成 0。只有代码明确
记录到的零调用、零 fallback 或零能耗才写数字 0。

| 候选 | 判断 | 后续动作 |
| --- | --- | --- |
| AndroidWorld direct Function / sequence / qualification replay | 非真实端执行旁路 | 删除；只保留 `OmniFlow.run(goal)` Planner 循环 |
| B-MoCA `script_replay.py` 与 runtime execution | 外部 benchmark 合同，不是 AndroidWorld 方法 | 保留，继续用测试锁住“无私有 mapping” |
| `data_index.py` 与 result ledger | 读写对象不同，不能粗暴合并 | 保留职责，拆出只在有测试证明时进行 |
| `batch_outcomes.py` 与 `result_registry.py` | 汇总和注册是两个不可互换的写入语义 | 先记录公共 path helper 重复，再局部收敛 |
| `omniflow/runlog.py` 与 `src/integrations/runlog.py` | canonical loader 与历史外部导入 adapter | 旧适配层按要求暂不清理 |
| `mobilegpt_source.py` / `appagent_source.py` | 两个外部协议不同 | 不合并；只共享纯证据 helper |
| AutoDroid DroidBot memory | 官方事件/UTG replay，不是 OmniFlow Function | 不转换为 OmniFlow schema；只校验 manifest、事件和 APK |
| 2430 行 shell、3093 行 scheduler、4619 行 launcher | 复杂文件，但都是现有唯一 owner | 先补 seam 和测试，再按模块内聚拆分；不复制入口 |
| 已删除的 packaged catalog、old source pool、direct launcher 文件 | 最近提交已清掉的死路径 | 不恢复 alias 或兼容副本 |

## 6. 变更顺序

每个重构遵循下面的顺序：

1. 先在本文件和 `FILE_EDIT_GUIDE.md` 写清 owner、接口和不变量。
2. 先改现有 owner；AndroidWorld 不增加测试专用执行入口。
3. focused tests 证明行为仍在，再删除无调用者的 helper/文件。
4. 不触及 schema、public result row、统计表；若必须触及，单独 commit。
5. 每个语义组独立 commit 并 push；最后再运行完整测试。

## 7. 新人最短路径

```text
README.md
  -> docs/ARCHITECTURE.md
  -> docs/FILE_EDIT_GUIDE.md
  -> scripts/exp/README.md
  -> 对应目录 README
  -> owner 文件及其 focused test
```

公共命令只有：

```bash
bash scripts/exp/run_androidworld.sh ...
```

Function 写入只有 `compile_runlog_to_store`；运行时 Function 只从已注册 Store 和
`data/current.json` 读取；OmniTransfer checkout 只有
`~/Projects/Omni/OmniTransfer`。
