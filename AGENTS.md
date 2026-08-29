# OmniFlow-exp 工作规则

## 修改长期原则（最高优先级）

- 不能直接改整体设计；保持既有架构边界、模块职责和实验合同不变。
- 不能改主线执行流；修复必须落在原有主线 owner 上，并保持原有调用顺序。
- 不能引入旁路、任务专用脚本或平行实现来绕过主线；配置、Checker 和 Transfer
  必须继续服从既有统一接口。
- 可以修复通用 BUG，但必须在主线上修复，并用 focused regression test 证明它对
  所有调用方都是通用修复，而不是针对单个任务的特判。

## AndroidWorld 极简运行长期规则（2026-08-25）

- 仓库只保留一个通用入口 `scripts/exp/run_androidworld.sh`，以及各 method 的
  Memory 转换实现。入口只支持两件事：`convert-memory` 和 `run`。
- `run` 只接收 task、method、device 和可选 Memory 路径；传入 Memory 就原样交给
  对应 method，不传就按无 Memory 启动或由真实运行自然报错。
- `convert-memory` 只接收 source RunLog、method 和目标 Memory 路径；禁止通过
  `data/current.json`、catalog、registry、ledger 或历史结果扫描选择输入。
- 禁止为 seed 111/113、固定 checkout 路径、固定模型、endpoint、AVD 状态或已完成
  结果增加启动硬门槛。通用运行时错误检查可以保留，但不能写死实验值。
- `data_index.py`、`data_migration.py`、启动 preflight、结果 registry、scheduler
  manifest/summary、旧数据搬运和重复校验层不属于真实实验运行，禁止重新引入。
- 临时调研、一次性教研脚本、排障文件、临时测试代码和生成产物不得写入仓库；放在
  系统临时目录或外部配置中。仓库不维护分散的专项测试脚本，只保留通用运行入口。
- 所有可复用的 Memory、方法常识、Function Store 和转换后的方法资产统一保存在
  `data/androidworld/<task>/<method>/<device_seed>/memory/<attempt>/`，按现有 task、
  method、device/seed 和 attempt 层级分目录；不得散落到仓库源码、`output/`、`tmp/`
  或额外 catalog。AndroidWorld 官方 RunLog/结果保存在同一 setting 的 `runlog/`。
- 转换中间文件、调度日志和一次性测试计划必须放系统临时目录并自动删除；它们不属于
  Memory。`fixed_replay` 的 Memory 就是 source RunLog/script，本身不做冷启动转换。
- MobileGPT 正式实验使用一份成功 source RunLog 转换出的 MobileGPT 官方 Memory；
  AndroidWorld 官方根据 task、固定 task parameters 和 evaluation seed 生成参数，
  目标 app 来自官方 task 的 `app_names`。已安装的官方 Accessibility client 必须直接
  复用；只有未安装或显式要求 rebuild 时才构建 APK，禁止每个 episode 重复
  Gradle/install。转换和执行都只使用这一份显式传入的 Memory，不读取历史结果自动选择。
- MobileGPT、AndroidWorld baseline 和 OmniFlow 共用同一套已初始化设备。OOB APK、
  benchmark APK、权限和 app snapshot 只在设备初始化阶段准备一次；正式
  task 只做 AndroidWorld 官方 task reset 和页面初始化，不重复 Gradle、APK
  install 或整套 app setup。只有 APK 版本变化或用户显式要求重建设备时
  才重新安装。
- OOB 只能在本地 canonical OpenOmniBot checkout 编译；4090 和 9207 只接收
  编译好的 APK，禁止向远程上传 OOB 源码、保留 OOB 编译 checkout 或
  执行 Gradle/OOB 构建。
- 实验唯一 OOB 发布件固定为权威 OmniFlow-exp 中的
  `data/runtime/oob/OOB-Experiment.apk`。新 OOB 版本只在本地编译一次并覆盖
  该发布件；同一版本的所有设备、4090 和 9207 只传输、安装这一个
  release，禁止从 OpenOmniBot build 临时目录重复打包或为每次 task 重建。

本节是当前 AndroidWorld 运行的最高优先级项目规则；下文涉及旧 index、migration、
seed/path preflight、结果注册或专项测试的历史描述不再适用。

本仓库只做论文 AndroidWorld 实验和 B-MoCA 验证。改代码前先读：

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/FILE_EDIT_GUIDE.md`
4. `scripts/exp/README.md`
5. 直接阅读对应 owner 代码和 focused tests

## 不可破坏的运行合同

- **唯一权威部署 checkout 是 `~/Projects/Omni/OmniFlow-exp`。** 本地对应
  `/Users/wuzewen/Projects/Omni/OmniFlow-exp`，9207 对应
  `/home/wuzewen/Projects/Omni/OmniFlow-exp`；以后所有部署、启动、Python
  环境、配置和 OmniFlow 实验数据都必须从这个 checkout 读取。
- `OmniFlow-exp-unified`、`OmniFlow-exp-official`、
  `OmniFlow-exp-mobilegpt-*`、`OmniFlow-exp-semantic-*` 和其他带日期/版本后缀
  的目录只作为历史证据或只读参考，禁止作为运行时 source、`PYTHON_BIN`、
  `OMNIFLOW_*_ROOT` 或部署目标。不要再从这些目录派生新的备份版本。
- OmniFlow 代码的版本一致性以权威 checkout 的 Git 工作树为准；远端部署必须
  先把该 checkout 同步到 9207 的同名路径，再从该路径的
  `scripts/exp/run_androidworld.sh` 启动。官方 MobileGPT、AndroidWorld 和
  OmniTransfer 若作为外部依赖，必须通过权威 checkout 的固定配置显式引用，
  不能从另一份 OmniFlow checkout 偷渡运行时代码或数据。
- 唯一公开入口是 `scripts/exp/run_androidworld.sh`。
- AndroidWorld 所有正式实验的应用启动、observe 和 act 物理层必须且只能走
  OmniFlow OOB；禁止使用 AndroidWorld native action、ADB/monkey 启动或
  MobileGPT/AppAgent 自带 Accessibility client。AndroidWorld 仍负责统一 setup
  和官方 validator；外部方法只保留其 Planner/Executor 决策逻辑。唯一启动器
  对非 OOB 后端必须在实验开始前硬失败，不能依赖临时 env 才保证这条合同。
- 唯一 task + method + device 调度器是 `src/experiment/run_tasks.py`。
- `src/experiment/run_task.py` 只执行一个原子 AndroidWorld 结果。
- `src/integrations/android_world/run_episode.py` 是唯一 native episode/lifecycle owner。
- `compile_runlog_to_store` 是唯一 Function 写 API；输出 v2 `store.json`、`compile_report.json` 和 sibling `transfer_states.json`。
- 成功 RunLog 由 authoring Agent 提取零个或多个连续局部 Function，并生成一个覆盖全部成功主流程动作的完整 Function。Agent 和 Compiler 都不得删除、重排或截断所选连续段的中间动作；`origin=checker` 恢复动作仍由独立共享 Checker 提取和执行。Compiler 只校验 schema、连续 source 顺序和可证明的参数提升。最终 Function 数量大于等于一，不增加嵌套或 parent/child schema。
- runtime 只读取注册的 Function Store 和 `data/current.json`，不能自动补 Store、建 catalog 或写平行 manifest。
- `data/current.json` 是唯一运行时本地索引；ledger、汇总和外部 manifest 只能作为证据。
- AndroidWorld 当前统一 E2E 方法为 `fixed_replay`、`omniflow`、`mobilegpt`、
  `appagent`、`t3a_hint`。AppAgent 通过同一入口和 OOB/validator 路径执行，
  不再作为历史专用方法；`script_replay` 仍不进入 AndroidWorld 方法。
- B-MoCA 的 replay selector 属于外部 benchmark 合同，不复制进 AndroidWorld method 名称。
- AndroidWorld 的 OmniFlow 正式执行只有完整 E2E Planner 循环：runner 只传 goal、
  Function Store 和设备环境；Planner 从 Function `input_schema` 生成参数，runtime
  通过 `bindings` 绑定后执行。禁止 direct Function id、Function 序列、预绑定参数、
  source qualification replay 或任务专用执行脚本。

## OmniTransfer 合同

- OmniTransfer 的 canonical checkout 永远是 `~/Projects/Omni/OmniTransfer`。
- 页面命中只用 `omniflow/transfer/embedding.py`：它调用 canonical OmniTransfer V10 `omnitransfer_point_conditioned_sparse_graph_v10` 统一模型的学习式 page-attention readout，输出归一化 1024D `PageEncoder`；不保留手工 64D、8-word pooling、512D 或第二 page encoder。动作映射仍只由 canonical OmniTransfer 统一 matcher runtime 负责。
- 映射失败是明确的 transfer failure，回到正常 Planner/VLM fallback；绝不执行 source 坐标。
- 禁止 resource-id/node-id lookup、坐标 passthrough、第二 page encoder、第二 action mapper。
- checker 是独立共享库，可跨 Function 使用；一个 RunLog 可注册并按顺序执行多个 Function。
- RunLog、Function 与在线 Planner 使用同一套 action 字段；不保留 `target_description`。映射只将 `source_state_id` 对应的完整 source 状态、source action point 和当前 target 状态交给统一 OmniTransfer。禁止 source 坐标直传、语义锚点、target-side 文本查找器或第二 action mapper。

## 证据、环境和数据

- 4090 长期部署规则：OmniFlow-exp 的唯一权威工作区固定为 `/home/zewen/Projects/OmniFlow-exp`。代码、`.git`、`.venv`、`data/current.json`、实验资产和结果都在这一工作区内直接修改、验证和运行；Git commit 是唯一版本历史。禁止再创建或从 `local-authoritative`、`current-snapshot`、按日期/任务命名的 checkout、`/data/omniflow-4090/OmniFlow-exp` 或其他复制目录部署和启动实验。旧目录只可作为只读历史证据，不得成为运行入口。所有 4090 命令必须先解析并校验 canonical workspace，路径不一致立即失败。
- AndroidWorld 时间合同：一个完整 task 的默认和最大 wall deadline 都是 600 秒；正式 OmniFlow Planner 单次模型请求默认最多 30 秒。禁止通过实验命令把整任务放宽到 10 分钟以上。
- Function v2 按成功 RunLog 动作顺序保存；步骤通过 `source_state_id` 引用 sibling `transfer_states.json`。新跑的正式 AndroidWorld source 默认使用 seed 111。
- B-MoCA env100 必须先通过 official success、method success、`model_calls=0`、`fallback_steps=0`，才可创建/运行其他环境。
- 所有 Python/Torch 命令使用 `~/Projects/Omni/OmniFlow-exp/.venv/bin/python`；正式执行不使用邻近环境。
- 长期训练规则：任何模型训练、微调或训练 smoke 都必须在 `9207` 远程环境执行；本地只允许数据清洗、质量检测、代码测试和评测入口验证，不得在本地启动训练。
- 所有实验资产、RunLog、截图、Store、transfer states、memory 和结果都在 `data/`；不要提交它们、credentials、APK、权重或 emulator image。
- 项目长期记忆：`OmniFlow-AndroidWorld-Experiments` 的唯一对外主表是
  `OmniFlow_AndroidWorld_116Tasks_15cell.xlsx`，固定为 116 个任务 × 15 个正式
  E2E 实验格 = 1740 个实验格。15 格由五种方法 `fixed_replay`、`omniflow`、
  `mobilegpt`、`appagent`、`t3a_hint` × Standard/Fold/Tablet 构成；转换阶段和
  Source 不占 E2E cell。每个任务只登记一份成功 Source 证据，作为五种方法共同的
  派生输入。该目录 `~/Desktop/OmniFlow-AndroidWorld-Experiments` 是权威归档源；
  更新主表时必须同时保存 RunLog、model usage、seed、设备、Function/Memory 路径和
  SHA-256 provenance，并更新同名 `.inspect.ndjson`。
- AndroidWorld 设备长期规则：source 固定为 Pixel small phone，API 33 / Android 13，
  720×1280，只用于 seed 111 source 证据。正式 target 始终且仅为 Standard + Fold +
  Tablet：`standard45562`/`OmniFlowTargetPixel6Pro` 是 Pixel 6 Pro，API 33 /
  Android 13，1440×3120；`fold45564`/`OmniFlowTargetFold` 是 7.6-inch foldable，
  API 34 / Android 14，1768×2208；`tablet45554`/`WXGA_Tablet_test_00` 是
  10.1-inch WXGA tablet，API 33 / Android 13，1280×800。`source5560` 永远是
  source-only，不得作为 target。`OmniFlowTargetSmall`、`pixel5576` 和
  `AndroidWorldAvd4090` 已从正式 target 协议退役，只能作为只读历史兼容输入。
- `Standard` 是实验平台角色，只能表示 Pixel 6 Pro / Android 13 / 1440×3120；它
  不能别名到 `small_phone` 或 `OmniFlowTargetSmall`。`small_phone` 只表示 source 的
  Pixel small phone / Android 13 / 720×1280。设备内部名字与该平台语义不一致时，
  必须修正启动的 AVD，禁止通过改 profile 或结果标签伪装成 Standard。
- 旧 10-cell/extended 结果和表格只作为历史证据放在桌面归档 `.archive/`；不得再作为
  对外主表或被后续任务更新。迁移时必须保留原始文件、来源路径、文件时间和 SHA-256
  provenance，并去重而不是覆盖冲突版本。
- 旧适配层和历史输入可按只读迁移路径保留；不要为了兼容重新加 alias、旧 writer、旧 index 或旧 runner。
- `source_calls` 只允许作为 Function 编译 provenance，不得成为 AndroidWorld 正式执行输入。
  B-MoCA 官方外部 replay 合同可使用共享底层 runtime，但不得接入 AndroidWorld 方法或入口。

## 通用数据目录合同

- `data/` 下的 benchmark 业务数据只分为 `androidworld/` 和 `bmoca/`；`data/current.json`、`data/inputs/` 等运行时元数据不算 benchmark 目录。AndroidWorld 与 B-MoCA 的任务、方法、设备和结果不得混放。
- AndroidWorld 的唯一可见层级是：

  ```text
  data/androidworld/<task_name>/<method>/<device_model>_seed<source_seed>[_eval<evaluation_seed>]/
    device.json
    runlog/
      attempt_NNN/
        run_log.json
        screenshots/
    memory/
      <attempt_id>/
        store.json
        compile_report.json
        transfer_states.json
        run_log.json
        screenshots/
  ```

- `<method>` 只使用统一名称：source 证据用 `source`；正式实验用 `fixed_replay`、
  `omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。禁止用 collector 名、模型昵称、
  时间戳前缀或历史 runner 名创建平行 method。
- 设备按 `device.json` 中探测到的真实 `device_model` 归类，不按 adb serial、端口、host alias 或人为昵称归类。设备目录保留实际运行 seed 作为 provenance；新跑的正式 source 默认使用 111，正式 target 结果同时写 `_eval<evaluation_seed>`。
- 可见 RunLog 目录只能命名为 `attempt_NNN`，同一 device 目录内从 `attempt_001` 递增。禁止保留 `manual_*`、`object_*`、`forced_live_plan_*`、hash 后缀或时间戳别名；导入时统一转换后再进入可见目录。
- setting 身份由 `task_name + method + 真实 device_model + 规范化 task_parameters + 协议配置` 决定，不由 RunLog seed、目录旧名、run_id、文件 hash 或设备别名决定。判断一个任务是否已有可转换 memory 的 source 时，只要求 `task_name` 相同且成功证据链完整，不要求 task parameters 或 seed 与当前实例一致。
- 同一 setting 的可见结果只保留一组：先选择 official success 且截图、XML/UI tree、native observation、action/result 链完整的版本；证据完整度相同时保留完成时间最新者。失败、旧副本和冲突版本移入 `data/androidworld/.archive/`，不得覆盖仍需保留的原始数据。
- 归档迁移保留原相对路径、文件时间和可核验 provenance；`.archive/` 不参与 runtime 选择和完成数统计。零 step、`status=running`、无截图/XML 的占位 RunLog 不得作为可用结果或 memory source。
- 每个 Function v2 只保存动作和 `source_state_id`；source observation 位于 sibling `transfer_states.json`，不得内联或手改 Store。
- `data/androidworld/COMPLETION_STATUS.md` 展示 116×15 正式 E2E cell 完成情况；
  `MEMORY_READY_SOURCES.md` 每个任务只展示一份可复用 source；`RUNLOG_INDEX.md` 和
  `ARCHIVE_AUDIT.json` 提供逐 RunLog 证据与机器审计。目录迁移、去重或导入结束后
  必须通过官方 refresh 入口更新 `data/current.json`，再重生成这些文档。

## 修改方式

- 先按 `docs/FILE_EDIT_GUIDE.md` 找唯一 owner；一个概念只有一个生产实现。
- 删除前先用 `rg` 查所有调用者和测试；行为保留就把测试迁移到 surviving owner。
- shell、scheduler、launcher 等大文件先通过接口收敛，再拆分；不要增加第二入口。
- schema、public result row、统计文件或输出格式的改动必须单独 commit，并同时更新 schema、owner、测试和 README。
- 纯代码清理不得顺手修改 schema 或统计合同。
- 每个语义组先 focused test，再 `git diff --check`，然后 commit 并 push；完整测试最后运行。
- 本地编辑使用 `apply_patch`；命令、测试和转换使用仓库 `.venv/bin/python`。

## 入口说明

```text
bash scripts/exp/run_androidworld.sh ...
  -> src/experiment/run_tasks.py
  -> src/experiment/run_task.py
  -> src/integrations/android_world/run_episode.py
```

Function authoring 只走 `compile_runlog_to_store`；管理工具只有 README 中列出的
Function/RunLog 工具。详细文件职责、E2E 入口和索引区分见
`docs/ARCHITECTURE.md` 与 `docs/FILE_EDIT_GUIDE.md`。
