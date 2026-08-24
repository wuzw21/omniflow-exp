# OmniFlow-exp 工作规则

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
- 唯一 task + method + device 调度器是 `src/experiment/run_tasks.py`。
- `src/experiment/run_task.py` 只执行一个原子 AndroidWorld 结果。
- `src/integrations/android_world/run_episode.py` 是唯一 native episode/lifecycle owner。
- `compile_runlog_to_store` 是唯一 Function 写 API；输出 v2 `store.json`、`compile_report.json` 和 sibling `transfer_states.json`。
- runtime 只读取注册的 Function Store 和 `data/current.json`，不能自动补 Store、建 catalog 或写平行 manifest。
- `data/current.json` 是唯一运行时本地索引；ledger、汇总和外部 manifest 只能作为证据。
- AndroidWorld 正式方法只有 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。
- B-MoCA 的 replay selector 属于外部 benchmark 合同，不复制进 AndroidWorld method 名称。
- AndroidWorld 的 OmniFlow 正式执行只有完整 E2E Planner 循环：runner 只传 goal、
  Function Store 和设备环境；Planner 从 Function `input_schema` 生成参数，runtime
  通过 `bindings` 绑定后执行。禁止 direct Function id、Function 序列、预绑定参数、
  source qualification replay 或任务专用执行脚本。

## OmniTransfer 合同

- OmniTransfer 的 canonical checkout 永远是 `~/Projects/Omni/OmniTransfer`。
- 页面命中只用 `omniflow/transfer/embedding.py`：canonical `UIGraph` 节点经原生手工 64D 表示和 8-word pooling 形成 512D `PageEncoder`；OmniTransfer runtime 仍负责动作映射。
- 映射失败是明确的 transfer failure，回到正常 Planner/VLM fallback；绝不执行 source 坐标。
- 禁止 resource-id/node-id lookup、坐标 passthrough、第二 page encoder、第二 action mapper。
- checker 是独立共享库，可跨 Function 使用；一个 RunLog 可注册并按顺序执行多个 Function。
- `click.target_description` 和 `input_text.target_description` 可绑定为 Function 参数，只作为 source-action 语义交给统一 OmniTransfer；禁止新增 target-side 文本查找器。

## 证据、环境和数据

- 4090 长期部署规则：OmniFlow-exp 的唯一权威工作区固定为 `/home/zewen/Projects/OmniFlow-exp`。代码、`.git`、`.venv`、`data/current.json`、实验资产和结果都在这一工作区内直接修改、验证和运行；Git commit 是唯一版本历史。禁止再创建或从 `local-authoritative`、`current-snapshot`、按日期/任务命名的 checkout、`/data/omniflow-4090/OmniFlow-exp` 或其他复制目录部署和启动实验。旧目录只可作为只读历史证据，不得成为运行入口。所有 4090 命令必须先解析并校验 canonical workspace，路径不一致立即失败。
- AndroidWorld 时间合同：一个完整 task 的默认和最大 wall deadline 都是 600 秒；正式 OmniFlow Planner 单次模型请求默认最多 30 秒。禁止通过实验命令把整任务放宽到 10 分钟以上。
- Function v2 按成功 RunLog 动作顺序保存；步骤通过 `source_state_id` 引用 sibling `transfer_states.json`。新跑的正式 AndroidWorld source 默认使用 seed 111。
- B-MoCA env100 必须先通过 official success、method success、`model_calls=0`、`fallback_steps=0`，才可创建/运行其他环境。
- 所有 Python/Torch 命令使用 `~/Projects/Omni/OmniFlow-exp/.venv/bin/python`；正式执行不使用邻近环境。
- 长期训练规则：任何模型训练、微调或训练 smoke 都必须在 `9207` 远程环境执行；本地只允许数据清洗、质量检测、代码测试和评测入口验证，不得在本地启动训练。
- 所有实验资产、RunLog、截图、Store、transfer states、memory 和结果都在 `data/`；不要提交它们、credentials、APK、权重或 emulator image。
- 项目长期记忆：`OmniFlow-AndroidWorld-Experiments` 的唯一关键主表事实是 `OmniFlow_AndroidWorld_116Tasks_10cell.xlsx` 固定为 116 个任务 × 10 个实验格 = 1160 个实验格；主矩阵不得因补充 baseline 或新实验而扩展。该目录 `~/Desktop/OmniFlow-AndroidWorld-Experiments` 是权威归档源；恢复或同步 10-cell 资产时，必须检查主表、对应的 `.inspect.ndjson` 和 `RESULT_GROUPS.md`。
- AndroidWorld target 长期规则：正式 target 始终且仅为 Standard AVD + Fold + Tablet，分别使用 `standard45562`/`OmniFlowTargetSmall`、`fold45564`/`OmniFlowTargetFold` 和 `tablet45554`/`WXGA_Tablet_test_00`；其中 `small_phone` 是普通小尺寸手机形态的 protocol profile，表示 Standard AVD 的设备形态，不是额外设备型号或 Pixel 5 别名。`source5560` 永远是 source-only，不得作为 target。`pixel5576`/`AndroidWorldAvd4090` 已从正式 target 协议退役，只能作为只读历史兼容输入。该拓扑调整不自动扩展历史 116×10 主表或归档矩阵。
- 10-cell 历史结果、测试结果和表格不能因目录迁移而丢失；合并到仓库时必须保留原始文件、来源路径、文件时间和 SHA-256 等 provenance，并去重而不是覆盖冲突版本。
- 桌面端 10-cell 归档是历史证据源，不是运行时选择器；归档副本放在 `data/`，运行时仍只读取注册的资产和 `data/current.json`。
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

- `<method>` 只使用统一名称：source 证据用 `source`；正式实验用 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。禁止用 collector 名、模型昵称、时间戳前缀或历史 runner 名创建平行 method。
- 设备按 `device.json` 中探测到的真实 `device_model` 归类，不按 adb serial、端口、host alias 或人为昵称归类。设备目录保留实际运行 seed 作为 provenance；新跑的正式 source 默认使用 111，正式 target 结果同时写 `_eval<evaluation_seed>`。
- 可见 RunLog 目录只能命名为 `attempt_NNN`，同一 device 目录内从 `attempt_001` 递增。禁止保留 `manual_*`、`object_*`、`forced_live_plan_*`、hash 后缀或时间戳别名；导入时统一转换后再进入可见目录。
- setting 身份由 `task_name + method + 真实 device_model + 规范化 task_parameters + 协议配置` 决定，不由 RunLog seed、目录旧名、run_id、文件 hash 或设备别名决定。判断一个任务是否已有可转换 memory 的 source 时，只要求 `task_name` 相同且成功证据链完整，不要求 task parameters 或 seed 与当前实例一致。
- 同一 setting 的可见结果只保留一组：先选择 official success 且截图、XML/UI tree、native observation、action/result 链完整的版本；证据完整度相同时保留完成时间最新者。失败、旧副本和冲突版本移入 `data/androidworld/.archive/`，不得覆盖仍需保留的原始数据。
- 归档迁移保留原相对路径、文件时间和可核验 provenance；`.archive/` 不参与 runtime 选择和完成数统计。零 step、`status=running`、无截图/XML 的占位 RunLog 不得作为可用结果或 memory source。
- 每个 Function v2 只保存动作和 `source_state_id`；source observation 位于 sibling `transfer_states.json`，不得内联或手改 Store。
- `data/androidworld/COMPLETION_STATUS.md` 展示 116×10 正式 cell 完成情况；`MEMORY_READY_SOURCES.md` 展示每个任务各 setting 的 source 可用状态；`RUNLOG_INDEX.md` 和 `ARCHIVE_AUDIT.json` 提供逐 RunLog 证据与机器审计。目录迁移、去重或导入结束后必须通过官方 refresh 入口更新 `data/current.json`，再重生成这些文档。

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
