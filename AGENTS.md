# OmniFlow-exp 工作规则

本仓库只做论文 AndroidWorld 实验和 B-MoCA 验证。改代码前先读：

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/FILE_EDIT_GUIDE.md`
4. `scripts/exp/README.md`
5. 直接阅读对应 owner 代码和 focused tests

## 不可破坏的运行合同

- 唯一公开入口是 `scripts/exp/run_androidworld.sh`。
- 唯一 task + method + device 调度器是 `src/experiment/run_tasks.py`。
- `src/experiment/run_task.py` 只执行一个原子 AndroidWorld 结果。
- `src/integrations/android_world/run_episode.py` 是唯一 native episode/lifecycle owner。
- `save_function` 是唯一 Function 写 API、compiler 和 Store writer；成功 RunLog 只生成一个完整 Function。
- runtime 只读取注册的 Function Store 和 `data/current.json`，不能自动补 Store、建 catalog 或写平行 manifest。
- `data/current.json` 是唯一运行时本地索引；ledger、汇总和外部 manifest 只能作为证据。
- AndroidWorld 正式方法只有 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。
- B-MoCA 的 replay selector 属于外部 benchmark 合同，不复制进 AndroidWorld method 名称。

## OmniTransfer 合同

- OmniTransfer 的 canonical checkout 永远是 `~/Projects/Omni/OmniTransfer`。
- 页面编码只用 `omniflow/transfer/page_embedding.py` 和协议指定 checkpoint。
- 映射失败是明确的 transfer failure，回到正常 Planner/VLM fallback；绝不执行 source 坐标。
- 禁止 resource-id/node-id lookup、坐标 passthrough、第二 page encoder、第二 action mapper。
- Function checker 是 Function-local registration；只在 source action 成功映射且达到全局阈值时执行一次。

## 证据、环境和数据

- source Function 必须来自完整、成功、带截图和 native observation 的 RunLog。新跑的正式 AndroidWorld source 默认使用 seed 111；历史 RunLog 的 seed 只作 provenance，不影响其作为 memory source 的可用性。
- B-MoCA env100 必须先通过 official success、method success、`model_calls=0`、`fallback_steps=0`，才可创建/运行其他环境。
- 所有 Python/Torch 命令使用 `~/Projects/Omni/OmniFlow-exp/.venv/bin/python`；正式执行不使用邻近环境。
- 所有实验资产、RunLog、截图、Store、transfer states、memory 和结果都在 `data/`；不要提交它们、credentials、APK、权重或 emulator image。
- 项目长期记忆：`OmniFlow-AndroidWorld-Experiments` 的唯一关键主表事实是 `OmniFlow_AndroidWorld_116Tasks_10cell.xlsx` 固定为 116 个任务 × 10 个实验格 = 1160 个实验格；主矩阵不得因补充 baseline 或新实验而扩展。该目录 `~/Desktop/OmniFlow-AndroidWorld-Experiments` 是权威归档源；恢复或同步 10-cell 资产时，必须检查主表、对应的 `.inspect.ndjson` 和 `RESULT_GROUPS.md`。
- 10-cell 历史结果、测试结果和表格不能因目录迁移而丢失；合并到仓库时必须保留原始文件、来源路径、文件时间和 SHA-256 等 provenance，并去重而不是覆盖冲突版本。
- 桌面端 10-cell 归档是历史证据源，不是运行时选择器；归档副本放在 `data/`，运行时仍只读取注册的资产和 `data/current.json`。
- 旧适配层和历史输入可按只读迁移路径保留；不要为了兼容重新加 alias、旧 writer、旧 index 或旧 runner。
- 单 Function 直跑是有价值的旁路，但必须复用共同 Host、OmniTransfer、checker、证据封存和结果路径；不要删除它，也不要复制执行实现。

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
        function_store.json
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
- 每个 Function memory bundle 必须自包含 `function_store.json`、`transfer_states.json`、对应 `run_log.json` 和该 RunLog 引用的截图；不得依赖已归档 source 目录中的绝对截图路径，也不得手改 Function Store。
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

Function authoring 只走 bridge 的 `save_function`；管理工具只有 README 中列出的
Function/RunLog 工具。详细文件职责、旁路方案和索引区分见
`docs/ARCHITECTURE.md` 与 `docs/FILE_EDIT_GUIDE.md`。
