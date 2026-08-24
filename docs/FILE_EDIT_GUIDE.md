# OmniFlow-exp 文件级编辑指南

这是唯一的 owner 指南。目录 README 只做入口导航；本表负责回答“改哪个
文件、为什么、不能改什么”。新增文件前先证明现有 owner 的接口无法承载；
如果只是为了复用，优先把实现放进已有 owner，而不是加一层旁路。

## Python 文件的修改权限

仓库当前有 140 个 Python 文件。下面的规则覆盖全部 `omniflow/**/*.py`、
`src/**/*.py`、`tests/**/*.py` 和 `tools/**/*.py`；新增 Python 文件必须先
放入一个已有 owner 的语义边界，并在本表补充路径。这里的“可改”不是任何人
可以随意改，而是允许修改实现；真正的判断还要看该文件所属的变更级别。

| 级别 | 可以做什么 | 必须满足 | 禁止做什么 |
| --- | --- | --- | --- |
| A · owner 可重构 | 修复实现、删重复代码、抽深模块、改善命名和测试 | 只改变该 owner 的语义；focused tests、`git diff --check`、完整回归 | 增加第二入口、第二 writer、第二 mapper 或私有旁路 |
| B · contract-controlled | 修改 runtime 合同、适配器协议、索引/ledger 行为 | 先更新 owner 文档和 focused tests；schema、public result row、统计输出必须另一个独立 commit | 在普通清理 commit 中悄悄改变字段、版本、统计口径 |
| C · frozen / read-only | 只能读取、诊断或迁移；不能为了新功能修补 | 如确实需要改变，先证明上游合同或用户授权已变化，并单独记录 | 改历史 baseline、外部 pinned runtime、生成证据、模型/数据文件 |
| D · test-only | 增补能证明 surviving owner 的行为测试 | 测试必须锁定真实不变量，不得为了过测删除断言 | 用测试替代生产逻辑，或把失败路径改成“通过” |

### 按 Python 路径的默认级别

| Python 路径 | 默认级别 | 语义 owner 和边界 |
| --- | --- | --- |
| `omniflow/core/model.py`, `schemas.py`, `trajectory.py` | B | 核心领域对象、wire/schema、RunLog 不变量；实现可修，合同变更单独 commit |
| `omniflow/core/config.py`, `omniflow/vlm/model_config.py`, `omniflow/vlm/planner.py`, `omniflow/vlm/usage.py` | B | runtime/model-facing 合同与 accounting；不能恢复 retired credential、tool 或隐式 retry |
| `omniflow/functions/assets.py`, `omniflow/functions/recall.py` | B | 唯一 Function compiler、validator、writer、recall；允许深模块重构，禁止第二 Store writer/catalog |
| `omniflow/runtime/*.py` | A/B | canonical execution、checker、fallback；必须复用 OmniTransfer，不能把 source 坐标当 target 执行 |
| `omniflow/transfer/*.py` | B/C | 唯一 page encoder 和 transfer-state reader；只能使用 canonical checkpoint，禁止新增 encoder/lookup 旁路 |
| `omniflow/bridge.py`, `omniflow/runlog.py`, `omniflow/vlm/context.py`, `omniflow/vlm_coordinates.py` | B | 对外 bridge、canonical RunLog 和 Planner evidence；适配必须回到现有合同 |
| `omniflow/**/__init__.py`, `src/**/__init__.py` | A | 只维护稳定导出和包边界，不放调度、业务实现或隐式兼容层 |
| `src/experiment/run_tasks.py`, `run_task.py`, `run_process.py` | A | scheduler、一个原子结果、统一子进程生命周期；不得新增 runner 或重复 `Popen` policy |
| `src/experiment/paths.py` | A/B | 唯一 repository-relative、index-relative 和 safe artifact path 规则；外部 roots 仍必须显式传入 |
| `src/experiment/data_index.py`, `checks.py`, `source_evidence.py`, `observation_evidence.py` | B | 唯一 `data/current.json`、source/设备 gate、证据转换；source 不负责 provider 分派；不得增加 index、snapshot 或 source pool |
| `src/experiment/offline_transfer_regression.py` | B | Page Embedding + canonical OmniTransfer 的离线 pair 数据集、标注和错误闭环；不得创建 Host/Planner/episode、第二 encoder/mapper 或 official result |
| `src/experiment/protocol.py`, `result_schema.py`, `result_registry.py`, `batch_outcomes.py` | B | 正式 protocol、public row、ledger、汇总；字段/版本/统计口径必须独立 commit |
| `src/experiment/development_emulator.py`, `emulator_processes.py`, `performance_metrics.py` | A/B | 开发 preflight、进程诊断、opt-in 性能侧通道；不能写 formal result 或改变 public row |
| `src/experiment/mobilegpt_contract.py`, `mobilegpt_source.py`, `appagent_source.py` | B | 外部 baseline 的 source/contract owner；适配可以改，不能把 baseline 变成 Function 或共用结果表 |
| `src/integrations/android_world/agent.py`, `methods.py`, `host.py`, `environment.py`, `apps.py`, `state.py` | A/B | AndroidWorld native adapter；可重构实现，但必须复用 Host、官方 validator 和唯一 method registry |
| `src/integrations/android_world/run_episode.py` | B | 唯一 native lifecycle；可修 setup/episode/evidence，但不能在这里增加 scheduler 或临时 executor |
| `src/integrations/android_world/oob_control.py` | C/B | 明确的开发/采集传输边界；不能复制成新适配层 |
| `src/integrations/bmoca.py`, `mobilegpt.py`, `mobilegpt_format.py`, `appagent.py` | B/C | 外部文件格式转换；不修改 pinned upstream 语义，不引入运行时适配 |
| `src/integrations/runlog.py`, `script_replay.py`, `skilldroid_replay.py` | C/B | 历史/官方 replay 薄适配；不能加私有 mapper、坐标 passthrough 或第二 executor |
| `tests/**/*.py` | D | 所有测试都可增补或随 surviving owner 迁移；不得削弱断言来掩盖功能变化 |
| `tools/manual_androidworld_harness.py` | C | 人工诊断工具；不能生成 formal result、刷新 canonical index 或替代公共 shell |

### 修改决策

对任意 Python 文件，先回答：

1. 这是实现重复，还是两个不同的领域语义？只有前者才能合并。
2. 这个文件是否拥有入口、Store、schema、ledger、统计或外部合同？若是，按 B
   处理，不要把改动混入普通清理。
3. 这个文件是否只是 adapter？先改共同 owner 或 adapter seam；不要在调用方
   加一层包装来掩盖重复。
4. 改动能否通过 surviving owner 的 focused test 证明？不能证明就不能删除。

快速检查全部 Python 文件是否仍落在本表范围内：

```bash
rg --files -g '*.py' | sort
```

## 入口和契约

| 文件 | owner / 修改方式 |
| --- | --- |
| `README.md` | 仓库使用说明和公共路径；只描述已实现的入口，不放局部实现细节 |
| `AGENTS.md` | 不可违反的短规则；详细设计放本文和 `docs/ARCHITECTURE.md` |
| `docs/ARCHITECTURE.md` | 顶层设计、旁路、索引语义、精简候选 |
| `docs/FILE_EDIT_GUIDE.md` | 文件 owner、变更边界、提交分组 |
| `config/paper_androidworld.json` | 唯一正式 protocol；方法、设备、seed、预算、endpoint、AVD 和 revision 只在这里新增/修改 |
| `pyproject.toml` / `uv.lock` | 依赖和锁定环境；依赖变更单独说明并验证 |

## `omniflow/`：可复用运行时

| 文件 | owner / 修改方式 |
| --- | --- |
| `omniflow/__init__.py` | 稳定导出；不要在这里放实验调度 |
| `omniflow/bridge.py` | JSON-line 管理接口和唯一执行工具 `run_gui`；Function 名不能作为隐藏直调工具 |
| `omniflow/runlog.py` | canonical RunLog 读取、截图和 native observation 校验；不写实验索引 |
| `omniflow/vlm_coordinates.py` | VLM 像素与 canonical action 坐标转换；不处理 OmniTransfer |
| `omniflow/core/__init__.py` | core 导出 |
| `omniflow/core/androidworld_accessibility.py` | native accessibility 投影 |
| `omniflow/core/config.py` | runtime 配置类型和默认值；正式 protocol 不在这里复制 |
| `omniflow/core/model.py` | Observation、Action、Function、Result 等领域接口 |
| `omniflow/core/schemas.py` | 外部 action/payload 规范化；无 retired alias |
| `omniflow/core/trajectory.py` | RunLog/trajectory 的结构与顺序不变量；旧截图 hash 只读兼容，新 RunLog 不写 hash |
| `omniflow/functions/__init__.py` | Function 导出 |
| `omniflow/functions/compiler.py` | 唯一 v2 RunLog compiler；物化 Agent 给出的 0+ 语义 Function 和一个最大安全组合 Function；允许省略 RunLog 重复/重试/checker/观察依赖动作，并校验 schema、顺序、原子边界与参数提升；写 `store.json`、`compile_report.json` 和 sibling `transfer_states.json` |
| `omniflow/functions/artifact.py` | v2 Function schema、binding 和参数校验 |
| `omniflow/functions/store.py` | v2 Store reader/writer；禁止手改 Store |
| `omniflow/functions/recall.py` | 只从已加载 Store 选择完整 Function；不创建 catalog |
| `omniflow/runtime/__init__.py` | runtime 导出 |
| `omniflow/runtime/core.py` | 单个 canonical action primitive |
| `omniflow/runtime/checker.py` | 独立共享 checker library、rule validation 和 deterministic recovery |
| `omniflow/runtime/engine.py` | 一个持久 Planner/Function 生命周期；不增加 method-specific loop |
| `omniflow/runtime/execution.py` | `execute_function`、`execute_robust_action`、`align_function_resume` 和 OmniTransfer 执行的唯一 owner |
| `omniflow/transfer/__init__.py` | transfer 导出 |
| `omniflow/transfer/admission.py` | 唯一 contextual mapping 准入策略；NULL、`0.8` 阈值和目标可执行性在此统一判断 |
| `omniflow/transfer/embedding.py` | 唯一 active page encoder；直接调用 canonical OmniTransfer v10 learned state-attention checkpoint 得到 1024D 页面命中表示，不在 OmniFlow 内生成手工向量或第二套 `UIGraph` encoder |
| `omniflow/transfer/runtime.py` | immutable transfer-state catalog 读取和 coverage 检查 |
| `omniflow/vlm/__init__.py` | VLM 导出 |
| `omniflow/vlm/context.py` | Planner evidence/context 组装 |
| `omniflow/vlm/model_config.py` | endpoint/credential profile 解析；不恢复旧 credential 名 |
| `omniflow/vlm/planner.py` | frozen model-facing tool space、调用校验和 Planner policy |
| `omniflow/vlm/usage.py` | token/model-call accounting |

## `src/experiment/`：论文工作流

| 文件 | owner / 修改方式 |
| --- | --- |
| `src/__init__.py` | 包标记 |
| `src/experiment/__init__.py` | experiment 导出 |
| `src/experiment/protocol.py` | `config/paper_androidworld.json` 的 typed view；不复制常量 |
| `src/experiment/run_tasks.py` | 唯一 task + method + device scheduler；AndroidWorld 只调度完整 E2E goal，不选择 Function 或绑定参数 |
| `src/experiment/run_task.py` | 一个 AndroidWorld `task + method + device` 结果；不是第二 scheduler |
| `src/experiment/paths.py` | 唯一路径解析、artifact component 和文件 SHA-256 owner；调用方不要重新实现 `Path(...).resolve()`、`_safe_component` 或文件哈希 |
| `src/experiment/run_process.py` | 所有 experiment command 的子进程组、timeout、终止和 immutable log seam；不要复制 `Popen` 生命周期 |
| `src/experiment/development_emulator.py` | 有界开发 emulator preflight |
| `src/experiment/emulator_processes.py` | managed emulator 进程识别；诊断命令不写结果 |
| `src/experiment/checks.py` | source、device、外部资产和 protocol gate；默认检查 root 并启用已安装的实验 Accessibility 服务；不生成 Store |
| `src/experiment/observation_evidence.py` | observation、截图、transfer coverage 证据封存 |
| `src/experiment/offline_transfer_regression.py` | 自包含 source/target pair 的去重收集、人工标签、全量回归报告和派生错误视图；只调用唯一 PageEncoder 与生产 transfer seam |
| `src/experiment/source_evidence.py` | source evidence 验证与 legacy input 的统一投影；不负责 AppAgent/MobileGPT 转换，不创建平行 source pool |
| `src/experiment/source_records.py` | `CanonicalRunLog`/`SourceRunLogProfile` 纯数据模型；不读取 index、不执行任务 |
| `src/experiment/data_index.py` | 唯一 `data/current.json` materializer/loader；不要增加 index/snapshot |
| `src/experiment/result_schema.py` | compact public result row；字段改动必须 schema/stat 独立 commit |
| `src/experiment/result_registry.py` | immutable result registration ledger；不负责调度或汇总 |
| `src/experiment/batch_outcomes.py` | 一次 attempt 的 outcome/summary；不成为运行时选择器 |
| `src/experiment/performance_metrics.py` | 明确 opt-in 的 performance side channel；不改 public result row |
| `src/experiment/mobilegpt_contract.py` | MobileGPT 证据常量和协议标签 |
| `src/experiment/mobilegpt_source.py` | AndroidWorld 的 MobileGPT source preparation |
| `src/experiment/appagent_source.py` | AndroidWorld 的 AppAgent source preparation |

## `src/integrations/`：外部契约适配器

| 文件 | owner / 修改方式 |
| --- | --- |
| `src/integrations/__init__.py` | 适配器包标记 |
| `src/integrations/runlog.py` | 外部/历史 RunLog 投影；canonical loader 在 `omniflow/runlog.py` |
| `src/integrations/mobilegpt.py` | 唯一 MobileGPT native memory converter |
| `src/integrations/mobilegpt_memory.py` | MobileGPT Prepared Memory 的统计、图检查、manifest/evidence 校验；不拥有 task 调度 |
| `src/integrations/mobilegpt_format.py` | 只调用 MobileGPT 官方 XML Encoder；不运行、不 patch MobileGPT |
| `src/integrations/appagent.py` | AppAgent native memory conversion/validation；不拥有执行调度 |
| `src/integrations/official_forward.py` | 唯一外部 baseline forwarder；只准备临时 workspace、设备和官方入口，不实现 parser/controller/action loop |
| `vendor/autodroid/androidworld_apps` + `official_forward.py` | AutoDroid 官方 DroidBot memory 与 replay forward；不转换为 OmniFlow schema，不复制 action loop |
| `src/integrations/bmoca.py` | B-MoCA DeviceDriver、episode 和 official reward adapter |
| `src/integrations/script_replay.py` | 仅 B-MoCA 外部 replay 合同的共享 runtime 薄适配器；禁止接入 AndroidWorld 方法或私有 mapper/executor |
| `src/integrations/skilldroid_replay.py` | DroidRun v0.5.6 官方 MacroPlayer adapter |
| `src/integrations/android_world/__init__.py` | AndroidWorld adapter 包标记 |
| `src/integrations/android_world/agent.py` | OmniFlow Host/runtime 构造和一个完整 `step()` cycle；只调用 `flow.run(goal)`，Planner 自行选择 Function 并填写 schema 参数 |
| `src/integrations/android_world/apps.py` | AndroidWorld app setup helper |
| `src/integrations/android_world/environment.py` | official task environment/validator adapter |
| `src/integrations/android_world/host.py` | native observe/act/reset Host |
| `src/integrations/android_world/run_episode.py` | 唯一 native lifecycle/launcher；不得暴露 direct Function、调用序列或测试专用执行模式 |
| `src/integrations/android_world/methods.py` | 正式方法 adapter registry；OmniFlow adapter 只传 Store、Planner 和 Host 配置，不传 Function intent |
| `src/integrations/mobilegpt_format.py` | MobileGPT 官方 XML Encoder 的最小转换入口 |
| `src/integrations/android_world/oob_control.py` | 仅显式选择的 development/source/OOB transport adapter |
| `src/integrations/android_world/state.py` | native state normalization |

## 脚本、schema、工具和测试

| 路径 | owner / 修改方式 |
| --- | --- |
| `scripts/exp/run_androidworld.sh` | 唯一公共入口；只做环境、协议读取、进程 dispatch |
| `scripts/exp/test_provider.sh` | provider 离线合同 harness；只编排现有 focused tests 和 CLI 检查，不启动 emulator、不写数据、不成为第二运行入口 |
| `scripts/exp/README.md` | shell 命令合同；flags 改动与脚本同 commit |
| `schemas/oob/*.json` | 外部 wire contract；任何字段/版本修改单独 schema commit |
| `schemas/oob/README.md` | schema 语义和禁止事项 |
| `tools/manual_androidworld_harness.py` | 人工诊断；不能创建 formal result、刷新 index 或代替 launcher |
| `tests/runlog_fixtures.py` | 共用 RunLog fixture |
| `tests/test_function_*.py` | Function compiler、writer、recall 和 management 合同 |
| `tests/test_function_store_migration.py` | 旧 Store/bundle 到新版 Store 的迁移合同；迁移行为改变时同步修改，不要把旧格式重新接回 runtime |
| `tests/test_runtime_*.py` | runtime/checker/transfer/fallback 合同 |
| `tests/test_transfer_*.py` / `tests/test_visual_transfer_pipeline.py` | OmniTransfer candidate/page encoder 合同 |
| `tests/test_offline_transfer_regression.py` | 成功/失败 pair 去重、负样本、人工标注、全量错误闭环和零设备/模型调用合同 |
| `tests/test_androidworld_*.py` | AndroidWorld Host、environment、method 合同 |
| `tests/test_run_tasks.py` / `tests/test_exp_script.py` | 唯一 scheduler 和 public shell 合同 |
| `tests/test_result_registry.py` / `tests/test_batch_outcomes.py` | ledger 与汇总分开验证 |
| `tests/test_mobilegpt_*.py` / `tests/test_appagent_source.py` | 外部 baseline adapter 合同 |
| `tests/test_script_replay.py` / `tests/test_skilldroid_replay.py` | B-MoCA/外部 replay adapter 合同，尤其验证无私有 mapper；不代表 AndroidWorld 入口 |
| 其余 `tests/test_*.py` | 按被测 owner 放置；删除 retired alias 时删除对应测试，不为兼容保留测试 |

### Provider 修改的最短路径

每个 provider 只维护一个公开 source owner：MobileGPT 改
`src/experiment/mobilegpt_source.py`，AppAgent 改
`src/experiment/appagent_source.py`。需要改变上游格式时，再进入对应的
`src/integrations/*.py`；不要把同一个变化复制到 scheduler、AndroidWorld
result runner 和 shell。

修改后运行：

```bash
bash scripts/exp/test_provider.sh mobilegpt
bash scripts/exp/test_provider.sh appagent
# 或一次运行两者
bash scripts/exp/test_provider.sh all
```

这个 harness 只测试现有 provider seam，不生成 memory、不调用模型、不启动
模拟器。正式端到端测试仍然只能从
`bash scripts/exp/run_androidworld.sh ...` 进入。

## 变更与提交规则

- 纯文档/owner 调整可以一个文档 commit；代码重构按一个 seam 一个 commit。
- schema、public result row、统计/汇总输出是独立 commit，不能和普通清理混在一起。
- 删除前必须 `rg` 找调用者，并把行为测试移到 surviving owner；无调用者 helper 才能删除。
- 任何代码/测试/数据写入都使用仓库 `.venv/bin/python` 和 `apply_patch`。
- 每个完成的语义组都执行 `git diff --check`、focused tests、commit、push；完整测试放在最后。
