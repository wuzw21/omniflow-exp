# OmniFlow-exp 文件级编辑指南

这是唯一的 owner 指南。目录 README 只做入口导航；本表负责回答“改哪个
文件、为什么、不能改什么”。新增文件前先证明现有 owner 的接口无法承载；
如果只是为了复用，优先把实现放进已有 owner，而不是加一层旁路。

## 入口和契约

| 文件 | owner / 修改方式 |
| --- | --- |
| `README.md` | 仓库使用说明和公共路径；只描述已实现的入口，不放局部实现细节 |
| `AGENTS.md` | 不可违反的短规则；详细设计放本文和 `docs/ARCHITECTURE.md` |
| `CONTEXT.md` | 领域词汇；新增概念先补这里 |
| `docs/ARCHITECTURE.md` | 顶层设计、旁路、索引语义、精简候选 |
| `docs/FILE_EDIT_GUIDE.md` | 文件 owner、变更边界、提交分组 |
| `config/paper_androidworld.json` | 唯一正式 protocol；方法、设备、seed、预算、endpoint、AVD 和 revision 只在这里新增/修改 |
| `pyproject.toml` / `uv.lock` | 依赖和锁定环境；依赖变更单独说明并验证 |

## `omniflow/`：可复用运行时

| 文件 | owner / 修改方式 |
| --- | --- |
| `omniflow/__init__.py` | 稳定导出；不要在这里放实验调度 |
| `omniflow/bridge.py` | JSON-line 管理接口和 `run_gui`；新增工具必须复用现有 Function/Store 合同 |
| `omniflow/runlog.py` | canonical RunLog 读取、截图和 native observation 校验；不写实验索引 |
| `omniflow/vlm_coordinates.py` | VLM 像素与 canonical action 坐标转换；不处理 OmniTransfer |
| `omniflow/core/__init__.py` | core 导出 |
| `omniflow/core/androidworld_accessibility.py` | native accessibility 投影 |
| `omniflow/core/config.py` | runtime 配置类型和默认值；正式 protocol 不在这里复制 |
| `omniflow/core/model.py` | Observation、Action、Function、Result 等领域接口 |
| `omniflow/core/schemas.py` | 外部 action/payload 规范化；无 retired alias |
| `omniflow/core/trajectory.py` | RunLog/trajectory 的结构、顺序、hash 不变量 |
| `omniflow/functions/__init__.py` | Function 导出 |
| `omniflow/functions/assets.py` | 唯一 compiler、validator、`save_function`、Store writer；禁止第二 writer 或手改 Store |
| `omniflow/functions/recall.py` | 只从已加载 Store 选择完整 Function；不创建 catalog |
| `omniflow/runtime/__init__.py` | runtime 导出 |
| `omniflow/runtime/core.py` | 单个 canonical action primitive |
| `omniflow/runtime/checker.py` | active Function 的 checker session；规则只执行一次、映射失败只跳过 |
| `omniflow/runtime/engine.py` | 一个持久 Planner/Function 生命周期；不增加 method-specific loop |
| `omniflow/runtime/execution.py` | OmniTransfer、target selection、执行和正常 VLM fallback 的唯一 owner |
| `omniflow/runtime/semantic_grounding.py` | 当前 observation 的可见目标 grounding；不替代 OmniTransfer |
| `omniflow/transfer/__init__.py` | transfer 导出 |
| `omniflow/transfer/page_embedding.py` | 唯一 active page encoder；只能使用 canonical OmniTransfer checkpoint |
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
| `src/experiment/e2e_task_pipeline.py` | 唯一 task + method + device scheduler；旁路应作为这里的请求模式进入共同 launcher |
| `src/experiment/androidworld.py` | 一个 AndroidWorld `task + method + device` 结果；不是第二 scheduler |
| `src/experiment/process_runner.py` | 所有 experiment command 的子进程组、timeout、终止和 immutable log seam；不要复制 `Popen` 生命周期 |
| `src/experiment/development_emulator.py` | 有界开发 emulator preflight |
| `src/experiment/emulator_processes.py` | managed emulator 进程识别；诊断命令不写结果 |
| `src/experiment/preflight.py` | source、device、外部资产和 protocol gate；不生成 Store |
| `src/experiment/observation_evidence.py` | observation、截图、transfer coverage 证据封存 |
| `src/experiment/source_assets.py` | source evidence 验证与 legacy input 的内存转换；不创建平行 source pool |
| `src/experiment/artifact_index.py` | 唯一 `data/current.json` materializer/loader；不要增加 index/snapshot |
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
| `src/integrations/mobilegpt_converter.py` | 唯一 MobileGPT native memory converter |
| `src/integrations/mobilegpt_runtime.py` | pinned MobileGPT upstream runtime seam |
| `src/integrations/appagent_adapter.py` | AppAgent native conversion/runtime；不拥有调度 |
| `src/integrations/bmoca.py` | B-MoCA DeviceDriver、episode 和 official reward adapter |
| `src/integrations/script_replay.py` | 完整 Function 的共享 replay 薄适配器；禁止私有 mapper/executor |
| `src/integrations/skilldroid_replay.py` | DroidRun v0.5.6 官方 MacroPlayer adapter |
| `src/integrations/android_world/__init__.py` | AndroidWorld adapter 包标记 |
| `src/integrations/android_world/agent.py` | OmniFlow Host/runtime 构造和一个完整 `step()` cycle；direct Function 必须在这里进入 canonical runtime |
| `src/integrations/android_world/apps.py` | AndroidWorld app setup helper |
| `src/integrations/android_world/environment.py` | official task environment/validator adapter |
| `src/integrations/android_world/host.py` | native observe/act/reset Host |
| `src/integrations/android_world/launch.py` | 唯一 native lifecycle/launcher；直跑 Function 只能复用此生命周期，不得 patch `agent.run` |
| `src/integrations/android_world/methods.py` | 五个正式方法的 adapter registry，以及 direct Function intent；不得增加第二个 executor |
| `src/integrations/android_world/mobilegpt_agent.py` | MobileGPT episode adapter |
| `src/integrations/android_world/oob_control.py` | 仅显式选择的 development/source/OOB transport adapter |
| `src/integrations/android_world/state.py` | native state normalization |

## 脚本、schema、工具和测试

| 路径 | owner / 修改方式 |
| --- | --- |
| `scripts/exp/run_androidworld.sh` | 唯一公共入口；只做环境、协议读取、进程 dispatch |
| `scripts/exp/README.md` | shell 命令合同；flags 改动与脚本同 commit |
| `schemas/oob/*.json` | 外部 wire contract；任何字段/版本修改单独 schema commit |
| `schemas/oob/README.md` | schema 语义和禁止事项 |
| `tools/manual_androidworld_harness.py` | 人工诊断；不能创建 formal result、刷新 index 或代替 launcher |
| `tests/runlog_fixtures.py` | 共用 RunLog fixture |
| `tests/test_function_*.py` | Function compiler、writer、recall 和 management 合同 |
| `tests/test_runtime_*.py` | runtime/checker/transfer/fallback 合同 |
| `tests/test_transfer_*.py` / `tests/test_visual_transfer_pipeline.py` | OmniTransfer candidate/page encoder 合同 |
| `tests/test_androidworld_*.py` | AndroidWorld Host、environment、method 合同 |
| `tests/test_e2e_task_pipeline.py` / `tests/test_exp_script.py` | 唯一 scheduler 和 public shell 合同 |
| `tests/test_result_registry.py` / `tests/test_batch_outcomes.py` | ledger 与汇总分开验证 |
| `tests/test_mobilegpt_*.py` / `tests/test_appagent_source.py` | 外部 baseline adapter 合同 |
| `tests/test_script_replay.py` / `tests/test_skilldroid_replay.py` | replay adapter 合同，尤其验证无私有 mapper |
| 其余 `tests/test_*.py` | 按被测 owner 放置；删除 retired alias 时删除对应测试，不为兼容保留测试 |

## 变更与提交规则

- 纯文档/owner 调整可以一个文档 commit；代码重构按一个 seam 一个 commit。
- schema、public result row、统计/汇总输出是独立 commit，不能和普通清理混在一起。
- 删除前必须 `rg` 找调用者，并把行为测试移到 surviving owner；无调用者 helper 才能删除。
- 任何代码/测试/数据写入都使用仓库 `.venv/bin/python` 和 `apply_patch`。
- 每个完成的语义组都执行 `git diff --check`、focused tests、commit、push；完整测试放在最后。
