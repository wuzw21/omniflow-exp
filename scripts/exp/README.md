# AndroidWorld experiment launcher

`run_androidworld.sh` 是唯一公开实验入口。普通使用只需要指定 task、method
和 device；protocol 中的 seed、模型和设备定义由代码读取，不需要手动复制。

正式 AndroidWorld 矩阵固定使用五个方法：`fixed_replay`、`omniflow`、
`mobilegpt`、`appagent`、`t3a_hint`。历史 AutoDroid/DroidBot replay 仅保留为
只读兼容边界，不进入正式矩阵。

## 一键测试一个 task

```bash
bash scripts/exp/run_androidworld.sh \
  --e2e-task TASK \
  --e2e-method omniflow \
  --e2e-device small5562:emulator-5562:5562,fold5564:emulator-5564:5564,small5554:emulator-5554:5554 \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113 \
  --control-backend oob
```

这个命令会：

1. 查找 task 对应的 Function；
2. 缺失时从成功 source RunLog 生成并校验 Function；
3. 刷新 `data/current.json`；
4. 只启动选中的 method/device E2E。

检查失败会安全停止，不会启动 target episode。

正式运行和 source collection 都会追加到
`data/androidworld/<task>/<method>/<device_model_seed>/runlog/attempt_NNN/`；每次
执行只写一份 `run_log.json` 和顺序命名的 `screenshots/`，不使用时间戳、截图
SHA 或对象仓库。不会再
生成平行的 `androidworld_10cell`、`androidworld_single_task_attempts` 或
`androidworld_validator` 顶层目录。可用
`./.venv/bin/python tools/audit_androidworld_archive.py` 刷新 116×10 完成表和
逐 RunLog 证据索引。

## 一键安装并启动设备

同一个入口也负责设备 setup；它会安装 OOB、MobileGPT 和 accessibility
forwarder，启用服务，检查 AndroidWorld/OmniTransfer/AppAgent/MobileGPT 的
host 环境，并用当前 OOB observe bridge 做最小通信探针：

```bash
PYTHON_BIN=/absolute/.venv/bin/python \
OMNIFLOW_ANDROID_SDK_ROOT=/absolute/Android/Sdk \
OMNIFLOW_ANDROIDWORLD_A11Y_APK=/absolute/accessibility_forwarder.apk \
bash scripts/exp/run_androidworld.sh --setup-device small5554
```

`--setup-device` 可接单个 label、逗号列表或 `all`；`all` 包含三个 target
和 source 设备。报告写到
`data/androidworld/.archive/setup/<UTC>/setup_report.json`。缺少 OOB
的当前 `OBSERVE_OMNIFLOW`/`CONTROL_OMNIFLOW` receiver、缺少 APK、协议版本
不匹配或 accessibility 未 bound 都会失败，不会开始实验。默认会补齐 Python
依赖；只做已有环境验收时设 `OMNIFLOW_SETUP_INSTALL_PYTHON=0`。

## 选择范围

```bash
# 只测 OmniFlow
--e2e-method omniflow

# 全部正式 method 和设备
--e2e-method all --e2e-device all

# 逗号分隔的子集
--e2e-method omniflow,mobilegpt \
--e2e-device small5562:emulator-5562:5562,fold5564:emulator-5564:5564
```

MobileGPT 的可复用 memory 由 seed 111 的成功 source RunLog 离线生成。转换器
直接保留 RunLog 的成功动作，并通过 MobileGPT 官方 `Memory` / `PageManager` API
写入页面、层级、subtask、action 和 task path；随后必须由官方 reader 逐动作回读、
完成 RunLog 对齐审计，才会封存并注册。这个阶段不启动 source emulator、MobileGPT
Server 或新的 AndroidWorld episode，也不使用 Function Store。只转换/校验 memory
而不启动 target 时使用：

MobileGPT 的 client/server handshake 和 Server handler 错误属于可重试的环境/
适配失败；step timeout、step budget exhausted 和官方 validator=false 才属于方法结论。
因此完成跳过不会把连接或 Server 崩溃误登记成 method failure。

```bash
bash scripts/exp/run_androidworld.sh \
  --e2e-task TASK \
  --e2e-method mobilegpt \
  --e2e-device all \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113 \
  --prepare-mobilegpt-memory-only
```

当前 AndroidWorld target 设备：

| label | serial | profile |
| --- | --- | --- |
| `standard45562` | `emulator-45562` | `small_phone` |
| `fold45564` | `emulator-45564` | `pixel_fold` |
| `tablet45554` | `emulator-45554` | `tablet` |

The formal target topology is always Standard AVD (`standard45562`), Fold
(`fold45564`), and Tablet (`tablet45554`). The Standard AVD is
`OmniFlowTargetSmall`; the Tablet uses the existing `WXGA_Tablet_test_00`
AVD (`tablet`, 10.1-inch WXGA). The source/original `source5560` remains an
internal source-only device on `OmniFlowSourceSmall` and is never a target cell.
The retired `pixel5576` / `AndroidWorldAvd4090` pair is historical read-only
compatibility only and is not part of the formal protocol.

## AutoDroid 补充基线（9207）

AutoDroid 不属于正式五方法，也不进入 `--e2e-method all` 或 116 × 10 主表。
它只能显式使用 9207 上的三个独立设备标签和独立结果命名空间：

```bash
OMNIFLOW_AUTODROID_ROOT=/absolute/OmniFlow-exp/vendor/autodroid/runtime \
OMNIFLOW_AUTODROID_MEMORY_ROOT=/absolute/OmniFlow-exp/vendor/autodroid/androidworld_apps \
OMNIFLOW_AUTODROID_POLICY=task \
bash scripts/exp/run_androidworld.sh \
  --e2e-task CameraTakePhoto \
  --e2e-method autodroid \
  --e2e-device all \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113
```

The 9207 supplemental devices are `autodroidsmall5554` (small),
`autodroidfold5564` (fold), and `autodroidandroidworld5594` (AndroidWorldAvd).

结果写入 `data/androidworld/<task>/autodroid/<device_model_seed>/`，调度与汇总元数据写入
`data/androidworld/.archive/`，使用
原生 DroidBot policy 和 AndroidWorld 官方 validator，不转换 Function、
不使用 OmniTransfer。默认 `replay` 保持历史 UTG 结果；设置
`OMNIFLOW_AUTODROID_POLICY=task` 才执行官方 online TaskPolicy，并把
`autodroid_stats.jsonl` 的 model calls、prompt/completion/total tokens 接入统一
outcome 和 summary 统计。online 模式还必须提供 `OMNIFLOW_ENV_FILE`；入口强制
使用 protocol 的 `GLM-4.6V`、`llmthu` endpoint 和默认 temperature `0.25`，不会
继承 `.env` 中的 Qwen/DashScope model。online 结果使用独立 attempt id/output root，不覆盖
历史 replay。完整公平比较合同见
[`docs/AUTODROID_9207_COMPARISON_PLAN.md`](../../docs/AUTODROID_9207_COMPARISON_PLAN.md)。

启动 AutoDroid 前，统一入口会收起系统面板；检测到 Fold 的多显示布局时，先将目标 app
放到逻辑 display `0`，再交回原生 DroidBot/TaskPolicy。该前置适配不修改 UTG、坐标或
官方策略，并在 attempt 中保存 `device_preflight.json`。

运行完整的 116-task supplemental campaign 时，使用批量入口并显式选择
supplemental method；这不会改变正式 `all` 矩阵：

```bash
OMNIFLOW_ANDROIDWORLD_SUPPLEMENTAL_METHOD=autodroid \
bash scripts/exp/run_androidworld.sh --all-tasks \
  --tasks TASK1,TASK2 \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113
```

省略 `--tasks` 会按 `data/current.json` 的 116 个 task 全量运行。完整
supplemental campaign 强制执行 AndroidWorld setup 和每 task snapshot restore；
`OMNIFLOW_ANDROIDWORLD_PERFORM_EMULATOR_SETUP=0` 只用于单 task 开发复跑，不能
用于完整 campaign。每个 task
单独初始化、单独封存 validator/replay evidence；AutoDroid 结果仍只写入
`androidworld/<task>/autodroid/<device_model_seed>/`。

固定实验值：source seed `111`、evaluation seed `113`、formal chat/vision model
`GLM-4.6V`；初始 VLM、VLM fallback 和 AppAgent 都使用同一原生图片输入链路。
`--control-backend oob` 用于 OOB observe/act transport。

## 环境变量

```bash
export OMNIFLOW_EXP_ASSET_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_EXP_RESULTS_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_EXP_MEMORY_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_ENV_FILE=/absolute/model.env
export OMNITRANSFER_ROOT="$HOME/Projects/Omni/OmniTransfer"
```

`model.env` 至少提供 `LLMTHU_API_KEY`。外部 AndroidWorld、Android SDK、
MobileGPT 和 AppAgent 路径由 launcher 的环境变量或机器默认值提供。
刷新索引时，launcher 默认同时扫描 `data/androidworld` 和其唯一的
`.archive/result_registry`；因此 scheduler 已注册的正式结果不会因后续
`--refresh-memory` 被遗漏。需要额外结果根时才设置
`OMNIFLOW_MEMORY_RESULT_ROOTS` 覆盖默认值。

## SSH 迁移到 4090

使用仓库内的构建器可以把代码、权威 `data/`、注册资产和模型环境迁移到
4090，并在服务器上构建运行环境：

```bash
bash tools/build_4090_resources.sh \
  --ssh user@4090 \
  --model-env /absolute/model.env \
  --run-smoke
```

默认 `--mode latest` 会更新 AndroidWorld、B-MoCA、AppAgent、MobileGPT、
OmniTransfer 及 B-MoCA 外部依赖到默认分支最新提交；实际 commit 会写入
`/data/omniflow-4090/deployment_manifest.json`，并通过部署环境变量让
AndroidWorld/AppAgent/B-MoCA revision 检查匹配这次部署。Python 依赖默认仍由
`uv.lock` 控制；需要主动升级时增加 `--upgrade-python-deps`。

脚本还会安装/检查 CUDA 无关的系统运行依赖、Android SDK/API 33/34、
Appium、AVD，并执行 `--check-only` 验证。首次迁移前请确保 SSH 已配置为
非交互登录、远端有 sudo 权限、4090 主机磁盘至少预留约 30 GB（不含模型和
完整数据归档）。`--skip-device-setup` 可用于先只构建代码环境；B-MoCA
的 env100/语料资产仍需按其官方资产合同另行准备，脚本不会伪造这些资产。

## 其他命令

```bash
bash scripts/exp/run_androidworld.sh --tasks TASK
bash scripts/exp/run_androidworld.sh --check-only --all-tasks
bash scripts/exp/run_androidworld.sh --refresh-memory
./.venv/bin/pytest -q
```

结果和中间证据写入 `data/`；Function 只通过 `save_function` 写入 Store，
运行时只读取 `data/current.json`。内部实现路径不是额外入口。

失败 source RunLog 的本地逐条重采集、系统提示词、截图和推理证据要求见
[`docs/MANUAL_RUNLOG_RECOLLECTION_WORKFLOW.md`](../../docs/MANUAL_RUNLOG_RECOLLECTION_WORKFLOW.md)。
