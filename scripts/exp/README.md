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
  --e2e-device small5554:emulator-5554:5554,small5562:emulator-5562:5562,fold5564:emulator-5564:5564 \
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
和 source 设备。报告写到 `data/setup/<UTC>/setup_report.json`。缺少 OOB
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

当前 AndroidWorld target 设备：

| label | serial | profile |
| --- | --- | --- |
| `small5554` | `emulator-5554` | `small_phone` |
| `small5562` | `emulator-5562` | `small_phone` |
| `fold5564` | `emulator-5564` | `pixel_fold` |

## AutoDroid 补充基线（9207）

AutoDroid 不属于正式五方法，也不进入 `--e2e-method all` 或 116 × 10 主表。
它只能显式使用 9207 上的独立设备标签和独立结果命名空间：

```bash
OMNIFLOW_AUTODROID_ROOT=/absolute/OmniFlow-exp/data/runtime/external/autodroid \
OMNIFLOW_AUTODROID_MEMORY_ROOT=/absolute/OmniFlow-exp/data/runtime/autodroid/androidworld_apps \
bash scripts/exp/run_androidworld.sh \
  --e2e-task CameraTakePhoto \
  --e2e-method autodroid \
  --e2e-device autodroid9207:emulator-5590:5590 \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113
```

结果写入 `data/androidworld_validator/supplemental/autodroid_9207/`，使用
原生 DroidBot/UTG replay 和 AndroidWorld 官方 validator，不转换 Function、
不使用 OmniTransfer，且 `model_calls=0`、`fallback_steps=0`。完整公平比较
合同见 [`docs/AUTODROID_9207_COMPARISON_PLAN.md`](../../docs/AUTODROID_9207_COMPARISON_PLAN.md)。

运行完整的 116-task supplemental campaign 时，使用批量入口并显式选择
supplemental method；这不会改变正式 `all` 矩阵：

```bash
OMNIFLOW_ANDROIDWORLD_SUPPLEMENTAL_METHOD=autodroid \
bash scripts/exp/run_androidworld.sh --all-tasks \
  --tasks TASK1,TASK2 \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113
```

省略 `--tasks` 会按 `data/current.json` 的 116 个 task 全量运行。每个 task
单独初始化、单独封存 validator/replay evidence；AutoDroid 结果仍只写入
`androidworld_validator/supplemental/autodroid_9207/`。

固定实验值：source seed `111`、evaluation seed `113`、formal model
`GLM-5.1`。`--control-backend oob` 用于 OOB observe/act transport。

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
