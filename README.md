# OmniFlow-exp

论文 AndroidWorld / B-MoCA 实验代码。正式实验只有一个入口：
`scripts/exp/run_androidworld.sh`。

## 最常用：一个 task，测试我们的 Function 跨设备复用

先准备一次环境变量：

```bash
export OMNIFLOW_EXP_ASSET_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_EXP_RESULTS_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_EXP_MEMORY_ROOT=/absolute/OmniFlow-exp/data
export OMNIFLOW_ENV_FILE=/absolute/model.env
export OMNITRANSFER_ROOT="$HOME/Projects/Omni/OmniTransfer"
```

模型和运行凭据只从显式的绝对路径 `OMNIFLOW_ENV_FILE` 加载；入口不会再读取
仓库 `.env`、第二个 runtime env 或旧的 `LLMTHU_KEY` alias。正式 llmthu 配置只需
提供 `LLMTHU_API_KEY`，endpoint/model 由 protocol 统一选择。

执行一个 task：

```bash
bash scripts/exp/run_androidworld.sh \
  --e2e-task TASK \
  --e2e-method omniflow \
  --e2e-device standard45562:emulator-45562:45562,fold45564:emulator-45564:45564,tablet45554:emulator-45554:45554 \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113 \
  --control-backend oob
```

流程只有四步：

1. 检查 `data/current.json` 中是否已有该 task 的 Function；
2. 没有时，可用 source RunLog 作为 observation authoring 输入；
3. `compile_runlog_to_store` 写入 v2 `store.json` 和同目录
   `transfer_states.json`；
4. 校验通过后，在指定设备上执行 E2E。

Function 步骤按成功 RunLog 动作顺序保存，并通过 `source_state_id` 关联 source
observation。source seed 固定为 `111`，
evaluation seed 固定为 `113`，正式 chat 和视觉模型统一为 `GLM-4.6V`；初始
VLM 与 fallback 都直接传当前 screenshot。

## 数据目录

AndroidWorld 可见实验证据统一写入：

```text
data/androidworld/<task>/<method>/<device_model>_seed.../
```

每个 setting 下按 `runlog/attempt_NNN/`、`memory/attempt_NNN/` 和 `result/`
保存执行；attempt 只用递增编号，不使用时间戳。每次 RunLog 执行直接保存为
该 attempt 下的一份 `run_log.json` 和 `screenshots/screenshot_NNNNNN.png`，
不创建 SHA 对象仓库。设备 CLI alias 只写 provenance，不进入目录名。B-MoCA
始终独立位于 `data/bmoca/`。完成情况见
[`data/androidworld/COMPLETION_STATUS.md`](data/androidworld/COMPLETION_STATUS.md)，
可直接转换 memory 的 source 候选见
[`data/androidworld/MEMORY_READY_SOURCES.md`](data/androidworld/MEMORY_READY_SOURCES.md)。

## 本地数据与测试资料索引

本项目和 OmniTransfer 共同组成跨设备学习器实验。两者的资料不要混用：

| 用途 | 位置 | 说明 |
| --- | --- | --- |
| AndroidWorld 正式证据 | `data/androidworld/` | 当前运行时使用的 task、RunLog、Function memory 和结果 |
| B-MoCA 正式证据 | `data/bmoca/` | B-MoCA 的环境、任务和评测结果 |
| 运行时索引 | `data/current.json` | 唯一的本地 Function/RunLog 运行时索引 |
| AndroidWorld 历史备份 | `data/.androidworld_legacy_backup/` | 只读历史资料，不参与当前运行时选择 |
| OmniTransfer 正式数据集 | `../OmniTransfer/runtime/datasets/` | canonical dataset；训练、清洗和复现实验的正式数据源 |
| OmniTransfer 评测资料 | `../OmniTransfer/runtime/evals/` | 映射测试集、评测输入、错误分析和评测证据 |
| OmniTransfer release | `../OmniTransfer/runtime/releases/` | 可运行的 release 包；release 内部数据必须保持自包含 |
| OmniTransfer 实验输出 | `../OmniTransfer/output/` | checkpoint、预测、review 和各次实验输出，不是 canonical dataset |
| OmniTransfer 临时资料 | `../OmniTransfer/tmp/` | 本轮清洗和统一评测的工作文件，完成同步后可清理 |

当前跨设备学习器清洗和统一测试资料位于：

```text
../OmniTransfer/tmp/unified_clean_audit_20260823/
```

其中：

```text
train/cleaned.jsonl          清洗后的训练数据
dev/cleaned.jsonl            清洗后的开发数据
test/cleaned.jsonl           清洗后的测试数据
test_new_method.json         新方法统一评测报告
test_control_no_local.json   对照方法评测报告
*.predictions.jsonl          对应的逐样本预测
```

本地只允许进行数据清洗、质量检测、代码测试和评测入口验证。任何模型训练、
微调或训练 smoke 都必须在远程 `9207` 环境执行；不要在本地启动训练。

测试代码位置：

```text
OmniFlow-exp/tests/
../OmniTransfer/tests/
```

离线测试入口：

```bash
./.venv/bin/pytest -q
```

重复文件清理记录（2026-08-23）：完全相同的 13 个评测压缩文件、一个重复查询
文件和一个重复 quarantine 文件已移到系统废纸篓，保留副本分别位于
`../OmniTransfer/runtime/evals/vision_widget_mapping/_repo/testset/`、
`../OmniTransfer/runtime/evals/vision_widget_mapping/splits/` 和
`../OmniTransfer/output/cleaned_ase_train_v2/`。release 内部的 hard link、
历史归档和用户代码没有清理。

## 选择方案或全矩阵

只测我们的方案：

```bash
--e2e-method omniflow
```

跑完整 method/device 矩阵：

```bash
--e2e-method all --e2e-device all
```

也可以使用逗号列表选择部分 method 或设备。method 和设备必须来自当前
protocol 配置；正常使用不需要手动修改配置文件。

正式 AndroidWorld 矩阵固定包含 `fixed_replay`、`omniflow`、`mobilegpt`、
`appagent` 和 `t3a_hint` 五个方法。历史 AutoDroid/DroidBot replay 仍保留在
代码中作为只读兼容边界，不进入正式方法矩阵。

设备第一次使用时先执行 setup：

```bash
PYTHON_BIN=/absolute/.venv/bin/python \
OMNIFLOW_ANDROID_SDK_ROOT=/absolute/Android/Sdk \
OMNIFLOW_ANDROIDWORLD_A11Y_APK=/absolute/accessibility_forwarder.apk \
bash scripts/exp/run_androidworld.sh --setup-device all
```

setup 会安装并启动 OOB、MobileGPT、AndroidWorld accessibility forwarder，
检查 AndroidWorld、AppAgent、MobileGPT、OmniTransfer 和模型环境，并通过
OOB observe bridge 验收设备；报告位于
`data/androidworld/.archive/setup/<UTC>/setup_report.json`。

## 其他入口

```bash
# 任务矩阵
bash scripts/exp/run_androidworld.sh --tasks TASK

# 静态检查，不启动 emulator
bash scripts/exp/run_androidworld.sh --check-only --all-tasks

# 离线测试
./.venv/bin/pytest -q
```

更详细的入口参数见 [`scripts/exp/README.md`](scripts/exp/README.md)。
架构和文件 owner 说明分别见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
和 [`docs/FILE_EDIT_GUIDE.md`](docs/FILE_EDIT_GUIDE.md)。
