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

执行一个 task：

```bash
bash scripts/exp/run_androidworld.sh \
  --e2e-task TASK \
  --e2e-method omniflow \
  --e2e-device small5554:emulator-5554:5554,small5562:emulator-5562:5562,fold5564:emulator-5564:5564 \
  --e2e-source-seed 111 \
  --e2e-evaluation-seed 113 \
  --control-backend oob
```

流程只有四步：

1. 检查 `data/current.json` 中是否已有该 task 的 Function；
2. 没有时，用成功的 source RunLog 调用 `save_function(enhance=True)`；
3. 校验 Function Store 和 OmniTransfer evidence；
4. 校验通过后，在指定设备上执行 E2E。

Function 检查失败时不会启动 target episode。source seed 固定为 `111`，
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
