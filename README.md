# OmniFlow-exp

AndroidWorld 和 B-MoCA 实验仓库。AndroidWorld 只有一个公开入口：

```bash
bash scripts/exp/run_androidworld.sh
```

入口直接调用统一 runner；runner 只使用调用者明确传入的 task、method、device 和
Memory，然后启动一次 AndroidWorld task。设备 lifecycle、task setup 和最终 validator
由 AndroidWorld episode 负责。

常用参数都是可选的：

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562 \
  --memory data/androidworld/_memories/omniflow/store.json
```

论文 target 为 Pixel 6 Pro、7.6 英寸 Fold 和 10.1 英寸 WXGA Tablet。一次选择
全部三台设备时，入口会复用已在线的 AVD、启动缺失的 AVD，并保持每台设备一个并发
worker：

实验平台统一为：Source 是 Android 13、`720x1280` 的 small-phone emulator；
Standard 是 Android 13、`1440x3120` 的 Pixel 6 Pro；Fold 是 Android 14、
`1768x2208` 的 7.6-inch foldable；Tablet 是 Android 13、`1280x800` 的
10.1-inch WXGA tablet。

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device all \
  --memory data/androidworld/_memories/omniflow/store.json
```

正式方法固定为 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。
其中 AppAgent 使用同一份 source RunLog 生成一次官方 demo Memory，再进入统一的
AndroidWorld/OOB/validator 流程；`script_replay` 不属于 AndroidWorld 方法。
设备和默认值来自 `config/paper_androidworld.json`。正式运行和 Memory 转换固定使用
`Qwen3.6-Plus`；显式传入其他模型会在入口处拒绝。

```bash
bash scripts/exp/run_androidworld.sh --help
```

Memory 保存和实验执行是两个独立协议，但都使用同一个入口，不读取 index。
仓库内配置和 Memory manifest 使用相对路径；外部依赖只在进程启动边界解析为本机路径。
`convert-memory` 只产生稳定的 Memory 地址；如果这个明确地址已经存在且校验通过，
入口直接复用，不再次调用模型；如果地址不完整或 source/model 不匹配，则报错并停止，
不会自动重跑或选择历史结果。

保存 Memory：

```bash
bash scripts/exp/run_androidworld.sh convert-memory \
  --task CameraTakePhoto \
  --method omniflow \
  --source-run-log data/androidworld/CameraTakePhoto/source/runlog/attempt_001/run_log.json \
  --memory data/androidworld/_memories
```

直接执行：

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562 \
  --source-run-log data/androidworld/CameraTakePhoto/source/runlog/attempt_001/run_log.json \
  --memory data/androidworld/_memories/omniflow/store.json
```

如果需要从一份 source RunLog 一次性生成三个需要 Memory 的方法，可使用固定的
目录布局；`fixed_replay` 和 `t3a_hint` 直接使用同一份 source RunLog：

```bash
bash scripts/exp/run_androidworld.sh convert-memory \
  --task CameraTakePhoto --method all \
  --source-run-log data/androidworld/CameraTakePhoto/source/runlog/attempt_001/run_log.json \
  --memory data/androidworld/_memories

bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto --method all --device all \
  --source-run-log data/androidworld/CameraTakePhoto/source/runlog/attempt_001/run_log.json \
  --memory data/androidworld/_memories
```

该布局只包含 `omniflow/store.json`、`mobilegpt/memory/` 和 `appagent/` 三份派生
Memory，不复制 source RunLog，也不扫描历史结果。manifest 内的 source、demo、日志和
校验文件均相对于 Memory 根目录记录，因而可以随仓库一起搬迁。

OmniTransfer 使用 canonical checkout `~/Projects/Omni/OmniTransfer`，页面检索统一调用
V10 `omnitransfer_point_conditioned_sparse_graph_v10` 模型的归一化 1024D
page-attention readout；不维护第二套页面编码器或旧 64D/512D 表示。

架构和文件 owner 见 `docs/ARCHITECTURE.md` 与 `docs/FILE_EDIT_GUIDE.md`。
