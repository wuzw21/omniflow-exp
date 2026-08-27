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
  --memory /path/to/store.json
```

论文 target 为 Pixel 6 Pro、7.6 英寸 Fold 和 10.1 英寸 WXGA Tablet。一次选择
全部三台设备时，入口会复用已在线的 AVD、启动缺失的 AVD，并保持每台设备一个并发
worker：

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device all \
  --memory /path/to/store.json
```

正式方法固定为 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。
其中 AppAgent 使用同一份 source RunLog 生成一次官方 demo Memory，再进入统一的
AndroidWorld/OOB/validator 流程；`script_replay` 不属于 AndroidWorld 方法。
设备和默认值来自 `config/paper_androidworld.json`。正式运行和 Memory 转换固定使用
`Qwen3.6-Plus`；显式传入其他模型会在入口处拒绝。

```bash
bash scripts/exp/run_androidworld.sh --help
```

Memory 转换也使用同一个入口，不读取 index：

```bash
bash scripts/exp/run_androidworld.sh convert-memory \
  --task CameraTakePhoto \
  --method omniflow \
  --source-run-log /path/to/run_log.json \
  --memory /path/to/output-memory
```

OmniTransfer 使用 canonical checkout `~/Projects/Omni/OmniTransfer`，页面检索统一调用
V10 `omnitransfer_point_conditioned_sparse_graph_v10` 模型的归一化 1024D
page-attention readout；不维护第二套页面编码器或旧 64D/512D 表示。

架构和文件 owner 见 `docs/ARCHITECTURE.md` 与 `docs/FILE_EDIT_GUIDE.md`。
