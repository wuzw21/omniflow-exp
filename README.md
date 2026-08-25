# OmniFlow-exp

AndroidWorld 和 B-MoCA 实验仓库。AndroidWorld 只有一个公开入口：

```bash
bash scripts/exp/run_androidworld.sh
```

入口直接调用 task scheduler；scheduler 为所选方法准备 Memory，然后启动一次
AndroidWorld task。设备 lifecycle、task setup 和最终 validator 由 AndroidWorld
episode 负责，入口不做 preflight。

常用参数都是可选的：

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562 \
  --memory /path/to/store.json
```

可选方法为 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`；
设备和默认值来自 `config/paper_androidworld.json`。传入的 seed、步数、fallback、
deadline 和 model 会原样进入本次运行，不要求等于配置中的默认值。

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

OmniTransfer 使用 canonical checkout `~/Projects/Omni/OmniTransfer`。

架构和文件 owner 见 `docs/ARCHITECTURE.md` 与 `docs/FILE_EDIT_GUIDE.md`。
