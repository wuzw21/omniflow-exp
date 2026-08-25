# AndroidWorld launcher

`run_androidworld.sh` 是唯一公开入口。它只选择 Python、加载可选 config，然后把
参数原样交给 `src.experiment.run_tasks`；不检查 seed、模型、endpoint、路径、磁盘、
依赖、AVD 或已完成结果。

```bash
bash scripts/exp/run_androidworld.sh \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562 \
  --source-seed 111 \
  --evaluation-seed 113
```

全部参数都可省略，默认值来自 `config/paper_androidworld.json`。也可换一份配置：

```bash
bash scripts/exp/run_androidworld.sh --config /path/to/experiment.json
```

五个 AndroidWorld 方法只在 Memory 准备方式上不同。Memory 就绪后都进入同一条
`run_task.py -> run_episode.py` task/validator 路径。
