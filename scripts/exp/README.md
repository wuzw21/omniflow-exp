# AndroidWorld launcher

`run_androidworld.sh` 是唯一公开入口。它只把参数原样交给
`src.experiment.run_tasks`；不保存调度文件，也不检查 seed、模型、endpoint、路径、
磁盘、依赖、AVD 或已完成结果。

```bash
bash scripts/exp/run_androidworld.sh \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562 \
  --source-seed 111 \
  --evaluation-seed 113
```

全部参数都可省略，默认值来自 `config/paper_androidworld.json`。

五个 AndroidWorld 方法只在 Memory 准备方式上不同。Memory 就绪后都进入同一条
`run_task.py -> run_episode.py` task/validator 路径。Memory 与 AndroidWorld 官方
结果是需要保留的实验产物；转换临时目录在任务结束后自动删除。
