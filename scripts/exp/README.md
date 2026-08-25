# AndroidWorld launcher

`run_androidworld.sh` 是唯一公开入口。它只把参数原样交给
`src.experiment.run_tasks`；不保存调度文件，也不检查 seed、模型、endpoint、路径、
磁盘、依赖、AVD 或已完成结果。

```bash
bash scripts/exp/run_androidworld.sh run \
  --task CameraTakePhoto \
  --method omniflow \
  --device standard45562 \
  --memory /path/to/store.json
```

全部参数都可省略，默认值来自 `config/paper_androidworld.json`。
除配置中的设备 label 外，临时设备可显式写成 `LABEL:SERIAL:PORT`。

`--device all` 选择论文的 Pixel 6 Pro、Fold 和 Tablet 三台 target。入口只启动尚未
在线的 AVD，已在线设备直接复用；不同设备并发，同一设备上的多个 method 顺序执行。

Memory 可省略；入口不会查 index 或自动寻找历史 Memory。转换 Memory：

```bash
bash scripts/exp/run_androidworld.sh convert-memory \
  --task CameraTakePhoto --method omniflow \
  --source-run-log /path/to/run_log.json \
  --memory /path/to/output-memory
```

五个 AndroidWorld 方法只在 Memory 准备方式上不同。Memory 就绪后都进入同一条
`run_task.py -> run_episode.py` task/validator 路径。Memory 与 AndroidWorld 官方
结果是需要保留的实验产物；转换临时目录在任务结束后自动删除。
