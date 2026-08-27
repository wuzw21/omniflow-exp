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

实验参数不通过命令行动态覆盖，固定值来自 `config/paper_androidworld.json`。
正式运行和 Memory 转换固定使用 `Qwen3.6-Plus`；传入其他模型会被入口拒绝。
设备只能使用配置中的 canonical label。

`--device all` 选择论文的 Pixel 6 Pro、Fold 和 Tablet 三台 target。入口只启动尚未
在线的 AVD，已在线设备直接复用；不同设备并发，同一设备上的多个 method 顺序执行。
每个 OmniFlow target 结束并封存官方 RunLog 后，单任务 runner 会立即打印一块可直接
复制的结果，统一包含 validator、物理动作、Function 复用、fallback、模型调用、token、
execution/lifecycle 时间、Store、RunLog 和 SHA-256；三端并行时按实际完成顺序输出。

`--method source --device source5560` 使用无 Memory 的 OmniFlow Planner、统一
OOB 物理层和官方 validator，采集 seed-111 Source RunLog。

Memory 可省略；入口不会查 index 或自动寻找历史 Memory。转换 Memory：

```bash
bash scripts/exp/run_androidworld.sh convert-memory \
  --task CameraTakePhoto --method omniflow \
  --source-run-log /path/to/run_log.json \
  --memory /path/to/output-memory
```

五个正式 AndroidWorld 方法只在各自定义的执行输入上不同。Memory
就绪后都进入同一条
`run_task.py -> run_episode.py` task/validator 路径。Memory 与 AndroidWorld 官方
结果是需要保留的实验产物；同一 setting 固定使用 `attempt_001`，不会自动创建新的
attempt 或自动选择“最好”的 attempt；转换临时目录在任务结束后自动删除。

AppAgent 通过同一入口执行，但其官方 executor 位于 disposable workspace；OOB
仍是唯一 observe/act 物理层，AndroidWorld 官方 validator 仍是唯一成功判据。

结果中的 `duration_ms` 是包含 task lifecycle、setup 和官方 validator 的完整 wall
time；论文中的方法执行时间使用 `execution_duration_ms`，它只累计 `agent.step`，明确
排除 setup 和官方 validator。`non_execution_duration_ms` 保留二者差值，便于审计，
不能当作方法推理时间。RunLog `diagnostics` 同时保存 Function recall、Planner rejection
和 LLM usage 汇总，模型调用与 token 统计不依赖终端输出。

9207 部署不再逐项复制代码、APK、权重和配置。使用
`scripts/package_9207_runtime.sh build` 生成单个 release archive；包内固定
OmniFlow、OmniTransfer V10、OOB APK、V10 checkpoint、无密钥运行配置、manifest
和 SHA256。`install` 更新 canonical checkout 与明确的 runtime 资产。API key 只保存
在被 Git 忽略的 `config/runtime.secrets.env`，不进入 release archive。
