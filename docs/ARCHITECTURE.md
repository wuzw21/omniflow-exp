# Architecture

AndroidWorld 只有一条运行链：

```text
scripts/exp/run_androidworld.sh
  -> src/experiment/run_tasks.py
  -> src/experiment/run_task.py
  -> src/integrations/android_world/run_episode.py
```

- shell 只转发参数。
- `run_tasks.py` 选择 task、五种 method 和 Standard/Fold/Tablet，并在系统临时目录调用对应 Memory 转换器。
- `run_task.py` 启动一个 method + device 的原子任务。
- `run_episode.py` 使用 AndroidWorld setup、OmniFlow OOB observe/act 和 AndroidWorld 官方 validator。

正式方法只有 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent` 和
`t3a_hint`。它们的差异只在 Memory 准备和 Planner/Executor adapter；设备 lifecycle
与官方结果路径共用。

运行时仅需要：

- `data/current.json`：task 到成功 source RunLog 的索引。
- source RunLog：为各方法生成 Memory。
- 各方法 Memory：实际执行输入。
- AndroidWorld 官方结果与 RunLog：论文实验输出。

scheduler manifest、完成结果扫描、seed/path preflight、重复 registry/ledger、启动日志和
转换缓存都不参与运行。Memory 转换中间文件使用系统临时目录，任务结束自动删除。

B-MoCA 是独立 benchmark，不进入 AndroidWorld 的五种 method。
