# Architecture

AndroidWorld 只有一条运行链：

```text
scripts/exp/run_androidworld.sh
  -> src/experiment/run_tasks.py
  -> src/experiment/run_task.py
  -> src/integrations/android_world/run_episode.py
```

- shell 只转发参数。
- `run_tasks.py` 只实现 `convert-memory` 和 `run`，Memory 路径由调用者直接传入；
  `run` 复用在线 AVD、启动缺失的所选 AVD，并按设备并发执行。
- `run_task.py` 启动一个 method + device 的原子任务。
- `run_episode.py` 使用 AndroidWorld setup、OmniFlow OOB observe/act 和 AndroidWorld 官方 validator。

正式方法固定为 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。
AppAgent 通过 official forwarder 接入，但 observe/act 仍由同一个 OmniFlow OOB
物理层提供；`script_replay` 和旧设备别名不进入新的 AndroidWorld 运行路径。

运行时仅需要：

- source RunLog：为各方法生成 Memory。
- 各方法 Memory：实际执行输入。
- AndroidWorld 官方结果与 RunLog：论文实验输出。

AppAgent 的 Memory 是其官方 demo 文档格式；它与 OmniFlow Store、MobileGPT
Memory 互不混用，但都只从同一份成功 source RunLog 派生一次。

历史结果扫描、自动 attempt 选择、scheduler manifest、重复 registry/ledger、启动日志和
转换缓存都不参与运行。Memory 转换中间文件使用系统临时目录，任务结束自动删除。

B-MoCA 是独立 benchmark，不进入此 AndroidWorld 入口。
