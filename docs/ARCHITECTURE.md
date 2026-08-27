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
Memory 互不混用，但都只从同一份成功 source RunLog 派生一次。新 manifest 的所有
资产引用相对于 Memory 根目录；旧绝对引用只作为兼容证据读取。

## 两个公开协议

实验入口只保留两个互相独立的协议。第一个协议产生地址，第二个协议消费地址；
执行协议不负责转换、复制、扫描或挑选历史 Memory。

### SaveMemory

请求：

```text
convert-memory
  task             = 一个 AndroidWorld task
  method           = omniflow | mobilegpt | appagent | all
  source_run_log   = 唯一成功 source RunLog 的显式路径
  memory           = 新 Memory 输出目录
```

单方法请求返回一个 `memory` 地址。`method=all` 返回一个 `memories` 映射：

```text
omniflow  -> <memory>/omniflow/store.json
mobilegpt -> <memory>/mobilegpt/memory/
appagent  -> <memory>/appagent/
```

`fixed_replay` 直接把 source RunLog 作为 Memory 地址，`t3a_hint` 使用同一份
source 证据；二者不生成第二份 source 副本。只有 SaveMemory 成功返回的地址才是
可用 Memory；失败调用留下的部分目录不得交给 DirectRun。

### DirectRun

请求：

```text
run
  task             = 同一 AndroidWorld task
  method           = 一个正式方法，或 all
  device           = canonical device，或 all
  source_run_log   = 同一 source 证据路径
  memory           = 单方法 Memory 地址，或 SaveMemory 返回的 Memory root
```

单方法执行把 `memory` 原样传给对应 method。`method=all` 只按固定目录约定将
Memory root 路由给各方法，然后并行使用配置中的设备；它不会重新转换 Memory，
也不会从历史目录推断地址。task parameters 从显式 source 证据保持一致，target
evaluation seed 仍由统一协议固定为 113。

两个协议的职责边界是：

```text
one source RunLog
        │
        ▼
SaveMemory  ──> stable Memory addresses
                         │
                         ▼
                 DirectRun(task, method, device, memory)
                         │
                         ▼
              AndroidWorld result + official validator
```

历史结果扫描、自动 attempt 选择、scheduler manifest、重复 registry/ledger、启动日志和
转换缓存都不参与运行。Memory 转换中间文件使用系统临时目录，任务结束自动删除。
显式传入的已有 Memory 会先进行完整校验（source SHA-256、内部文件哈希、task、模型、
协议字段）；校验通过即复用，校验失败不会触发隐式重转换。

B-MoCA 是独立 benchmark，不进入此 AndroidWorld 入口。
