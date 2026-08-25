# File edit guide

真实运行 owner：

| 需求 | 唯一 owner |
|---|---|
| 公开启动参数 | `scripts/exp/run_androidworld.sh` |
| task/method/device 调度和 Memory 转换调用 | `src/experiment/run_tasks.py` |
| 单个 AndroidWorld 任务启动 | `src/experiment/run_task.py` |
| AndroidWorld lifecycle、OOB I/O、官方 validator | `src/integrations/android_world/run_episode.py` |
| OmniFlow Function 编译 | `src/experiment/function_v2.py` |
| MobileGPT Memory 转换 | `src/integrations/mobilegpt.py` |
| AppAgent Memory 转换 | `src/experiment/appagent_source.py` |
| 运行时协议默认值 | `config/paper_androidworld.json`、`src/experiment/protocol.py` |
| OOB 设备就绪（episode 内） | `src/experiment/checks.py` |
| AndroidWorld 证据写入 | `src/experiment/observation_evidence.py` |

判断一个文件是否保留，只问：正式实验入口能否到达它，以及删除后真实 task、Memory、
OOB 控制或官方 validator 是否会失效。仅用于旧 preflight、测试、兼容参数、结果扫描、
scheduler summary、离线回归或重复索引的文件不保留。

不要新增第二 launcher、第二 scheduler、第二 Function writer、第二结果注册器或路径
校验层。task、method、device、seed、步数、fallback、deadline、model 都通过入口的
可选参数传入；入口不为特定 seed 或机器路径设置硬门槛。
