# File edit guide

宿主重构 owner：`omniflow/runtime/engine.py` 管任务循环与单次调用；
`control.py` 管共享预算和取消；`protocol.py` 生成调用反馈；
`src/integrations/gui_agent_tools.py` 管外部会话和重试去重。
`gui_agent_mcp.py` 只提供 MCP 传输；`gui_agent_oob_host.py` 复用 OOB。
不得在宿主适配器内新增 Planner 或 action mapper。协议见 `docs/HARNESS_PROTOCOL.md`。

真实运行 owner：

| 需求 | 唯一 owner |
|---|---|
| 公开启动参数 | `scripts/exp/run_androidworld.sh` |
| task/method/device 调度和 Memory 转换调用 | `src/experiment/run_tasks.py` |
| 单个 AndroidWorld 任务启动 | `src/experiment/run_task.py` |
| AndroidWorld lifecycle、OOB I/O、官方 validator | `src/integrations/android_world/run_episode.py` |
| OmniFlow Function 编译 | `src/experiment/function_v2.py` |
| MobileGPT Memory 转换 | `src/integrations/mobilegpt.py` |
| AppAgent Memory 转换 | `src/experiment/appagent_source.py`、`src/integrations/appagent.py` |
| 运行时协议默认值 | `config/paper_androidworld.json`、`src/experiment/protocol.py` |
| AndroidWorld 证据写入 | `src/experiment/observation_evidence.py` |

判断一个文件是否保留，只问：正式实验入口能否到达它，以及删除后真实 task、Memory、
OOB 控制或官方 validator 是否会失效。仅用于旧 preflight、测试、兼容参数、结果扫描、
scheduler summary、离线回归或重复索引的文件不保留。

不要新增第二 launcher、第二 scheduler、第二 Function writer、第二结果注册器或路径
校验层。task、method、device 和显式 Memory 是唯一运行输入；seed、步数、fallback、
deadline、模型和 retry policy 固定在协议 owner 中，正式模型固定为 `Qwen3.6-Plus`。

## Harness 接入 owner

- `src/integrations/gui_agent_harness.py`：统一 `TaskHarness.arun(HarnessContext)`，内置
  Planner 与外部 CLI 的决策宿主适配；外部工厂实现此接口即可接入同一 AndroidWorld 入口。
- `src/integrations/gui_agent_mcp.py`：两个工具的唯一 MCP 分发；`--connect` 只转发字节到
  episode 已持有的运行时，不创建 Host、Store 或新的执行循环。
- `src/integrations/android_world/agent.py`：在既有 step 中调用选定 Harness，仍由
  `run_episode.py` 管理环境、记录、预算和官方判定。
