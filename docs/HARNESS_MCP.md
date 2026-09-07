# 外部宿主 MCP 接入

协议见 [HARNESS_PROTOCOL.md](HARNESS_PROTOCOL.md)。实现是 `src/integrations/gui_agent_mcp.py`，
仅调用已有 `GuiAgentToolRuntime`，不创建 Planner。当前支持 MCP stdio，依赖 MCP Python
SDK 2.x；HTTP 服务、远程身份认证和跨进程恢复不在当前实现内。

在 canonical checkout 中安装可选依赖并查看启动参数：

```bash
.venv/bin/python -m pip install -e '.[mcp]'
.venv/bin/python -m src.integrations.gui_agent_mcp --help
```

服务需要已初始化且运行 OOB 的 Android 设备；不会自行构建或安装 APK。`--device`
是明确的 adb serial。`--store` 是明确的 v2 Store；省略表示 Memory OFF，只暴露原始
动作，不编译或扫描任何 Memory。canonical OmniTransfer 使用既有配置和模型安装。

Codex 配置示例（将 serial 和路径替换为实际值；不需要在此配置模型 endpoint）：

```toml
[mcp_servers.omniflow]
command = "/absolute/OmniFlow-exp/.venv/bin/python"
args = ["-m", "src.integrations.gui_agent_mcp", "--device", "DEVICE_SERIAL", "--store", "/absolute/memory/store.json", "--evidence-root", "/absolute/OmniFlow-exp/data/runtime/gui_agent/codex"]
env = { PYTHONPATH = "/absolute/OmniFlow-exp" }
tool_timeout_sec = 600
```

宿主超时要覆盖一个 Function 的调用；内核仍共享最多 600 秒的整任务预算，模型请求
最多 30 秒。默认 Python 环境不会自动携带远端设备工具；需要时显式设 `--adb-path`。
服务进程持有本机同 serial 的 MCP 排他锁；此锁不控制未采用本协议的其他工具或机器。
不要让原生 computer use 与此服务同时操作同一 Android 会话。

把仓库中的 `skills/omniflow-gui/` 目录复制到宿主的技能目录即可分发。Skill 只描述
发现、执行、反馈与停止，不携带设备驱动、Memory、模型权重或硬编码任务。服务端随项目
安装，Skill 本身不能替代 MCP 服务。

## 工具协议

1. `omniflow_tools` 返回实时可用的 canonical action / Function input schemas。
   `omniflow_status` 返回当前 `session_id`、剩余时间、执行计数、in-flight 状态。
2. `omniflow_observe` 返回一份当前 UI 信息和一张 MCP image；调用文本去掉 base64，
   原图仍保留在证据目录。坐标沿用 canonical action schema 的语义和原始 display。
3. `omniflow_execute` 输入 `session_id`、唯一 `request_id`、`tool_name`、`arguments`。
   外层与工具参数均按 schema 校验；它调用现有原始动作或 Function，返回 invocation v1。
4. 传输超时后先查询 status；重取已完成结果时只能重用同一 request id 和参数。若状态
   为 unknown/in-flight，不生成新 id 重试同一副作用。Function 内部失败不是 transport
   重试；宿主根据当前状态恢复，不从头盲目重发。
5. `omniflow_finish(session_id, content)` 调用配置的 verifier。独立 Android 服务默认
   没有官方 task verifier，返回 `task_status=unknown` 并关闭会话；宿主应报告其观察
   证据，不把它称为 official success。嵌入 AndroidWorld 时仍由官方 lifecycle 验证。
6. `omniflow_cancel(session_id)` 阻止后续动作；已发出的动作收尾前 `in_flight=true`。
   旧任务关闭并且没有在途操作后，才可用 `omniflow_start(session_id)` 明确开始新任务。

MCP 为 `execute` 返回 `isError=true` 时仍可能带有成功执行的前缀与最新观察；宿主必须
读取反馈。服务重启会产生新 session id，旧请求拒绝续接。没有持久 resume token。

默认从 canonical checkout 启动，证据目录为当前工作区的 `data/runtime/gui_agent/<id>/`。
从其他目录或安装包启动时显式传 `--evidence-root`，避免向安装目录写入数据。
`events.ndjson` 是已有 step fact 的流式调试记录，不是官方 RunLog，不能直接冒充成功
source 输入。原始 PNG 按内容去重。数据不随 Git 或 Skill 分发。

## 验证状态

已验证真实 MCP stdio 子进程的 initialize、list、status、cancel、start 和过期 session
拒绝；fake Host 用于验证执行去重和参数拒绝。当前可见设备均为模拟器或未连接；
Android 真机端到端、跨设备 Function 映射和 Codex 实际使用验收均为**待真机验证**。

官方宿主参考：[Codex MCP](https://learn.chatgpt.com/docs/extend/mcp)、
[Codex computer use](https://learn.chatgpt.com/docs/computer-use)、
[Reusable Codex skills](https://learn.chatgpt.com/use-cases/reusable-codex-skills)。
接入 Android 是本项目 MCP 的能力，不表示 Codex 自带 CUA 已暴露 Android 后端。
