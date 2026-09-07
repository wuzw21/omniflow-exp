# 两工具 Function 服务接入

当前版本为 1.1.0.dev1。对外仅暴露两个工具，定义统一来自
GuiAgentToolRuntime.function_tools()，MCP 和 Python harness 不各自复制 schema：

| 工具 | 输入 | 返回 |
|---|---|---|
| omniflow_recall | task_id、goal、limit（1–32） | 候选 Function、input_schema、当前 observation、session_id、召回审计 |
| omniflow_execute | session_id、request_id、function_id、arguments | invocation v1 执行事实、失败位置、当前 observation |

召回调用既有 page-aware recall 与 canonical OmniTransfer；它不执行设备动作。
执行只接受注册且可见的 Function，使用同一个 Check → Transfer → Act → Observe
闭环，不启动内部 Planner，也不把 Function 成功当作任务成功。

## 安装与连接

在 canonical checkout 中运行：

    .venv/bin/python -m pip install -e '.[mcp]'
    .venv/bin/python -m src.integrations.gui_agent_mcp --device DEVICE_SERIAL --store /absolute/memory/store.json

省略 --store 表示空 Memory；召回返回空候选，不扫描历史、不隐式编译 Store。
默认后端仍是已安装的 OOB，不重新构建或安装 APK。截图和 step fact 调试日志默认在
当前工作区 data/runtime/gui_agent/ 下；在其他目录启动时显式传 --evidence-root。

Codex MCP 配置（替换为实际路径和 serial）：

    [mcp_servers.omniflow]
    command = "/absolute/OmniFlow-exp/.venv/bin/python"
    args = ["-m", "src.integrations.gui_agent_mcp", "--device", "DEVICE_SERIAL", "--store", "/absolute/memory/store.json", "--evidence-root", "/absolute/OmniFlow-exp/data/runtime/gui_agent/codex"]
    env = { PYTHONPATH = "/absolute/OmniFlow-exp" }
    tool_timeout_sec = 600

Python 宿主可直接使用 OmniFlow.arecall / aexecute_function，也可调用
GuiAgentToolRuntime.call_function_tool 获得与 MCP 相同的 schema 校验、会话与请求去重。
GUI-Owl 和 V-Droid 的既有工具分发可以接这两个服务；Mobilerun 的
build_omniflow_custom_tools 只暴露这两个。正式实验的 native-action adapter 仍可用
build_runtime_custom_tools 组合原有 OOB 原始动作和这两个服务。

## 后端接入

Python 通过 build_runtime(host_factory=...) 注入后端；命令行使用
--host-factory package.module:factory。factory 接收 device、evidence_root、
source_states，返回实现 observe、act、get_state 的 Host，方法可同步或异步。
工厂是由部署者明确选择的 Python 代码，不接受模型工具调用中的任意代码路径。
source_states 来自显式 Memory 的 canonical loader，不是另一套 mapper。

执行 open_app 的 Host 还可提供 installed_apps()，同步或异步返回标签到包名的
字典。显式传入 OmniFlow 的清单优先；否则在受 deadline/取消控制的首次执行中
读取并缓存 Host 清单。OOB 后端从当前设备只读查询已安装包，应用启动仍走 OOB。
没有清单或包未安装时明确失败，不推测安装状态。设备或安装集合改变后重建 runtime。

这只替换物理后端；Function compiler、recall、Checker、OmniTransfer 和执行反馈不替换。
AndroidWorld 的正式入口和 OOB-only 合同不受该独立宿主接口影响。

## 生命周期与恢复

同一逻辑任务始终复用 task_id。首次 recall 建立会话并返回 session_id；
同任务再次 recall 不重置 deadline。新的 task_id 明确开始新任务，在途执行期间拒绝切换。
旧 task id 退役，旧 session id 不能续接到新任务。

execute 的传输重试必须保持 request_id 和参数相同，以获取已保存的结果。不要把
Function 部分失败当作 transport retry：读取成功前缀、失败位置和当前状态，再由宿主
决定如何恢复。v1 没有持久化 resume API，不能从失败 index 直接授权重放。

宿主负责整个任务的规划、完成判定、用户输入、原始动作与取消。嵌入式宿主通过
runtime.cancel() / flow.cancel() 取消；MCP 宿主使用协议请求取消通道。正在执行的
I/O 先收尾，之后禁止新增动作。状态和当前反馈包含在结果中，不另加 model-visible
status、observe、finish、start 或 cancel 工具。进程重启不恢复旧 session。

若 Codex 只有桌面 CUA，没有 Android 原始动作通道，则不能宣称具备 Android 的任意
fallback 能力。两工具服务仍可召回与执行 Function；失败后交还宿主或报告阻塞。
不能为隐藏这一接入缺口偷偷启动内置 Planner。

## 验证范围

协议测试覆盖真实 stdio initialize/list、严格两工具发现、过期 session 拒绝；
适配矩阵覆盖 MCP、GUI-Owl、V-Droid、Mobilerun 与同步 Host、异步 Host、OOB Host
适配器的组合，检查召回零 act、Function 执行、部分失败、请求去重和宿主控制权。

这些矩阵使用受控 Host 与编码器测试替身，只证明接入合同。真实 canonical 模型、
真实上游 harness 运行及物理设备仍需要独立验证，不能拿矩阵通过代替。
旧版七工具的 OOB 模拟器 smoke 仅为旧协议证据，不能冒充当前两工具验收。

当前已验证 dev1 隔离 wheel 的导入、两工具发现和 Skill 打包；真实 canonical
1024D 召回在 9207/OOB 模拟器上通过。完整执行与真机验收分别记录，详见发布说明。
上游 Droidrun 0.5.6 的 ToolRegistry 已验证成功与部分失败两条分发路径，使用
mobilerun-sdk 2.1.0；该约束已加入 bmoca 可选依赖。5.x SDK 将模块名改为
mobilerun_sdk，不能满足这个旧版 Droidrun 的 mobilerun 导入。安装时使用项目的
可选依赖组合 `.[bmoca,mcp]`，不要用同名 mobilerun 新框架替代 SDK。
这项验证调用真实上游 registry，设备和编码器仍是合同测试替身，不是上游 LLM E2E。

取消测试通过真实 MCP ClientSession/Server 的 JSON-RPC 取消通知，确认在途动作
收尾、下一步不执行、原请求重试返回取消事实、会话关闭后拒绝新执行。

[内核协议](HARNESS_PROTOCOL.md) · [Skill](../skills/omniflow-gui/SKILL.md)
