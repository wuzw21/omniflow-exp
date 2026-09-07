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

宿主必须允许所需 MCP 工具经过正常审批。Codex CLI 0.153.4 的本次测试中，
`read-only` 加 `approval_policy="never"` 在模型首次调用 recall 时返回
`MCP tool call requires approval, but approval policy is never`，没有执行设备动作。
用户明确授权本次测试后，临时会话使用 `approval_policy="on-request"` 和
`approvals_reviewer="auto_review"`，保留只读沙箱并通过审批。
不要把权限错误当作 Function 失败重试，也不要修改全局权限来掩盖接入问题。
自动审批的配置与边界见 [Codex 官方说明](https://learn.chatgpt.com/docs/sandboxing/auto-review)。

Claude Code 使用相同的 stdio server，配置格式为：

    {"mcpServers":{"omniflow":{"command":"/absolute/OmniFlow-exp/.venv/bin/python","args":["-m","src.integrations.gui_agent_mcp","--device","DEVICE_SERIAL","--store","/absolute/memory/store.json","--evidence-root","/absolute/OmniFlow-exp/data/runtime/gui_agent/claude"],"env":{"PYTHONPATH":"/absolute/OmniFlow-exp"}}}}

测试时通过 `--strict-mcp-config --mcp-config <file>` 只加载该服务，
`--tools ""` 关闭内置工具；用户授权后用 `--allowedTools`
`mcp__omniflow__omniflow_recall,mcp__omniflow__omniflow_execute` 限定允许的工具。
`--no-session-persistence` 避免留下可恢复的测试会话。以上配置不需要第二套内核。

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

Function 名称不是幂等承诺。蓝牙 live 检查表明，同一份“turn on”历史动作在开关已经
开启时仍会点击并关闭它。宿主应在调用前判断目标是否已满足，并核对状态敏感操作的
起始条件；不确定时使用自己的观察通道。完成后继续核验目标，不以 actions completed
代替目标达成。当前 Function v2 没有额外的条件动作/持久化 resume schema。

宿主负责整个任务的规划、完成判定、用户输入、原始动作与取消。嵌入式宿主通过
runtime.cancel() / flow.cancel() 取消；MCP 宿主使用协议请求取消通道。正在执行的
I/O 先收尾，之后禁止新增动作。状态和当前反馈包含在结果中，不另加 model-visible
status、observe、finish、start 或 cancel 工具。进程重启不恢复旧 session。

若 Codex 只有桌面 CUA，没有 Android 原始动作通道，则不能宣称具备 Android 的任意
fallback 能力。两工具服务仍可召回与执行 Function；失败后交还宿主或报告阻塞。
不能为隐藏这一接入缺口偷偷启动内置 Planner。

## 验证范围

必须区分模型驱动的完整宿主、框架工具分发、模型输出适配器和物理后端。
GUI-Owl/V-Droid 在本仓库的接入是输出适配器，不能以适配测试通过宣称两个完整
上游 Harness 已经验收。Standard/Fold/Tablet 是同一 OOB 后端的不同设备，
同步/异步测试 Host 也不是已经落地的独立操作系统后端。

| 路径 | 实际已验证层级 | 尚不能推出的结论 |
|---|---|---|
| Codex CLI 0.153.4 | 模型自主 recall → execute；本机五步 OOB 关闭蓝牙，模型观察与系统状态均确认关闭 | Codex 自带 Android 原始动作 fallback、模型驱动失败恢复 |
| Claude Code 2.1.237 | 模型自主 recall → execute；同一模拟器五步 OOB 开启蓝牙，模型观察与系统状态均确认开启 | 其他任务、模型驱动取消与失败恢复 |
| 通用 MCP stdio 客户端 | 本机模拟器真实召回、执行、取消、重启隔离、去重、空 Memory | Codex/Claude 模型自主使用工具 |
| Droidrun 0.5.6 | 实际 ToolRegistry 分发；Tablet 上五步 OOB 执行并核验蓝牙开启 | 完整 DroidAgent 模型循环 E2E |
| GUI-Owl 适配器 | Fold 上真实 OOB 执行并核验蓝牙开启 | 上游模型自主规划与恢复 |
| V-Droid 适配器 | 受控 Host 上的成功与部分失败合同 | 上游模型或真实设备 E2E |
| Minitap | 保留既有适配代码 | 上游框架接入验收 |
| OOB 后端 | 本机及远端 Android 模拟器，canonical OmniTransfer | 物理手机验收、其他操作系统后端 |

完整宿主验收必须留下模型实际工具调用、返回事实和独立设备状态证据；进程退出码
为零或工具可发现都不足以判定通过。性能比较另需冻结配对任务与组件时延，当前
接入诊断不能支撑“比原生 Computer Use 更快”的结论。

2026-09-07 的两个真实模型宿主测试使用同一 MCP 服务和 canonical OmniTransfer，
没有替换 Store、增加专用动作或启动内置 Planner。Codex 先关闭蓝牙，Claude 再恢复
开启；两者各自主调用一次 recall 和一次 execute，各完成五个动作。服务返回
`task.status=unknown`、`control.next=host`，由宿主根据返回 XML/截图判断目标，
并由测试主机只读查询系统状态独立核验 1→0→1。测试运行在现有本机 Pixel 6 Pro
模拟器（Android 13、1440×3120、OOB 0.6.0.3/versionCode 10），不安装 APK。
记录位于 `data/runtime/validation/20260907-model-harness-acceptance/summary.json`，
包含首次审批阻塞、成功调用链、五步事实及 checkpoint/event SHA-256。
这补齐了两工具成功路径的真实模型宿主证据，仍不代表模型驱动故障恢复、论文结果或
真机验收；服务的取消/去重/重启证据来自独立的 MCP 客户端测试。

后续真实模型空 Memory 测试中，Codex 召回两次、Claude 召回一次，均收到
`empty_memory` 后停止，没有调用 execute、创建 Store 或改变蓝牙状态；Codex 的
两次请求复用相同 task_id，没有重置任务预算。记录：
`data/runtime/validation/20260907-model-empty-memory/summary.json`。

两种真实 CLI 还完成了步骤边界中断检查：模型开始执行五步 Function，在首步
open_app 的完成事实出现后发出 SIGINT，CLI 退出后观察五秒，没有新增动作记录；
随后重新取得设备锁，OOB 确认仍在设置首页，系统蓝牙仍开启。
Codex 退出码为 1，未返回 execute 结果；Claude 退出码为 0，却返回
`error_during_execution` 和通用工具拒绝消息。两者均不能据此推断任务成功或零动作。
记录：`data/runtime/validation/20260907-model-stop-acceptance/summary.json`。
这只证明本次步骤边界的 CLI 停止行为，不证明 Android 动作在途时的收尾，也不证明
桌面应用停止按钮、断电恢复或跨进程 exactly-once。Skill 明确要求缺少 invocation
反馈时将副作用视为未知，恢复前核验当前设备；真机验收仍待完成。

协议测试覆盖真实 stdio initialize/list、严格两工具发现、过期 session 拒绝；
适配矩阵覆盖 MCP、GUI-Owl、V-Droid、Mobilerun 与同步 Host、异步 Host、OOB Host
适配器的组合，检查召回零 act、Function 执行、部分失败、请求去重和宿主控制权。

这些矩阵使用受控 Host 与编码器测试替身，只证明接入合同。真实 canonical 模型、
真实上游 harness 运行及物理设备仍需要独立验证，不能拿矩阵通过代替。
旧版七工具的 OOB 模拟器 smoke 仅为旧协议证据，不能冒充当前两工具验收。

当前已验证 dev1 隔离 wheel 的导入、两工具发现和 Skill 打包。同一份蓝牙 Function
在 4090 Standard、Fold、Tablet 上经真实 canonical 1024D 召回与 OmniTransfer 映射，
各完成五步 OOB 执行；重复请求不产生设备 I/O。最终 XML 中只有 Standard 的蓝牙开关
已开启，Fold 与 Tablet 仍关闭，不能把动作执行完成当作目标达成；宿主必须核验目标。
这是真实模拟器上的 Function 接入检查，没有调用官方 task validator，不是论文结果；
真机验收仍待完成。完整记录见发布说明。
后续由宿主先观察确认关闭状态，GUI-Owl/Fold 和真实 Droidrun ToolRegistry/Tablet
分别完成五步执行并观察到开启。该复测还修复了共享 Checker 在 open_app 前错误恢复
source Launcher 的问题；原始 Store 和统一映射器未变。这些接入路径由测试驱动调用，
尚不代表上游 LLM 端到端或真机验收。
上游 Droidrun 0.5.6 的 ToolRegistry 已验证成功与部分失败两条分发路径，使用
mobilerun-sdk 2.1.0；该约束已加入 bmoca 可选依赖。5.x SDK 将模块名改为
mobilerun_sdk，不能满足这个旧版 Droidrun 的 mobilerun 导入。安装时使用项目的
可选依赖组合 `.[bmoca,mcp]`，不要用同名 mobilerun 新框架替代 SDK。
这项验证调用真实上游 registry，设备和编码器仍是合同测试替身，不是上游 LLM E2E。

取消测试通过真实 MCP ClientSession/Server 的 JSON-RPC 取消通知，确认在途动作
收尾、下一步不执行、原请求重试返回取消事实、会话关闭后拒绝新执行。

[内核协议](HARNESS_PROTOCOL.md) · [Skill](../skills/omniflow-gui/SKILL.md)
