# OmniFlow 内核与 Harness 协议

2026-09-07；基线为 main 上的 `v1.0.0`（`30dcea39`）。本轮目标是把可变宿主从
已有执行内核中分离，保留论文的核心机制：从成功轨迹提取显式参数化 Function，
利用 canonical OmniTransfer 适应目标设备，每步观察并验证，失败交还 Agent。

当前 `1.1.0.dev1` 将对外服务收敛为 **Function Recall + Function Execute**：
`omniflow_recall(task_id, goal, limit)` 调用 `OmniFlow.arecall`，
`omniflow_execute(session_id, request_id, function_id, arguments)` 调用同一执行内核。
`aexecute_function` 拒绝未注册的 Function，原始动作不能借此入口执行。MCP 默认仅有这
两个工具；取消由宿主控制通道或 SDK 负责，任务完成由宿主判断。dev0 的七工具接口已移除。

同一逻辑任务复用 `task_id`，Recall 返回此次会话的 `session_id`。新的 `task_id` 明确
表示开始新任务；在途操作未结束时不能切换。旧 task id 被退役，旧 session id 不能重发
到新任务上。不能用更换 task id 绕过任务预算或未知副作用处置。

## 1. 一个任务只有一个外层循环

```mermaid
flowchart TD
  A[内置 Planner 或 Codex 宿主] --> B[单次调用协议]
  B --> C[绑定显式 Function 参数]
  C --> D[共享 Checker]
  D --> E[canonical OmniTransfer]
  E -->|可靠映射| F[OOB Act]
  F --> G[Observe 与证据]
  G -->|还有内部步骤| D
  E -->|失败| H[返回状态、失败位置和执行事实]
  G -->|内部步骤结束| H
  H --> A
  A --> I[任务完成判定]
  I -->|通过或取消| J[终止会话]
```

| 层 | Owner | 合同 |
|---|---|---|
| Function 作者与编译 | 既有 `compile_runlog_to_store` | 显式 stable/task_parameter/online_observation；v2 连续步骤不改 |
| 内核执行 | `runtime/execution.py`、`core.py` | 同一 Checker → Transfer → Act → Observe；映射失败绝不重放 source 坐标 |
| 内置任务 Harness | `OmniFlow.arun` | 唯一内置 Planner/Router 循环；官方完成判定、预算、停止 |
| 对外核心 | `OmniFlow.arecall` / `aexecute_function` | 召回不 act；执行仅限注册的 Function，失败返回宿主 |
| 原始动作适配 | 既有 `acall_tool` / GUI-agent adapters | harness 的 OOB 原始动作接入，不进入两工具服务 |
| 会话 | `GuiAgentToolRuntime` | 同一控制预算、串行执行、请求去重、显式关闭和新任务 |
| 设备 | AndroidWorldHost / OobGuiAgentHost | 复用 OobControlClient；宿主决策与设备传输分别替换 |
| 证据 | 既有 recorder/state owner | 无损 PNG 内容寻址、紧凑索引、必要时才转 base64 |

AndroidWorld 保持原来的 shell → run_tasks → run_task → run_episode 链；MCP 是外部
宿主的传输接口，不能作为第二个 AndroidWorld 实验 runner，也不能替换官方 validator。

## 2. 本轮找到的问题与修改

1. **双重循环。** 外部 `acall_tool` 原本复用会自动 fallback 的任务循环。外部 Agent
   等待工具返回时，内部 Planner 又继续执行，控制权和失败语义混乱。现在单次调用与
   `arun` 分开，但共用原来的 `execute_function`，没有复制 Function 内核。
2. **Router 无步数消耗。** 只有 Planner 分支增加运行步数，Router 成功、页面变化重试
   可重复绕过该限制。现在每轮都消耗同一预算；相同页面、参数和决策最多允许三次，
   第四次返回 `no_progress`。这是一条保守的停止规则，不证明任务不可能继续。
3. **完成判定不统一。** 原始动作路径的 `finished` 和 Function 完成分支有不同处理。
   现在内置 Harness 统一调用当前任务 checker；通过立即终止，checker 出错立即返回
   `verifier_error`，不会为了“修正页面”盲目追加动作。无 checker 的完成声明保持未验证。
4. **阻塞取消与预算重置。** 同步模型/OOB 调用阻塞事件循环，外部每次调用缺少共享预算。
   现在同步 I/O 进入工作线程，当前会话共享不超过 600 秒的 deadline；模型 HTTP 请求
   不超过 30 秒且受剩余预算约束。取消阻止下一次 dispatch，已发出的动作先收尾。
5. **不明确的失败可能被重试。** 已 dispatch 的动作若无法确认结果，返回 `effect_unknown`
   并关闭会话；映射失败则返回明确的未 dispatch 事实。两者不得混为“可安全重放”。
6. **重复截图和错误的后态引用。** Host/recorder 重复生成截图，索引长期保存完整观察；
   Fast Pass 还读取了不存在的 Host 属性。现在共享一次捕获、引用 recorder 的当前后态，
   相同 PNG 用同一 SHA-256 文件，索引仅保存引用。原图尺寸、source 状态和动作序列不改。
7. **核心导入依赖实验目录。** 独立 wheel 安装后，`import omniflow` 会读取未分发的
   `config/paper_androidworld.json` 而失败。实验配置读取现已移回 `src/experiment/protocol.py`，
   核心采用通用默认值；AndroidWorld owner 仍显式传入自己的协议预算。

以上是代码证据和回归结果；**全部行为修复仍待真机验证**，不能据此宣称用户问题已验收。

## 3. 一次调用反馈 v1

机器 schema：`schemas/omniflow_invocation.v1.json`。反馈分为四种独立事实：

- `execution`：本次 Function/动作是否完成，实际成功动作数，失败步骤，以及失败动作
  是否已经 dispatch；`next_step_index` 是位置证据，不是授权重放的 resume token。
- `observation`：最新页面、原始 display、截图引用；不在历史上下文重复携带 base64。
- `task.status`：`unknown` / `verified_success` / `verified_incomplete` / `verifier_error`。
  Function 成功不自动变成 task 成功。
- `control`：交还 `host` 或 `stop`，附明确原因；`automatic_retry` 永远为 false。

外部模式由宿主判断下一步。Function 部分成功后，宿主读取失败位置与当前状态，执行
必要的恢复动作或改用当前页面的原始动作。v1 **没有对外持久化 resume API**，不能仅凭
失败 index 重发整个 Function；内置续执行仍走原有 owner，并禁止续执行时改变绑定参数。

## 4. 生命周期、并发与重试

外部会话首次 observe/execute 时开始计时；一个任务中的多次调用不重置 600 秒预算。
关闭后只能显式开始新任务。`cancel`、完成、预算用尽或未知副作用都不允许追加动作。
不同层的控制对象通过 ContextVar 共享；Checker trigger budget 也跨本会话调用共享。

`execute_request(session_id, request_id, tool_name, arguments)` 对当前进程内、当前会话
的传输重试去重：同一 id 和参数返回原结果；同一 id 不同参数报冲突；正在执行或未获知
结果的 id 返回明确状态。每会话最多 128 个请求，缓存紧凑结果，不无限积累完整 trace。
进程重启后 session id 改变，旧请求不能续接。协议不承诺跨进程 exactly-once。

宿主中断可能只产生通用工具错误，甚至没有返回 invocation。此时副作用未知，不能
根据“工具被拒绝”或 CLI 退出码推断零动作。2026-09-07 的真实 CLI 测试中，Codex
中断后没有 execute 返回，Claude 返回通用拒绝消息；两者的首步 open_app 都已完成。
恢复前必须通过宿主观察通道核验当前设备，无观察能力则报告未知并停止；不能盲目
重发 Function 或换 task_id 绕过未知副作用。原始事实与测试范围见 MCP 接入文档。

同一 runtime 同时只执行一个 observe/act/Function；并发调用返回 busy。
取消是协作式的：不会撤回已经发生的点击，也不会在旧调用仍运行时把设备锁交给下一调用。
OOB/模型传输有 timeout；任意第三方插件若永久卡住，Python 线程不能被安全强杀。
正式 AndroidWorld 的 lifecycle deadline 仍由既有外层 owner 强制实施。

## 5. 截图与经验保留

截图保存在原证据目录的 `observations/objects/<sha256>.png`，并发写入只发布完整文件。
相同编码内容复用一份文件；像素不同的页面不做感知去重。当前后态和完整 source 状态
仍保留原始分辨率，避免损害 Transfer。模型传输只发送当前一张图；长期历史保存引用。
这降低重复磁盘写入和 Python 对象保留，尚未测量真机 RSS 或整个系统的加速比例。

历史清理仅作用于本地 `data/androidworld/.archive`。首轮删除约 16.72 GB，保守保留
731 个引用/经验文件。用户随后授权清空历史目录：确认唯一现存引用是过时的 result
registry 搜索根；TasksDueNextWeek 的归档轨迹与正常目录中的轨迹在路径归一化后相同，
截图逐字节一致，并非真正独有的经验。因此移除旧搜索根并删除剩余 790 个文件（含重建
的 Finder 元数据，约 11.41 MB），整个 `.archive` 目录已删除。当前经验没有被改写。
两阶段清单、原始时间和 SHA-256 日志分别在 `data/runtime/archive_cleanup/20260907/`
与 `20260907-final/`；审计属于本地数据，不提交 Git。

## 6. 跨设备与 Codex 接入边界

Function、bindings、source states 和 OmniTransfer 不依赖宿主的 Planner prompt。
宿主通过同一反馈协议更换，设备通过 Host 合同更换；两种变化不应互相牵连。
当前真正实现的设备适配是 Android OOB；不能把接口可插拔表述为已经完成 iOS/Windows
跨平台 Transfer。新增设备必须提供真实 Observe/Act/状态与坐标语义，并验证映射模型覆盖。

两工具服务不是完整的 Android Agent。需要原始动作 fallback 的宿主必须另有适用于当前
设备的动作通道，并协调同一设备 owner。Codex 自带桌面 CUA 不能被默认为 Android
动作通道；缺少该能力时，Function 失败必须交还宿主或报告阻塞，不能暗中恢复内置 Planner。

Codex 可通过 MCP 调用 Android OOB，而 Codex 自带 computer use 处理其支持的桌面。
两者由 Codex 宿主协同，不接管或替换 Codex 的私有 CUA 后端。Codex 主观上响应快不等于
已经证实某个内部组件更快；要比较同任务、设备、参数、模型与 endpoint 的 Memory ON/OFF，
分别记录模型、recall、Transfer、OOB、stabilization、checker、fallback 的耗时。

## 7. 验收与后续顺序

先验证取消等待中的模型/动作、重复取消、预算到期、成功后零新增动作、Function 部分失败、
请求重发、会话重启和相同页面重复决策；再做同任务 Memory OFF/ON 的交替真机测试。
记录设备、APK 版本、动作、结果、重启恢复与证据路径。没有真机不做已验收结论。

2026-09-07 补充联调已在 9207 的 `emulator-45562`（1440×3120）完成，使用已安装的
OOB 0.6.1 / versionCode 7。实际 OOB 截图和一个 wait 动作成功；重复 request id 没有
新增设备 I/O；取消后新动作被拒绝；新会话和未配置 verifier 的 finish 语义符合协议。
证据在 `data/runtime/validation/20260907-oob-protocol/`。这不是物理设备、Function
跨设备映射或模型性能验收，也不能将包含 SSH 传输的 smoke wall time 计入论文时延。

下一阶段的真机验收顺序与通过条件：

| 顺序 | 验证项 | 必须观察到的结果 |
|---|---|---|
| 1 | 模型等待与动作在途时取消、重复取消 | 收尾前不释放设备 owner，之后零新增动作；未知副作用明确报告 |
| 2 | Function 完成、checker 拒绝与异常 | 通过即停止；拒绝反馈宿主；异常不触发盲目纠正动作 |
| 3 | Transfer 失败与部分执行恢复 | 保留已执行前缀和当前页面，不重发前缀或 source 坐标 |
| 4 | 会话断开、重连与进程重启 | 同会话重发不重复动作；过期 session id 拒绝续接 |
| 5 | Memory OFF/ON 交替运行 | 同任务/参数/设备/seed/模型/endpoint，两组均通过官方验证才组成时延配对 |

以上真机项尚未验收。冻结配对设置后再采集结果，不为得到成功样本临时更换模型；
若模型权限、设备或 validator 不可用，保留环境失败证据并排除出有效时延配对。

性能结论保持现有论文合同：主文只用 paired official-success 的 `execution_duration_ms`，
完整系统必须纳入 Function 失败但 fallback 恢复成功的成本。当前还不能宣称完整系统更快。
下一阶段再根据组件计时决定是否需要独立可中断 worker、持久 resume 或新的设备后端；
避免在没有实测前引入第二套执行循环。
## 统一 AndroidWorld Harness 接口

AndroidWorld `agent.step` 只调用 `TaskHarness.arun(HarnessContext)`；默认适配器调用
现有 `flow.arun`，CLI 适配器把同一 `flow` 的两工具服务交给外部宿主。其他宿主以
`package.module:factory` 接入这个接口，不复制 episode、Host 或 Function 执行。
统一入口是 `run_androidworld.sh run ... --harness builtin|codex|claude|package.module:factory`。

独立服务允许宿主显式开始新任务；在 AndroidWorld episode 内锁定一个逻辑任务，
新 task_id 不能重置预算或完成状态。注册的官方 checker 在 Function 成功后由共享
判定函数调用，通过则立即返回终止事实；没有 checker 的独立服务仍返回未知任务状态。
外部 CLI 的请求次数若未报告，RunLog 与 task result 保留 null，不能把宿主 turn 数
或工具数当作模型调用数。CLI 的接入结果不进入冻结模型的正式结果晋升。
