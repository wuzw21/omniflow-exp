# 同 harness 基线（2026-10-12）

## 为什么要有同 harness 基线

官方 MobileGPT/AppAgent 复现同时改变了太多变量：各自的 Planner prompt、动作空间、
Accessibility client、步数预算、模型调用方式，以及 MobileGPT 需要的大量兼容补丁。
主表若直接拿 OmniFlow 和这些复现比，审稿人无法判断差距来自"复用机制"还是来自
"工程实现"。因此主表改成同 harness 对比：所有行共享同一个 OmniFlow Planner、同一个
OOB Host、同一个模型（Qwen3.6-Plus）、同一步数预算，官方 validator 只在 episode 结束后
调用一次，`--completion-oracle` 必须为 off。各行唯一的差别是"从 source 演示里复用什么、
怎么复用"。

| 行 | 复用的东西 | 运行方式 |
| --- | --- | --- |
| zero-shot | 无 | `--method omniflow`，不传 `--memory` |
| demo_hint（T3A+hint 的同 harness 版） | 同一份脱敏演示步骤文本追加到 goal | `--method omniflow --baseline demo_hint --source-run-log <run_log.json>` |
| fixed_replay | 原坐标逐步重放 | `--method fixed_replay` |
| MobileGPT-style | 子任务 Function + 属性匹配重定位 | `--method omniflow --baseline mobilegpt_style --memory <store.json>` |
| AppAgent-style | 元素文档注入 Planner prompt | `--method omniflow --baseline appagent_style --memory <store.json> --element-docs <appagent>/apps/<app>/demo_docs` |
| MobiMem-style（AgentRR） | 录制的交互序列 + 安全条件回放 | `--method omniflow --baseline mobimem_style --memory <store.json>` |
| OmniFlow | Function + OmniTransfer + Checker | `--method omniflow --memory <store.json>` |

注：原 `--method t3a_hint` 仍走 AndroidWorld 官方 T3A agent，只用于附录保真度检查；主表用
`demo_hint`，它复用 t3a_hint 完全相同的脱敏 hint 文件（`_source_action_hint_path_for_item`），
只把执行者换成 OmniFlow Planner，并关闭 Function 记忆。

## Baseline registry and suite runner

基线定义集中在 `src/experiment/baselines.py`。该注册表同时维护展示名、输入依赖、
Function memory 开关、Transfer 替换和结果槽位，Agent 与 launcher 都从这里读取，避免
新增基线时分别修改多处条件分支。

可以用 `--baseline all` 在同一个 task、device、evaluation seed、Memory、source RunLog
和 element-docs 输入上按注册表顺序运行全部同 harness 条件：

```bash
bash scripts/exp/run_androidworld.sh run \
  --method omniflow \
  --baseline all \
  --memory <store.json> \
  --source-run-log <run_log.json> \
  --element-docs <appagent>/apps/<app>/demo_docs
```

suite runner 逐个设置 baseline 环境并调用同一个 `_run`，每个条件仍写入独立的
`<task>/<baseline>/` 结果槽位。返回值保留每一行的状态和设备结果，单行失败会使 suite
标为失败，同时保留此前各行的证据。当前 runner 只提供基建，尚未宣称任何新的真实设备
成功率或成本结果。

## MobileGPT-style

MobileGPT 的核心是"记住子任务、在新屏幕上按属性重新找到记录过的元素、不重新规划元素选择"。
在同 harness 里，它就是 OmniFlow 的 Function Store 加上一个不同的 Transfer：
`omniflow/transfer/attribute.py::attribute_transfer`。它先在 source 页面上找到点击点下面
最小的带标签元素，再在目标页面上按 resource-id+text、resource-id、text、content-desc
的顺序精确匹配，多个候选时选离原位置最近的；找不到就返回 `attribute_no_match`，交给同一个
Checker/Planner 恢复路径。Function 的召回、参数绑定、执行和 Planner 主线与 OmniFlow 完全相同，
所以 MobileGPT-style 与 OmniFlow 的差就是 OmniTransfer 本身的贡献。

## AppAgent-style

AppAgent 的核心是探索/演示阶段为每个 UI 元素生成一份文档（tap/text/swipe/long_press 各做什么），
执行时把当前屏幕上出现元素的文档拼进 prompt。同 harness 实现直接复用 `convert-memory --method appagent`
已经产出的官方 `demo_docs/`：`src/integrations/android_world/agent.py` 的 `_TaskHost` 按 AppAgent
自己的元素 id 规则（`_appagent_element_id`）匹配当前 XML，把命中的文档放进
`observation.extra["element_docs"]`，Planner 以 "Documentation of UI elements on this screen"
一节呈现。AppAgent-style 强制关闭 Function 记忆，因此它与 zero-shot 的差就是元素文档的贡献。

## MobiMem-style（AgentRR）

MobiMem（arXiv:2512.15784，*Beyond Training: Enabling Self-Evolution of Agents
with MOBIMEM*）把 agent memory 分成三个原语。只有第三个能和 OmniFlow 的
Transfer 直接对照，所以这里只复现它：

- **Profile Memory**（DisGraph 用户偏好图）：AndroidWorld 任务里没有可检索的
  用户偏好信号，复现出来也测不出差别。
- **Experience Memory**（多级模板）：这是 MobiMem 版的 Function 资产，而
  Function 资产本来就归 OmniFlow 所有。换掉它改变的是"测什么"，不是"谁映射得更准"。
- **Action Memory + AgentRR**（录制回放）：回放**录制的动作序列**，而不是为单个
  动作从源界面重新推导一个目标点。这才是 OmniTransfer 的直接替代，因此它是本行。

实现是 `omniflow/transfer/agentrr.py::agentrr_transfer_with`，通过
`PluginSet(transfer=...)` 注入，与 `mobilegpt_style` 走同一个接缝。Planner、
Checker、Function 召回、参数绑定、步数预算、官方 validator 全部不变，所以
这条行与 OmniFlow 行的差**只有 mapper 一个函数**。

它和 MobileGPT-style 的区别要保留，不要合并：

- 属性重定位为**单个**录制元素在新屏按 resource-id / text / content-desc 重新
  定位，并重新绑定本动作自己的参数。
- AgentRR 按身份回放**录制的交互**，每次派发前检查安全条件；条件不满足时**弃权**
  （`agentrr_guard_*`），而不是去找一个更松的匹配——一个开始猜的回放机制就不再是
  回放机制了。

安全条件（`AgentRRPolicy`）与它们产生的失败原因：

| 条件 | 不满足时的 reason |
| --- | --- |
| 源状态必须存在 | `agentrr_source_state_missing` |
| 包名未改变 | `agentrr_guard_package_changed` |
| 录制身份在当前屏仍唯一存在 | `agentrr_guard_signature_absent` |
| 录制点可解析 | `agentrr_invalid_source_point` / `agentrr_recorded_element_missing` |

`AgentRRPolicy(require_package_match=False, require_same_signature=False)` 是纯
位置回放，也就是本基线最弱的形式：回放录制的点本身，不要求录制身份仍在。
它是消融，不是默认行。

包名比较取**整屏** package（`observation.package_name` 或窗口根节点的），不能取
录制叶节点的——多数 Android 节点不带 `package` 属性，只有窗口根带，取叶节点会让
这条条件变成空条件。这个 bug 在离线回归里暴露过一次。

## 运行与归档

`--baseline` 只允许 `--method omniflow`、builtin harness、oracle 关闭；`mobilegpt_style` /
`appagent_style` / `mobimem_style` 需要 `--memory`，`demo_hint` 需要 `--source-run-log`。
每个基线有自己的正式结果槽位
`data/androidworld/<task>/<baseline>/<setting>/runlog/current`，失败按同一规则进 `.archive/<task>/<baseline>/`，
与 omniflow 互不覆盖。汇总：`scripts/exp/generate_task_table.py` 目前只读 `omniflow`，需要加 `--methods` 参数才会出基线行（该脚本未入库，补丁另附）。

## Planner 看到的任务信息

`_goal_with_task_parameters` 以前把 AndroidWorld `task.params` 整体附在 goal 后，其中包括剪贴板内容、
预置数据、QA 答案等基线看不到的隐藏 setup 数据。现在只复述 goal 原文里已经出现的参数值，其余一律不给。

## 原版复现的定位

原版 MobileGPT/AppAgent 不再进入主表，只在 10–20 个任务上跑一次作为保真度检查，
报告在附录：说明同 harness 版本与原版在同一批任务上的成功率差距，并引用 MobileGPT 论文与
V-Droid 报告的数字。正文中的跨设备结论只来自同 harness 主表。

## 环境：整机 AVD 快照

AndroidWorld 自带的是按 app 的数据快照（`app_snapshot`），只恢复 app 数据目录，系统设置、
IME、OOB 服务状态和时间仍会在方法之间漂移。现在启动器支持整机快照：

```bash
# setup 完成后存一次（每台设备各存一份）
bash scripts/exp/run_androidworld.sh save-snapshot --device all --avd-snapshot omniflow_clean
# 之后每个 (method, device) 运行前自动 load
bash scripts/exp/run_androidworld.sh run --method all --device all --avd-snapshot omniflow_clean ...
```

实现是 `run_tasks._avd_snapshot`，通过模拟器 console（`adb emu avd snapshot save|load`）完成，
load 后等待 `sys.boot_completed`。`-no-snapshot-save` 只关闭退出时的 quick-boot 保存，不影响
命名快照。episode 内仍会照常执行 `_ensure_oob_control_app` 和按 app 的数据恢复，因此快照只是
更强的起点保证，不替代已有 setup。这与 MobileWorld 的 Docker AVD + 快照做法一致。

**待真机验证**：快照 load 后 OOB Accessibility 是否保持绑定、adb 是否短暂 offline、
设备时间回退是否影响依赖日期的任务，都需要在 4090/真机上各跑一轮确认。

## Google ARTEMIS / 外部 harness（2026-11-06 更新）

ARTEMIS（github.com/google/artemis，Apache-2.0，Python 3.12+）已读到源码，此前"未读到源码"
的表述作废。它是**完整的外部应用**：自带 Planner / Operator / Explorer 图、自己的 adb
驱动、自己的 accessibility helper APK、自己的完成判定。它不调用任何 OmniFlow API，所以
它是 external harness，不是同 harness 的复用变体。

接入形态：`src/integrations/artemis_harness.py::factory`，走已有的
`--harness module:factory` 缝，主线代码零改动。

```bash
export OMNIFLOW_ARTEMIS_ROOT=/abs/path/to/artemis
bash scripts/exp/run_androidworld.sh run --method omniflow \
  --harness src.integrations.artemis_harness:factory \
  --task <TASK> --device standard45562
```

适配器做的几件必须做的事：

- `--standalone` 必给。不给的话 ARTEMIS 会把任务提交给常驻 daemon 排队，设备就脱离本
  episode 的控制了；一个 episode 必须独占一个进程。
- 每个子 agent 节点都钉到正式 OpenAI 兼容 endpoint（`FORMAL_MODEL`、
  `FORMAL_MODEL_BASE_URL`），并清空 `GOOGLE_API_KEY` / `GEMINI_API_KEY`，避免某个
  sub-agent 悄悄回落到 Gemini 让这条行不再是同模型对比。
- 录制点必须存在；缺失时返回 `artemis_device_serial_required` / `artemis_root_required`
  这类 episode 失败，而不是中断整批运行。
- ARTEMIS 自己的 `task_status` 只写进 `artemis_task_status`，**不写进 `done_reason`**。
  episode 包装层只对 `done_reason == "finished"` 发出官方完成信号，而 validator 需要那个
  信号；把 ARTEMIS 的判定塞进 `done_reason` 会让改法悄悄改变成功定义。

**定位：只做附录，不进主表。** 三条理由：

1. ARTEMIS 公布的 99%+ 是它自己 Checker 的 exit review，不是 AndroidWorld 官方
   validator。两个数不能相减。
2. 它没有 Function、Transfer、复用率这些产物，所以除了任务成功率，任何 OmniFlow 指标
   都无从对照。
3. 每个 episode 要多维护一个第三方应用栈 + 它的 helper APK + 它的配置漂移，成本远高于
   换来的一条成功率数字。

附录里的写法：报官方 validator 结果，并明确"该数字与主表不可相减，完成判定不同"。

2026-11-08 已完成 4090 AVD 预验证。`emulator-45562` 上的 ARTEMIS helper v6
以 protocol v2 建立连接，随后通过 OmniFlow 唯一入口完成 `SystemBluetoothTurnOn`
fixed replay 的 5 个 OOB 动作，官方 validator 为 0/1。OOB episode 收尾会停用 helper，
因此外部进程结束后已显式卸载 helper 并停止该 AVD。ARTEMIS Pro 通过 DashScope
`qwen3.6-plus` 产生了多次 200 响应，最后在 Hopper 的 JSON 输出请求处收到兼容性错误，
没有生成官方成功结果。adapter 已同时传入 `--test-name` 和 `ARTEMIS_TRACES_DIR`，
一次 DashScope trace probe 已把 `data_engine.db` 与 `steps.json` 写入请求的 evidence 目录。

ARTEMIS 的物理真机验收仍待验证。DashScope structured output 在 ARTEMIS 外部 checkout
中需要把非 Google endpoint 的 `with_structured_output` 默认方法设为 `function_calling`。
4090 临时 checkout 应用这一补丁后，Pro 流程完成了自己的任务循环和成功收尾，trace 也已
写入 episode evidence。该补丁尚未写入 OmniFlow 主线或官方 ARTEMIS 发布件，部署时必须
显式使用已打补丁的外部 checkout。附录表格只能引用 AndroidWorld 官方 validator 结果，
并保留 external harness 的完成判定边界。
