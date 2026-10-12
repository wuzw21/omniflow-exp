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
| t3a_hint | 演示步骤文本放进 prompt | `--method t3a_hint`（官方 T3A agent，见下方注） |
| fixed_replay | 原坐标逐步重放 | `--method fixed_replay` |
| MobileGPT-style | 子任务 Function + 属性匹配重定位 | `--method omniflow --baseline mobilegpt_style --memory <store.json>` |
| AppAgent-style | 元素文档注入 Planner prompt | `--method omniflow --baseline appagent_style --memory <store.json> --element-docs <appagent>/apps/<app>/demo_docs` |
| OmniFlow | Function + OmniTransfer + Checker | `--method omniflow --memory <store.json>` |

注：t3a_hint 目前仍走 AndroidWorld 官方 T3A agent，而非 OmniFlow Planner，严格说不是同
harness；论文里应如实标注，或后续改为 OmniFlow Planner + 演示文本。

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

## 运行与归档

`--baseline` 只允许 `--method omniflow`、builtin harness、提供 `--memory` 且 oracle 关闭。
结果按 `archive_kind=baseline_<name>` 归档到 `.archive/<task>/omniflow/<device>/`，不会覆盖
OmniFlow 的正式结果槽位。

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

## Google ARTEMIS / 外部 harness

ARTEMIS（公开信息来自博客，未读到源码，以下需核实）是基于 adb 的 Android agent，带 MCP server，
Apache-2.0。它有两种用法：作为一个额外的外部基线（用它自己的 agent 跑同一批任务，validator
仍在 episode 后调用一次），或作为宿主通过 MCP 调用 OmniFlow 的 Function 工具，走
`docs/HARNESS_MCP.md` 已有的外部 harness 路径（`--harness module:factory`，归档为
`external_harness`）。在核实其接口前不写适配代码。
