# OmniFlow 项目 Report

Status: project report draft
Last updated: 2026-06-10

## 1. 执行摘要

OmniFlow 是一个面向 GUI Agent 的 cache system。它解决的问题是：AI
助手在手机或桌面应用里做重复任务时，通常每次都要重新观察界面、重新规划、
重新调用模型。OmniFlow 把一次成功执行产生的 `run_log` 整理成可复用的
`Function` asset，让后续相似任务可以 cache-first 命中并直接执行已经验证过
的 GUI 操作片段。

OmniFlow 的目标不是替代上层 AI 助手，也不是做新的 foundation planner。它更像
是给 GUI Agent 增加一层可复用的操作经验缓存：命中时跳过命中段的在线 reasoning，
未命中时交回原有 agent / planner 路径。执行阶段仍保留页面起点检查、route-only
bridge、action transfer、terminal / effect verify 和统一 run event stream，因此
它是 control-aware execution，不是裸录屏重放。

当前项目已经完成核心 replay runtime、Function 资产化链路、privacy-safe
`UDEG node` 表示、run_log 写回、多用户复用和 OpenOmniBot / MCP / HTTP 接入边界。论文
主线正在准备投稿；代码仓库、源码包和论文材料已经整理完成。

## 2. 项目背景

GUI Agent 能通过视觉、无障碍树或 UI 自动化接口操作现有应用，但在真实应用中会遇到
几个重复问题：

- 相同或相似任务会被反复规划，造成时间和模型 token 成本浪费。
- 直接坐标 replay 很脆弱，页面、布局、弹窗和设备差异都会导致错误点击。
- 保存 raw XML、截图或用户文本会带来隐私和共享边界问题。
- 单步 action 是否下发成功不等于任务真的完成，仍需要业务效果验证。

OmniFlow 针对这些问题，把成功轨迹沉淀为可复用的 `Function`，并用结构化页面状态、
受控执行和结果验证来提高 replay 的稳定性。

## 3. 项目目标

本项目的目标可以概括为四点：

1. 将成功 GUI 执行从一次性日志转成可复用 `Function` asset。
2. 在后续相似任务中优先使用 cache hit，减少重复在线 reasoning。
3. 用 `UDEG node` 和 page vector 支持隐私安全的状态匹配与共享。
4. 在 replay 中保留 start-node precondition、route-only bridge、action
   transfer 和 terminal / effect verify，保证命中执行不是盲目重放。

非目标：

- 不声称 OmniFlow 能替代所有 GUI Agent 或 planner。
- 不声称所有任务都能零成本完成。
- 不把 raw XML / screenshot 作为长期共享主对象。
- 不把尚未冻结的大规模跨设备结果写成最终结论。

## 4. 核心方案

OmniFlow 的核心对象如下：

| 对象 | 外行解释 | 系统角色 |
|---|---|---|
| `Function` / `function asset` | 一段可复用的操作能力 | cache 的主对象，命中后可执行 |
| `UDEG node` | 一个隐私安全的页面状态节点 | 用于判断当前页面是否适合复用历史操作 |
| `run_log -> function asset` | 成功经验变成能力 | 将真实执行日志写回为可复用资产 |
| `cache-first` | 先查缓存再规划 | 命中后跳过命中段在线 reasoning |
| `control-aware execution` | 带检查和验收的执行 | 防止 wrong-start、错误点击和伪成功 |

系统运行流程：

1. 用户或宿主 agent 完成一次 GUI 任务。
2. OmniFlow 记录 canonical `run_log`。
3. `run_log` 被导入并整理成 `Function` asset 和相关 `UDEG node`。
4. 后续相似 goal 到来时，OmniFlow 先做 Function lookup。
5. 如果 cache hit，系统检查当前 `UDEG node` 是否满足 `Function.start_node_id`。
6. 如果起点不一致，只允许通过 route-only navigation / goto 边桥接到正确起点。
7. 执行目标 `Function`，并做 action transfer 与 terminal / effect verify。
8. 成功、失败、fallback 和事件流都写入统一 run event stream，供调试和论文证据使用。

## 5. 当前交付物

代码与源码包：

- OmniFlow 代码仓库: https://github.com/omnimind-ai/OmniFlow
- 当前整理分支: `codex/omniflow-open-source-slim`
- 当前已推送 commit: `52e13bf8` (`Slim OmniFlow open-source runtime`)
- 本地干净 clone: `/Users/wuzewen/Projects/Omni/clone/OmniFlow`
- 源码包: `/Users/wuzewen/Projects/Omni/omniflow-source-codex-open-source-slim-52e13bf8-20260610.zip`
- 源码包 SHA-256: `/Users/wuzewen/Projects/Omni/omniflow-source-codex-open-source-slim-52e13bf8-20260610.zip.sha256`

论文与材料：

- 论文 PDF: `UTG_papers/nsdi/omniflow.pdf`
- Overleaf 源文件: `UTG_papers/nsdi/overleaf/`
- 论文 claim 真源: `UTG_papers/nsdi/core_claims.md`
- 评测矩阵真源: `UTG_papers/nsdi/eval_matrix.md`
- 对外说明文档: `OMNIFLOW_OOB_OPEN_SOURCE_NOTE_CN.md`

OpenOmniBot 相关：

- OpenOmniBot 公开仓库: https://github.com/omnimind-ai/OpenOmniBot
- 最新 release: `v0.5.4`
- Stars: GitHub 当前显示约 `1.7k` stars（2026-06-10 精确值为 `1,694`）
- 接入状态: OpenOmniBot native 执行主线正在接入 OmniFlow 的 Function asset、run_log
  ingest/import 和 MCP / HTTP 工具面。

## 6. 已冻结证据与当前结果

以下结果来自当前 `UTG_papers/nsdi/eval_matrix.md` 登记的评测边界。

| 维度 | 当前证据 | 边界 |
|---|---|---|
| End-to-end replay | 当前 strict AndroidWorld latest index: OmniFlow native 29/33 official-validator success, replay rows 0 online tokens/model calls | 旧 accepted-30 30/30 是 archived preserved set，不进入当前 paper-facing 主结果 |
| 表示与召回 | cross-device reviewed subset 中 OmniFlow UDEG/PageVector top-1 98.63%，top-3 100.0%；screen-only VL baseline top-1 69.44%，top-3 87.50% | 当前不是 30 goals x 3 devices headline |
| 编码成本 | OmniFlow avg encode 4.13 ms/item，screen-only VL baseline avg encode 188.69 ms/item；报告 speedup 45.73x，token reduction 267,254 total tokens | screen-only total encode time按登记口径保留 |
| run_log 资产化 | asset-yield compression 7x (7 -> 1)，slots=2，cache_hit=True | 只报告已登记的可运行聚合指标 |
| Function transfer | bounded live 覆盖 16 safe Contacts cases，positive action success=93.3%，negative false positive=1/1 | 不能写成大规模 live success |
| 多用户复用 | cross-user HIT=100%，isolation=100%，token reduction 88.1%，58 multi_user tests pass | 不写成生产级网络效应 |
| Boundary ablation | wrong-start semantic tail 底层 action 可返回 true，但 terminal verify 失败；route-only bridge tests 由 runner 验证 | 证明需要 start-node precondition、route-only bridge 和 terminal / effect verify |
| Case study | 当前真实 nonzero event stream 只观察到 cache_hit | recovery / fallback 三 regime 图尚未冻结 |

这些证据支撑当前项目主张：OmniFlow 能把成功 GUI 执行转成可复用 `Function`，
并在受控验证下减少重复在线 reasoning。仍未冻结的跨设备大规模 headline、生产级
OpenOmniBot trace 和完整 recovery / fallback event-stream 结果，不应被提前写成最终结论。

## 7. OpenOmniBot 接入说明

OpenOmniBot 是一个公开的移动端开源助手项目，已有约 1.7k GitHub stars。当前 OpenOmniBot native
主线负责设备侧执行、Project / workbench、MethodChannel 和内部 run log；OmniFlow
负责资产层和工具面：

- `GET /functions`: OpenOmniBot 获取 OmniFlow Function 列表。
- `POST /run_logs/ingest`: OpenOmniBot 将 canonical run log 回写给 OmniFlow。
- `POST /mcp`: 暴露 `omniflow.recall` 工具契约。

这个接入方向的意义是：OpenOmniBot 可以作为真实移动端开源助手入口，OmniFlow 可以作为其背后
的 replay cache / Function runtime。OpenOmniBot 负责执行体验和端侧闭环，OmniFlow 负责将
成功执行沉淀为可复用资产。

## 8. 项目价值

对普通用户：

- AI 助手做重复任务更快。
- 已经成功过的流程可以复用。
- 失败时仍能回到原有 agent / planner 路径。

对开发者：

- 不需要 App 开发者提前提供正式 API，成功 GUI 轨迹可以生成 tool-like API surface。
- Function asset 可以被测试、导入、导出和多用户共享。
- MCP / HTTP 工具面让外部 agent 能调用 OmniFlow 的 recall 和资产能力。

对论文：

- 论文主线是 GUI Agent cache system with co-evolution through trajectory reuse。
- 贡献点集中在 `Function`、`UDEG node`、`run_log -> function asset`、
  cache-first 和 control-aware execution。
- 评测重点是端到端 replay、表示召回、run_log 资产化、多用户复用和执行边界消融。

## 9. 当前限制与风险

当前 report 保持以下边界：

- OmniFlow 不替代宿主 agent；miss 仍交回现有 agent / planner。
- 当前不能声称所有任务全流程零 token 或零成本。
- 当前不能声称大规模跨设备 live headline 已经完全冻结。
- 当前不能声称 OpenOmniBot 生产级接入已经完成；状态是正在接入。
- 当前不能把 OpenOmniBot stars、release 或外部仓库状态写成论文实验结果。
- 当前不能把 raw XML / screenshot 长期共享写成系统主线。

## 10. 下一步计划

1. 完成 OpenOmniBot native 与 OmniFlow Function asset / run_log ingest 的真实端到端 trace。
2. 补齐更多 device / app / task sweep，并只在 runner 输出冻结后进入论文。
3. 继续整理源码包、论文 PDF、Overleaf zip 和 artifact appendix。
4. 将 recovery / fallback event-stream case study 补齐为完整三 regime 证据。
5. 完成论文投稿前的 claim / eval / figure 一致性检查。

## 11. 对外一句话

OmniFlow is a GUI Agent cache system with co-evolution through trajectory
reuse. It turns successful GUI run logs into reusable Function assets, enabling
future agents to execute matched task segments through cache-first and
control-aware replay instead of repeatedly reasoning from scratch.
