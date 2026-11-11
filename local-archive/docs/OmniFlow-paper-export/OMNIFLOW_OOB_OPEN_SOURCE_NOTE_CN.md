# OmniFlow / OOB 开源与论文说明

Status: draft release note
Last updated: 2026-06-10

## 核心说明

我们已经完成了一个鲁棒的 GUI 重放系统。OmniFlow 将成功执行产生的
`run_log` 写回为可复用的 `Function` asset，并在后续任务中通过
cache-first lookup 命中可执行片段。命中执行仍受 `start_node_id`
前置条件、route-only bridge、action transfer 和 terminal / effect
verify 约束，因此它是 control-aware execution，不是裸 replay。

当前论文主线正在准备投稿。论文、Overleaf 源文件、图和评测说明都保留在
本仓库的 NSDI 主线目录中：

- 论文 PDF: `UTG_papers/nsdi/omniflow.pdf`
- Overleaf 源文件: `UTG_papers/nsdi/overleaf/`
- 论文 claim 真源: `UTG_papers/nsdi/core_claims.md`
- 评测矩阵真源: `UTG_papers/nsdi/eval_matrix.md`

## OmniFlow 代码仓库

OmniFlow 的代码仓库入口：

- GitHub: https://github.com/omnimind-ai/OmniFlow
- 当前开源整理分支: `codex/omniflow-open-source-slim`
- 当前已推送 commit: `52e13bf8` (`Slim OmniFlow open-source runtime`)
- 本地干净 clone: `/Users/wuzewen/Projects/Omni/clone/OmniFlow`

本地 clone 命令：

```bash
git clone --branch codex/omniflow-open-source-slim \
  https://github.com/omnimind-ai/OmniFlow.git \
  /Users/wuzewen/Projects/Omni/clone/OmniFlow
```

这个代码仓库包含 OmniFlow 开源框架、核心 runtime、tests、demo、文档和论文
相关材料。当前 paper-only export 仓库只保存论文材料；代码实现以 OmniFlow
代码仓库为准。

## OOB 开源助手

OOB 的公开仓库入口：

- GitHub: https://github.com/omnimind-ai/OpenOmniBot
- 默认分支: `main`
- 最新 release: `v0.5.4`
- Stars: GitHub 当前显示约 `1.7k` stars（2026-06-10 精确值为 `1,694`）

OOB 是一个已有开源用户基础的移动端开源助手。OOB native 执行主线正在接入
OmniFlow：OOB 负责原生执行、Project / workbench、MethodChannel 和内部
run log；OmniFlow 提供 `Function` asset、run_log ingest/import、多端协作
和 MCP / HTTP 工具面。

当前接入边界：

- `GET /functions`: OOB 获取 OmniFlow Function 列表
- `POST /run_logs/ingest`: OOB 将 canonical run log 回写给 OmniFlow
- `POST /mcp`: 暴露 `omniflow.recall` 工具契约

## 对外表述建议

OmniFlow is a GUI Agent cache system with co-evolution through trajectory
reuse. We have built a robust replay system that turns successful GUI run logs
into reusable Function assets, enabling cache-first execution under
control-aware verification. The paper is being prepared for submission, and
the open-source framework, code, and paper artifacts are organized in the
OmniFlow repositories. OpenOmniBot (OOB), a public open-source assistant with
about 1.7k GitHub stars, is being connected to OmniFlow through Function
assets, canonical run-log ingest, and the MCP / HTTP tool surface.

## 维护边界

- 论文正文、图和表只能使用 `UTG_papers/nsdi/core_claims.md` 与
  `UTG_papers/nsdi/eval_matrix.md` 中已经冻结的主张和数字。
- OOB 只作为接入环境与外部工具面说明，不作为论文核心概念。
- 代码实现以 OmniFlow 代码仓库为准；本仓库是 paper-only export。
- 若 OOB 接入状态、release 版本或 stars 数变化，更新本说明文档即可；不要把
  这些外部项写成论文实验结果。
