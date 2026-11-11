# AndroidWorld 补实验设计 v2（2026-11-10）

以论文定稿 `../mobicom27_gui_agent.pdf` 与 `../rebuttal.md` 为唯一核心，替换"一张 116×15 大表跑到底"的旧思路。
新方案把实验拆成 **L0 前置 + A/B/C 三层**，每层只回答对应的审稿问题；所有产出写 `new_exp/runs/`，不回填 9/3 冻结结果，不改动 `../src/OmniFlow-exp/` 里的源码与冻结文档。

## 0. 旧设计为什么要改

| 问题 | 出处 | 后果 |
|---|---|---|
| 任务集三层混用 | `rebuttal_matrix.sh` 默认约 55 个 canonical asset 任务；`EXPERIMENT_SCENARIOS.md` 冻结 5 个探针；论文主表 116 | 补实验数字容易被写成"116 全量"；`PAPER_FREEZE.md` 实际只有 137 completed episodes |
| L3 缺档 | `exp_l3()` 明确无 `--replay-unit trajectory` 开关 | L1→L2→L4 把 granularity / re-grounding / fallback 混在一起，正是 Reviewer B 要求拆开的 |
| oracle 条件不一致 | 源码 2026-10-10 起 `--completion-oracle` 默认 **off**（`run_tasks.py` 1348–1351、`AGENTS.md` 157–163）；论文 53.2%/37.1% 是旧 oracle-on 条件 | 矩阵跑出的新结果本来就是 off，与论文数字不是同一条件，不能直接相减；须显式标注条件，或补一组 `oracle_ablation` 对照 |
| 配对身份不完整 | `EXPERIMENT_SCENARIOS.md` 要求 12 项共享身份（task / real device / params / seed / model+endpoint / source / code / Transfer / protocol / Store / transfer-state catalog / hashes） | 矩阵目前只保证 L1/L4 同 `--memory`，差异可能来自资产而非方法 |
| 单 seed、单重复 | 矩阵每格 1 次、seed 113；只有场景文档有 OFF/ON/ON/OFF 两次配对 | 无方差、无置信区间 |
| 资产同步事故 | 4090 上 `store.json`(5 states) 与 `transfer_states.json`(4 states) 曾不一致，2026-11-08 才同步 | 不先校验 bundle，跑出的差异无法归因 |
| 阈值语义 | `page` 门才是真正变化的门（cache confidence 在 admission 时 ≈1.0）；operating-point coverage 只是 policy simulation | 没有 reviewed label 就不能写成 false-accept/false-reject 率 |
| 缺层 | 真机 latency/energy、工作流中断、partial memory、PAPT/OmniTransfer 人工标注、AutoDroid-V2、全 attempts token | 对应 rebuttal 承诺 2/3/5/6 与 369 行基线承诺，矩阵里没有 |

## 1. 三层任务集（先固定身份，再跑）

| 代号 | 任务 | 用途 | 能否写 SR |
|---|---|---|---|
| `S5` | 冻结的 5 个探针：`CameraTakePhoto`、`RecipeAddSingleRecipe`、`SimpleSmsSendClipboardContent`、`RecipeAddMultipleRecipes`、`RecipeDeleteDuplicateRecipes` | 机制/场景验证、中断触发 | 否，只报机制证据 |
| `S55` | `rebuttal_matrix.sh list` 解析到的 canonical asset 任务（约 55） | 消融、阈值、同 harness 主表 | 可，但必须标注 "canonical asset 子集" |
| `S116` | 全 116 | 论文级 SR | 仅在 oracle off 同 harness 全量重跑后才可 |

先跑 `list` 把 S55 名单与数量存档；**禁止看到结果后换任务或换历史 Memory**（场景文档 54–56 行）。S5 的资产资格问题（monolithic three-recipe Function、static replay）仍未清，只能当历史证据，需 `convert-memory` 重授权成新 bundle 并冻结新 batch identity 后才能用于机制结论。

## 2. L0 前置（不通过不开工）

1. canonical Memory 同步：`store.json` + 同级 `transfer_states.json` 作为一个显式 bundle，记录两个 SHA-256（`EXPERIMENT_SCENARIOS.md` 228–235）。
2. `--completion-oracle off` 已是默认，仍要在 launcher 公共参数里**显式固定**（防止环境变量污染）；`on` 只作 `oracle_ablation` 独立归档，不进正式结果表。官方 validator 只在 episode 结束后调一次。
3. 固定 Qwen3.6-Plus 与 endpoint、Planner、OOB Host、步数预算、代码 hash、Transfer 协议与 V10 checkpoint。
4. 每格把 12 项配对身份写进 RunLog 侧车；`--function-reentry` / `--checker` 的默认开态也要记录。
5. 先跑 `../src/OmniFlow-exp/tools/rebuttal_artefact_audit.py` 审计哪些 artefact 真能支撑 rebuttal，缺失的不排跑。

## 3. A 层：同 harness 主表（承诺 1、6；A/B/C/D）

以 `BASELINES_SAME_HARNESS.md` 的 7 行为骨架，全部共享同一 Planner / Host / 模型 / 步数预算，唯一差别是"复用什么、怎么复用"：

`zero-shot` / `demo_hint` / `fixed_replay` / `MobileGPT-style` / `AppAgent-style` / `MobiMem-style(AgentRR)` / `OmniFlow`

- 规模：S55 × 3 设备 × 2 重复 ≈ 2310 格；用 `--baseline all`，每个条件写独立结果槽位。
- **AutoDroid-V2**：rebuttal 369 行承诺加入，但同 harness 注册表里尚未实现 → 要么先实现，要么在回复里明确声明未覆盖，不能留空行。
- 指标：official success、validator coverage、model calls、prompt/completion/total tokens、`execution_duration_ms`、fallback steps、reuse rate。
- **token 节省按全部 attempts 统计**（含失败与恢复），不再只取双方成功的 83 对（-50.8%）。
- 原版 MobileGPT/AppAgent 只在 10–20 个任务上做保真度检查，放附录，不进主表。

## 4. B 层：阶梯与消融（承诺 4；B/C/D）

| 组 | 条件 | 约束 |
|---|---|---|
| 阶梯 | `L1 memory_off` / `L2 fixed_replay` / `L4 omniflow` | 三者共享同一 `--memory`（`rebuttal_matrix.sh` 227–228 行）；S55 × 3 设备 |
| 组件 | `no_reentry`、`no_checker`、`no_reentry_no_checker` | 只在对应 trigger 真实发生时才算机制证据；未触发记 `not_triggered` |
| 配对重复 | OFF1 / ON1 / OFF2 / ON2 | 两次配对，按 task 聚类 bootstrap 给 CI |
| 阈值 | `threshold` {0.80,0.90,0.95,0.99} + `page` {0.60,0.70,0.80,0.90} | deployed 0.95/0.80 与 sweep 分开报；page 门为主要变量，每次记录有效值到 RunLog |
| 部分记忆 | 删 first/last 25%/50%、随机删 1 个 GUI node（`rebuttal.md` 373 行） | SR / reuse / model calls 随 source coverage 变化 |
| 参数 | `source_parameter_seed` 111 | 只隔离参数生成，不能标成 same-device exact replay |

## 5. C 层：专项证据（承诺 2、3、5；A/C/D/E）

- **真机**：full-task success、paired latency、energy、action reuse、fallback、内存增长；记录 device identity、APK/version hash、操作与结果。`pending_physical_device` 不能写成 passed，emulator（如 4090 AVD）不能冒充真机。
- **中断**：app 版本漂移、重复弹窗、登录/加载中断、网络间歇；记录 trigger 是否发生、Checker 动作、resume index、recovered success。
- **OmniTransfer 人工验证**：PAPT 128 组相似页面（top 15 apps）+ ASE 配对，产出 reviewed correspondence label → false-positive/negative 率；与 policy simulation 的 operating-point coverage 分开报（`EXPERIMENT_SCENARIOS.md` 148–155 行）。
- **语义失败**：Function 成功但 official validator 失败的行单独成表，支撑"结构重放 ≠ 语义完成"与 62% 分析。
- **写作型结论**（不用跑）：Fold 低成功率归因、53% 代表性、failure taxonomy、Table 1 三属性挑战、OmniTransfer 训练细节，直接从冻结数据与 `omniflow_unified_results.xlsx` 统计。

## 6. 指标与统计验收

- 每张表都报 planned / completed / failed / unreached / missing / not_triggered / pending_physical_device。
- paired full system、paired fast path、paired recovered 分开报；不能只看最后一个 Function 成功就判 fast path。
- success latency 用 `execution_duration_ms`，缺 timing 记 unknown 而非 0。
- 用 `src.experiment.performance_metrics.summarize_paired_experiments` + 冻结 expected pair list，12 项身份齐全才配对。
- 禁止：policy simulation 当误接受率、55 子集写成 116、不同 oracle 条件的 SR 相减、把环境失败（如 403 / stale-state）记成方法失败。

## 7. L3 决策点（需拍板）

`exp_l3()` 明确禁止伪造这一格，二选一：

- **(a)** 实现 `--replay-unit trajectory`（约一天）→ 阶梯变四档，可拆开 granularity / re-grounding / fallback；
- **(b)** 报告三档阶梯，并在 rebuttal 里明确承认该 confound。

建议先按 (b) 提交，时间允许再补 (a)。

## 8. 执行顺序

1. `rebuttal_matrix.sh list` + artefact audit → 固定 S55 名单与可用资产；
2. L0：bundle hash、oracle off、12 项身份；
3. A 层 `--baseline all`（含 AutoDroid-V2 的实现/声明决策）；
4. B 层 ladder → 组件消融 → 配对重复 → 阈值 sweep → partial memory；
5. C 层 真机 + 中断 + 人工标注；
6. 汇总：全 attempts token、置信区间、failure taxonomy、E 的两项纯分析。

## 9. 输出与命名

4090 上写 `data/androidworld/.archive/rebuttal-<date>/<layer>/<label>/`，同步到 `new_exp/runs/<date>-<name>/`。
每格保留 RunLog / RunLog SHA-256、Result Evidence / SHA-256、condition 标签、12 项身份与 `label_status`。
不覆盖 `../AndroidWorld-Experiments/omniflow_unified_results.xlsx`（那是 9/3 冻结数据的统一视图），不回填冻结论文结果。
