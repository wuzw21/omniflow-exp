# Rebuttal 可直接引用的冻结数据证据（2026-11-10）

数据源：`../AndroidWorld-Experiments/omniflow_unified_results.xlsx`（唯一权威结果，`Cells` 1740 行）。
只收录**不需要新实验、可从 9/3 冻结数据直接统计**的数字，用于 MobiCom'27 rebuttal 回复写作。
新实验结果一律写 `runs/`，不回填本文件、不改动冻结数据。

## 1. 结构重放成功但 validator 失败（Reviewer C，承诺 5 的语义失败分析）

| 项 | 数值 |
|---|---|
| OmniFlow 348 格 | PASS 185 / FAIL 150 / PLANNER_TIMEOUT 11 / NOT_REACHED 1 / MAX_STEPS 1 |
| validator coverage | 347/348 |
| `outcome_class=official_validator_failure` | 150 格 |
| 其中 `replay_status=succeeded` | **93 / 150 = 62.0%** |
| 其余 | replay failed 49、not_started 8 |

精确复现 `rebuttal.md:183` 的观察（"Most failed episodes complete their selected function without action-level failure but fail task validation"）。
可直接写：150 个到达 validator 却失败的 episode 中，93 个（62.0%）所选 Function 的动作重放全部成功，失败发生在任务语义校验而非动作级重放——结构正确性不等于语义正确性。

边界：`replay_status` 是 episode 级汇总，不能拆成 "state matching 错" 还是 "action transfer 错"，更不能当作 OmniTransfer 的 false-positive/negative 率；承诺 5 的人工标注仍需 C 层做。

## 2. Fold 低成功率归因（Reviewer E: 251、348）

| device | PASS/116 | SR | replay/动作重放失败 | not_started | planner_timeout | validator 失败中 replay 成功 |
|---|---|---|---|---|---|---|
| standard | 63 | 54.3% | 26 | 12 | 4 | 34/49 (69.4%) |
| fold | 57 | 49.1% | 32 | 12 | 3 | 33/54 (61.1%) |
| tablet | 65 | 56.0% | 34 | 15 | 4 | 26/47 (55.3%) |

- Fold 比 standard 少 6 个 PASS、比 tablet 少 8；18 个任务 "fold FAIL 但 tablet PASS"，反向仅 10 个。
- 掉分任务集中在 `Markor*` / `Recipe*WithNoise` / `SimpleCalendar*` / `Contacts*` 等需要精确元素定位与多步状态的任务。
- 代理指标：Fold 的**动作重放失败从 26 增至 32（+6）**，检索未命中不变（12），planner timeout 反而最少（3）。
  → 可写：差异主要出现在动作重放与元素定位（re-grounding）环节，而非 host agent 的推理超时；折叠屏布局与多窗口变化更易让录制轨迹的元素坐标/层级失配。

边界：`replay_status` 只是代理，**不能精确三分** "GUI-state retrieval / action grounding / agent reasoning"（Reviewer E 的原问法）。要精确回答需 B 层 per-invocation 机制统计（`tools/rebuttal_mechanism_stats.py`）与 `not_triggered` 标注；不要写成直接测量。

## 3. 53.2% 的代表性（Reviewer E: 252、352）

| difficulty | scheduled | OmniFlow | Zero-Shot | AppAgent | MobileGPT | AutoDroid |
|---|---|---|---|---|---|---|
| Easy | 183 | 129 (70.5%) | 107 (58.5%) | 53 (29.0%) | 52 (28.4%) | 47 (25.7%) |
| Medium | 108 | 44 (40.7%) | 21 (19.4%) | 15 (13.9%) | 8 (7.4%) | 2 (1.9%) |
| Hard | 57 | 12 (21.1%) | 1 (1.8%) | 3 (5.3%) | 0 | 0 |

- SR 随难度单调下降 70.5% → 40.7% → 21.1%；Medium+Hard 占 165/348 = 47.4% 的格子，是拉低总体 SR 的主因，这些任务多为长时序、精确状态与参数约束。
- 相对增益在各难度都成立：对 Zero-Shot +12.0 / +21.3 / +19.3 pp；在 Medium/Hard 上相对其他基线接近"从 0 到可用"。
  → 可写：53.2% 是 AndroidWorld 任务难度分布的结果而非系统瓶颈，memory reuse 的收益在难任务上更显著，因此不影响核心结论。

边界（**必须写进回复**）：论文 53.2%/37.1% 是带官方 completion checker 的 oracle 条件；`BASELINES_SAME_HARNESS.md` 要求 `--completion-oracle off`。上表是冻结数据的**历史口径**，不能作为同 harness 结论；同口径数字必须由 A 层 `--baseline all` 重跑产出。

## 4. 全部 attempts 的开销（Reviewer C: 184，承诺 6）

| 组 | 格数 | mean total_tokens | mean model_calls | mean execution_sec | mean fallback_steps |
|---|---|---|---|---|---|
| PASS | 185 | 25,592 | 6.1 | 78.9 | 0.4 |
| 非 PASS（含失败/超时/恢复） | 162* | 52,177 | 10.0 | 139.2 | 0.8 |
| 全部 348 | 347* | 38,003 | 7.9 | 107.1 | 0.6 |

\* 有 1 格缺 timing/token，按非空计。

- 失败尝试的平均 token 是成功尝试的 **2.04×**、model calls 1.64×、耗时 1.76× → 直接回应 Reviewer C 的 "Failed attempts and recovery can materially affect the benefit experienced by users"。
- 348 次 attempts 总 token 13,187,114；三设备均值 standard 30,703 / fold 38,235 / tablet 45,074 tokens。

边界：冻结数据**只有 OmniFlow 侧逐格 token**，没有同 harness 基线侧的逐格开销，**不能从本表算出 token 节省比例**（-50.8% 仍是"双方均成功的 83 对"口径）。承诺 6 的"全 attempts 节省"必须靠 A 层 `--baseline all` + OFF/ON 配对重跑（v2 设计 §3/§6）。本表只能报绝对开销分布。

## 5. 纯写作项（无数据需求）

Reviewer E 的 "Table 1 三属性同时实现的挑战"（`rebuttal.md:250/344`）、A 的 OmniTransfer 训练细节与 XML baseline（`rebuttal.md:370–371`，注意 418 apps / K=8 vs 16 / NULL class 三处口径仍待拍板）、`rebuttal.md:373` 的 partial memory 承诺——都不用跑，直接写。

## 6. 数据答不了、必须去 4090 的

| rebuttal 承诺 | 缺什么 | v2 设计位置 |
|---|---|---|
| 1 state-matching / action-transfer 与 prior methods 对比 | 同 harness 7 行基线真实结果 | A 层 §3 |
| 2 工作流中断鲁棒性 | app 更新 / 重复弹窗 / 登录中断构造 | C 层 §5 |
| 3 真机 full-task success / latency / energy | 真机配对测量、device identity | C 层 §5 |
| 4 Function 分解 / 中间态重放消融、阈值敏感性 | L3 缺档、no_reentry/no_checker、threshold/page sweep | B 层 §4 |
| 5 false-positive/negative 人工量化 | PAPT 128 组 reviewed label | C 层 §5 |
| 6 全 attempts token 节省 | 配对基线侧逐格 token | A 层 §3 |

---

出处与口径：`Cells` 表 `final_20260903_frozen`（OmniFlow 348 格，185 PASS，validator coverage 347/348）、`zeroshot_recovered`、`matrix_20260903`。难度统一为冻结分层 Easy/Medium/Hard（61/36/19 tasks）。三套 SR（163/329、185/348、208/348）与 oracle 警告见 `omniflow_unified_results.xlsx` 的 `README` / `Excluded` sheet。补实验默认约 55 个 canonical asset 任务，不能写成 116 全量。
