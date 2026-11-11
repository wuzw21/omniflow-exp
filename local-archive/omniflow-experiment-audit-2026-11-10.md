# OmniFlow 实验梳理：需要哪些东西 / 哪些没测

整理日期：2026-11-10。范围：`/Users/zewenwu/Projects/OmniFlow`（只读索引仓库，本地不跑测试，评测在 CVM/9207）。

## 0. 先说结论

仓库里实际有**两条互不相关的实验线**，容易混淆：

1. **AndroidWorld 主线**（`AndroidWorld-Experiments/`）：OmniFlow 方法与基线的 116 任务对比矩阵，最后更新 2026-09-03。这是通常说的"OmniFlow 实验"。
2. **Agent 迁移新方向**（`omni-device-agent/`，`docs/HANDOFF.md` 2026-11-02）：研究草案，尚无正式运行结果。

按当前工作簿（比旧的 md 审计更新）看，AndroidWorld 主线**主矩阵基本跑满了**，真正的缺口集中在四类：AppAgent 的 27 格、AutoDroid 的 seed 114/115、Zero-Shot 基线的 81 格、以及 15cell 表里 696 个全空的旧基线列。更大的问题不是"没跑的量"，而是**口径不统一、缺重复实验/消融、证据链不完整**。

一个重要纠正：`HISTORY_RECONCILIATION_20260831.md` 里"MobileGPT 33 格 / AppAgent 310 格待运行"已经过时。9/3 的 12cell 主表里 MobileGPT 348 格全部有正式结论，AppAgent 只剩 27 格无正式结果。不要按旧数字安排重跑。

---

## 1. 完整实验需要哪些东西

### 1.1 固定协议（已有，必须锁定）

| 项 | 当前值 | 出处 |
|---|---|---|
| 任务集 | AndroidWorld 116 tasks | 各工作簿 `116 Tasks` sheet |
| 设备 ×3 | Standard/Pixel 6 Pro (`standard45562`)、Fold (`fold45564`)、Tablet (`tablet45554`) | `new-final-verision.txt` |
| Source | `OmniFlowSourceSmall_seed111`，每 task 一份 official-success RunLog | 各 checkpoint |
| Target seed | evaluation seed 113（AutoDroid 计划 113/114/115） | `AUTODROID_OFFICIAL_ANDROIDWORLD_AUDIT.md` |
| 判据 | 只认 AndroidWorld 官方 validator（`official_validator_used=true`、coverage=1.0、reward=1.0）；自报完成/启动失败不计 PASS | `RESULT_GROUPS.md`、`HISTORY_RECONCILIATION_20260831.md` |
| 执行入口 | `scripts/exp/run_androidworld.sh run --method <...> --device all` | 各审计文档 |
| 规则 | 已写入格不重跑、不覆盖；缺失经验记 unavailable/preflight，不能合成 | `RESULT_GROUPS.md`、`AGENTS.md` |

### 1.2 方法 / 基线集合（当前 12cell 协议）

`OmniFlow × MobileGPT × AppAgent × AutoDroid`，4 方法 × 3 设备 = 12 cells/task，共 1392 格。
`12cell_autodroid.xlsx` 的 `Legacy Baselines` 明确：Fixed Replay / T3A Hint / Script Replay 已从当前矩阵移除。

另需保留的对照：
- **Planner-only Zero-Shot**（`omniflow-zeroshot.xlsx`）：Memory/Function/Transfer = None，116×3=348 格，Max steps=30。
- **旧基线**：Fixed Replay、T3A Hint（15cell 表里保留列）、Script Replay（旧 12cell 变体）——若要和历史结果对比，必须补齐或明确废弃。

### 1.3 指标（部分已有，需统一到一张可比表）

成功率（SR）、validator coverage、model calls、prompt/completion/total tokens、physical actions、Function reuse、fallback steps、checker triggers、wall/execution time、失败分类（official_validator_failure / setup_or_validator_init_failure / planner timeout / max steps / validator not reached）、success-only 统计。

`15cell.xlsx` 的 `Rerun Summary` 是唯一给出冻结版指标的表（"Frozen 2026-09-03"），其余都是分批次统计。

### 1.4 证据链（每个 cell 应具备）

Source RunLog（official success + SHA-256）→ Memory / Function store / compile_report / transfer_states → `run_log.json` + `task_results.jsonl` → 结果 SHA-256 → Run Receipts / 主表 → `readme-list/<Task>/` 的 `README.md`、`summary.csv`、`steps.csv`、`<method>_executions.csv`。

本地只有 xlsx / ndjson / csv / md 索引；原始 runlog、task_results、部分截图在云端（`/home/wuzewen/...`、`/home/zewen/...`）。

### 1.5 运行资产（复跑必备）

`portable/manifest.json`：AndroidWorld revision `632ac959`、OmniFlow commit `b959d3b`、OmniTransfer `point_conditioned_sparse_graph_v10` 权重 SHA、OOB-Experiment.apk SHA、模型 `Qwen3.6-Plus`；AutoDroid 另需 Instructor XL embedding；CVM/9207 与 rsync 通道（`README.md`）。

---

## 2. 已经测了什么

### 2.1 当前权威主表：`OmniFlow_AndroidWorld_116Tasks_12cell_autodroid.xlsx`（9/3 14:04）

逐格解析 `116 Tasks` sheet（116 行 × 12 列）：

| 方法 | Standard | Fold | Tablet | 合计 PASS | 无正式结论 |
|---|---:|---:|---:|---:|---:|
| OmniFlow | 71 | 67 | 70 | **208** | 0 |
| MobileGPT | 22 | 18 | 20 | **60** | 0 |
| AppAgent | 30 | 18 | 23 | **71** | **27** |
| AutoDroid | 19 | 14 | 16 | **49** | 17 格 init-failure |

其余格均为 `official validator failed`（FAIL）。也就是说 1392 格中 **1365 格有 PASS/FAIL 结论，27 格无正式结果**，没有空白"待运行"格。

注意：AutoDroid 的 49 PASS 与 `AUTODROID_OFFICIAL_ANDROIDWORLD_AUDIT.md` 的 49/348 一致；MobileGPT 60 PASS 与 8/31 审计的 60 PASS 一致。

### 2.2 其它已完成的批次

- **OmniFlow 9/3 最终重跑**（`15cell.xlsx` → `Rerun Summary` / `Rerun Cells`）：348 格，185 PASS（**53.16%**），validator coverage 0.9971，1 个 Markor timeout 未达 validator；失败分类 150 validator failure / 11 planner timeout / 1 max steps / 1 未达。
- **OmniFlow 早期归档统计**（`readme-list/omnimind-gpt55-new/all-results/README.md`）：348 cells，329 覆盖（94.54%），163/329 PASS（49.54%），三设备 59/53/51；Function reuse 52.33%，2-call cells 24.14%。
- **AutoDroid seed 113**：348/348 已填，49 PASS，`rerun-20260831/new-final-verision.txt` 结尾"剩余 0 个待运行"。
- **AppAgent / MobileGPT 历史整理**（8/31）：回填 283 格；AppAgent 另有若干任务的官方审计（SystemBluetoothTurnOn 2/3、SystemBluetoothTurnOnVerify 3/3 等）。
- **Zero-Shot**：129 PASS / 138 FAIL / 72 undetermined / 9 Missing RunLog，113 个 tri-device groups，22 组三端全 PASS。
- **GPT-5.5 隔离批次**：仅 SystemBluetoothTurnOn 3 格（3/3 PASS），3 个显式 retry 0/3 成功。
- **旧 10-cell 结果组**（`RESULT_GROUPS.md`）：1160 格中 479 完成、127 通过、681 未完成（历史矩阵，已被新协议排除）。

---

## 3. 没测的 / 缺口清单

### P0 — 主矩阵里明确缺格的

1. **AppAgent 27 格无正式结果**（3 设备 × 9 任务左右）
   - 20 格 `未执行（Memory 转换失败）`：source 无物理 demo，`appagent_official_demo_actions_required`，连续重试后终止，未伪装成 FAIL。
   - 6 格 `按协议跳过`（不适用）、1 格 `未完成（AppAgent 结果证据缺失）`。
   - 涉及任务：`OpenAppTaskEval`、`RetroPlaylistDuration`、`RetroSavePlaylist`、`SimpleCalendarAnyEventsOnDate`、`SimpleCalendarEventsInNextWeek`、`SimpleSmsReplyMostRecent`、`SportsTrackerLongestDistanceActivity`、`TasksIncompleteTasksOnDate`、`VlcCreatePlaylist`、`VlcCreateTwoPlaylists`。
   - 按 `RESULT_GROUPS.md` 规则这些应记为 unavailable/preflight，**要么补齐合法 Memory，要么在论文里明确记为不可用**，不能当失败率分母。

2. **AutoDroid seed 114 / 115：696 格完全没跑**
   - `AUTODROID_OFFICIAL_ANDROIDWORLD_AUDIT.md` 计划 1044 = 116×3×3 seeds，实际 348（seed 113），114/115 两行全是 `—`，全计划完成度 4.69%。
   - 且仓库中**没有说明其它方法是否也要跑 114/115**，多 seed 重复性目前只 AutoDroid 有规划。

3. **15cell 表的旧基线 6 列全空：696 格**
   - `Fixed Replay ×3` 和 `T3A Hint ×3` 在 `15cell.xlsx` 的 `116 Tasks` sheet 里 116 行全为空白。
   - 若论文需要与 replay/hint 类基线比较，这 696 格要么补跑，要么明确宣布废弃（12cell 已将其排除）。

4. **Zero-Shot 基线 81 格无有效结论 + 3 个任务组缺失**
   - 72 `Preflight/undetermined` + 9 `Missing RunLog`；只记录了 113/116 个 tri-device groups。
   - 作为核心消融基线，这个完整度不足以支撑与 OmniFlow 的严格对比。

5. **OmniFlow 冻结版 1 格未达 validator**（Markor timeout，coverage 99.71%）。早期统计另有 19 格无 validator 覆盖。

6. **有记录但结论不可比**：AutoDroid 17 格、AppAgent 2 格为 `official initialization failure`（0 action / 0 call / 0 token），按协议不计 PASS，但也没有有效任务级结论；严格比较时需要说明是否补跑。

### P1 — 可比性与说服力（比"格子数"更关键）

7. **除 AutoDroid 外，所有方法只有 1 个 eval seed、每格只跑 1 次**：没有重复实验、没有方差/置信区间/显著性检验。`omniflow-zeroshot` 也只有 seed 113。
8. **OmniFlow 成功率三套数字不一致，没定唯一权威口径**：
   - 早期归档 163/329（49.54%）
   - 9/3 冻结重跑 185/348（53.16%）
   - 12cell 主表逐格 208/348（59.77%）
   - 三份都是"OmniFlow AndroidWorld 116 tasks"，必须先确定用哪一批、为什么其它两批不作为最终结果。
9. **12cell 与 15cell 都自称 "current"**，且 15cell 保留 6 个全空基线列；仓库里没有宣布切换/废弃关系的文件。
10. **组件级消融缺失**：目前只有"整体 Zero-Shot（Memory/Function/Transfer 全关）"这一档；缺 Memory-only / Function-only / Transfer-only、Transfer admission 阈值（0.800）、max steps、fallback 策略等单因素消融。
11. **跨模型系统性对比缺失**：主模型 Qwen3.6-Plus；GPT-5.5/API 只跑了 1 个任务 3 格 + 3 个 retry（0/3），`multisegment-qwen-4090`、`transfer_algorithm_retry`、`transfer_regression` 都是零散变体，没有 116 task 全量。
12. **效率/成本缺统一对照表**：耗时、calls、tokens、fallback、reuse 只在各自的 README/Rerun Summary 里；没有四方法同口径的表格，也没有能耗、内存峰值、真机/端侧数据。
13. **失败分析未闭环**：Transfer 低置信度 → fallback 膨胀、official validator 的 `forest=None` 边界异常、SMS initializer 找不到 received message、AppAgent/MobileGPT Memory 转换失败等，散落在 checkpoint 里，没有汇总成 failure taxonomy + 占比。

### P2 — 证据链 / 工程

14. 本地不跑测试，原始 `run_log.json` / `task_results.jsonl` / PNG 在 CVM 或另一台机器路径；本地只有索引。AutoDroid per-task CSV 只有 113/116 个任务，约 3 个任务缺本地证据；`AUTODROID_AUDIT.md` 也承认部分截图路径指向临时目录、未保留完整视觉证据包。
15. `outputs/` 下只有 `20260902-omniflow-function-e2e/` 一个目录，且与主目录的 12cell 工作簿不是同一文件（md5 不同），没有其它正式产出归档。
16. 复跑依赖 CVM/9207、`ssh 9207`、四张 4090、`OOB-Experiment.apk`、OmniTransfer 权重；`HANDOFF.md` 记录旧远端曾有 ENOSPC / RAM 目录消失 / cuDNN 失败，开始前需只读复查环境。

### P3 — 新方向（2026-11-02，几乎全部未测）

`omni-device-agent/docs/AGENT_MIGRATION_SPEC.md` + `experiments/agent_migration/PROTOCOL.md` 是**草案，状态"尚未运行"**：

- 已做：仅一个脚本化 engine 的 Harness 语义探针（同一 `move_file` 任务在 CodeAct / ReAct 下最终状态一致，但输出协议、消息角色、engine 参数、控制循环不同），**没有真实模型**。
- 未跑：2×2 因素设计（模型不变/改变 × Harness 不变/改变）；3 类 fixture 任务（`organize_files` / `patch_repo` / `recover_failure`） × 3 条件（`direct-copy` / `manual-port` / `compiled`）；适配集 + 隐藏任务；10 项指标（最终状态正确率、schema 合法率、postcondition、错误恢复、隐藏任务成功率、人工修改行数、延迟/内存/能耗/云端数据暴露）；8 类必须保存的证据。
- 未加入：失败恢复、验证失败、turn limit、真实模型、以及后续因素（同 Harness 换模型、跨模型+跨 Harness、OpenHands SDK、PalmClaw typed tools、不同 engine/hardware）。
- `HANDOFF.md` 明确："当前没有剩余运行中的实验，不要自动重启"；旧账本里仍未测 SVMemAgent、连续摄像头、功耗、真机 Android 闭环、完整模型常驻服务、并发负载、崩溃恢复；Stage A 的 384 道 held-out 未运行。

---

## 4. 建议的优先级

1. **先定口径**：宣布唯一权威表（建议 12cell_autodroid 为当前主表，15cell 仅作历史/冻结重跑存档），统一 OmniFlow SR 到 9/3 冻结版 185/348，并说明 163 与 208 的批次归属。
2. **补齐 AppAgent 27 格**或把它们明确标为 unavailable，不让"缺结果"混入通过率分母。
3. **补 AutoDroid seed 114/115**（或明确声明只用 seed 113 并放弃 3-seed 计划），给出多 seed 方差。
4. **补 Zero-Shot 的 81 格 + 3 个任务组**，并把 Zero-Shot 与 OmniFlow 做成同口径对照。
5. **补组件级消融和跨模型全量对比**（至少 1 个非 Qwen 模型的 116×3）。
6. **整理四方法统一指标表 + failure taxonomy**，把 checkpoint 里的定性诊断变成可引用的定量分析。
7. **同步云端原始证据**，补齐本地缺失的 3 个任务 CSV 和截图清单，确保可离线复核。
8. 新方向 Agent Migration 按 PROTOCOL 从"模型不变 × Harness 改变"一格起步，先证明存在真实迁移损失，再扩展；不要把草案写成已验证结论。

---

## 5. 主要证据文件

- `OmniFlow/README.md` — 本地索引、CVM 工作流
- `OmniFlow/AndroidWorld-Experiments/OmniFlow_AndroidWorld_116Tasks_12cell_autodroid.xlsx(.inspect.ndjson)` — 当前主表（9/3）
- `OmniFlow/AndroidWorld-Experiments/OmniFlow_AndroidWorld_116Tasks_15cell.xlsx(.inspect.ndjson)` — 15cell + `Rerun Summary`/`Rerun Cells`（9/3 冻结）
- `OmniFlow/AndroidWorld-Experiments/omniflow-zeroshot.xlsx(.inspect.ndjson)` — Zero-Shot 基线
- `OmniFlow/AndroidWorld-Experiments/new-final-verision.txt` — 12cell 协议与 checkpoint 046–050
- `OmniFlow/AndroidWorld-Experiments/RESULT_GROUPS.md` — 旧 10-cell 规则（含 baseline 边界）
- `OmniFlow/AndroidWorld-Experiments/readme-list/HISTORY_RECONCILIATION_20260831.md` — 8/31 回填审计（数字已过时）
- `OmniFlow/AndroidWorld-Experiments/readme-list/AUTODROID_OFFICIAL_ANDROIDWORLD_AUDIT.md` — 1044 计划 / seed 114/115 缺口
- `OmniFlow/AndroidWorld-Experiments/readme-list/APPAGENT_OFFICIAL_ANDROIDWORLD_AUDIT.md` — AppAgent Memory 转换失败说明
- `OmniFlow/AndroidWorld-Experiments/readme-list/omnimind-gpt55-new/all-results/README.md` — 早期 348 格统计
- `OmniFlow/OmniFlow-AndroidWorld-Experiments/OmniFlow_AndroidWorld_116Tasks_12cell.xlsx` — 旧 12cell 变体（含 Script Replay）
- `OmniFlow/portable/manifest.json` — 运行时资产与版本
- `OmniFlow/omni-device-agent/docs/HANDOFF.md`、`EXPERIMENT_LEDGER.md`、`AGENT_MIGRATION_SPEC.md`
- `OmniFlow/omni-device-agent/experiments/agent_migration/PROTOCOL.md` — 新方向草案

---

## 6. 清理清单（用户第 1 条决策：旧数据、旧文档可清空）

状态：**尚未执行任何删除**。原因：`/Users/zewenwu/Projects/OmniFlow` 与 `AndroidWorld-Experiments/` 都不是 git 仓库（`git rev-parse` 报 NO GIT REPO），直接 `rm` 不可恢复；且本条以"1、"开头，后续条目可能改变保留范围。

### 6.1 判为"旧数据"、可清

| 路径 | 体积 | 判断依据 |
|---|---:|---|
| `AndroidWorld-Experiments/.history/` | 8.6M | 10cell / 12cell 的历史快照（4 个 xlsx/ndjson） |
| `AndroidWorld-Experiments/.archive/` | 114M | 24 个 `_before_*` 日期快照 + `20260903_before_latest_source_refresh` |
| `AndroidWorld-Experiments/old/` | 13M | 旧 12cell 变体、`_副本.xlsx`、`.numbers`、旧 preview 图、G07 manifest 副本 |
| `AndroidWorld-Experiments/outputs/20260902-omniflow-function-e2e/` | 8.3M | 与主目录 12cell 工作簿 md5 不同的旧版本 |
| `OmniFlow-AndroidWorld-Experiments/`（整目录） | 122M | 旧 12cell 变体（含 Script Replay）+ `RESULT_GROUPS.md` 副本 + 2 张 preview 图 |
| `AndroidWorld-Experiments/*.numbers` | 3.1M | `12cell_autodroid3.numbers`、`OmniFlow_AndroidWorld_bruteforce.numbers` |
| `readme-list/omnimind-gpt55*` | 544M | 早期批次 + GPT-5.5 隔离批次（`-new` 内 524M，含唯一 `cells.csv` 明细） |
| `readme-list/rerun-20260831/` | 37M | AutoDroid 历史整理与逐任务流水 |
| `readme-list/multisegment-qwen-4090/`、`_transfer_review_r13/`、`data/`、`transfer_errors.snapshot.jsonl` | ~20M | 零散变体/回归批次 |

合计约 870M，占 `AndroidWorld-Experiments/`（1.1G）的大部分。

### 6.2 判为"旧文档"、可清

- `RESULT_GROUPS.md` ×2（两个根目录各一份，旧 10-cell 规则，明确写"不包含新实验"）
- `new-final-verision.txt` ×2（根目录 113KB checkpoint 046–050 + `readme-list/rerun-20260831/` 版）
- `HISTORY_RECONCILIATION_20260831.md`（8/31 的 33/310 待运行数字已被 9/3 主表推翻）
- `APPAGENT_OFFICIAL_ANDROIDWORLD_AUDIT.md`（旧执行审计）
- `G07_PROMOTION_MANIFEST.json` ×2
- `readme-list/<Task>/README.md`（117 个）+ 其余 46 个 md / 4 个 txt（共 163 md、4 txt）

### 6.3 必须保留（清掉就无法复现结论）

- `AndroidWorld-Experiments/OmniFlow_AndroidWorld_116Tasks_12cell_autodroid.xlsx` + `.xlsx.inspect.ndjson` — 当前权威主表，1392 格逐格结论
- `AndroidWorld-Experiments/OmniFlow_AndroidWorld_116Tasks_15cell.xlsx` + `.ndjson` — `Rerun Summary` / `Rerun Cells` 是 9/3 冻结的 OmniFlow 最终结果（185/348）
- `AndroidWorld-Experiments/omniflow-zeroshot.xlsx` + `.ndjson` — Zero-Shot 基线
- `readme-list/<Task>/*.csv`（796 个）— 逐格结果证据，是数据的主体，不属于"文档"
- `portable/`（231M）— 运行时资产与版本锁
- `omni-device-agent/`（33M）— 源码、FluxMem 实验、Agent Migration 新方向
- `OmniFlow/README.md`

### 6.4 需要先拍板的边界项

1. **`15cell.xlsx`**：`116 Tasks` 里 Fixed Replay / T3A Hint 六列 696 格全空，看起来"旧"；但 `Rerun Cells` 是唯一的 9/3 冻结结果。建议先把 `Rerun Cells` 导出成独立 CSV，再决定整表是否清。
2. **`omnimind-gpt55-new/`（524M）**：占空间最大，但里面有唯一逐格明细 `cells.csv` 和 163/329 早期统计。若 SR 口径统一到 185/348，可整目录清；若要保留"批次演化"证据则留。
3. **`AUTODROID_OFFICIAL_ANDROIDWORLD_AUDIT.md`**：虽是旧审计，但它是 seed 114/115（1044 计划）的唯一出处，删掉后"没测什么"就没依据了。建议保留。
4. **`OmniFlow-AndroidWorld-Experiments/authoring/`（2 个 authoring JSON）和 `task_tables/ContactsAddContact/`**：确认不再需要旧 authoring 后再清。
5. **清理后是否重写一份新的 `RESULT_GROUPS.md` / `README`**：旧文档清空后，"哪张表是权威表、哪些基线已废弃、SR 用哪一批"需要有一处明文说明，否则口径问题仍然存在。

### 6.5 建议执行方式（等你确认后）

1. 先确定唯一权威表与 SR 口径（第 4 节第 1 条），并导出 `Rerun Cells`。
2. `mkdir -p _trash_20261110`，把 6.1 / 6.2 整体 `mv` 进去（可恢复），保留 6.3。
3. 复核：主表 116 行 × 12 列仍可读、796 个 CSV 仍在、1392 格结论没丢。
4. 确认无缺后再物理删除 `_trash_20261110`。

**执行结果（2026-11-10）**：已按 6.1/6.2 移入 `_trash_20261110`（873M，可恢复），未物理删除。导出 `AndroidWorld-Experiments/OmniFlow_RerunCells_20260903.csv`（348 行 × 19 列）保住 9/3 冻结结果。复核：主表 117 行 × 14 列完好、116 个任务目录齐全、116 份 `summary.csv` 齐全、逐任务 CSV 753 个（其余 43 个在被移走的历史批次内）、`AUTODROID_OFFICIAL_ANDROIDWORLD_AUDIT.md` 已保留。

---

## 7. 下一步（以 rebuttal 为准，2026-11-10）

### 7.1 语境校正

清理后在根目录发现今天同步进来的材料，说明**当前真实任务不是"补齐 AndroidWorld 主表"，而是补 MobiCom'27 rebuttal 实验**：

- `rebuttal.md`（385 行，11/10 17:19）、`mobicom27_gui_agent.pdf`（论文定稿）、`paper.md`（只是 README 副本）
- `src/OmniFlow-exp/`（实验源码，11/10 16:52）、`docs/OmniFlow-exp/`（副本）、`docs/OmniFlow-paper-export/`（另一套论文导出，AGENTS.md 指向 NSDI'27 材料，勿与本次 rebuttal 混用）
- `src/OmniFlow-exp/AGENTS.md`（权威叙事与口径警告）、`docs/PAPER_FREEZE.md`、`docs/BASELINES_SAME_HARNESS.md`、`docs/EXPERIMENT_SCENARIOS.md`、`docs/REBUTTAL_DRAFT_20261107.md`
- `scripts/exp/rebuttal_matrix.sh`（已写好的实验矩阵）、`tools/rebuttal_mechanism_stats.py`、`tools/rebuttal_artefact_audit.py`

因此第 4 节里"补 AppAgent 27 格 / AutoDroid seed 114/115"降级为索引整理，不再是主线的下一步。

### 7.2 必须先裁决（不裁决就跑，数字会白跑）

1. **oracle 条件**：`AGENTS.md` 明确论文 OmniFlow 53.2%（185/348）与 no-memory 37.1%（129/348）产生于会调用 AndroidWorld 官方 completion checker 的旧合同，属 oracle 条件；AutoDroid/MobileGPT 没有该信号，`BASELINES_SAME_HARNESS.md` 要求 `--completion-oracle off`。**这些数字不能与基线无条件对比**，需在 oracle off 下重跑正式数字，或在 rebuttal 里显式标注条件。
2. `REBUTTAL_DRAFT_20261107.md` 里三处必须由用户拍板的矛盾：PAPT 训练口径（418 apps / 是否零样本）、K=8 vs K=16（影响 1024 维）、NULL class / partial assignment 是否真在 loss 里（写错会被 A 要求贴 loss 代码）。
3. `PAPER_FREEZE.md` 记录 "137 completed OmniFlow episodes"，`rebuttal_matrix.sh` 默认只解析约 **55 个有 canonical target asset 的任务**。补充实验是子集，不能写成 116 任务全量。
4. 定稿数值以 `mobicom27_gui_agent.pdf` 为准，不以本地索引三套 SR 数字（163/185/208）为准。

### 7.3 代码已就绪、可直接跑（最快出数，都在 4090/CVM）

| 动作 | 回答谁 | 入口 |
|---|---|---|
| L1 memory-off / L2 fixed-replay / L4 OmniFlow 阶梯（同一 `--memory` 配对） | B、D 的"增益是否来自 memory / Function" | `rebuttal_matrix.sh l1/l2/l4` |
| `no_reentry`、`no_checker` 全子集消融 | B、D 的 intermediate-state replay / Checker 隔离 | `rebuttal_matrix.sh no_reentry no_checker` |
| cache 阈值 {0.80,0.90,0.95,0.99} + page 门 {0.60,0.70,0.80,0.90} sweep | B4/C4/D4 阈值敏感性、accepted-match error rate | `rebuttal_matrix.sh threshold page` |
| `--baseline all` 同 harness 基线（zero-shot / demo_hint / fixed_replay / MobileGPT-style / AppAgent-style / **MobiMem-style AgentRR** / OmniFlow） | A/B/C/D 要求的新基线对比 | `BASELINES_SAME_HARNESS.md`；runner 目前"只提供基建，尚无真实结果" |
| OFF/ON/ON/OFF 两次配对重复 + task-cluster bootstrap | D 的 confidence interval、C 的 token savings across all attempts | `EXPERIMENT_SCENARIOS.md` + `summarize_paired_experiments` |
| per-invocation 机制统计与资产审计 | B4/C4/D4 的 false-accept / false-reject、rebuttal 可用证据边界 | `tools/rebuttal_mechanism_stats.py`、`tools/rebuttal_artefact_audit.py` |

注意 `EXPERIMENT_SCENARIOS.md` 的警告：operating-point coverage 只是基于已记录分数的策略模拟，**没有 reviewed correspondence label 就不能写成 false-accept/false-reject 率**。

### 7.4 需要新实现或新环境（不能假造）

- **L3 缺档**：`rebuttal_matrix.sh` 明确 L3（trajectory 级单元 + OmniTransfer + fallback）没有开关，需要实现 `--replay-unit trajectory`（约一天）或在 rebuttal 里承认三档阶梯的 confound。Reviewer B 问的正是这个，假造比承认缺档更糟。
- **部分记忆实验**：删除 first/last 25%/50%、随机删一个 GUI node，测 SR / reuse rate / model calls 随覆盖率变化（B/C/D 的 cold-start）。
- **工作流中断**：app 更新、重复弹窗、登录中断、网络间歇（B/C/D），需构造更真实环境，现有只有 onboarding pop-up 一例且由共享 Checker 处理。
- **真机完整任务指标**：配对 full-task latency、energy、完整任务成功率、action reuse、fallback rate、内存增长、重复吞吐（C3/D1/D5/D6）。现有只有组件级 65.9ms/0.046mWh 等。
- **PAPT 难例 + 人工验证**：布局几乎相同但功能不同的页面、OmniTransfer 的 false-positive/negative 人工标注（A、E）。
- **AutoDroid-V2 基线**：rebuttal 承诺加入，但 `BASELINES_SAME_HARNESS.md` 目前只实现了 MobiMem-style / AgentRR，没有 AutoDroid-V2 行。
- **全 attempts 的 token 统计**：现在只有双方均成功的 83 对（-50.8%），C 要求含失败与恢复的全部尝试。

### 7.5 不需要跑、纯写作/统计即可

E 的四个问题（Table 1 三属性同时实现的挑战、Fold 低成功率归因、53% 代表性的实践含义、failure taxonomy / 结构重放 vs 语义正确）、A 的 OmniTransfer 训练细节与 XML baseline 说明、C 的 62% "结构完成但 validator 失败" 分析——其中 Fold 归因和失败分类可以直接从 `OmniFlow_RerunCells_20260903.csv` 与 12cell 主表统计得出。

### 7.6 建议的第一步

在 4090 上，用 `src/OmniFlow-exp` 仓库：

1. `bash scripts/exp/rebuttal_matrix.sh list` 确认实际解析到多少任务（预期约 55），先固定这个子集身份；
2. `/usr/bin/python3 -I tools/rebuttal_artefact_audit.py` 审计哪些 artefact 真能支撑 rebuttal，避免在缺失资产上跑；
3. 先裁决 7.2 的四项口径，再按 7.3 顺序跑 L1/L2/L4 → `no_reentry`/`no_checker` → 阈值 sweep → `--baseline all`；
4. 所有输出写 `data/androidworld/.archive/rebuttal-<date>/`，不覆盖冻结论文结果，也不与旧 348 格混算。

本地只保留索引与报告；下一步的实际执行位置是 CVM/9207，`README.md` 的 rsync 通道不变。

---

## 8. 已合并为一份数据（2026-11-10）

按"清理成一份数据"执行，唯一产物：`AndroidWorld-Experiments/omniflow_unified_results.xlsx`（844K，6 个 sheet）。

| sheet | 数据行 | 内容 |
|---|---:|---|
| `README` | 33 | 口径规则、字段字典、清理记录、禁止跨协议混算的警告 |
| `Cells` | 1740 | 权威格级结果：OmniFlow 348（9/3 冻结）+ MobileGPT/AppAgent/AutoDroid 1044（12cell 主表）+ Zero-Shot 348 |
| `Summary` | 20 | 按 protocol × method × device 与 method 合计的 PASS/FAIL/未测/SR |
| `Missing` | 149 | 140 个格级未测/需重跑 + 2 planned + 1 retired + 6 rebuttal 缺口 |
| `Evidence` | 3046 | 1492 run_receipt + 348 rerun_cell + 453 zeroshot_receipt + 753 csv_index（含 16 个嵌套 `autodroid_native/` CSV） |
| `Excluded` | 165 | 25 处 SR 口径逐格冲突、被排除的旧工作簿/列、`_trash_20261110` 归档清单 |

**统一口径**：OmniFlow 采信 185/348 = 0.5316（`final_20260903_frozen`），12cell 主表的 208/348 只保留差异记录；Zero-Shot 129/348；难度统一用冻结分层 Easy/Medium/Hard（61/36/19 tasks，915/540/285 格）。三套 SR、oracle 条件警告、55 任务子集限制都写在 `README` sheet，禁止跨协议直接相减。

**归档**：旧三套工作簿、3 个 ndjson、2 张 preview、RerunCSV、`task_tables/` 共 10 项移入 `_trash_20261110/AndroidWorld-Experiments/sources/`（可恢复）；生成脚本存 `_trash_20261110/build_unified_results.py` 便于复现。`AndroidWorld-Experiments/` 根目录现在只剩 `omniflow_unified_results.xlsx` 与 `readme-list/`（753 个原始 CSV + `AUTODROID_OFFICIAL_ANDROIDWORLD_AUDIT.md`）。未物理删除任何文件。


