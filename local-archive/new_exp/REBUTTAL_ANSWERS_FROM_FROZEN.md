# Rebuttal 正式回复文字（冻结数据可定稿，2026-11-10）

配套：`REBUTTAL_EVIDENCE_FROM_FROZEN.md`（数字与口径）、`../src/OmniFlow-exp/docs/REBUTTAL_DRAFT_20261107.md`（已有 5 块）。

本文件补的是现有草稿**未覆盖**的四个问题：Reviewer C.183 / C.184 / C.192、Reviewer E.251 / E.252 / E.253。
四段全部来自 9/3 冻结 `Cells`（348 格）的**再分析**，不是新跑的实验；因此英文用 "we add a breakdown of the frozen log"，
不写成 "we ran / we measured in a new study"。每段附中文依据与**不能写的口径**，与现有草稿体例一致。
需要 4090 重跑才能填的数字，一律留 `will report in the revised evaluation`，不预填、不估算。

---

## C/E.1 — Structural replay vs. semantic correctness（C.183、C.192、E.253，承诺 5）

The 62.0% figure in your comment matches our frozen log, and we now make the underlying
breakdown explicit rather than leave it as an example. Of the 348 OmniFlow cells, 347 reach
the official validator and 150 fail it. In 93 of those 150 (62.0%) the selected Function
replayed to completion without any action-level failure; 49 ended in a replay failure and 8
never started replay. The gap is therefore at task semantics — parameter binding, state
preconditions, and end-to-end validation — not at reproducing the recorded action sequence.
We agree this is a real limitation: OmniFlow delegates semantic completion to the host, and
correct structural replay is not evidence of a correct task. We will state this boundary in
§5 and add an end-to-end validation discussion. We do **not** read the 62.0% as a rate of
state-matching or action-transfer errors: `replay_status` is an episode-level summary and
cannot separate retrieval from re-grounding. Quantifying false-positive and false-negative
rates of the matcher requires the reviewed human labels we commit to add (C layer), not this
log.

> **中文依据 / 不能写**
> - 150 / 93 / 62.0% / 49 / 8 / 347 / 348 全部来自 `REBUTTAL_EVIDENCE_FROM_FROZEN.md` §1，
>   源表 `omniflow_unified_results.xlsx` `Cells` 的 `outcome_class=official_validator_failure`。
> - 审稿人自己在 comments 里写了 62.0%，所以这段是**确认并给出分解**，不要写成"我们新发现了一个 62%"。
> - **绝对不能**把 62.0% 说成 OmniTransfer 的误接受/误拒绝率，也不能拆成 "state matching 错 X% /
>   action transfer 错 Y%"；承诺 5 的人工标注还没做，正文不得暗示已有 reviewed label。
> - 承认 limitation 的那两句别删：E.253 要的就是"correct replay ≠ semantic completion"被明说，
>   辩解式措辞会招来下一轮追问。

---

## E.2 — Why Fold is lower（E.251）

We add a per-configuration breakdown of the frozen log. Success is 54.3% on Standard (63/116),
49.1% on Fold (57/116) and 56.0% on Tablet (65/116); 18 tasks fail on Fold but pass on Tablet,
while only 10 go the other way, so the gap is task-specific rather than a uniform offset. The
proxy signals point to re-grounding, not to retrieval or host reasoning: action-replay failures
rise from 26 (Standard) to 32 (Fold), retrieval misses are unchanged at 12, and planner timeouts
are actually lowest on Fold (3). The affected tasks — Markor note editing, noisy Recipe flows,
calendar and contacts entry — need precise element location across multi-step state. Fold's
layout and multi-window transitions are the most likely to invalidate a recorded trajectory's
element coordinates and hierarchy at transfer time. We state this as an episode-level proxy
only: these fields cannot cleanly separate GUI-state retrieval, action grounding and agent
reasoning, which is exactly the three-way split you ask for. We will add per-invocation
mechanism statistics (which admitted invocation failed, at which stage) in the revised
evaluation to answer that question directly.

> **中文依据 / 不能写**
> - 63/57/65、54.3/49.1/56.0、18 vs 10、26→32、12、3 来自证据文件 §2；任务名同处。
> - `replay_status` 是**代理**，英文里已用 "proxy signals" + "episode-level proxy only" 兜住。
>   不要写成 "we measured that grounding causes the gap"（那是直接测量，没做过）。
> - 精确三分（retrieval / grounding / reasoning）必须写成 will add（B 层
>   `tools/rebuttal_mechanism_stats.py`），不能拿这三列数字硬凑一个三分比例。
> - 掉分任务名可做例子，但别声称"全部"集中在这些 app；证据只说"集中在…等"。

---

## E.3 — What 53.2% means in practice（E.252）

We clarify both the criterion and the practical reading. A task counts as successful only when
AndroidWorld's official validator passes after the episode; there is no partial credit. In the
frozen log, success falls monotonically with difficulty — 70.5% on Easy (129/183), 40.7% on
Medium (44/108), 21.1% on Hard (12/57) — and Medium plus Hard account for 165 of 348 cells
(47.4%). They are predominantly long-horizon tasks with precise state and parameter
constraints, so the headline number reflects the benchmark's difficulty mix rather than a
ceiling of the mechanism. The relative gain over Zero-Shot holds at every level (+12.0, +21.3,
+19.3 points) and is largest on Medium and Hard, where replay-based reuse matters most; the
conclusion that memory reuse reduces runtime cost is therefore not an artifact of easy tasks.
We do not claim the remaining failures are harmless: roughly half of attempts still need user
intervention or a retry, and we will say so explicitly in §5. One comparability caveat: the
53.2% / 37.1% figures in the paper come from the earlier configuration that consulted the
completion checker; the revision reports the same-harness, oracle-off numbers from a fresh
baseline run, and the two conditions must not be subtracted or compared as if identical.

> **中文依据 / 不能写**
> - 129/183、44/108、12/57、165/348=47.4%、+12.0/+21.3/+19.3 来自证据文件 §3。
> - **oracle 口径那句是硬要求**：论文 53.2%/37.1% 属旧 oracle 条件（源码 2026-10-10 起
>   `--completion-oracle` 默认 off），不能写"我们复现了 53.2%"或拿上表当同 harness 结论。
>   同口径数字由 A 层 `--baseline all` 重跑产出，这里只写 will report。
> - "roughly half still need user intervention" 是主动承认，别删成"可接受";E 问的就是
>   "what does that mean in practice / how much user intervention"。
> - 不要引入冻结数据之外的成功率（163/329、208/348 那两套），否则三个 SR 并列会自相矛盾。

---

## C.4 — Cost across all attempted tasks（C.184，承诺 6）

We accept that the headline 50.8% reduction is computed on the 83 pairs solved by both
variants, and that this understates what a user experiences. We therefore add the absolute
cost distribution over all attempts in the frozen log. Across 347 cells with complete
accounting (348 scheduled, one missing timing), a successful attempt averages 25,592 tokens,
6.1 model calls and 78.9 s; a failed, timed-out or recovered attempt averages 52,177 tokens,
10.0 model calls and 139.2 s — 2.04×, 1.64× and 1.76× respectively. The full run totals
13.19M tokens. Failed attempts are thus materially more expensive, exactly as you note, and
any saving computed on successes alone overstates the net benefit. We cannot yet turn this
into a net saving ratio, because the frozen log records per-cell token cost for OmniFlow only;
the matched baseline-side costs do not exist in it. The revised evaluation reruns all seven
rows under one harness (`--baseline all`) and reports total cost over every attempted task,
including failures and recovery, paired per task and device.

> **中文依据 / 不能写**
> - 25,592 / 6.1 / 78.9、52,177 / 10.0 / 139.2、2.04× / 1.64× / 1.76×、13,187,114 来自
>   证据文件 §4；"347 有完整记账、1 格缺 timing"同处，英文已注明。
> - **绝对不能**从本表算出或暗示任何"token 节省比例"——冻结数据只有 OmniFlow 侧逐格 token，
>   没有同 harness 基线侧。50.8% 仍只能标注为 83 对口径，并承认它偏乐观。
> - "we cannot yet turn this into a net saving ratio" 这句别删：它把 C 的问题接住、又划清
>   数据边界，比硬给一个数字安全。承诺 6 的净节省由 A 层 OFF/ON 配对重跑补，写 will report。
> - 不要用 "we measured a 2× overhead on failures" 暗示这是新实验；这是**冻结日志再分析**。

---

## 合起来用之前

- 四段合计约 540 词，加进现有 5 块（约 1,030 词）会超 500 词窗口。若窗口有限，优先级：
  **E.3（53.2% 代表性，E 点名）> C.4（全 attempts，承诺 6）> C/E.1（62% 语义失败，承诺 5）>
  E.2（Fold）**。C/E.1 与 E.2 可各压成两句并进前两段。
- 四段**全部是冻结数据再分析**，正文不得出现任何"新跑出来"的数字；三处 `will add / will
  report` 对应的是 B 层机制统计、C 层人工标注、A 层同 harness 主表，必须等 4090 重跑后回填。
- 与 `REBUTTAL_DRAFT_20261107.md` 的 B/C/D.2（partial memory）不冲突：那段是**实验计划**，
  本文件四段是**已有数据的分析**，措辞一个是 will add、一个是 we add a breakdown，别混用。
- 跨设备口径：现有草稿只有 B/C/D.1 带框，本四段按设备拆了 Fold/Standard/Tablet，若整份
  rebuttal 采用跨设备叙述，这几段可顺势呼应，不必另起。
- 落盘约定：本文件是可定稿文字；补实验一旦跑出结果写 `runs/<date>-<name>/`，**不回填**本文件
  与冻结数据表。
