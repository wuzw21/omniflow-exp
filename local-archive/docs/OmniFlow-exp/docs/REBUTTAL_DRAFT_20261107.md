# Rebuttal draft — 2026-11-07

Five answer blocks, each 150–250 words of pasteable English followed by Chinese
notes on provenance and on what must not be written. Sources: `mobicom27_gui_agent.pdf`
§3–5 (authoritative per AGENTS.md), `figs/data/omnitransfer_page_recall.csv`,
`figs/data/omnitransfer_action_comparisons.csv`, `docs/BASELINES_SAME_HARNESS.md`.

---

## A/C.1 — How OmniTransfer is trained

We will add a complete training paragraph and a hyperparameter table to §4. OmniTransfer
is a single 1,194,582-parameter model, trained once from random initialization (seed 17)
for 15 epochs with AdamW (learning rate 3×10⁻⁴, weight decay 0.05), batch size 4 for ASE
matching updates and 8 for page batches, in PyTorch; full training takes 0.82 GPU-hours on
one RTX 4090. The only training data is the ASE training split — 835 source–target GUI
pairs expanded to 5,944 samples, with a 115-pair validation split containing 1,566 directed
widget-mapping samples. The ASE Test split we report on (98 pairs, 196 directions, 1,286
reviewed correspondences) contributes no gradient. Page supervision comes only from PAPT,
and our evaluation split is application-disjoint from it: the 128 pairs come from 15
applications that appear nowhere in the 418 PAPT training applications. We will state this
split boundary explicitly in §5.3, because it determines what the 93.75% R@1 can be read as.
One shared encoder feeds both heads — the page
readout (K=8 slots × 128 = 1,024 dimensions) and the action matcher with local-context
calibration and bidirectional verification. At deployment the same frozen checkpoint is
exported to NumPy and runs on the mobile CPU: there is no per-device, per-application or
test-time adaptation, so the on-device latency and the offline accuracy come from identical
weights.

> **中文依据**
> - 训练配置逐项来自定稿 §4 "Training Data / Training Setup"（835 / 5,944 / 115 / 1,566 /
>   seed 17 / 15 epochs / AdamW / 3e-4 / 0.05 / batch 4 与 8 / 0.82 GPU-h / 4090）。
>   不要改写成 epoch 数或 lr 的"约值"。
> - 参数量 1,194,582、4.8 MB float32、NumPy 端侧 runtime 来自 §4 与 AGENTS.md。
> - **训练数据口径，这里我改过一次，务必按改后的写**：`tmp/eval_compile/main.tex` 第 389 行
>   写的是 "128 pairs from 15 apps, disjoint from the 418 PAPT training apps"——即 page head
>   **在 PAPT 上训练过**，heldout-128 只是应用不相交。所以不能说 "no PAPT pair is used for
>   training / zero-shot"，那样 A 只要对照正文就能反驳，比不回答更糟。
> - `papt_page_embedding/README.md` 第一句的 "do not train or fine-tune OmniTransfer for
>   it" 指的是**那一次 716-pair 评测实验**不做训练，不是"模型从未见过 PAPT"。两句不矛盾，
>   但引用时必须引后者（应用不相交），不能引成前者。
> - 如果实际是零样本（即 418 这个数字已作废），那要先把 `main.tex` 那句改掉，再回来改这段。
>   现在这两种说法在仓库里同时存在，是**必须由用户裁决**的一项，不能由我按某一个版本写死。
> - **K 值有冲突，必须统一**：定稿 §3 写 K=8、8×128=1024；而
>   `papt_page_embedding/README.md` 写 "K=16"。上面按定稿写 K=8。如果实际 checkpoint
>   是 K=16，那 1024 维就不成立，两个数字必须一起改，不能只改一个。
> - "115-pair validation 用于选超参" 这句我**没有写**——定稿只说该 split 存在，没说
>   用它做模型选择。如果确实用它选过超参，可以加一句 "hyperparameters were fixed on the
>   115-pair validation split before the test split was evaluated"，这是加分项；如果没选过
>   就不要写，否则等于承认调过 test。

---

## A/C.1 rev.2 — 把两个 design decision 并进训练细节（推荐用这版贴）

We will add a training-details paragraph and a hyperparameter table to §4. OmniTransfer is
trained once, from random initialization (seed 17), for 15 epochs with AdamW (learning rate
3×10⁻⁴, weight decay 0.05), batch size 4 for matching updates and 8 for page batches, in
PyTorch, taking 0.82 GPU-hours on a single RTX 4090. Supervision comes from ASE page pairs
of semantically corresponding iOS and Android pages: 835 page pairs expanded to 5,944
samples, with a 115-pair validation split of 1,566 directed widget-mapping samples. The
reported ASE Test split (98 pairs, 196 directions, 1,286 correspondences) and the 15 PAPT
evaluation applications are disjoint from training and contribute no gradient. Two design
decisions determine what the objective supervises. First, each node is encoded with its
local graph context, and each candidate pair carries a correspondence state refined through
local-graph layers with bidirectional cross-page message passing; the supervised unit is
therefore a context-level correspondence rather than an isolated node's features or
coordinates, which is why the failure modes in §5.3 — duplicated labels, platform-specific
grouping — cannot be resolved from node features alone. Second, the prediction head performs
partial assignment over all target nodes with an explicit NULL class, so abstention is a
trained outcome rather than a post-hoc threshold on a similarity score. Because training is
cross-platform (iOS↔Android) while deployment is cross-device (Android↔Android), no
retraining is required: the frozen 1,194,582-parameter checkpoint is exported to NumPy and
runs on the mobile CPU.

> **rev.2 与上一版的差别，以及为什么这样改**
> - 你原来那两段是**架构新颖性**，不是 training details。A 问 training，回里没有一个数字，
>   他只会认为细节被藏了。rev.2 把两个 design decision 改写成"**监督什么**"——它们回答的
>   是 loss 的定义，正好是 training details 的一部分，这样两段并非白写。
> - 补上了三段具体问题：**数据从哪来（ASE 训练 split）、测试集有没有参与（不相交）、
>   训练多久（0.82 GPU-h）**。这三个是 A 真正会核查的。
> - 新增一句 "training is cross-platform while deployment is cross-device"。你那句
>   "iOS and Android pages" 本身会招来 domain gap 反问，主动写成优势（跨平台更难，部署
>   时不重训也成立）比等他问好。
> - 新增 NULL class 的用处：**abstain 是训练出来的，不是事后加阈值**。这是全篇最有价值的
>   一句之一——它直接解释 §5.3 的低误接受率，也把你们和"top-1 + 阈值"那类方法分开。
> - **前提：NULL class 必须真的在 loss 里。** 我在 `omniflow/transfer/` 的 runtime 里只看到
>   confidence 判据（`omniflow/transfer/errors.py` 的 `confidence` / `pair_confidence` /
>   `absolute_contextual_confidence` 字段），那是运行时分发门，不是训练目标。checkpoint
>   在 OmniTransfer 仓库、不在这个 checkout 里，我无法验证 Sinkhorn / dustbin / NULL 的
>   存在。**这一句能用与否只能由你确认**：如果实现是 top-1 + 阈值，这句话会被要求贴 loss
>   代码，代价远大于收益。
> - 同样，"partial assignment" 只有在确实用了全局分配（如 OT/Sinkhorn）时才成立。如果只是
>   每个 query 独立 softmax，应改写成 "the head scores all target nodes jointly against
>   an explicit null candidate"，语义更窄但不失分。
> - "graph-alignment model" 这个说法我**没保留**：正文 §3 与 §5.3 通篇没有 alignment，
>   概念是 graph refinement + matching head。换名字会让 A 去正文找对应段落而找不到。
>   已在正文有这个术语之前，不要引入。

---

## A/C.2 — The XML baseline, in detail

We will specify the XML baseline precisely rather than leave it as a label. It is
MobileGPT's published screen representation, not a strip of our own: the Accessibility
Service hierarchy is pruned of nodes that are neither interactive nor semantically
informative, every surviving node is mapped to an HTML tag (`<button>`, `<input>`,
`<scroll>`) carrying its text and content-desc, and each element receives an index;
MobileGPT reports an 84.6% average token reduction for this procedure. The resulting page
string is embedded by GLM-Embedding-2 through the vendor API into one vector per page, and
pages are ranked by cosine similarity over exactly the same 128 queries, the same same-app
target gallery, and the same Top-k definition as OmniTransfer. Measured cost: 95,685 tokens
over 32 requests, 214.1 ms mean and 1,309.3 ms P95 per page, CNY 0.0478 for the whole split;
accuracy is 86.72 / 91.41 / 94.53 R@1/3/5. Neither this baseline nor the screenshot
baseline (Qwen3-VL; 256 requests, 1,004.2 ms, 93.75 / 97.66 / 99.22) was tuned on PAPT.
Both are general-purpose encoders run at their published defaults, which is the point of the
comparison. We will also state explicitly what the XML path cannot see and a hierarchy-aware
encoder can: cross-platform attribute drift, where the same control exposes different
resource-ids and nesting on the source and the target device.

> **中文依据**
> - 压缩流程（剪枝 → 映射 HTML 标签 → 元素编号）与 84.6% token 降幅来自 MobileGPT 原文
>   §3.1，证据在 `tmp/pdfs/mobilegpt.txt` 第 165–200 行。这是"不是我方削弱基线"的关键，
>   务必引它的原文而不是概括。
> - 214.1 / 1309.3 / 32 / 95,685 / CNY 0.0478 / 86.72 / 91.41 / 94.53 来自
>   `figs/data/omnitransfer_page_recall.csv` 第 2 行。截图行 256 / 1004.2 / 93.75 同行。
> - **不要写 GLM-Embedding-2 的向量维度**。仓库里没有它的维度记录，`omnitransfer_latency.csv`
>   的 2048 维是另一件事（9207 服务器上 W4 量化的通用 VLM embedding），混用会自相矛盾。
> - **两个 PAPT 协议绝对不能串**：`papt_page_embedding/README.md` 报的是 716 pairs /
>   94 apps 的 strict exact-pair 协议，Top-1 是 79.61 / 25.42 / 29.47；论文用的是
>   heldout-128 + same-app gallery，是 93.75 / 86.72 / 93.75。审稿人若拿到 README 的数字
>   会以为我们自相矛盾。只引用 heldout-128 那一套。
> - 最后一句"attribute drift"是解释而非借口；如果要更保守，可删掉，只保留结果与成本。

---

## B/C/D.1 — The pre-built memory assumption, and fairness across memory methods

We accept the premise and will make it explicit instead of leaving it implicit. §5.1 will
state that every memory-based row starts from a memory built out of **one successful source
execution on the source device**, not from an accumulated personal history. This is a
deliberate protocol choice, and it is also what makes the result deployable: one source
execution is amortised over the target devices, so the cost of the memory is paid once.

Fairness follows from holding that object fixed. Every row consumes the same asset:
zero-shot sees nothing; demo_hint sees the same redacted step text; fixed replay replays the
same source RunLog by coordinate; MobileGPT-style, AppAgent-style and MobiMem-style (AgentRR)
receive the same Function Store and differ from OmniFlow in exactly one mapper function;
and the original-protocol AutoDroid and MobileGPT rows additionally receive their own native
construction procedure (full-UTG exploration; native cold start) so that they are not
disadvantaged by our asset. Because the memory is identical across rows, a difference
between rows cannot be attributed to memory quality — only to how each method re-grounds and
reuses it. What this design does not measure is how each method builds memory from scratch;
we will scope the claim to reuse accordingly and say so in the limitations.

> **中文依据**
> - "same memory for every row" 的逐行清单来自 `docs/BASELINES_SAME_HARNESS.md` 第 13–21 行
>   的表格，以及定稿 §5.1 "For methods that directly accept prior executions, we use the
>   same successful execution trace to construct their memory for a fair comparison"。
> - AutoDroid 的 UTG 全量探索与 MobileGPT 的 native cold start 是定稿 §5.1 明写的，写进来
>   是为了挡住"你把基线饿死了"的反问。
> - 第一段最后一句"one source execution amortised over the target devices"接的是 11-04
>   定的跨设备框（见 `project_omniflow_mobicom_rebuttal` 记忆）。如果不打算在 rebuttal 里
>   提跨设备，这半句可以删；但留着能把"pre-built 不现实"转成"这正是可部署性"。
> - 不要在英文里说 "we assume a complete memory is realistic"——那是我们承认不了的话。
>   承认的是"我们假设了 pre-built memory 这个前提"，不是"这个前提在现实中成立"。

---

## B/C/D.2 — The partial-memory experiment we will add

We will add this experiment in the next revision. Starting from the same frozen Store, we
will build partial memories by manual removal, applied identically to every memory row:
prefix removal (first 25%, first 50%), suffix removal (last 25%, last 50%), and structural
removal (randomly drop one GUI state from a Function, which severs the step chain at a
random point; the sampling seed is fixed and published). These families test different
failures and will be reported separately rather than collapsed into one coverage curve.
Prefix removal does not delete a task, it deletes the entry point: the Function can no longer
be recalled, but a mid-function state may still be reached and resumed. Suffix removal keeps
recall and truncates completion. Single-state removal leaves a Function almost intact and
asks whether replay survives a hole in the middle. For each subset we will report official
success, reuse rate (function-served steps over all executed steps), model calls, fallback
rate and re-entry count, per method. Because the removed content is defined by index and
seed rather than by a method's own success, no row gains information another row lacks.

> **中文依据**
> - 这个实验**确实还没跑**。`docs/EXPERIMENT_SCENARIOS.md` 第 184 行只有
>   "Cold-start and partial Memory coverage" 这一行占位（验收输出写的是 success / reuse /
>   model calls as coverage grows），记忆里的覆盖率 25/50/75/100 四档也还挂在待办上。
>   所以措辞必须是 will add，不能写任何数字。
> - **"randomly remove one of the GUI node" 这句是歧义源**，审稿人原话可能指两种东西：
>   (a) 从 Function 的步骤序列里删掉一个 GUI 状态（我按这个写）；(b) 从 Store 里删掉一个
>   Function。建议在英文里就把它定义清楚（我写了 "drop one GUI state from a Function"），
>   否则下一轮他们还会问。如果两种都想答，就说明两档都跑。
> - **粒度风险，必须先想好再动手**：本地 104 个 store 实测，每个 store 中位 1 个 Function、
>   每个 Function 中位 6 步（总 122 个 Function / 1083 步）。按这个粒度，25% ≈ 1–2 步，
>   五档之间的差别会落在噪声里。建议：(i) 在表里同时报"绝对删除了几步"，(ii) 或者只对
>   ≥8 步的 Function 做梯度、(iii) 或者把档位改成 1 步 / 2 步 / 一半。
> - "prefix 删的是入口不是任务"这句是设计要点，别删——它解释了为什么不能把五档并成一条
>   coverage 曲线，也正是 B 真正想问的东西。

---

## A.3 — Representativeness of the PAPT benchmark

PAPT does not by itself claim that a successful match corresponds to a successful match in
practice, and the paper keeps three measurements apart for exactly that reason. PAPT measures
page identity; ASE (98 pairs, 196 directions, 1,286 reviewed correspondences) measures
whether a recorded action can be re-grounded on the live target; AndroidWorld under the
official validator measures task success. Reading a PAPT hit as task-level evidence would be
the conflation we avoid, and §5.3 says so.

What makes the PAPT result non-trivial is its protocol: the 15 applications are held out and
disjoint from training, the gallery is restricted to the same application so page identity
cannot be recovered from the app name, the pairs are cross-device (phone against tablet, with
different resolution, aspect ratio and OS version), and the 15 applications in the recall
gallery are absent from the 418 applications used to train the page head. What it does not
cover is equally clear: the gallery contains no
same-layout-different-function negatives, so 93.75% R@1 is a retrieval accuracy, not a
precision under near-duplicates. We will add a human verification of OmniTransfer in the next
revision: reviewers will label accepted matches as true or false positives and rejected
matches as true or false negatives at the deployed operating point, and we will report the
false-positive rate, false-negative rate, and their categories — duplicated controls,
same-layout-different-function, and list-item identity.

> **中文依据**
> - 三个测量分工来自定稿 §5.1（micro-benchmarks 的作用）与 §5.3 开头"page recall 不等于
>   动作已经 grounded"。这一段的作用是把 A 的问题接住并转成"你问的那一层由谁回答"，
>   不要写成辩解。
> - PAPT 协议要素（应用不相交、同 app gallery、phone–tablet 跨设备）来自 `figs/omnitransfer_section.tex`
>   第 4–10 行与 §5.3。"zero-shot"在第 98 行指的是**没有记忆的那一行**，不是模型没见过
>   PAPT，两处不要混。
> - "no same-layout-different-function negatives" 是记忆里 11-04 已认定的事实（两个编码器
>   都 93.8% R@1，page recall 的 claim 应从"精度"改成"代价"）。主动承认这一条比被追问好，
>   而且它顺手解释了为什么 OmniTransfer 与截图基线打平却仍有价值（33.0 ms vs 1004.2 ms，
>   0 API 调用）。
> - **有一个可选加强句，我没写进正文**：`tools/rebuttal_mechanism_stats.py` 在 167 个
>   run 上跑出 admitted invocation 的 false-accept 是 17/179 = 9.50%。这是"practice 侧
>   真的会错到什么程度"的直接读数，比 PAPT 更有说服力。**但它是子集不是 348 格全量**，
>   要用必须先写 "on the subset of runs we have analysed (167 episodes, 179 admitted
>   invocations)"，或者重算全量再引。不要裸引 9.5%。
> - 人工验证这条要写成 next revision，本轮不能暗示已经做过。

---

## 合起来用之前

- 五段合计约 1,030 词。如果实际窗口是 500 词，优先级：A/C.1 训练细节（最被追问）>
  A.3 PAPT 代表性 > B/C/D.2 partial memory 计划 > A/C.2 XML baseline 细节 >
  B/C/D.1 公平性。后两条彼此可以压缩成一句并进前两条。
- 三条"我们会加的实验"全部是 will add，正文任何一处都不要出现未跑出来的数字。
- 上一轮 2/2/3/2/3 的重心是跨设备执行迁移；这五段里只有 B/C/D.1 那半句带框。若整份
  rebuttal 都用跨设备口径，其余几条也应至少各有一句呼应，否则读起来像两篇论文。
