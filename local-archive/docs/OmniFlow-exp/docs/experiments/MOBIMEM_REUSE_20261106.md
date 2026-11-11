# 同一 workload 上的三种记忆设计（2026-11-06）

对照 arXiv:2512.15784（MobiMem）。本文只报**从已封存 RunLog 离线可算的量**，不报需要设备
的新结果；设备结论留给下一轮实验，见最后一节。

## 先说不可比的地方

MobiMem §6.3 报 37.5%（ActTree）/ 59.7%（LLM 模板 ActChain）/ 77.3%（人工模板 ActChain）。
它测的 workload 是 **8 个类别、~454 个同族参数化变体**：同一任务的参数不同，界面几乎不变。
本仓库的 workload 是**同一任务跨设备迁移**：目标与参数固定，只有屏幕变。

两者不是同一个量，**数字不能并排放**。所以这里不复述它的数，而是把它的**定义**搬过来，
在我们的 155 条轨迹上重算同一个量。

## 三个设计的定义

| 设计 | 复用单位 | 判定 |
| --- | --- | --- |
| ActTree | 历史轨迹的共享前缀 | 前缀一断即为 0，branch 之后不再假设环境一致 |
| ActChain | 不变量步 / 变量步 | 历史中只出现在单一位置的步视为不变量，可任意位置复用；重复出现的步视为变量，必须位置也匹配 |
| Function（本方法） | 由 Function 执行的步 | `execution_trace` 中带 `function_id` 的步 |

ActTree / ActChain 各算两次历史池：**同任务**（该任务自己的 source 轨迹）与**同 app**
（同 app 内其它任务），后者才是 MobiMem 的 per-application Action Memory。

## 结果（同一分母：155 条运行、1004 个执行步）

| 设计 | 复用率 |
| --- | --- |
| ActTree（同任务池） | 0.197 |
| ActChain（同任务池） | 0.433 |
| ActTree（跨任务池，中位 10 个历史任务） | 0.214 |
| ActChain（跨任务池） | 0.468 |
| **Function（本方法）** | **0.576** |
| Function + Transfer 证据 | 0.369 |

按任务等权（macro）复核，结论不变：ActChain 跨任务 0.500、同任务 0.464。短任务没有
污染结论——≤3 步的运行占 18.1% 的条数但只占 7.8% 的步数。

复现：

```bash
python3 tools/memory_design_comparison.py --root data/androidworld --json-out /tmp/mem_cmp.json
python3 tools/cross_task_memory.py --root data/androidworld --json-out /tmp/cross_task.json
```

## 读法

**Function 的覆盖率高于两种回放结构（0.576 vs 0.468），这是本方法的正面证据。** 原因不是
Function 更会找元素，而是回放只能复现已录制的步，而 Function 由参数绑定和 Checker 承接
录不到的部分。MobiMem 的 ActChain 在它自己的 workload 上做到 77.3%，靠的是**人工把不变量
步与变量步排成相邻**——一个本仓库没有的、需要人工标注的前提。

**但覆盖率不是成功率。** 155 条运行全部通过官方 validator，因为 `current/` 只保存成功
结果，失败的在 `.archive/`。所以上表能说"这条记忆能替代多少步模型调用"，**不能说**"复用率
高就更容易成功"。任何把 0.576 读成"准确率高 15 个百分点"的写法都是错的。

**跨任务池不低反高**（0.468 vs 0.433）说明池子更大时回放更容易撞上相同控件。这也意味着
回放类方法的复用率可以用**扩大历史池**刷高，与记忆原语的好坏无关——审稿人会问这一点，
我们的回答只能是"所以我们报 Function+Transfer 的 0.369，而不只报覆盖率"。

## 能不能写进论文

**能写的只有"关系"，不能写"数字"。** 具体分三档。

### 可以写（关系性结论，条件随身带）

> 在固定的同 harness 条件下，Function 覆盖的执行步比例高于两种回放结构
> （同任务池与跨任务池均如此），且差距不随历史池扩大而消失。

这句话成立的前提要一起给：同一批轨迹、同一定义、同一分母。它是一个**比较结果**，不是某个
常数。

> 回放类方法的复用率随历史池扩大而上升（本 workload 上 ActChain 从 0.433 升到 0.468），
> 因此复用率必须与池子大小同时报告。

这一条我认为比上面的数字更有价值，因为它是关于**测量方法**的结论，不会被新数据推翻。

### 只能在附录写（条件数字，必须标注前提）

上表那些具体数字（0.197 / 0.468 / 0.576）可以进附录，但每一行都要带上：轨迹数量、
分母定义、历史池构造方式、以及"分母是执行步不是任务数"。**绝不能**把它们写成一个
与方法绑定、换数据集仍成立的常数。

### 不能写

- "Function 覆盖率高 → 任务成功率更高"。失败侧对照不足（8 条、单一任务、
  `control_is_conclusive: false`），这条因果现在没有证据。
- 和 MobiMem 的 37.5 / 59.7 / 77.3 直接并排或做减法。workload 不同（同族变体 vs 跨设备）。
- 任何把复用率当作"高效率"或"高准确率"的转述。它只说明"这一步可以不调用模型"。

最短的诚实写法：**"我们不主张复用率是方法的常数；我们主张在同等条件下，Function 能覆盖的
执行步多于回放，且这个差距来自覆盖范围而非复用精度。"**

## 跑实验前的自检

`scripts/exp/preflight.sh` 一次报出所有会让一批任务白跑的环境问题：

```bash
PYTHON_BIN=/path/to/.venv/bin/python bash scripts/exp/preflight.sh
```

2026-11-06 在沙箱里的输出（两项失败是环境，不是代码）：

```
interpreter                        OK   .../python (3.12.15)
python deps                        OK   openai PIL numpy pandas cv2 absl json_repair
android_world                      OK   116 tasks
OmniTransfer                       FAIL missing: .../OmniTransfer (set OMNIFLOW_OMNITRANSFER_ROOT)
adb                                FAIL not found (set OMNIFLOW_ADB_PATH)
protocol budget                    OK   1200s<=1200s, device=standard45562
baseline mapper                    OK   mobimem_style -> agentrr_transfer_with.<locals>.transfer
```

在 4090 上 OmniTransfer 与 adb 都在，这一屏应当全绿。`--device` 那个 bug 就是被这个自检
暴露出来的：不带 `--device` 时启动器直接死在 `unknown_configured_device`，一个任务都跑不起来。

回归状态：`test_agentrr_transfer.py` + `test_same_harness_baselines.py` 共 26 项全过；
`test_kernel_configuration.py` 15 过 5 失败，5 个失败全部是同一个原因
`omnitransfer_root_missing`（该 checkout 不含 OmniTransfer 仓库），不是代码回归。

## 下一步实验（为下一结果做保证）

上表把问题定位到一个具体缺口：我们缺**带失败对照**的复用率。

### 第 1 步已经跑了：失败侧对照（结论：不充分）

```bash
python3 tools/failure_side_coverage.py --root data/androidworld --json-out /tmp/fail_cov.json
```

| | 成功侧 | 失败侧 |
| --- | --- | --- |
| 运行数 | 155 | 8（另有 6 条 0 步，根本没执行） |
| Function 覆盖率 | 0.576 | 0.370 |
| 逐条中位覆盖率 | 0.571 | 0.500 |

方向符合预期——失败侧覆盖率更低——但**这个对照目前不成立**，脚本自己给出
`control_is_conclusive: false`，理由有两条：

1. 只有 8 条有执行步的失败，其中 6 条一步未走（超时/启动失败），本来就没有覆盖率可言。
2. 8 条失败全部来自**同一个任务** `SystemBluetoothTurnOn`。失败侧没有跨任务变异，覆盖率
   差异可能是任务难度而非记忆设计造成的。

所以现在**不能**写"Function 覆盖率导致成功"，也不能写"低覆盖率导致失败"。能写的只有
"覆盖率高的运行成功率也高，但样本不足以排除任务难度这一混淆"。要把它变成可用的对照，
必须让失败侧跨任务。

### 已修：失败侧的可见性

原先诊断这条时说的是"失败没进 evidence"，这不准确——失败其实**已经**被
`_archive_failed_run` 归档了（本次 14 条）。真正的问题是它们**没有可分析的索引**：只能靠
遍历 `.archive/<task>/<method>/<device>/<kind>_<n>` 并从目录名猜种类，而且
`external_harness_001..022` 这种命名里既有成功也有失败（19 条成功 / 8 条失败），
光看目录无法区分。任何离线分析都可能只报成功侧而不自知。

修法是加一个只追加的 `data/androidworld/archive_index.jsonl`：

```json
{"task":"...","method":"...","device":"...","archive_kind":"failure",
 "path":".archive/.../failure_001","success":false,"validator_success":false,
 "steps":3,"failure_reason":"vlm_planner_failed:Request timed out."}
```

既有归档布局完全不动（旧结果路径仍然有效），索引只是新增一条可机读记录，写明**归档原因与
真实结果**。`tools/failure_side_coverage.py` 现在优先读索引，并在输出里标明数据来源是
`archive_index` 还是 `directory_walk`（后者是历史批次没有索引时的回退，分析者应当知道
结果是从 payload 反推的，不是从索引读的）。

下一次跑批之后，失败侧会自动带上跨任务记录，第 1 步的对照才有可能成立。

### 第 2 步：同族变体 workload

构造 MobiMem 式的参数化变体（同一 app、同一任务族、参数不同），与跨设备 workload 并列
报告。这是唯一能和它的 37.5/77.3 对话的形式，也是回答"你们为什么不用扩池提分"的唯一方式。

注意本文件的跨任务池结果（0.468 > 同任务 0.433）已经表明：回放类方法的复用率**可以靠扩大
历史池刷高**，与记忆原语好坏无关。所以这一步必须同时报池子大小，否则两边都在比池子。

### 第 3 步：设备侧只跑一件事

`--baseline mobimem_style`（AgentRR）与 OmniFlow 在同一批任务上的成功率对照。复用率已离线
算完，设备只用来验证"覆盖率优势是否转成成功率优势"，不再用来复测复用率。
