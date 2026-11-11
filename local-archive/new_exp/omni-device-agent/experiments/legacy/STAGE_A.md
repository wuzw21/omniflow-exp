# 阶段 A：入口 E 是否值得修正

状态：已实现实验入口并完成本机 0.5B 检查及服务器两题 7B pilot；完整开发/评估集未运行，没有选层。没有跨规模模型、WAN 加速或 agent 成功率结论。目标仍是筛选适合端云 agent 的系统机制，不能以“脚本跑通”替代 OSDI 研究问题。

适用范围：本协议针对 Qwen2.5 完整注意力模型。新近混合注意力模型还维护递归/短卷积状态，不能只交换 KV 和 E 就假设可恢复；固定上游状态池检查见 [部署源码与状态边界](results/edge_setting_source_audit.md)。这项补充没有改变既有消融数据，也不构成入口修正有效性的证据。

## 当前实测结果

后续 [精确 KV 替换复核](results/stage_a_splice_findings.md)进一步收紧结论：八题 152 组层掩码完成，Joe Frazier 的有害修正在精确目标区间 KV 下仍复现；上一轮唯一联合 oracle 相对加法 oracle 的 F1 改善，则在精确替换下消失。同进程两路径各三次复核确认这一数值敏感性。完整开发/评估尚未运行，不能据此选择最终修复区间或训练入口预测器。

新增 [固定目标八题双因素消融](results/stage_a_factorial_findings.md)：A_s/A_t/B_s/B_t 的共享缓存全部接到同一 B_t prefix，补齐此前缺失的 A_t 缓存对照；七个四层区间均运行来源、前文、模型、加法、联合入口。联合入口没有稳定超越更简单 oracle，部分分数改善主要是答案格式，并出现有害实体改写。56 个原生区间外校准输出与目标一致。未运行完整开发集/评估集，未选层或训练修正器；原定 gate 不变。以下两题 pilot 是历史证据，不能代替最新八题结果。

首次 7B pilot 已完成，详见 [7B 结果与边界](results/stage_a_7b_findings.md)。两道开发题 × P0/P2、七个 4 层区间均跑完；P2 上原生、直接复用、R2/R6 全部答对，没有入口修正的答案收益证据，也没有选层。全部 28 个纯目标模型原生入口重算校准保持输出 token 不变，但 KV 仍存在约 1.1% 最大数值误差；不能宣称精确等价或正式质量 gate 通过。

同一 4090 上两段 1.2K/1.3K 共享材料的原生前向约 0.132/0.141 秒，而未压缩 KV+E 的 100 Mbps 解析序列化时间为 5.45/5.93 秒。网络不是实测；在无预取重叠的该设置下，传输成本已明显盖过潜在计算节省。后续不应将这一同架构原样 KV 方案直接包装为端云加速框架。

以下保留协议与准备过程；其中“尚未运行”等表述是对应阶段的历史记录，以本节和实际结果文件为当前状态。完整开发/评估实验仍未完成。

## 要证伪的假设

在模型 A/前文 H_s 下得到共享材料 X 的状态，在模型 B/前文 H_t 中使用它时，少数连续层的入口 E 是可修复的主要误差来源。先用 oracle E 判断质量恢复空间；若理想入口也没有收益，就不训练校正器。中间层 oracle 来自完整前向，不能计为在线加速；第 0 层是明确例外，其入口可由目标 token embedding lookup 精确获得，见下方基线说明。

主模型对仍为 Qwen2.5-7B Base → Instruct。同架构便于控制变量，不代表小模型 → 大模型。当前 0.5B 对只检查算子语义；它的层数是 24，不能把其固定层区间当成 7B 的层选择结果。

文献依据：[DroidSpeak](https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf) 的连续层重算需要来源 E；[KVCMAS](https://arxiv.org/html/2609.34060v1) 处理同模型不同前文的 KV 校正。本实验不声称复现了两者的完整算法或性能。

### 原文复核后的创新边界与因果比较边界

[DroidSpeak §4.1–4.3](https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf#page=7) 已明确讨论来源 E 的误差传播、E 的大载荷、连续层重算和按模型对离线建立配置；线上还会根据 SLO 选配置并流水化传输。因此“入口误差重要”“连续重算”“E/KV 与计算重叠”均不能作为本工作的新增 insight。第 0 层 embedding 可直接计算是当前架构的基本性质，也不是新方法。

阶段 A 的 R6−R2 是**给定联合错配时，理想入口能改善多少**的上限诊断。它不证明联合修正优于已有分离方法，也不证明端云场景独有。当前 P0/P2 还改变了支持文档位置、共享段内容及前文长度；即使 P2 降幅更大，也不能仅据此归因于“模型×前文”交互。

后续若 pilot 值得继续，必须在同一道 P2 的同一目标输入 B(H_t || X || Q) 上比较四套来源缓存：B_t（原生）、A_t（仅模型来源不同）、B_s（仅来源前文不同）、A_s（两者同时不同）；所有拼接都保持原生 B_t prefix。现有 smoke 已捕获四套状态，但缺少 A_t 的完整共享缓存拼接对照；当前冻结的 7B pilot 也没有这一对照，不能将其结果写成完整双因素实验。此缺口应在 pilot 结束后补齐，不能修改正在排队的冻结代码。

同样，R3/R4/R5/R6 是入口消融，区间外仍使用 A_s KV；只替换入口不等于已经做了上述纯模型/纯前文缓存对照。R5 若接近 R6，应优先考虑分离修正；仅当联合 oracle 有实质额外收益时，才有理由研究联合机制。所有 oracle 的可获得性仍需单独解决。

可以从既定七区间结果补充观察：同一模型对、固定重算预算下，不同前文是否使静态配置失效。但不能只比较 P0 选层与 P2 表现；在 P2 开发集重新 profiling 的最佳静态配置也是必需基线。逐题用金答案挑最佳区间只能作为不可部署上限，不能当在线策略结果。本轮没有据此新增算法、修改原定 go/no-go 门槛或查看评估集。

## 固定数据与输入契约

HotpotQA distractor validation 的 HF 镜像 revision 为 `1908d6afbbead072334abe2965f91bd2709910ab`。CMU 原下载入口本次失败，未验证镜像与原文件逐字节一致；HF Parquet 已与发布的 SHA-256 匹配。数据许可证 CC-BY-SA-4.0。

固定种子 20261024，按题型比例选择 512 题：开发 128（102 bridge、26 comparison），评估 384（307 bridge、77 comparison）。按题目划分，所有前文变体同属一个集合。没有按任何模型输出筛题。

- P0：模型不同，前文完全相同。
- P1：角色说明不同，材料相同。
- P2：目标历史含一个支持文档 D1，源历史无已有证据；共享段含另一支持文档 D2 和干扰文档，排除 D1。
- 问题在前文中，输出指令在共享材料之后。输入使用相同的实际 token ID，不分别应用两个 chat template。
- 每一对前文使用可见的“无关填充”做到同 token 长度；没有 masked padding 或 RoPE 重定位。
- 2K 是共享段预算上限，不是每题固定长度。实际 P0/P1 中位 1386、P2 中位 1267 token；短样本照实保留。
- 用支持文档标注控制材料位置，是诊断性构造，不是真实检索能力评测。并未证明每道题都必须依靠 D1 才能回答；不得将此当作事实。
- 原文文档顺序保留；超预算时裁去干扰文档，不静默截断支持证据。若支持文档本身超预算则明确失败。

数据及其派生 token 文件保存在 `~/.agent-reach/edgecloud-stage-a/`。工作目录的 `results/stage_a_data_manifest.json` 固定全部题目 ID、划分、tokenizer 词表 hash、文件 hash 和长度分布。评估集尚未被模型运行。

## 执行与对照

四次完整前向提取 A_s、A_t、B_s、B_t。E 是 decoder block 首个 RMSNorm 之前的 residual stream，覆盖全部共享 token；K 是 post-RoPE，V 为实际 cache 值。

对固定连续区间 [s,s+k)，仅运行 B 的这些 decoder blocks，使用 B_t 的原生 prefix KV。区间内 X-KV 替换为重算值，区间外仍为来源 KV。最终 Q 和答案均由 B 按所有层正常执行，不跳过 decode 层。

| 对照 | 入口/缓存 |
|---|---|
| R0 | 原生目标缓存与正常后续计算 |
| R1 | 直接复用来源 X-KV |
| R2 | E(A,H_s) |
| R3 | E(A,H_t)，前文 oracle |
| R4 | E(B,H_s)，模型 oracle |
| R5 | E(B,H_s) + [E(A,H_t) - E(A,H_s)]，加法 oracle |
| R6 | E(B,H_t)，联合 oracle |

补充：区间外换原生 KV 的 R2/R6；B→B 的直接复用、来源入口、目标入口。这些均已接入本机 smoke。

R5 的差分和求和必须在 FP32 中计算，然后转回目标 dtype。相同前文 P0 下，它必须严格退化为 R4/R6。直接 BF16 左结合加减会破坏这个恒等关系；本轮发现并修正了该实验实现问题。

7B 主实验候选起点 0/4/8/12/16/20/24，主预算 4 层，辅助预算 1/2 层。开发集选区间后冻结；当前 0.5B 固定 [8,12) 仅为实现检查，没有做选择。开发集描述性 paired bootstrap 已实现并通过合成分数回归，但尚无 7B 结果可供分析。独立于源前文的错误目标历史控制、完整层搜索运行、锁定评估集和正式 go/no-go 判定仍未完成。

## 指标和不能混淆的结论

官方答案 F1/EM；在同一目标原生生成序列上 teacher-force 的 KL/NLL；区间 K/V 相对误差。原生模型是行为参考，不是答案质量的数学上界。

实际 FP32 开发样本 `5a7780e855429949eeb29e9f` 的 P2：原生回答 Garrett M. Brown，金答案为 Aaron Taylor-Johnson，复用变体答对。它只证明“更接近原生状态”不能自动推出“答案更好”；不能据此宣称复用更优。不得删除此类样本或对负的原生质量差距计算恢复比例。

预注册的继续门槛保持不变：7B 锁定评估集 P2、最多 4 层，R6 相对 R2 至少 +3 F1 点且 paired 95% CI 下界 > 0；在原生参考明确更好的前提下恢复至少 30% 剩余差距；行为分布支持，EM 无明确恶化。R5≈R6 则先考虑分离修正；R6 额外胜出才支持联合修正研究。任何 oracle 成功都不证明廉价预测器或端云收益存在。

## 当前实际检查与数值问题

- `test_stage_a_core.py`：原生入口重建、分段等价、缓存重复使用、区间替换边界、非法输入、BF16 加法恒等式。
- `test_stage_a_data.py`：确定性划分、重复 ID 拒绝、共享 token/位置、P2 支持文档迁移、答案标签不进入输入、预算保护。
- 连同已有路由/解析测试，27 项回归通过，日志在 `results/stage_a_regression_tests.log`。随机小模型测试不等于任务质量结果。
- 本机设备为 Apple Silicon、16 GiB、macOS 26.5.1，PyTorch 2.9.1 MPS、Transformers 4.57.6，官方 0.5B Base/Instruct 权重的 revision 与权重 SHA 均固定。修正后真实预训练权重复测结果单独记录。
- 修正后的 `stage_a_smoke_bf16.json` 已完成：4 道开发题 × P0/P1/P2，共 12 个检查；所有原生区间 KV 重建相对误差为 0，重复 continuation 不变，P0 下 R5/R6 结果完全相同。属于本机真实权重复测；正式状态为 `completed_with_numerical_caveat`，不能写成全部质量验收通过。
- `stage_a_smoke_pilot.json`：先前单题 FP32 检查，分段/完整 logits 相对误差约 1.9e-6，原生区间 KV 重建误差为 0。其 R5 使用修正前表达式，不作为正式比较依据。
- `stage_a_smoke_bf16_before_additive_fix.json`：保留的修正前结果；R5 不可用于判断模型/前文是否可分离。不能将该文件的 completed 状态理解为科学有效性验收。
- `stage_a_numerical_calibration.json`：4 道开发题 P2 上，无任何 KV 迁移、固定金答案 continuation 的分段/完整 prefill 比较。FP32 相对 logits 误差约 1.7e-6–2.5e-6；BF16 为 0.012–0.014，KL 为 0.00022–0.00071；此小组 argmax 相同。KL 的约 1e-8 负值来自浮点舍入，按数值零解读。
- 在原生生成 continuation 上还观察到 BF16 部分位置的 argmax 分歧。因此正式质量实验必须先在 CUDA 上校准无干预误差，不把 MPS BF16 的漂移归因于迁移机制。严格校准与结构检查分开，报告可能为 `completed_with_numerical_caveat`。

这是真实 Mac 上的实验脚本验证，尚无 CUDA/7B、端云网络或手机 agent 验收。原生区间 KV 恢复为 0 误差只证明实现符合语义，不是入口修正有价值的证据。

服务器 `ssh 9207` 已连通，检查时有 4 张 RTX 4090、显存基本空闲、驱动 580.178.04；这是瞬时观察，不是资源预约。系统 Python 3.10 没有 Torch；现有 miniconda 为 Python 3.13.5、Torch 2.7.1、Transformers 4.31.0。Transformers 需要在隔离目录更新到 4.57.6，再校准 CUDA 数值基线。不能把 0.5B 结果用作 7B 的 go/no-go。

### 2026-10-24 服务器准备进展

- 已上传本实验代码和公开数据；`run_stage_a_development.py` 只接受开发集 ID，支持 A/B 各占一张 GPU、固定预算下遍历七个区间，比较原生、直接复用、来源入口和 oracle 入口。它尚未在 7B 上运行，不能视为已验证的主实验入口。
- `stage_a_7b_models.json` 固定 Base revision `d149729398750b98c0af14eb82c78cfe92750796`、Instruct revision `a09a35458c702b33eeacc393d103063234e8bc28` 及八个权重分片 SHA-256。共约 30.48 GB；两者 tokenizer.json 的 Git blob hash 与本机 0.5B 对一致，运行时仍会核对词表 hash。
- 原定目录 `/home/wuzewen/experiments/edgecloud-stage-a-01a15efc` 写小文件即报 ENOSPC，尽管 df 有约 198 GiB。Btrfs metadata 为 98.63%，设备未分配空间只有 1 MiB。没有执行磁盘平衡、清理他人文件、改挂载选项等操作；不把这些观测当成已完成的文件系统诊断。
- 当前临时目录为 `/dev/shm/edgecloud-stage-a-wuzewen-01a15efc`，可写、可执行、初始约 55 GiB 空闲；主机当时约 101 GiB 可用内存。该目录重启即失，必须回收结果到本地。系统 Python 的下载器可以正常在此工作。
- 直接访问 HF 超时后，镜像配置文件与固定版本 hash 相符，开始经 hf-mirror.com 下载。所有最终文件仍必须匹配发布的 SHA-256/Git blob hash；镜像不改变模型版本或校验要求。
- 捕获到 curl 对小 JSON 的重定向正文也应用大小限制的问题。已修正传输上限、保留最终精确大小/hash 检查；真实服务器上 Base generation_config.json 已成功下载并通过校验。对应 HTTP 回归已加入。
- 首次环境安装进程在根盘文件系统的 `handle_reserve_ticket` 等待中，尚未完成。用 O_NOATIME 读取该进程持有的库文件可以立即返回，正尝试以系统 rsync 的 `--open-noatime` 复制现有公开 Python/Torch 运行时到内存目录。此绕行尚未完成验证，不宣称根盘问题已修复。
- 随后，基础 Python 运行时已通过 rsync 复制到内存目录，能启动并加载 SSL、pip。原安装进程曾恢复下载后再次等待根盘，后续已不再存在。复制的运行时没有 Torch，不能把复制完成误称为 CUDA 环境可用；当前新进程在完全独立的 `runtime/` 中经清华 PyPI 镜像安装 Torch 2.9.1、Transformers 4.57.6、NumPy 2.4.2、ujson 5.11.0。安装、CUDA 自检与 7B 推理仍未完成。所有临时文件与缓存指向内存目录，不修改 HOME 或根盘挂载。
- 最新完整回归为 32 项通过（新增开发/评估隔离和重定向回归），日志仍为 `results/stage_a_regression_tests.log`。这不能替代 CUDA 真机主实验。

远程 PID/命令/日志以及当时的进程存在性保存在 `results/stage_a_remote_live_status.json`。它是历史快照；继续时必须重新检查 `/proc/<pid>/cmdline`、日志、分片大小和逐模型 verified_manifest，不能仅凭旧句柄启动重复下载。原镜像进程的 Base worker 已因 curl 63 失败，Instruct worker 仍可继续；Base 由独立角色限定的进程补跑。因此原镜像进程最后即使返回失败，也不能直接推断 Instruct 分片无效。

### 一次性 7B pilot 已排队；不要另行启动重复实验

服务器 RAM 目录中的 `queue_stage_a_pilot.py` 已作为后台作业启动，启动 PID 为 `2913645`，句柄与状态分别为 `results/queued_pilot_job.json`、`results/queued_pilot_state.json`。这是当前实验的单次依赖流程，不是定时任务，不安装或重启任何依赖。最近实际检查中 `/proc/2913645/cmdline` 存在，状态为 `waiting_for_existing_preparation`，没有子进程；继续时须重新检查，不能只信本段或 JSON。

它最多等待两小时：既有安装进程退出、两个固定模型 verified_manifest 均符合预期后，校验已固定的八个源码/输入 manifest 文件，检查 GPU 0/1 各有至少 20,000 MiB 空闲且利用率小于 10%，执行真实 CUDA BF16 矩阵计算检查、core 语义回归并记录 pip freeze，最后运行已有 7B 入口（2 道开发题 × P0/P2，七个 4 层区间，max_new_tokens=64）。这是 pilot，不会运行完整开发集、评估集或自动冻结层区间。

安装失败会在运行时检查中被拒绝；GPU 繁忙、源码变化、既有输出、检查失败或子进程失败均使作业停止并留日志，不自动重试。等待超时不停止原始下载。子进程超时会请求终止并保留 PID，重试前必须确认其实际退出。源码 hash 已在真实服务器逐项核对；在此作业终止前不要修改/覆盖已固定的运行文件。

可执行回归新增等待条件不满足、重复任务/输出保护、源码变化拒绝四个案例，完整 42 项通过。这些是启动条件检查，不是 CUDA 验收。排队作业日志为 `results/queued_pilot.log`；若开始执行，CUDA/核心/pilot 日志分别为 `queued_cuda_check.log`、`queued_core_checks.log`、`stage_a_7b_pilot.log`，真实模型结果为 `stage_a_7b_pilot.json`。目前尚无这些后续步骤的成功记录。RAM 目录在重启后消失，继续时应将实际结果同步回本地。

随后安装进程 `2908999` 已退出，`results/ram_runtime_install.log` 明确记录 Successfully installed。为提前发现运行时问题，在权重下载期间独立执行了一次真实 CUDA 检查（PID 2913894，已退出）：Python 3.13.5、Torch 2.9.1+cu128、Transformers 4.57.6，Torch 路径位于独立 RAM runtime，GPU 0/1（RTX 4090）均完成 256×256 BF16 矩阵乘法并通过 finite 检查。命令和输出分别保存在 `results/early_cuda_probe_job.json`、`results/early_cuda_probe.log`。这确认 CUDA 基础环境可运行，不是 7B 或分段 prefill 数值校准，也不表示根盘故障已经修复。排队作业仍等待两套权重完成；它将在真实模型运行前再次检查资源状态。

### 前文敏感性对照：已加入并完成本机真实权重检查

仅把支持文档 D1 放进目标前文，不足以证明任务需要不同上下文。新增 `H0_native_source_history`：B 完整计算 H_s || X，后续回答也保持 H_s，不注入目标前文。它与原生 B(H_t || X) 比较前文敏感性；现有 S1 则将 B(H_s || X) 的 X-KV 接到原生 H_t prefix 上，检查 decode 能否利用目标前文补救。两者必须分别报告，不能将 S1 当作移除目标证据的实验。

7B 开发入口已加入对应 `history_controls.native_source_history` 和 `source_history_shared_with_target_prefix` 字段。所有题目均保留，不根据这些对照是否答对筛选样本；相同前文必须严格恢复相同输出和指标。脚本已上传到 9207，7B 执行尚待模型/环境准备。

本机真实 Apple Silicon / macOS 26.5.1、Torch 2.9.1 MPS、Transformers 4.57.6、固定 0.5B 权重下运行了前 4 道开发题 × P0/P2，共 8 个检查。结果为 `results/stage_a_history_control_smoke.json`，摘要为 `results/stage_a_history_control_findings.json`。全部结构恒等检查通过；状态仍是 `completed_with_numerical_caveat`，不代表数值校准或 7B 科学判定通过。

这 4 道 P2 中原生目标答案均错（平均 F1=0），来源前文原生 B 平均 F1=0.25；其中 Aaron Taylor-Johnson 一题去掉目标历史反而答对。固定 [8,12) 的 R2/R6 平均 F1 分别为 0.4167/0.25。这些极小开发样本既不能支持也不能否定 7B 入口机制，但明确说明：当前 0.5B 检查没有建立原生模型的任务能力与正向恢复空间，不能从 KV 重建成功推导任务收益，也不能计算这里的“恢复原生质量差距”比例。

新增可执行回归检验来源前文 continuation 等于其完整原生前向，并与拼接目标 prefix 的路径区分；完整 33 项回归通过。实际预训练权重检查包含重复 continuation、同前文恒等、层入口与 KV 重建，结果均保存在上述文件；CUDA/7B 检查仍未运行。

复现本轮增量检查（输出必须使用未存在的新文件名）：

```bash
.venv-stage-a/bin/python -m unittest -v test_stage_a_core.py test_stage_a_data.py test_stage_a_development.py test_fetch_stage_a_7b.py test_feasibility.py test_window_oracle.py
.venv-stage-a/bin/python run_stage_a_smoke.py --limit 4 --conditions P0 P2 --dtype bfloat16 --output results/stage_a_history_control_smoke.json
```

### 必须区分第 0 层 embedding 基线与中间层 oracle

当前 full-attention Qwen2 的 E 定义为 block 首个 RMSNorm 前的 residual。第 0 层的 E_B 等于 `B.get_input_embeddings()(X)`，与 H_s/H_t 无关；从第 0 层重算前四层时无需学习入口修正，也无需把入口 E 经网络发送到目标端。此结论不意味着中间层入口也可从 embedding 直接获得，更不意味着无需原生目标 prefix KV。

开发入口已对 layer-zero lookup 与完整前向捕获的 E 做严格相等检查，并记录 `target_entry_provenance`。第 0 层 R6 实际使用 lookup 值；其余起点仍使用完整目标前向的 oracle。分析输出区分这两类可获得性。这个低成本基线必须参加比较，不能把所有 R6 统一包装为需要新预测器。

已在本机真实 0.5B Base/Instruct、BF16 MPS 上运行前 4 道开发题 × P0/P2、[0,4) 重算：`results/stage_a_embedding_entry_smoke.json`。8 个检查均验证 R2=R3、R4=R5=R6（同源模型的 layer-zero E 不依赖前文）、embedding lookup 等于完整目标入口。状态仍为 `completed_with_numerical_caveat`；P2 的 R2/R6 平均 F1 同为 0.4167，原生目标为 0。这个小组不支持任何质量收益判断，只验证可获得性与算子语义。最新 38 项回归通过。

```bash
.venv-stage-a/bin/python run_stage_a_smoke.py --limit 4 --conditions P0 P2 --start 0 --width 4 --dtype bfloat16 --output results/stage_a_embedding_entry_smoke.json
```

固定 7B revision 的 config 已从服务器复制并与模型 manifest 的 Git blob hash 核对，保存在 `results/stage_a_7b_instruct_config.json`。使用既有 `feasibility.kv_bytes` / `serialization_seconds` 计算的解析结果见 `results/stage_a_payload_bounds.json`，不是设备网络实测：

| 2K 共享 token、BF16 | 传输载荷 | 假设 100 Mbps 的纯序列化时间 |
|---|---:|---:|
| 全部 28 层 KV | 112 MiB | 9.40 s |
| 省掉 4 层，只传其余 KV | 96 MiB | 8.05 s |
| 省掉 4 层，但再传一份中间入口 E | 110 MiB | 9.23 s |

原因是 `hidden / (2 × kv_heads × head_dim) = 3584 / 1024 = 3.5`：一份 E 等于 3.5 层 KV 的字节数。第 0 层 lookup 避免这份入口传输，但剩余 KV 仍大。中间层源 E 的传输也不等于已经获得目标 oracle E。这些数值不含 prefix、编码、修复、排队和 RTT；预取重叠可改变可见等待，压缩与跨尺寸模型也会改变载荷，不能由该表直接宣称端云加速或否定全部变体。

## 重现

7B 开发结果分析使用 `analyze_stage_a_development.py`：按题目成对 bootstrap R6−R2 的 F1/EM；完整报告前文对照、数值漂移及各区间结果，不按答案准确率排除题目。只有固定全部开发 ID、P0/P1/P2 和七个 4 层区间均齐全时，才提出 P2 上 R2 平均 F1 最高的区间候选（同分取较低起点）。候选不会自动冻结；必须先审查 CUDA 数值误差。小样本 pilot 不产生层选择候选。

若原生平均 F1 不高于 R2，恢复比例输出 null；即便分母为正，比例也仅为描述性数值，不能替代“原生更好”的统计证据。逐区间置信区间未做多重比较校正，不能用于在开发集上宣布机制通过。当前新增 5 项统计/样本完整性回归，完整 38 项通过；使用的是明确标注的合成分数，不是新增模型结果。

```bash
.venv-stage-a/bin/python analyze_stage_a_development.py --input results/stage_a_7b_development.json --output results/stage_a_7b_development_analysis.json
```

上述分析命令需等待对应 7B 输出生成，尚未在真实 7B 结果上执行。

```bash
uv venv --python 3.12 .venv-stage-a
uv pip install --python .venv-stage-a/bin/python -r stage_a_requirements.lock
python3 fetch_stage_a_models.py
python3 fetch_stage_a_data.py
.venv-stage-a/bin/python stage_a_data.py
.venv-stage-a/bin/python -m unittest -v test_stage_a_core.py test_stage_a_data.py test_feasibility.py test_window_oracle.py
.venv-stage-a/bin/python run_stage_a_smoke.py --limit 4 --dtype bfloat16 --output results/stage_a_smoke_bf16.json
.venv-stage-a/bin/python stage_a_calibration.py
```

以上本机重现命令仅用开发题、下载约 2 GB 的公开 0.5B 权重。远程 7B 权重准备已按用户提供的 9207 资源另行启动；尚无付费云资源。核心模型计算复用上游 Transformers，不实现新模型、压缩算法或 Agent loop。
