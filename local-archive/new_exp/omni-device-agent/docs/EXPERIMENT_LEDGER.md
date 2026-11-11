# 实验账本与证据入口

记录截止 2026-10-31。表内“完成”仅指所述范围；服务器是实际硬件，但不因此成为手机/穿戴端侧验收。历史文件保留原文与日期，最终报告优先于早期进度。

| 分支 | 真实完成内容 | 结果 / 尚缺 | 冻结入口 |
|---|---|---|---|
| C 去除历史拼接＋B/C 重复计时 | M4：41 前缀＋9 次完整生成精确一致；12 对维护、6 对实际视觉前端 | 追加 cat 280→0；C 状态 57.87→14.45 MiB（B 19.22）；维护中位数 3.75→2.39 秒；前端 5/6 对更快，尚不称稳定部署加速 | [结果](../experiments/fluxmem_incremental_compact/RESULTS.md)、[证据](../experiments/fluxmem_incremental_compact/evidence/) |
| Stage A：0.5B / 7B pilot | 层入口、历史、embedding 可获得性与两题 7B | 早期小样本，无正式质量判定 | [协议](../experiments/legacy/STAGE_A.md)、[7B](../experiments/legacy/results/stage_a_7b_findings.md) |
| Stage A：八题双因素 | 固定目标、四套共享 KV、五类入口、七区间 | 有帮助也有有害改写，不能由此训练新预测器 | [结果](../experiments/legacy/results/stage_a_factorial_findings.md) |
| 精确 KV 复核 | 152 组层掩码、同进程路径重复 | 部分分数对数值路径敏感 | [结果](../experiments/legacy/results/stage_a_splice_findings.md) |
| Stage A：128 题 P2 | 四张 RTX 4090、固定开发题、两分片完成 | 加法最佳 55.34，联合 55.39；held-out 与跨尺寸未跑 | [最终报告](../experiments/legacy/results/stage_a_full_p2_findings.md)、[原始合并](../experiments/legacy/results/stage_a_full_p2_001/merged.json) |
| XKV 同模型检查 | M4、真实 0.6B、第三方公开译器、八题 | 拆分算子一致；四对问题未出现预设复用失效；非网络实测 | [报告](../experiments/legacy/results/latent_question_probe_findings.md) |
| Qwen 0.6B→4B | config、形状与译器接口 | 未训练、未测异构质量 | [边界](../experiments/legacy/results/qwen_xkv_pair_decision.md) |
| 公共段 KV 私有依赖 | 64 次 0.5B MPS forward，合成公开段和标签 | 文本边界不是状态依赖边界；非隐私攻击/保证 | [报告](../experiments/legacy/results/public_kv_dependency_findings.md) |
| Minion 首次协作 | Mac→9207 实际生成与单模型对照 | 协作失败 | [报告](../experiments/legacy/results/minions_collaboration_findings.md) |
| Minion 3B | 三题，22 次主生成＋6 次远端文件对照 | 目标答案一致；长文件主要只在本地读，未确立跨模型 KV 收益 | [报告](../experiments/legacy/results/minions_3b_findings.md) |
| PAPILLON 重叠 | 两公开样本、真实本地/远端生成 | 冷暖混杂与 P95 输出差异，不能报加速 | [报告](../experiments/legacy/results/papillon_overlap_findings.md) |
| PAPILLON 接续/格式 | 原生分块对齐与 JSONAdapter 诊断 | 不能漏计提前计算或混淆输出长度与效率，未重跑端到端质量 | [报告](../experiments/legacy/results/papillon_replay_findings.md) |
| PAPILLON prompt 布局 | 上游实际布局与重复区间分析 | 源码/输入结构，不是模型加速 | [布局](../experiments/legacy/results/papillon_layout_findings.md) |
| 手机公开计数 | 691 轨迹、13,379 步计数核对 | 第四步现象非因果归因，未命中不全是重复 | [报告](../experiments/legacy/results/mobile_cache_findings.md) |
| GUI 算子诊断 | 实际模型状态和输入依赖探针 | 单模型诊断，非端云闭环 | [报告](../experiments/legacy/results/mobile_operator_findings.md) |
| 真实视觉 prefill | M4 2B VLM、原版历史构造、四图回放 | 原生前缀接续 logits 一致，裁剪缩短复用 | [报告](../experiments/legacy/results/mobile_prefill_replay_findings.md) |
| CUDA 视觉塔 | 4090 固定视觉权重与一/三图，原生 cuDNN 路径 | 中位约 10.61/24.82 ms；未测完整任务或网络 | [报告](../experiments/legacy/results/mobile_vision_cuda_findings.md) |
| 公开 MoE 路由 | 路由统计、窗口驻留 oracle | 公开记录分析，非实际执行大模型 | [报告](../experiments/legacy/results/public_trace_findings.md) |
| 流式专家换入 | 允许换入的字节下界与穷举回归 | 解析/轨迹回放，不是在线调度收益 | [报告](../experiments/legacy/results/streaming_expert_findings.md) |
| 远端执行专家 | 放置回放、校准/评估分割、依赖链分析 | RTT 为模型，不是实际 WAN | [报告](../experiments/legacy/results/remote_expert_dependency_findings.md) |
| EC² | 原格式化函数结构测试 | 未运行 benchmark 或 API | [报告](../experiments/legacy/results/ec2_collaboration_audit.md) |
| 车载 / FreeToken | AST 与 CPU 原状态池常数张量 | 非真车、非模型推理质量 | [报告](../experiments/legacy/results/edge_setting_source_audit.md) |
| τ² | 环境、原生评估器、真实 Qwen 工具调用、重启检查 | 完整 agent 任务数为 0；旧远端 ENOSPC | [报告](../experiments/legacy/results/tau2_native_readiness.md) |
| PAAC / 隐私合作 | 合约和来源审计 | 未完成任务级复现 | [PAAC](../experiments/legacy/results/paac_operator_boundary.md)、[来源](../experiments/legacy/results/trusted_processing_source_boundary.md) |
| KVCMAS 原版 | 固定代码、161 题顺序、Coder-7B 配置 | 模型未跑；旧存储/环境阻塞；不作为当前任务重启 | [复现计划](../experiments/legacy/REPRODUCE_KVCMAS.md) |
| DFlash / FocusGUI / MiniCPM-o | 阅读与源码审计 | 无相关端侧模型实跑 | [文献表](RELATED_WORK.md)、[MiniCPM-o](MINICPM_O_AUDIT.md) |
| 像素门控试跑 | M4 72 条视觉计时，完整模型失败案例 | 更快但未过质量门槛；无峰值内存、功耗或摄像头测量 | [结果](../experiments/legacy/stream_gate/RESULTS.md) |
| **SVMemAgent 原版** | **资源检索＋两套本机配置加载实际失败；模型尚未运行** | **发现来源未核验的 HF 候选，缺自定义模型实现和原始入口，性能未测；后续按用户授权换 FluxMem** | [执行证据](../experiments/svmemagent/README.md) |
| **FluxMem 3B 端侧部署** | **M4 上原算法完整短/长期路径；三题×两条件×三重复，共 18 次正式问答；视觉/压缩/语言时间与阶段内存** | **82 秒输入 5056→3193 token，压缩中位 2.173 秒，生成 footprint 峰值 11.06→9.61 GiB；时延明显漂移，不能报稳定加速；非独立 writer** | [实际结果](../experiments/fluxmem/RESULTS.md)、[原始记录](../experiments/fluxmem/evidence/fluxmem_profile_002/results.json) |
| **FluxMem 固定采样三对照** | **M4 上 41 前缀、三题×三条件完整 QA 精确一致；实际调用、持久 storage、拼接复制、reset 与故障注入** | **三题视觉 69→41 块；C 比 B 少 48 次匹配/22 次剪枝，长期合并未减少；状态 19.22→57.87 MiB，未确立附加加速** | [结果](../experiments/fluxmem_incremental/RESULTS.md)、[计数](../experiments/fluxmem_incremental/evidence/audit_results.json) |

## 归档与重跑

`experiments/legacy` 是原目录中脚本、协议、Markdown 报告、JSON/JSONL 输出、锁文件及日志的字节一致快照。来源与 SHA-256 在 [manifest](../provenance/legacy_snapshot.json)。没有复制模型、虚拟环境、上游 checkout、私有原始轨迹或编译产物。

许多脚本仍依赖原缓存和绝对路径；归档意味着证据不丢，不意味着它们已成为跨机器的一键工具。记录了真实生成文本的公共基准输出是研究证据，不能被当作新用户输入执行。若未来公开发布仓库，还需单独审核第三方数据/输出的再分发许可；当前仅建本地仓库，没有声明整个快照统一适用某个许可证。

没有为了迁移重跑耗时模型实验。新仓库运行的完整性与回归检查见 [validation](../validation/README.md)。这部分只证明归档与已记录结果的约束，不增加科学样本，也不代表 SVMemAgent 或设备端到端验收。
| 长视频×查询密度 scaling | 188 秒真实视频、94 个前缀、40/10/4 秒读出；C2 与全量逐前缀等价；B/C 记忆及前端曲线 | 188 秒 B/C2 状态 44.06/19.03 MiB；维护中位数 17.45/5.79、56.70/5.69、104.11/4.92 秒；前端单对均 C2 更快，但未做稳定性区间 | [结果与曲线](../experiments/fluxmem_scaling/result_001/RESULTS.md) |
