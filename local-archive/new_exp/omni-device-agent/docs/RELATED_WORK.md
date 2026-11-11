# 相关工作：用途、假设与证据等级

截至 2026-10-31 的探索索引，不是穷尽检索或所有论文的正式复现。这里的“阅读”是历史研究记录；“实跑”必须另有实验链接。此轮交接重新打开核对了 SVMemAgent、RaReCache、JointServe 原文与固定本地审计目录，其余沿用已存记录，进一步写论文时应回原文核对具体数字和版本。未确定的论文 ID 不猜测。

## 当前基线与直接挑战

| 工作 | 已有能力 / 交换或保留的对象 | 我们的证据与复现边界 |
|---|---|---|
| [FluxMem](https://arxiv.org/abs/2603.02096) / [作者代码](https://github.com/ShareLab-SII/FluxMem) | 短期保留、中期相邻帧剪枝、长期局部聚类；原实现嵌在 QA prefill 内 | 作者源码和完整官方权重已固定，M4 上完整短/长期路径实际运行；不是常驻后台 writer。见 [部署测量](../experiments/fluxmem/README.md) |
| [SVMemAgent](https://arxiv.org/html/2609.18540v1) | 不看未来帧与查询，在线替换/丢弃记忆帧；之后交给 VideoLLM 问答 | 原文已读；作者代码/checkpoint 尚未固定，未复现。论文 Table 8 报告 writer 16.2 ms/帧，不能当作本机结果或完整 QA 成本 |
| [ActiveSight](https://doi.org/10.1145/3795866.3844451) | 音频前端、文本锚点与选择性视觉处理，面向低功耗主动感知 | 历史证据主要为出版方摘要，未取得完整实现并复现；不把摘要功耗当完整 VLM agent 功耗。与 SVMemAgent 的组合应是强挑战，而非预设失败 |
| [StreamMem](https://arxiv.org/html/2508.15717v1) | 查询无关的有界流式视觉/KV 记忆，通过选择与合并控制状态 | 原文阅读；视觉编码与后续过滤成本要分开；未复现 |
| [MemEvo](https://arxiv.org/html/2609.36581v1) | 搜索查询无关的记忆维护程序 | 历史原文阅读，服务器 writer 成本记录不能直接当端侧可用性；未复现 |
| [StreamAgent](https://arxiv.org/html/2508.01875v3) | 任务感知的时机/位置选择与持续记忆，规划和精确回答分工 | 已有主动时机选择，不可把 task-aware anticipation 笼统视为空白；未复现 |
| [TGL / Do Proactive Agents Really Need an LLM to Decide When to Wake and What to Anchor?](https://arxiv.org/html/2605.30152v1) | 根据结构化 OS 活动触发与选实体，避免每个事件调用大模型 | 输入是结构化活动，不是原始音视频；不能将触发延迟直接比较摄像头感知成本；未复现 |
| [StreamSense](https://arxiv.org/html/2601.22738v1) | 轻量模型、选择性 VLM 与延后处理模糊事件 | 已知类别任务与通用助手有区别；低调用率不等于低驻留内存；未复现 |
| [MMEdge](https://arxiv.org/html/2510.25327v1) | 感知/编码流水、自适应配置、跨模态选择性跳过 | 端侧已有多阶段协作；不能只提统一流水作为创新；未复现 |
| [Nanomind / Tiny but Mighty](https://arxiv.org/html/2510.05109v6) | 异构模块与内存管理、按电量切模式、低占空运行 | 续航估计的工作模式必须核对，不能当作持续重推理；未复现 |
| [React When You Need To](https://arxiv.org/html/2609.22587v1) | 依据行动相关变化控制 VLA 执行 | 少触发不代表前端编码免费；未复现 |
| [SentinelBench](https://www.microsoft.com/en-us/research/articles/sentinelbench-a-benchmark-for-long-running-monitoring-agents/) | 长时间监控、条件等待与变化触发 | 网页状态监控不是短暂视觉证据的等价负载；未复现 |
| [AoII 相关采样工作](https://arxiv.org/abs/2304.00875) | 在采样/能量成本下管理语义信息过时 | “旧状态可能失效且检测有成本”已有形式化基础；未复现 |
| [Certified World Models as Sensing Clocks](https://arxiv.org/html/2607.01537v1) | 预测有效期与再次感知时机 | 历史阅读未建立某个特定项普遍优于强基线的结论；不能将 sensing validity 宣称为全新问题 |
| [可穿戴助手部署经验](https://arxiv.org/html/2511.11811v2) | 配对手机与穿戴传感器，实际常驻/唤醒成本 | 设备和占空比特定，不能推成所有助手统一的功耗下界；未复现 |

其他探索中检索到 R3-Streaming、SPOTBench / Don't Pause 等名字，但未保留足够的全文与固定源码证据，不据此写具体机制或性能结论；无需在当前阶段继续扩充它们。

## 跨模型、跨前文与多智能体通信

| 工作 | 分工 / 状态假设 | 对探索的影响 |
|---|---|---|
| [DroidSpeak](https://www.usenix.org/system/files/nsdi26-liu-yuhan.pdf) | 同架构、不同微调权重；复用共享前缀状态，连续层重算并传入口 E | 入口误差、选层、离线 profiling、流水传输已有；我们的 Stage A 不是其忠实复现；历史检索未定位可核验作者实现，不等于没有发布 |
| [KVCMAS](https://arxiv.org/html/2609.34060v1) / [固定代码](https://github.com/hjeon2k/KVCMAS/tree/e2170cf4b1790bd93f98d15e926397999d590867) | 同模型不同角色前文，锚点与低秩 KV 校正 | 已固定原始 HumanEval＋Qwen2.5-Coder-7B 设置，未运行模型。固定 checkout 未发现 LICENSE，未将其源码复制进本仓库 |
| [RaReCache](https://arxiv.org/html/2610.11358v1) | 校准跨模型线性映射，再按 rank disagreement 选择 token 重算；同 tokenizer | 原文报告某些模型对需要重算 30%–40% 位置；比例不是运行时间，不能称“修几个 token”；未复现 |
| CacheBlend / KVComm | 已有块复用与前文相关修正，是同模型不同上下文讨论中的参照 | 本仓库未收录新的作者实现复现；精确版本与公式应从历史记录引用继续核对，不补造链接或统一假设 |
| [LCF-X](https://arxiv.org/html/2605.22863v1) | 池化来源状态，结合目标状态产生残差；历史审计中的跨上下文实验为同模型 | 不同上下文的语义通信已有，adapter 参数量不是通信量 |
| [XKV](https://arxiv.org/html/2608.20617v1) | 两侧分别有证据，池化、层对齐、联合 memory，更新接收方已有 KV | 已覆盖异构与独立上下文；不是完整文件缓存重建。我们的实跑用第三方 kvloom，不冒称作者代码 |
| [CacheLink](https://arxiv.org/html/2610.15509) | 映射同一可复用 prefix 的跨模型状态，接收方行为监督 | 不再声称 zero-prefill 跨模型交接全新；其设置不能自动覆盖目标私有前文或 WAN |
| [kvloom](https://github.com/batuhanzorbeyzengin/kvloom/tree/146ae229d3a1d5f0c2aa972aeb3096cb409b9b09) | 第三方 XKV 译器与 session 接口 | 已做 0.6B 同模型真实小样本；0.6B→4B 仅接口检查。未作为生产运行时集成 |

更详细的算子与载荷计算见[冻结审计](../experiments/legacy/results/latent_communication_audit.md)，选择重算位置的创新边界见[prior-art 记录](../experiments/legacy/results/repair_selection_prior_art_boundary.md)。

## 端云合作与真实工作负载

| 工作 | 实际分工 / 传递内容 | 本地状态 |
|---|---|---|
| [Minions](https://github.com/HazyResearch/minions/tree/a87d0ee9063485ae2c065679c0471fd559aa5f4c) | 本地读文档，远端发问题并综合短答案 | 实跑原 Minion 流程；本地 3B、远端 7B 是诊断用模型对，非前沿 API |
| PAPILLON / PRISM | 本地敏感信息处理、云回答、本地整合等不同分工 | PAPILLON 已做两样本实际运行与格式/接续诊断；PRISM 仅相关工作。具体来源在[边界报告](../experiments/legacy/results/privacy_collaboration_execution_boundary.md) |
| [EC²-Bench](https://arxiv.org/html/2610.17816v2) | 本地执行与云顾问，传格式化历史 | 原格式化器结构检查；未运行 benchmark。公开发布与论文功能范围不同 |
| [AgBench](https://arxiv.org/html/2609.38652v1) | 个人设备本地、混合、云部署比较 | 历史 artifact 获取受阻；未复现。云调用本地工具是必要对照 |
| [AWS 车载助手](https://docs.aws.amazon.com/solutions/in-vehicle-ai-assistant-on-aws/) / [MongoDB 示例](https://github.com/mongodb-industry-solutions/voice-car-assistant-v2/tree/d90e4f9981344c773a59dd94eafdbdccd90670e1) | 架构指导与可读示例展示不同的“离线”语义 | 示例 AST 检查不是真车、断网或双模型测试；offline 数据源与本地推理不能混用 |
| τ² / PAAC | 工具依赖、结构化规则、本地语义与隐私边界候选 | τ² 做到环境与真实工具调用检查，未完成任务；PAAC 为合约审计 |
| [Hcompany trajectories](https://huggingface.co/datasets/Hcompany/trajectories/tree/ede2a362c99ddba0d03f439c47a0b8fea6abae22) / [Qwen-CUA](https://github.com/xlang-ai/Qwen-CUA/tree/8a295fd7aaf7c4164a4c56dfb7b2d52ba6040bac) | 公开手机轨迹与实际历史构造 | 做过计数审计、真实 VLM 回放、像素门控；未闭环重跑原 Android 任务 |

## 多模型、异构芯片和多模态部署

| 工作 | 借鉴点 | 证据边界 |
|---|---|---|
| [JointServe](https://arxiv.org/html/2610.19822) | 权重分片与各会话 KV 联合驻留；从调用者输出中提前加载下个专家 | 原文核对：A800 80GB、1 TB host RAM。会话恢复不等于异模型 KV 共用；未端侧复现 |
| [DFlash](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a) | 目标模型与条件化草稿的分工、验证与缓存生命周期 | 固定源码检查；未跑量化 Qwen＋草稿、跨芯片或工具多轮。旧 checkout 在 `/tmp/dflash-audit-20261031/dflash` |
| [FocusGUI](https://github.com/fyzhang1/FocusGUI/tree/5633650f90a35feeb68302e1a562977d2b95547d) | 局部结构与多模态注意范围启发 | 阅读/源码候选，无本机性能。旧 checkout 在 `/tmp/focusgui-audit-20261031` |
| [MiniCPM-o](https://arxiv.org/html/2604.27393) | 实时多模态、已有双工与多后端 | 详见[源码审计](MINICPM_O_AUDIT.md)，无完整本机运行 |
| [FreeToken](https://arxiv.org/html/2608.16157v1) | 专家/缓存预算与混合模型状态管理 | 抽取原状态池做常数张量检查，未运行完整模型；CPU 状态复制不等于迁移质量 |
| [MARS](https://arxiv.org/html/2604.26963v1) / [EdgeAgent](https://arxiv.org/html/2610.03394v1) | 工具期状态机会成本、端侧执行/调度 | 已有工作限制“工具期保留 KV”或“端侧调度”的宽泛新颖性；未复现 |
| ktransformers、DSpark、多 batch dLLM | 异构执行与并行草稿的讨论入口 | 未在本仓库建立固定复现，不列论文性能或声称已验证 |

引用某项工作不等于已把它集成。新增实验应先选当前基线，不把这个索引变成并行启动全部候选的任务列表。
