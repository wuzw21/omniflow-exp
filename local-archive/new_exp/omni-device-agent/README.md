# omni-device-agent

端侧持续视觉助手的推理部署研究。最新完成 **C 的存储布局修正与 B/C 重复计时**：保持 FluxMem 原策略和固定采样，M4 上 41 个前缀及三题×三条件的完整模型输出仍精确一致。追加历史拼接由 280 次变为 0；82 秒末持久状态由旧 C 的 57.87 MiB 降至 14.45 MiB，B 为 19.22 MiB。12 对维护计时中位数 B/C=3.75/2.39 秒；6 对含视觉前端为 32.45/31.76 秒，5 对更快、1 对更慢，尚不宣称稳定部署加速。[最新结果](experiments/fluxmem_incremental_compact/RESULTS.md)、[复跑入口](experiments/fluxmem_incremental_compact/README.md)。

前一轮 [固定采样三个对照](experiments/fluxmem_incremental/RESULTS.md) 仍保留；C1 是那一轮的旧 C，C2 是本轮布局。旧单次计时不与本轮直接相减计算修复收益。

此前已完成 FluxMem 作者代码＋官方 Qwen2.5-VL-3B 的原算法部署剖析：三题×两条件×三重复，共 18 次正式问答。[基线结果](experiments/fluxmem/RESULTS.md)。

这是一项端侧部署剖析，不是新方法或通用 agent 框架。FluxMem 公开代码在 QA prefill 内进行记忆压缩；分别测视觉编码、压缩和语言计算，不能称作持续后台 writer。3B 权重完整公开，但为非商业研究许可。[此前 SVMemAgent 的真实阻塞](experiments/svmemagent/README.md) 保留，不以新实验冒充其复现。

这个仓库收拢了此前从端云 KV 迁移、多模型部署、多模态流水线，到持续感知与记忆的探索；不是已经实现的通用 agent 框架。状态更新于 **2026-10-31**。

- **[完整探索与交接主文档](docs/EXPLORATION.md)**：问题如何变化、哪些证据改变了判断、保留哪些方向。
- **[下一位接手入口](docs/HANDOFF.md)**：当前边界、环境、依赖、直接开始的步骤。
- [研究定位与 Intro](docs/INTRO.md)、[相关工作及证据等级](docs/RELATED_WORK.md)、[MiniCPM-o 部署审计](docs/MINICPM_O_AUDIT.md)。
- [科研写作约束](docs/WRITING_PROTOCOL.md)：保留用户要求的证据、图表、超参数比较与三轮编辑规则。
- [实验账本](docs/EXPERIMENT_LEDGER.md)：实际执行、失败、只检查源码与尚未运行分别列出。
- [历史代码和结果快照](experiments/legacy/README.md)：保留原文，不把其中过期的“下一步”作为当前任务。

最值得先看的两组结果：128 道 P2 开发题中，联合入口 oracle 最佳 F1 为 55.39，简单加法为 55.34，未建立复杂联合修正的可靠优势；M4 上的像素门控试跑减少了视觉编码时间，但丢失短暂文字证据，未通过预定质量门槛。后者**不是 SVMemAgent 复现**。

## 检查交接材料

```sh
python3 scripts/verify_snapshot.py
```

它只核验归档字节完整性，不运行模型。已有实验的可执行测试、输入引用、环境与真实设备边界见 [验证说明](validation/README.md)。模型、虚拟环境与公开数据缓存未收入 Git；没有配置远端仓库或发布权重。

最新的 scaling 实验把同一真实视频延长到 188 秒，并把读出间隔设为 40、10、4 秒。94 个前缀仍逐位验等；B 的记忆维护中位数从 17.45 秒增到 104.11 秒，C2 约 5 秒；188 秒状态为 B 44.06、C2 19.03 MiB。含真实视觉塔的前端单对计时为 B/C2=83.45/71.24、114.35/71.83、198.99/82.11 秒。见 scaling 结果与曲线，以及 Intro/Story。
