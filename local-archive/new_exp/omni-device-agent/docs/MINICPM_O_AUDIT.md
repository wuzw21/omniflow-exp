# MiniCPM-o 端侧执行审计

这是前一阶段的固定源码检查，尚未实际运行完整模型。记录是为了避免接手者重复提出上游已有的多模态流水线。

| 来源 | 固定版本 |
|---|---|
| [OpenBMB/llama.cpp-omni](https://github.com/OpenBMB/llama.cpp-omni/tree/a64932ec615e1463158129feaf2f0752d0c07a2e) | `a64932ec615e1463158129feaf2f0752d0c07a2e` |
| [MiniCPM-o-Demo](https://github.com/OpenBMB/MiniCPM-o-Demo/tree/47709a9210dfd71afa76c058e017fc8c4db5c8d2) | `47709a9210dfd71afa76c058e017fc8c4db5c8d2` |
| [技术报告](https://arxiv.org/html/2604.27393) | 历史阅读来源；服务器指标不是本机测量 |

本机历史 checkout 在 `/tmp/minicpmo-deploy-audit` 与 `/tmp/minicpmo-demo-audit`，临时目录可能清理；长期引用以上固定 commit，而不是假定 `/tmp` 永久存在。

## 执行路径与状态

```text
音频 / 视频采集
  → encoder 队列
  → VPM 与 APM 可并行，汇合成输入 packet
  → 同一 LLM context 中按序 prefill / decode
  → 生成文本 token 与 hidden states
  → TTS 跨 chunk 生成 mel
  → token2wav / vocoder
  → 播放
```

`tools/omni/omni.cpp` 中的 DuplexPipeline 已有 encoder/prefill 队列，VPM/APM 的异步并行和 join，LLM worker 对同一上下文的顺序处理，TTS worker 与输出队列。后续编码可以和前序阶段重叠，不能据此推断同一 LLM 状态可任意并发。TTS 保持跨 chunk KV，并在 turn 边界重置；token2wav 的窗口与 lookahead 也约束交付顺序。高层 session 还有 prefill/decode worker 与有界 pending 请求。

上游已有 GPU layer 放置、VPM CoreML 和可选 T2W DiT CoreML；非 CUDA 路径的 vocoder 默认 CPU，注释提到 Metal 小 kernel 调度成本。将 CPU/GPU/ANE 统一列在图里不是新增功能，也不证明最优放置。

## 值得测量但尚未成立的候选

在 `token2wav/token2wav-impl.cpp` 与 `coreml/coreml_t2w_dit.mm` 的固定版本中，CoreML 路径仍有状态输入输出、主机端 CFG/Euler 步骤和缓存更新；GGUF 与 CoreML 加载路径也值得检查实际驻留是否重复。按代码形状计算，每 ODE step 的 attention 状态约 75 MiB，五步约 375 MiB。这个数是逻辑状态量/复制规模的估计，**不是测得的物理内存流量、PCIe 传输量或耗时**，更不能仅凭大小认定瓶颈。

已有 `tools/omni/perf/DUPLEX_PROFILING.md` 和 profiling 脚本区分 decision、first audio 与 TTS RTF。关闭 TTS 的测试不覆盖完整语音交互。暂停生成可以让出计算，但不会自动释放权重/状态；要判断是否值得驱逐，必须实际测驻留与恢复成本。

尚未测量：本机完整 MiniCPM-o 的内存峰值、不同后端选择的质量与性能、CoreML 状态复制占比、连续运行功耗、模型暂停恢复、手机闭环交互。历史 issue 中的卡顿或高 FPS 异常不能直接归因为这些候选。

这条线现在是归档候选。下一步优先 SVMemAgent，不同时开始 MiniCPM-o 优化。
