# FluxMem 端侧执行验证与成本分解

2026-10-31，在权重下载和首次生成之前固定。本协议不宣称已经跑通；以实际运行结果为准。

- 作者代码：ShareLab-SII/FluxMem，`6fdbf65b6f2af1a94ca41d542be5be86a565d338`，Apache-2.0；不改 `FluxMem` 或模型 forward。
- 官方权重：Qwen/Qwen2.5-VL-3B-Instruct，`66285546d2b821cf421d4f5eb2576359d3770cd3`，完整 BF16，两分片 SHA-256 核验。权重遵循其独立 LICENSE，不称其与代码同为 Apache-2.0。
- 本机：Apple M4 / 16 GiB / macOS 26.5.1；MPS、BF16、SDPA。作者示例是 7B/CUDA/FlashAttention2，因此这是同一方法的 3B 端侧部署实验，不是论文表格的严格复现。
- 环境：独立 Python 3.12、Torch 2.6.0、torchvision 0.21.0、Transformers 4.50.0、accelerate 1.4.0。先实际导入和生成；兼容失败照实保存。
- 数据：mjuicem/StreamingBench，`48872fa707124474ce4c5172ddc58efb8bc88058`，原始 Real_Time_Visual_Understanding.csv 前三行；sample_1 的 4、53、82 秒问题。视频通过 remotezip 读取发布 zip 中唯一 `sample_1/video.mp4`，不重编码，不构造假视频。
- 每次只提供提问时刻之前的前缀；原作者模板、视频读取/采样和 processor。1 FPS，最多 96 帧，min_pixels=16×28²，max_pixels=128×28²；分辨率是明确的端侧设置。
- 方法开/关对照；同一模型、问题、帧、生成参数。FluxMem short_frames=8、medium_frames=64，默认 Otsu 与原阈值不调参。作者已有 video_chunk_t=2 控制视觉塔分块，两条件一致。
- greedy，max_new_tokens=32；记录真实输出和是否到达长度上限。三题仅用于执行与成本诊断，不报总体准确率或论文性能提升。
- 先以短前缀做完整模型 smoke；正式记录三次交错重复，条件顺序随重复交换。模型只加载一次，形状首次运行与后续重复分别保留；不把内核首次开销藏掉。

## 测量边界

公开实现每次问答重建输入，并在 prefill 内调用记忆压缩。不得把这个入口伪装成已经实现持续后台 writer。分别同步计时：视频读取/processor、视觉编码、FluxMem 压缩、完整首轮 forward、后续 decode、完整 generate。首轮 forward 扣除嵌套视觉/压缩时间后，是语言部分及剩余处理，不冒称纯 attention kernel。

记录权重实际张量字节、视觉/压缩输入输出 token 数、KV 实际张量字节、进程 RSS 与 MPS allocated/driver memory。阶段峰值用 10 ms 轮询加边界采样，标明“采样峰值”，可能漏掉更短尖峰；macOS 统一内存下 RSS 与 MPS 指标不能相加。另留进程高水位，但不能当每个阶段的独立峰值。

视频预处理可在加载模型前单独进程执行并保存实际 CPU 输入，以避免一次解码整个视频与权重同时驻留。它的耗时与峰值必须另外报告，不能计作零；不声称这等同原 evaluator 的内存峰值。没有摄像头、功耗、语音或 GUI 动作测试。

先不新增缓存、量化、持续 writer 或压缩方法。若 MPS 操作不受支持，仅在明确记录后允许 PyTorch 原生 CPU fallback；失败与额外设备搬运计入成本。源码、输入、模型和运行脚本均保留哈希。

### 执行前的读取器适配（首次生成之前）

真实视频为 1920×1080 / 25 FPS。原 torchvision reader 将 82 秒前缀全部变成 RGB 张量后才抽帧，原始 RGB 约 12.8 GB，尚未计堆叠副本。在本机使用 PyAV 两次顺序解码：第一次统计截止时刻以内的实际帧数；按作者 `smart_nframes` 和原 `linspace(...).round()` 计算索引；第二次仅保留这些帧。原 resize、processor、FluxMem 和模型 forward 不变。先对 4 秒前缀执行逐像素及 sample_fps 相等测试；不通过不继续。两次解码时间计入预处理。此适配不是作者原始 I/O 峰值，不能据此声称作者原版读取器适配 16 GiB。

完整 smoke 同时跑 4 秒及 82 秒前缀，短前缀成功不能证明压缩/长期合并路径可用。

### 真机 smoke 发现的运行时兼容问题

`fluxmem_smoke_001` 已完整加载全部权重，但 Torch 2.6.0 的 MPS SDPA 不接受作者视觉塔的三维 Q/K/V，报 `Dimension out of range ... got 3`，尚未生成回答。相同形状的真实 MPS 算子在现有 Torch 2.9.1 环境成功。因此后续使用 Torch 2.9.1 / torchvision 0.24.1，保留 Transformers 4.50.0 与作者源码；旧失败记录和环境锁保留。PyAV 19 与 torchvision `metadata_errors` 参数不兼容，固定 PyAV 16.1.0 后读取器逐像素回归通过。

`fluxmem_smoke_002`（Torch 2.9.1）完成 4 秒前缀并回答 D，但 82 秒前缀在长期合并的整数 `scatter_reduce_(reduce="amin")` 报 `not supported for torch.int64`；不能将短样本视为完整验收。该整数 MPS 算子在本机 Torch 2.13.0 已验证可用，后续采用 Torch 2.13.0 / torchvision 0.28.0，保持作者算法和 Transformers 4.50.0。输入张量仍来自此前通过像素等价测试的预处理，不重新挑帧或改问题。前两次失败及独立环境锁留存。
