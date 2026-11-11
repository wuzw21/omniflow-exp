# FluxMem：公开实现的 M4 执行与成本测量

使用 [FluxMem 作者代码](https://github.com/ShareLab-SII/FluxMem)、[官方 Qwen2.5-VL-3B 权重](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct) 与 [StreamingBench 原视频](https://huggingface.co/datasets/mjuicem/StreamingBench)。上游实现未改动；没有训练新模型或替换记忆策略。具体结果见 [RESULTS.md](RESULTS.md)，执行边界在 [PROTOCOL.md](PROTOCOL.md)。

代码 Apache-2.0；3B 权重完整公开且无需授权下载，但采用 Qwen Research License，仅限非商业研究/评估，不能称为不受用途限制的开源权重。该方法不需要额外学习的记忆 checkpoint。固定代码 commit、权重 revision/SHA 与输入来源分别在 `upstream_manifest.json`、`evidence/weights_verified.json`、`evidence/prepared.json`。

本实验是 3B/BF16/MPS/SDPA 的部署变体，论文示例是 7B/CUDA/FlashAttention2。公开实现将记忆压缩嵌在每次 QA prefill 中，不能把这次测试称为持续后台 writer 的运行。真实覆盖了视觉编码、短/中/长期压缩、完整语言模型与 greedy 回答；不涉及摄像头实时采集、功耗或手机闭环。

## 复跑

从仓库根执行。以下路径与本机缓存布局一致，权重约 7.5 GB。首次获取可用镜像下载 Python 包，来源和版本仍以锁文件为准。

```bash
uv venv --python 3.12 .venv-fluxmem
uv pip install --python .venv-fluxmem/bin/python -r experiments/fluxmem/requirements.lock
# remotezip 仅用于按 Range 提取视频，不需要下载整个 9 GB 数据包。
uv pip install --python .venv-fluxmem/bin/python remotezip==0.12.3

git clone https://github.com/ShareLab-SII/FluxMem "$HOME/.agent-reach/research/fluxmem-repro"
git -C "$HOME/.agent-reach/research/fluxmem-repro" checkout 6fdbf65b6f2af1a94ca41d542be5be86a565d338
.venv-fluxmem/bin/python experiments/fluxmem/fetch_weights.py
.venv-fluxmem/bin/python experiments/fluxmem/fetch_video.py

FORCE_QWENVL_VIDEO_READER=torchvision .venv-fluxmem/bin/python experiments/fluxmem/prepare_inputs.py \
 --upstream "$HOME/.agent-reach/research/fluxmem-repro" \
 --model "$HOME/.agent-reach/research/fluxmem-qwen25vl3b/66285546d2b821cf421d4f5eb2576359d3770cd3" \
 --data-manifest "$HOME/.agent-reach/research/fluxmem-streamingbench-sample1/manifest.json" \
 --output "$PWD/runs/fluxmem_prepared_001"

bash experiments/fluxmem/run_local.sh smoke fluxmem_smoke_new
bash experiments/fluxmem/run_local.sh profile fluxmem_profile_new
python3 experiments/fluxmem/summarize.py runs/fluxmem_profile_new/results.json \
 --output runs/fluxmem_profile_new/summary.json
python3 -m unittest discover -s tests -v
```

已有输出拒绝覆盖。复跑前确保没有其他占用 GPU 的实验；不要为了测量关闭用户其他应用。当前测量是在正在使用的物理 Mac 上完成，并非隔离实验室设备。

`prepare_inputs.py` 使用作者模板、抽帧数量与索引、resize 和 processor；只调整解码保留方式，避免先把 82 秒 1080p 前缀整体转成 RGB 张量。该读取器与作者 torchvision 读取器的短前缀逐像素等价检查已实际通过。输入在独立 CPU 进程准备，预处理时间和峰值独立报告，不把它计作零。

读取器等价证据是在 Torch 2.6/torchvision 0.21/PyAV 16.1 下取得；需要重跑原 torchvision reader 比对时，使用对应旧版本的独立环境（不要降级正在运行的主环境）。主环境 Torch 2.13 用于执行作者算法所需的 MPS 三维 SDPA 和整数 scatter-reduce。两次旧后端的真实失败与一次通过的完整 smoke 均保留在 `evidence/`。

`run_profile.py` 包装原函数，仅同步计时、记录输入输出长度、实际 KV 字节和内存；阶段事件保存在 `phases.jsonl`。每次运行保存脚本快照和 SHA。RSS、physical footprint、MPS allocator 和 driver 四种统计独立保留，不相加。

回归测试检查已发生的真实运行证据、前缀截止、完整权重加载、长期路径触发、失败不被覆写成成功及运行入口的防覆盖。它不替代新机器上的完整生成 smoke。

真实回归命令：

```bash
.venv-fluxmem/bin/python experiments/fluxmem/verify_backend_ops.py --output runs/backend_ops_new.json
.venv-fluxmem/bin/python experiments/fluxmem/compare_prepared.py  runs/fluxmem_prepared_001/prepared.json runs/fluxmem_prepared_torch213_check/prepared.json  --output runs/input_equivalence_new.json
```

第二条比较两次实际预处理产生的张量；需先用 `prepare_inputs.py` 在新输出目录重建后一份输入。当前已执行并通过，记录在 `evidence/preprocessing_runtime_equivalence.json`。
