# 先复现 KVCMAS 的原始千问实验

2026-10-25：优先级调整为先复现作者已有基线。暂停扩展 τ² 接入和新的联合入口机制；已有诊断结果保留，但不计作 DroidSpeak/KVCMAS 复现。当前仅完成源码、数据与配置固定，模型实验尚未执行。

## 为什么选这一项

[作者仓库](https://github.com/hjeon2k/KVCMAS/tree/e2170cf4b1790bd93f98d15e926397999d590867) 的 `scripts/run_accuracy.sh` 直接包含 HumanEval＋Qwen2.5-Coder-7B-Instruct。它既符合用户指定千问的约束，又覆盖同模型、不同角色前文的共享上下文校正；无需先构造端云任务、替换模型或实现第二套 agent loop。

固定源码 revision `e2170cf4b1790bd93f98d15e926397999d590867`，原始 checkout 在 `/Users/wuzewen/.agent-reach/research/kvcmas-e2170cf`。75 个跟踪文件的 SHA256、原始 161 题的顺序、模型 revision 和参数保存在 [复现清单](results/kvcmas_reproduction_manifest.json)。没有修改作者代码；仓库未发现 LICENSE 文件，当前只在本机保留研究副本。

模型是 **Coder-7B-Instruct**，不能直接拿此前的 Qwen2.5-7B-Instruct 权重替代。HF revision 固定为 `c03e6d358207e414f1eca0bb1891e29f1db0e242`，尚未下载。该版本固定当前公开权重；作者未提供可用于核对的实验权重 revision，不能保证与论文训练时文件逐字节相同。

## 原始运行配置

- 方法：NonShared 与 KVCMAS，先保留上游两项原始入口。
- HumanEval 161 题，仓库自带数据，无重排、删题；先用前 2 题检查执行，再全量运行。
- 3 个 CodeWriting agent＋FinalRefer，FullConnected，1 round，batch 1；原始提示、生成和评分器保持不变。
- KVCMAS：chained；PH/base-key/base-value rank 32，PF rank 0；anchor pool 10，hot anchors 10，threshold 0.34。
- 原始环境要求 Python 3.10、Torch 2.8.0+cu128、Transformers 4.50.2、FlashAttention 2.8.3.post1。先验证原 requirements 可安装；不能沿用 τ²/MLX 环境并声称精确复现。
- 第一对运行 seed=42，与上游默认一致。论文为三次平均，但确切三组 seed 未从当前源码确认；全量重复实验应明确记录这一差别。

## 直接执行作者入口

以下命令需要在可写持久目录、已验证依赖和固定模型 snapshot 的 CUDA 主机上执行，目前未运行。snapshot 的完整路径应保留 `Qwen` 字样，因为上游 registry 根据模型名字符串选择本地 LLMChat；否则会进入 API 类路径。

```bash
cd /PATH/TO/kvcmas-e2170cf
git diff --exit-code HEAD --
export HF_HUB_OFFLINE=1
export KVCMAS_ATTN_IMPL=flash_attention_2
export SEED=42
export QWEN_SNAPSHOT=/CACHE/models--Qwen--Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242
OUT=/RESULTS/smoke_nonshared_s42 bash scripts/run_accuracy.sh humaneval nonshared 0 --llm_name "$QWEN_SNAPSHOT" --num_samples 2
OUT=/RESULTS/smoke_kvcmas_s42 bash scripts/run_accuracy.sh humaneval kvcmas 0 --llm_name "$QWEN_SNAPSHOT" --num_samples 2
```

两项真实模型运行、评分、缓存复用/回退路径通过后，用新的输出目录删除 `--num_samples 2` 执行全量。小样本只验收执行链路，不计算论文复现成功。记录逐题输出与失败、正确率、实际复用率、端到端时间、TTFT、峰值显存，并检查计时是否包含 anchor 更新。

若使用 4090，属于硬件变更后的复跑；可以测本机两方法的相对结果，不能把绝对延迟直接等同论文 A100 的数字。论文数字和本次结果必须分列，失败/OOM 不删。

## 不先做的实验及原因

1. 不将此结果称为跨模型或端云加速。先复现同一模型的不同前文校正，再改变一个条件做迁移实验。
2. 不首先复跑 vLLM QPS 图作为真实 agent 成绩。作者 `scripts/vllm/qps_lkv.py` 使用随机 token stream、设定的 dense/reuse 比例和计算成本参数构造负载，并启用原生 prefix caching；这是受控服务实验，不能替代原始准确率 runner 的真实交互与校正验收。
3. DroidSpeak 仍是后续跨模型参照，但本轮在正式论文、作者主页和 GitHub 检索中尚未定位可核验的作者实现；没有代码不能把自写层重算脚本称为它的完整复现。该结论仅表示本轮未找到，不断言作者未发布。

## 当前阻塞

9207 上专属持久目录写入 errno 28；旧内存盘权重和运行时消失。已有 Conda 导入 Torch 缺少 libcudnn.so.9。需要可写持久存储或另一可用 CUDA 主机，之后建立上述作者要求的隔离环境。没有在故障环境反复下载模型，也没有重写上游算法绕过此阻塞。

参考：[KVCMAS 论文](https://arxiv.org/html/2609.34060v1)、[原始精度入口](https://github.com/hjeon2k/KVCMAS/blob/e2170cf4b1790bd93f98d15e926397999d590867/scripts/run_accuracy.sh)、[DroidSpeak 作者主页](https://yuhanliu11.github.io/)。
