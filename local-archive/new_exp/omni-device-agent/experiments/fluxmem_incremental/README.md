# 固定采样与增量执行等价性实验

已完成的真机结果见 [RESULTS.md](RESULTS.md)：41 个连续前缀及三题×三条件的完整模型核对均精确一致。

本目录只改变 FluxMem 的执行方式。原 Otsu、相邻匹配、剪枝、长期聚类均从固定作者 checkout 调用，没有新策略、量化、训练或 KV 复用。A 全部重算；B 缓存视觉特征、重算记忆；C 缓存视觉并增量维护记忆。实验边界先写入 [PROTOCOL.md](PROTOCOL.md)，复用范围见 [UPSTREAM.md](UPSTREAM.md)。

使用现有 `.venv-fluxmem` 和公开 sample_1 视频；首次获取环境、官方权重及数据见 [基线复跑入口](../fluxmem/README.md)。以下命令在仓库根执行，所有输出目录必须是新目录：

```bash
TOKENIZERS_PARALLELISM=false .venv-fluxmem/bin/python experiments/fluxmem_incremental/prepare.py \
 --output runs/fluxmem_fixed_inputs_new

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-fluxmem/bin/python experiments/fluxmem_incremental/audit.py \
 --inputs runs/fluxmem_fixed_inputs_new/manifest.json \
 --output runs/fluxmem_incremental_audit_new

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-fluxmem/bin/python experiments/fluxmem_incremental/qa.py \
 --inputs runs/fluxmem_fixed_inputs_new/manifest.json \
 --audit runs/fluxmem_incremental_audit_new/results.json \
 --output runs/fluxmem_incremental_qa_new

.venv-fluxmem/bin/python experiments/fluxmem_incremental/replay_regression.py \
 --audit runs/fluxmem_incremental_audit_new/results.json \
 --output runs/fluxmem_incremental_replay_new.json

python3 experiments/fluxmem_incremental/summarize.py \
 --audit runs/fluxmem_incremental_audit_new/results.json \
 --qa runs/fluxmem_incremental_qa_new/results.json \
 --replay runs/fluxmem_incremental_replay_new.json \
 --output runs/fluxmem_incremental_summary_new.json
```

`audit.py --limit 5` 只验证到首次短期淘汰，不能替代默认 41 个前缀的长期路径验收。QA 要求完整 audit 成功，拒绝拿短 smoke 生成完整结果。

连续前缀审计加载官方权重中的完整视觉塔，另外读取两个真实特殊 token embedding，使用作者 RoPE 和完整记忆函数作 oracle。比较的是实际视觉输出及压缩结果，不是模拟分数。原始三题另加载完整语言模型，验证真实文本位置、RoPE、attention mask、首 token logits 和生成 token。

`state.py` 只维护原算法的执行队列、距离、mask 与特征，所有选择委托给原 helper。读出时在 mask 副本上标记短期保留，防止查询污染下一次淘汰。当前为验等保留原始、归一化、合并后三份特征；额外状态不是最小内存下界。追加使用 `torch.cat`，复制开销单独记录，不将它描述为所有开销都与新增帧数成正比。

统计区分：持久 tensor storage（去重）、Python 对象估计、临时输入/拼接/输出、模型权重、审计副本和落盘证据。共同的可重读视频/预处理输入不计作某条件的额外缓存；A 的零持久神经状态不代表零总内存或原始历史免费。审计中三种状态同时存在，进程峰值不能当作单条件部署峰值。

QA 通过已验证的状态文件衔接两个进程，目的是验证完整模型等价性。B/C 的视觉与更新成本已经在前一阶段记录，不能把 QA 阶段缓存命中说成零维护成本。QA 计时包含导出中间结果用于核对的开销，不报告服务加速倍数。
