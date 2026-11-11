# 固定采样下去除历史状态拼接

此目录只修改上一轮 C 的执行状态，旧实现和证据保留在 `../fluxmem_incremental/`。C1 指旧 C，C2 指本轮版本，B 仍为缓存视觉后调用原版完整记忆函数。采样、短/中期窗口、模型与原策略不变。实验边界见 [PROTOCOL.md](PROTOCOL.md)。

`compact.py` 用五个时间块容量的环形缓冲保存短期匹配需要的归一化特征、距离和 mask。剪枝仍调用作者函数；剪枝结束后只保存该帧被保留的特征和绝对索引。长期合并调用作者的逐帧函数，局部索引按原顺序映射，空间位置不变。读出时按帧顺序拼接保留结果。它没有保留完整原始视觉副本，不支持从已丢弃信息回退到另一套策略；长期保留输出仍随历史增长，不是有界的整个记忆系统。

执行依赖边界来自固定版本作者实现：相邻匹配与剪枝只读短期邻帧；长期合并只读当前移入长期区帧内的已保留 token。原 BF16 路径在长期合并时重新以 FP32 归一化，C2 传 `hidden_norm=None` 沿用同一路径。没有复制策略实现或更改阈值。上游版本与许可证沿用 [复用记录](../fluxmem_incremental/UPSTREAM.md)。这些边界经过本视频所有 41 个前缀验等，不据此声称适配其他 FluxMem 版本或 dtype。

在仓库根运行；每次给新输出路径，拒绝覆盖历史结果。环境为既有 `.venv-fluxmem`，权重与固定输入沿用上一轮。以下验证读取真实模型的视觉输出，并不重新编码视频；前端计时则对每个条件独立运行真实视觉塔。

```bash
# 旧版应失败，新版应通过：两块真实特征、实际 MPS 算子。
.venv-fluxmem/bin/python experiments/fluxmem_incremental_compact/check_append.py --legacy
.venv-fluxmem/bin/python experiments/fluxmem_incremental_compact/check_append.py

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-fluxmem/bin/python experiments/fluxmem_incremental_compact/validate.py \
 --audit runs/fluxmem_incremental_audit_001/results.json --output runs/fluxmem_compact_validate_new

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-fluxmem/bin/python experiments/fluxmem_incremental_compact/benchmark.py \
 --audit runs/fluxmem_compact_validate_new/results.json --inputs runs/fluxmem_fixed_inputs_002/manifest.json \
 --output runs/fluxmem_compact_memory_new --pairs 12

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-fluxmem/bin/python experiments/fluxmem_incremental_compact/benchmark.py \
 --audit runs/fluxmem_compact_validate_new/results.json --inputs runs/fluxmem_fixed_inputs_002/manifest.json \
 --output runs/fluxmem_compact_frontend_new --pairs 6 --frontend

PYTORCH_ENABLE_MPS_FALLBACK=1 .venv-fluxmem/bin/python experiments/fluxmem_incremental/qa.py \
 --inputs runs/fluxmem_fixed_inputs_002/manifest.json --audit runs/fluxmem_compact_validate_new/results.json \
 --output runs/fluxmem_compact_qa_new

python3 -m unittest discover -s tests -v
```

重复计时包括三个原问题到达时的新特征入库、记忆执行及输出组装。每次从空状态开始，两条件各一次完整预热不纳入统计，轮换 BC/CB 顺序。视觉计时包括真实编码和输入上传；记忆重放使用同一真实特征，不代表完整前端。GPU 同步计时边界，算子跟踪和验等不在计时内。整次前端不包含语言模型回答。

持久状态报告按实际 tensor storage 去重；ring 的所有已分配槽均计入，而非只算当前占用。共同视频输入、验证所用参考特征、模型权重、临时输出与 allocator pool 另计。状态组件中有别名，不能直接把组件字节相加。仪器记录的进程峰值包括审计参考，不当作部署峰值。完整 QA 仍用跨进程状态文件核对，不能据此宣称已实现常驻流式服务。
