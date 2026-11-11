# 原实现复用边界

来源：ShareLab-SII/FluxMem，commit `6fdbf65b6f2af1a94ca41d542be5be86a565d338`，Apache-2.0，许可证副本为 `LICENSE.upstream`。作者源码 checkout 未修改。具体实现位于 `models/qwen2-5-vl/src/qwen2_5_vl_fluxmem/fluxmem.py`。

`state.py` 的队列推进顺序依据作者 `process_memory_streaming`，将原来的函数局部变量变为跨调用状态；直接调用作者 `_frame_grids`、`_max_sim_against_frames`、`_apply_adjacent_pruning`、`_long_term_memory_merge_per_frames`。不重写 Otsu、阈值、邻接、聚类、锚点或合并规则。原 API 没有可续接的状态参数，故需要这一执行适配。

全量对照始终调用作者完整 `process_memory_streaming`。QA 读出继续调用作者 `_pack_sample`、`right_pad_and_stack`，复用真实文本位置和 RoPE。归一化仍是作者同一 `torch.nn.functional.normalize(p=2,dim=1,eps=1e-8)`，只将处理范围改成新增块；实际逐元素核对其结果。

约束：当前仅验证单视频、batch=1、固定分辨率、固定采样时钟、完整两帧块、单调前缀。没有宣称通用流式服务、跨模型复用、并发快照、崩溃恢复或有界长期状态。reset 后重放与同状态重复读取由真机回归验证。

权重与环境沿用 `../fluxmem` 的锁文件和模型 manifest；3B 权重仍遵循其非商业研究许可，不能把上游代码许可证套用到权重。
