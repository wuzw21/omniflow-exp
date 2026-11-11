# 本次建仓验证

执行日期：2026-10-31。以下均从新仓库路径执行，没有改变旧研究目录或重跑 GPU/MPS 模型实验。

| 检查 | 实际结果 | 证明范围 |
|---|---|---|
| 冻结快照 SHA-256 | 529/529 文件一致 | 迁移未改变归档字节 |
| `test_stream_gate` | 10 项通过，无 skip | 因果前缀、容量与淘汰、公开 139 图哈希、记录中的逐步选择、72 条计时与实际 QA 失败案例一致性 |
| Stage A 最终结果相关回归 | 21 项通过，无 skip | 128 题归档、合并、静态与 oracle 选择、固定目标双因素约束 |

日志：[完整性](snapshot.log)、[门控回归](stream_gate.log)、[Stage A 回归](stage_a_final.log)。命令、工作目录、退出码保存在 [results.json](results.json)。未执行的其他历史回归没有被计作通过。

```sh
# 从仓库根运行
python3 scripts/verify_snapshot.py
cd experiments/legacy
../../../edgecloud-research/.venv-latent/bin/python -m unittest -v test_stream_gate
../../../edgecloud-research/.venv-stage-a/bin/python -m unittest -v \
  test_stage_a_full_p2_artifacts test_merge_stage_a_factorial \
  test_stage_a_selection test_stage_a_factorial
```

依赖原机器环境和公开缓存。原生算子的真机记录发生在历史实验，设备/版本/设置在各报告中；本次回归读取其证据，没有再次执行推理。SVMemAgent 原版模型、手机/穿戴设备闭环、持续摄像头、功耗与新方法效果均**未验证**。本次工作是建仓交接，不是 bug 修复验收。

新撰写文档的仓库内链接另做了存在性检查；旧归档保持原样，其中外部缓存、上游源码和相对旧环境的引用不保证跨机器可用。没有将复制成功称为干净环境完整复现成功。

全仓库 `git diff --cached --check` 检出了原始日志和四份历史 Python 文件中的尾随空白/末尾空行，未通过该项格式检查。为保持归档哈希，不清洗历史证据；仅针对新增文件排除 `experiments/legacy` 后的格式检查通过。这不影响上述实际回归结果。

## 2026-10-31：FluxMem 真机运行

- 物理设备：Apple M4 / 16 GiB / macOS 26.5.1；完整 Qwen2.5-VL-3B BF16 参数在 MPS。
- 作者源码未改。Torch 2.6 的三维 SDPA、2.9 的长期合并 int64 scatter-reduce 失败均保留；Torch 2.13 在新进程完成短/长期 smoke 与 18 次正式问答、两次预热。两项故障算子的 MPS 回归也实际通过。
- 原视频短前缀与限内存读取器像素/FPS 相同；更换运行时后重新准备三题，所有实际输入张量逐元素相同。没有未来帧。
- `python3 -m unittest discover -s tests -v`：14 项通过，含 10 项 FluxMem 证据约束及 4 项旧 SVMemAgent 约束。检查真实运行记录，不以 mock 代替模型验收。
- `python3 scripts/verify_snapshot.py`：529 个历史文件仍完整；没有改冻结实验。
- 运行记录、时间与内存口径见 [FluxMem 结果](../experiments/fluxmem/RESULTS.md)。工作中设备存在明显时间漂移与内存压力，未建立稳定加速结论。未测手机、摄像头、功耗或独立后台 writer。

## 2026-10-31：固定采样三个执行对照

- 物理 M4 / Torch 2.13：41 个连续前缀、三题×三条件的真实完整模型核对通过。视觉、保留位置、合并值、实际 QA 输入张量、首 token logits、输出 token 精确相同。
- 原策略 helper 不变；首次短期剪枝与首次长期合并边界均执行。重复读出不污染状态，reset 后重放一致。向独立对象注入短期 mask 写回错误，真实 MPS 回归能检测到差异。
- 验证程序 meta buffer 加载失败、QA 加载/清理顺序下的 footprint 膨胀均保留。正常构造视觉塔、首个输入上传前清理缓存后，在新进程完成全部核对；不是只靠源码检查验收。
- `python3 -m unittest discover -s tests -v`：23 项通过（新增 9 项）。证据见 [实验结果](../experiments/fluxmem_incremental/RESULTS.md)，实际执行脚本可重跑，不以这些记录检查代替新硬件验收。
- 未测有界长期状态、多视频、摄像头、功耗、并发读写或持久化崩溃恢复；没有据单次计时宣称可靠加速。

## C 去除历史拼接的真机验收

- 设备：同一物理 Apple M4 / 16 GiB / macOS 26.5.1；Torch 2.13、Transformers 4.50、原 Qwen2.5-VL-3B BF16。固定此前视频与采样，不改原策略 helper。
- 最小实际操作：向执行状态追加两块真实视觉特征。旧 C 的回归因 7 次历史 cat 失败，新 C 为 0 次并通过；41 块全流程中旧 280 次、新 0 次。真实 MPS 算子跟踪，不是源码断言。
- 41 前缀与原版全重算/C1 逐位相同；环形缓冲跨槽复用、首次剪枝/长期合并、重复读出、读出结果外部修改隔离、reset 后五块重放均通过。完整模型三题九次输出/logits 全一致。
- 12 对空状态维护、6 对独立视觉编码前端按 BC/CB 交替重复；每轮重新建立状态。计时结束后才运行完整 QA 和回归，避免本任务内部 GPU 争用。
- `python3 -m unittest discover -s tests -v`：31 项通过（新增 8 项）。新回归同时执行旧红新绿真机入口并核验完整模型证据；[执行日志](../experiments/fluxmem_incremental_compact/evidence/regression_tests.log)。
- 报告的是最终持久 tensor storage，不是峰值。未验收完整模型常驻流式服务、手机、能耗或任意策略切换。没有把 6 对前端计时包装为稳定部署加速。[结果与原始证据](../experiments/fluxmem_incremental_compact/RESULTS.md)。

## 长视频与查询密度 scaling 真机验收

- 同一 Apple M4 / 16 GiB，真实 sample_1.mp4 扩展至 188 秒；1 fps、两帧块、Qwen2.5-VL-3B BF16；前 41 个视觉块哈希与上一轮一致。
- 94 个前缀中 C2 与原版全量记忆逐位一致，末端 4258 个保留 token。测试使用真实视觉塔输出和原作者 FluxMem helper。
- 40/10/4 秒查询间隔分别为 6/20/48 次读出。B 的记忆维护随查询变密从 17.45 增至 104.11 秒中位数；C2 为 5.79、5.69、4.92 秒。含视觉前端的单对 BC/CB 均由 C2 更快，但每密度只有一对，不能当稳定性区间。
- 188 秒末持久 tensor state B/C2 为 44.06/19.03 MiB；C2 状态仍随长期保留 token 增长。物理 footprint 记录在各前端 JSON，不等同部署峰值。
- python3 -m unittest tests/test_fluxmem_scaling.py -v：5 项通过；曲线与原始记录见 [scaling 结果](../experiments/fluxmem_scaling/result_001/RESULTS.md)。
