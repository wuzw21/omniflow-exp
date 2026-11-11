# 给下一位接手者

状态：2026-10-31。先读本文，再读 [EXPLORATION](EXPLORATION.md)；原始文件按需查阅，不必重新开展一轮广泛选题。

## 2026-11-02：研究主线转为 Agent 迁移与 Harness 变化

用户已将下一步主线改为：研究云端 Agent 迁移到个人异构设备时，模型、工具接口和 Harness 同时变化如何被验证和适配。这个方向的规格与第一轮最小实验协议见 [Agent Migration Spec](AGENT_MIGRATION_SPEC.md) 和 [Harness Migration Protocol](../experiments/agent_migration/PROTOCOL.md)。这两份文件是研究草案，尚无运行结果，不能把它们写成已验证方法。

第一轮必须固定模型、设备和工具，只比较同一 OpenJarvis 版本内的 `NativeOpenHandsAgent`（CodeAct）与 `NativeReActAgent`（Thought-Action-Observation）。直接复制、人工移植和 canonical event IR 编译三者都要以 fixture 最终状态验收；若 Harness-only 对照没有真实迁移损失，停止扩展，不进入模型、Engine、KV 或端云路由。

已完成一个脚本化 engine 的 Harness 语义探针：[probe results](../runs/agent_migration_harness_probe_001/RESULTS.md)。同一 `move_file` 任务在两个 Harness 中最终状态一致，但模型输出协议、消息角色、engine 参数和控制循环不同。该结果只证明迁移对象确实不同，尚未证明真实模型下存在任务损失或编译收益；下一步必须加入失败恢复、验证失败和 turn limit，并使用真实模型。

## 已确定的任务与最新结果

最新用户要求：去除 C 的历史拼接，保持记忆策略和采样，再验等并与 B 重复计时。已完成，详见 [最新结果](../experiments/fluxmem_incremental_compact/RESULTS.md)。不要自动设计新记忆策略、扩展框架或包装贡献。

- C1 为前一轮稠密历史状态；C2 用固定五槽短期环形缓冲、逐帧保留特征与绝对索引。原匹配、Otsu、剪枝、长期合并函数直接调用，作者 checkout 未改。
- 固定 1 FPS、两帧块，前缀 2–82 秒共 41 个；原问题 4、53、82 秒。53 秒只读 52 个配对帧。C2 与作者全重算及 C1 在全部前缀精确一致；九次完整生成的记忆张量、首 logits、生成 token 均一致。
- 同一 41 块：追加 cat 从 280 次/808.91 MiB 逻辑输出变为 0 次。新状态仍有入库复制、索引选择和查询时输出组装，不能称零复制。
- 82 秒末持久 tensor storage：B 19.22、C1 57.87、C2 14.45 MiB。排除共同输入、验证参考、权重和临时输出。短前缀预分配可能使 C2 大于 B；长期保留输出仍增长，不是有界整个系统。
- 12 对独立空状态维护重放：B/C2 中位数 3.750/2.388 秒，12/12 对 C2 更快。6 对实际视觉前端：中位数 32.453/31.761 秒，5/6 更快；平均配对节省 1.948 秒，bootstrap 95% 区间 [0.115,3.125]。同视频/单设备重复，前端整体尚不足以称稳定部署加速。
- 原三题视觉均 41 块；匹配 B/C2=125/77、剪枝=59/37、长期合并均 5。没有减少更多策略调用；变化是执行状态存储与原增量路径。
- 物理 M4 上重复读取、读出修改隔离、reset 重放、旧红新绿两块回归均通过。31 项仓库测试通过。旧故障及两轮证据保持原样。

## 继续工作的入口

1. 读 [compact/README](../experiments/fluxmem_incremental_compact/README.md)、[协议](../experiments/fluxmem_incremental_compact/PROTOCOL.md) 和实际结果。当前没有剩余运行中的实验，不要自动重启。
2. 新原始运行均完成：`runs/fluxmem_compact_validate_001`、`runs/fluxmem_compact_memory_001`、`runs/fluxmem_compact_frontend_001`、`runs/fluxmem_compact_qa_001`。小证据归档在 `experiments/fluxmem_incremental_compact/evidence`，大张量留在 runs，SHA 已记录。
3. 输入仍为 `runs/fluxmem_fixed_inputs_002`；真实参考特征来自 `runs/fluxmem_incremental_audit_001`。旧完整 QA 为 `runs/fluxmem_incremental_qa_003`。复跑必须指定新目录，不能覆盖旧结果。
4. 独立 `.venv-fluxmem`：Python 3.12 / Torch 2.13 / Transformers 4.50。作者代码 `6fdbf65`、权重 `66285546`、视频 `48872fa7`。Qwen 权重为非商业研究许可，不是无限制开源许可。
5. C2 可删除历史 raw/norm 的条件是固定原策略依赖已结束；它不支持从已丢弃信息回退到另一套策略。完整 QA 仍用状态文件交接验证，没有测只持有 14.45 MiB 状态的完整模型常驻服务、并发摄像头、能耗、多视频或崩溃恢复。
6. 上轮 meta buffer 加载失败、QA_001 footprint 近 15 GiB 中断及加载后清理修复保留在旧证据。不要把测试程序修正计为方法收益。SVMemAgent 阻塞、FluxMem 原始 18 次问答及早期 KV 探索也仍保留。

## 已有资产与环境

| 资产 | 已知位置 / 状态 | 接手注意 |
|---|---|---|
| 本仓库 | `~/Desktop/个人思考/omni-device-agent` | 本地 Git，尚无远端 |
| 原研究目录 | `~/Desktop/个人思考/edgecloud-research` | 未改动；历史文本、脚本、结果已冻结复制到 `experiments/legacy` |
| 本机 | Apple M4，Mac16,12，16 GiB，macOS 26.5.1 | 既有实验真实运行设备；不是手机/眼镜 |
| 当前实验 Python | `.venv-fluxmem/bin/python` | Python 3.12 / Torch 2.13 / Transformers 4.50；见 experiments/fluxmem/requirements.lock |
| 旧视觉实验 Python | `../edgecloud-research/.venv-latent/bin/python` | 保留，不修改 |
| 旧 Stage A 环境 | `../edgecloud-research/.venv-stage-a/bin/python` | 与 latent 环境不同；不要混装 |
| 公开研究缓存 | `~/.agent-reach/research/` | 没有收入 Git；manifest 中保留了原机器绝对路径 |
| Qwen3-VL-2B | `~/.agent-reach/research/mobile-prefill-replay/model_path.txt` 指向固定 snapshot | 旧试验用，不是 SVMemAgent QA 原配置 |
| 公开 139 张截图 | `~/.agent-reach/research/stream-gate-pilot/` | 哈希在旧 manifest；未复制图片进 Git |
| CUDA 入口 | 用户此前授权 `ssh 9207` | 既往使用四张 RTX 4090；不是端侧结果的替身 |

旧远端曾遇到 ENOSPC、RAM 目录重启消失、cuDNN 依赖失败。之后曾观察到 GPU 空闲，但不能据此认定存储和运行时都恢复。开始新任务前只读复查磁盘、环境、现有任务和模型路径；旧 PID、队列 JSON 和 `/dev/shm` 路径不是当前状态。不要复制 SSH 配置、token 或凭据进仓库。

候选 HF 仓库 `temptemp1234/SVMemAgent` 已固定元数据 revision `7b7ba15a2d35064287f449259607759e36259440`，取得配置，未下载大权重；没有作者正式发布确认。缺 `VideoStreamingMemoryAgent` 实现与配置注册，没有已验证可执行的原版命令。不要把 Qwen、DINO 缓存或同名权重存在等同于可运行的作者策略。

## 快速核验交接

在仓库根运行：

```sh
python3 scripts/verify_snapshot.py
cd experiments/legacy
../../../edgecloud-research/.venv-latent/bin/python -m unittest -v test_stream_gate
```

第二条测试依赖旧 manifest 中的公开图像缓存、NumPy；换机器需先恢复文件或重定位派生 run，不能改冻结结果来让测试绿。图像来源和 SHA-256 已记录；下载公开数据应固定 revision，原 `prepare.py` 还依赖 Hcompany 索引与原轨迹 JSON，并非无依赖安装器。

新一轮运行须复制必要代码到新的 `runs/<run-id>/`，给它独立输出，不在冻结目录直接执行会改写 analysis/manifest 的脚本。benchmark 与 QA 脚本已有“输出存在则拒绝覆盖”保护。遗留完整模型重跑还依赖抽取的视觉权重和原环境；仓库迁移完整性不等于干净机器一键复现。

本轮验证结果见 [validation](../validation/README.md)。未重跑全部历史实验，也未测 SVMemAgent、连续摄像头、功耗或真机 Android 闭环。

## 不要遗失的判断

- 128 道 P2 已完成；384 道 held-out 未运行。联合 oracle 与简单加法的最佳静态 F1 几乎相同，暂不训练复杂联合修正器。
- Minions 的长文件并未被端云共同反复读取。独立上下文不自动意味着无前缀复用；本地工具也不自动意味着本地模型必要。
- Pixel gate 的时间减少伴随证据损失；它不是新方法成功，不是 ActiveSight/SVMemAgent 复现。
- MiniCPM-o 已有流水、双工和异构后端；源码候选问题没有实际耗时归因。不要重新把“一体化多模态”当空白。
- ActiveSight＋SVMemAgent 可能已经足够好；必须用强基线挑战想法，不能人为使其失败。
- 用户允许建立在已有方法上，扩大规模、完成系统。不要因为已有文献就不断换题。

## 建议使用的技能

- `agent-reach`：定位作者官方项目与权重；当前旧 CLI 曾报模块缺失，必要时使用可用检索工具，不擅自修全局环境。
- `codebase-recon`：取得作者代码后识别原始入口与依赖。
- `diagnosing-bugs`：出现可复现兼容或性能问题时先定位原因，保持作者基线。
- `academic-plotting`：有真实分解数据后绘制成本图并实际渲染检查。
- `handoff`：下一次交接更新本文和实验账本，引用已有资产，不复制私人对话。

按需使用，不是必须把全部技能走一遍。仓库 AGENTS.md 保存了用户的真机验收、回归测试、复用上下文压缩实现与科研写作规则。

## 2026-11-01：长视频与查询密度 scaling

最新任务已完成：同一真实 sample_1 视频扩展至 188 秒，94 个前缀，查询间隔 40/10/4 秒。详见 scaling 结果和 Intro/Story。

- 固定采样和 FluxMem short=8/mid=64 策略不变；前 41 块哈希与上一轮一致。C2 与原版全量记忆在 94 个审计前缀逐位一致，最终 4258 tokens。
- 188 秒读出数为 6/20/48。B 记忆维护中位数 17.45/56.70/104.11 秒；C2 为 5.79/5.69/4.92 秒。B 的增长来自每次读出重新扫描历史；C2 的策略维护只随新增块发生，但查询组装仍保留。
- 188 秒持久 tensor state 为 B 44.06、C2 19.03 MiB；C2 从 82 秒 14.45 MiB 增长，长期保留输出不是有界状态。
- 含真实视觉编码的前端只做一对 BC/CB/密度：B/C2 为 83.45/71.24、114.35/71.83、198.99/82.11 秒；这是 scaling 观察，不是稳定性区间。
- 原始运行：runs/fluxmem_scaling_inputs_002、runs/fluxmem_scaling_features_001、runs/fluxmem_scaling_audit_001、runs/fluxmem_scaling_mem_{40,10,4}s_001、runs/fluxmem_scaling_front_{40,10,4}s_001。报告已归档在 experiments/fluxmem_scaling/result_001。本轮没有执行完整语言模型 QA，因为 94 个前缀的视觉记忆验等已直接覆盖执行状态；没有声称 agent 质量。
- 物理前端峰值分别约为 B 3.07/4.16/6.34 GiB、C2 3.05/3.12/3.12 GiB（按 40/10/4 秒密度的一对运行），包含视觉权重与运行时公共状态，不是设备部署峰值。
