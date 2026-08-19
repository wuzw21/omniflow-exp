# OmniFlow-exp 顶层设计与精简边界

本文回答三个问题：仓库真正的运行路径是什么、哪些文件拥有哪种语义、哪些
“旁路”必须保留但不应再复制实现。文件级编辑位置见
[`FILE_EDIT_GUIDE.md`](FILE_EDIT_GUIDE.md)；本文件不替代它。

## 1. 仓库职责

本仓库只承载论文 AndroidWorld 实验和 B-MoCA 验证。它不是通用产品仓库，
也不保存 RunLog、截图、模型、APK、模拟器镜像或正式结果数据。运行时资产
统一位于 `data/`，代码只读取 `data/current.json` 这一入口。

顶层可以压缩成四个概念：

| 概念 | 唯一拥有者 | 责任 |
| --- | --- | --- |
| 任务执行 | `scripts/exp/run_androidworld.sh` → `src/experiment/e2e_task_pipeline.py` → `src/experiment/androidworld.py` | 选择任务、方法、设备，并运行一个原子结果 |
| Native episode | `src/integrations/android_world/launch.py` | 创建 AndroidWorld/B-MoCA 环境、调用官方 validator、封存 RunLog |
| Function 生命周期 | `omniflow/functions/assets.py::save_function` | 从一份成功 RunLog 编译、验证并原子写入一个 Function Store |
| 本地证据索引 | `src/experiment/data_index.py` | 物化并读取 `data/current.json`；运行时不扫描替代索引 |

## 2. 唯一运行路径

```text
scripts/exp/run_androidworld.sh       # 唯一公开入口
  ├─ source / development / formal
  ├─ B-MoCA campaign preparation
  └─ src/experiment/e2e_task_pipeline.py  # 唯一 task + method + device 调度器
       └─ src/experiment/androidworld.py  # 一个原子 AndroidWorld 结果
            └─ src/integrations/android_world/launch.py  # 一个 native episode
                 └─ src/integrations/android_world/methods.py
                      ├─ fixed_replay
                      ├─ omniflow
                      ├─ mobilegpt
                      ├─ appagent
                      └─ t3a_hint
```

The boundaries inside this path are intentionally different:

- `e2e_task_pipeline.py` schedules one task-major attempt and interprets the
  result; it does not own AndroidWorld lifecycle or a second replay runner.
- `androidworld.py` translates protocol records into command specifications and
  collects result evidence; it does not own child-process lifecycle.
- `paths.py` owns repository-relative resolution, index-relative evidence
  references, and safe artifact components. This unifies path rules without
  moving external AndroidWorld, OmniTransfer, or B-MoCA roots into `data/`.
- `process_runner.py` is the single process-group, timeout, and immutable-log
  seam shared by AndroidWorld and B-MoCA experiment commands.
- `launch.py` owns one native AndroidWorld episode, including setup,
  observation/action recording, official validation, and teardown.
- `methods.py` resolves one method adapter. Direct Function replay is carried
  into `agent.py` as an execution decision, not implemented by launcher-side
  mutation of an agent instance.

普通 `omniflow` 的运行核心继续是：

```text
OmniFlow.run()
  -> Planner / Function selection
  -> Function-local checker evaluation
  -> OmniTransfer candidate mapping
  -> target execution
  -> mapping failure => normal VLM fallback
```

Formal Function 动作绝不能回放 source 坐标。OmniTransfer 映射失败就是一次
Function 步骤失败，并回到 Planner 的正常 fallback；任何隐藏的 resource-id、
node-id、坐标直通分支都违反这个接口。

## 3. 旁路的正确形态

“旁路”指的是相同执行语义被不同入口各写一遍，不是“有一种不走 Planner 的
合法模式”。单 Function 直跑有价值，因为它用于 B-MoCA source gate、离线
重放和零 model-call 验证；它不应被删除，也不应拥有自己的 mapper、executor、
resume state 或结果格式。

当前保留的复用点是 `src/integrations/script_replay.py`：它只选择一个完整
Function，随后调用 `agent_instance.call_tool()` / canonical runtime，测试中
必须能证明它没有私有 action mapping。AndroidWorld source qualification
通过 `build_e2e_command(function_id=..., function_arguments=...)` 进入同一
launcher；`methods.py` 把这个意图交给 `agent.py`，由正常 `step()` cycle
注入 `ToolCall`。因此 direct replay 和普通 `OmniFlow.run()` 共享 Host、
OmniTransfer、checker、证据封存和结果归档，且不会因为 CLI 入口不同而产生
第二个 executor。

## 4. 索引、ledger 和历史输入的区别

| 对象 | 是否运行时入口 | 语义 |
| --- | --- | --- |
| `data/current.json` | 是，唯一 | source RunLog、Function Store、transfer states、method memory、result rows 的内联索引 |
| `result_registry` ledger | 否 | 不可变注册证据；由 `current.json` 引用，不能成为第二个选择器 |
| `batch_outcomes.py` 的汇总 | 否 | 从一次尝试和注册结果生成报告/compact row，不拥有调度 |
| MobileGPT/AppAgent manifest | 否 | 外部 baseline 的不可变证据，不是 Function catalog |
| 旧 source/index/manifest | 否 | 只可在迁移或兼容读路径中使用；不得写入新的运行副本 |

因此“只保留一套索引”不是删除所有 manifest 或 ledger，而是保证只有
`current.json` 参与运行时解析；其他文件各自只承担证据或统计语义。

## 5. 当前精简候选与保留决定

### 五个大文件的直白职责

这五个文件不是五个重复的 runner，而是 Source RunLog 从输入证据变成可运行
baseline 记录时经过的五个位置：

| 文件 | 只应该回答的问题 | 不应该知道的事情 |
| --- | --- | --- |
| `source_evidence.py` | 这份 Source RunLog 的截图、XML、动作和 revision 是否可信？ | AppAgent/MobileGPT 的转换规则 |
| `appagent.py` | 如何把可信 source 转成 AppAgent 的 Prepared Memory？ | task 调度、Local Index、AndroidWorld episode 生命周期 |
| `mobilegpt.py` | 如何把可信 source 转成 MobileGPT 的 Prepared Memory？ | AppAgent 规则、Local Index 选择策略 |
| `checks.py` | 这次运行的依赖、设备和 Prepared Memory 是否 ready？ | 具体 provider 的转换实现 |
| `data_index.py` | 如何物化和读取唯一 Local Index？ | AndroidWorld runner 和 provider 内部校验细节 |

`source_evidence.py` 曾经暴露 `convert_runlog_memory(method=...)`，让共享 source
模块根据字符串选择 provider。这是已经删除的浅 seam：AppAgent 直接调用
`convert_runlog_to_appagent_memory`，MobileGPT 直接调用
`convert_runlog_to_mobilegpt_bundle`。新增 provider 时应新增自己的 adapter，
而不是把分派字符串重新塞回 source 层。

| 候选 | 判断 | 后续动作 |
| --- | --- | --- |
| `launch.py` 中的 direct Function 调用与普通 OmniFlow 运行 | 真旁路：生命周期相似、调用语义可共享 | 通过 E2E 请求 seam 收敛，不删除直跑能力 |
| `script_replay.py` 与 runtime execution | 不是重复 mapper；前者是薄适配器，后者是核心实现 | 保留，继续用测试锁住“无私有 mapping” |
| `data_index.py` 与 result ledger | 读写对象不同，不能粗暴合并 | 保留职责，拆出只在有测试证明时进行 |
| `batch_outcomes.py` 与 `result_registry.py` | 汇总和注册是两个不可互换的写入语义 | 先记录公共 path helper 重复，再局部收敛 |
| `omniflow/runlog.py` 与 `src/integrations/runlog.py` | canonical loader 与历史外部导入 adapter | 旧适配层按要求暂不清理 |
| `mobilegpt_source.py` / `appagent_source.py` | 两个外部协议不同 | 不合并；只共享纯证据 helper |
| 2430 行 shell、3093 行 scheduler、4619 行 launcher | 复杂文件，但都是现有唯一 owner | 先补 seam 和测试，再按模块内聚拆分；不复制入口 |
| 已删除的 packaged catalog、old source pool、direct launcher 文件 | 最近提交已清掉的死路径 | 不恢复 alias 或兼容副本 |

## 6. 变更顺序

每个重构遵循下面的顺序：

1. 先在本文件和 `FILE_EDIT_GUIDE.md` 写清 owner、接口和不变量。
2. 先改现有 owner，优先让旁路通过共享接口表达。
3. focused tests 证明行为仍在，再删除无调用者的 helper/文件。
4. 不触及 schema、public result row、统计表；若必须触及，单独 commit。
5. 每个语义组独立 commit 并 push；最后再运行完整测试。

## 7. 新人最短路径

```text
README.md
  -> docs/ARCHITECTURE.md
  -> docs/FILE_EDIT_GUIDE.md
  -> scripts/exp/README.md
  -> 对应目录 README
  -> owner 文件及其 focused test
```

公共命令只有：

```bash
bash scripts/exp/run_androidworld.sh ...
```

Function 写入只有 `save_function`；运行时 Function 只从已注册 Store 和
`data/current.json` 读取；OmniTransfer checkout 只有
`~/Projects/Omni/OmniTransfer`。
