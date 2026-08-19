# OmniFlow-exp 工作规则

本仓库只做论文 AndroidWorld 实验和 B-MoCA 验证。改代码前先读：

1. `README.md`
2. `docs/ARCHITECTURE.md`
3. `docs/FILE_EDIT_GUIDE.md`
4. `scripts/exp/README.md`
5. 对应目录 README 和 owner 测试

## 不可破坏的运行合同

- 唯一公开入口是 `scripts/exp/run_androidworld.sh`。
- 唯一 task + method + device 调度器是 `src/experiment/e2e_task_pipeline.py`。
- `src/experiment/androidworld.py` 只执行一个原子 AndroidWorld 结果。
- `src/integrations/android_world/launch.py` 是唯一 native episode/lifecycle owner。
- `save_function` 是唯一 Function 写 API、compiler 和 Store writer；成功 RunLog 只生成一个完整 Function。
- runtime 只读取注册的 Function Store 和 `data/current.json`，不能自动补 Store、建 catalog 或写平行 manifest。
- `data/current.json` 是唯一运行时本地索引；ledger、汇总和外部 manifest 只能作为证据。
- AndroidWorld 正式方法只有 `fixed_replay`、`omniflow`、`mobilegpt`、`appagent`、`t3a_hint`。
- B-MoCA 的 replay selector 属于外部 benchmark 合同，不复制进 AndroidWorld method 名称。

## OmniTransfer 合同

- OmniTransfer 的 canonical checkout 永远是 `~/Projects/Omni/OmniTransfer`。
- 页面编码只用 `omniflow/transfer/page_embedding.py` 和协议指定 checkpoint。
- 映射失败是明确的 transfer failure，回到正常 Planner/VLM fallback；绝不执行 source 坐标。
- 禁止 resource-id/node-id lookup、坐标 passthrough、第二 page encoder、第二 action mapper。
- Function checker 是 Function-local registration；只在 source action 成功映射且达到全局阈值时执行一次。

## 证据、环境和数据

- source Function 必须来自完整、成功、带截图和 native observation 的 RunLog；AndroidWorld source seed 是 111。
- B-MoCA env100 必须先通过 official success、method success、`model_calls=0`、`fallback_steps=0`，才可创建/运行其他环境。
- 所有 Python/Torch 命令使用 `~/Projects/Omni/OmniFlow-exp/.venv/bin/python`；正式执行不使用邻近环境。
- 所有实验资产、RunLog、截图、Store、transfer states、memory 和结果都在 `data/`；不要提交它们、credentials、APK、权重或 emulator image。
- 旧适配层和历史输入可按只读迁移路径保留；不要为了兼容重新加 alias、旧 writer、旧 index 或旧 runner。
- 单 Function 直跑是有价值的旁路，但必须复用共同 Host、OmniTransfer、checker、证据封存和结果路径；不要删除它，也不要复制执行实现。

## 修改方式

- 先按 `docs/FILE_EDIT_GUIDE.md` 找唯一 owner；一个概念只有一个生产实现。
- 删除前先用 `rg` 查所有调用者和测试；行为保留就把测试迁移到 surviving owner。
- shell、scheduler、launcher 等大文件先通过接口收敛，再拆分；不要增加第二入口。
- schema、public result row、统计文件或输出格式的改动必须单独 commit，并同时更新 schema、owner、测试和 README。
- 纯代码清理不得顺手修改 schema 或统计合同。
- 每个语义组先 focused test，再 `git diff --check`，然后 commit 并 push；完整测试最后运行。
- 本地编辑使用 `apply_patch`；命令、测试和转换使用仓库 `.venv/bin/python`。

## 入口说明

```text
bash scripts/exp/run_androidworld.sh ...
  -> src/experiment/e2e_task_pipeline.py
  -> src/experiment/androidworld.py
  -> src/integrations/android_world/launch.py
```

Function authoring 只走 bridge 的 `save_function`；管理工具只有 README 中列出的
Function/RunLog 工具。详细文件职责、旁路方案和索引区分见
`docs/ARCHITECTURE.md` 与 `docs/FILE_EDIT_GUIDE.md`。
