# MobileGPT RunLog 转换记录

本文件只记录 `RunLog → MobileGPT Prepared Memory` 的离线转换工作。真实
AndroidWorld 实验结果仍以运行结果、官方 validator 和独立实验工作簿为准；
离线转换测试不占用、也不填充实验 cell。

## 2026-08-23：RunLog 约束的官方 authoring

- 状态：完成离线实现与 focused tests；未启动模拟器，未执行真实任务。
- 输入：完整且 official validator 成功的 AndroidWorld RunLog。
- authoring：继续调用 MobileGPT 官方 Server、Agent、Memory、XML encoder 和
  reader；在官方 prompt 的 user-message 边界追加当前 RunLog 的权威动作证据。
- 保存条件：官方执行流实际发出的动作必须与源 RunLog 逐步对齐。`click` 等
  动作严格比较目标 index；`scroll` 允许官方补充执行所需的容器 index，但
  动作类型和方向不得改变。
- 失败条件：模型改选目标、改选动作、提前 `finish`、生成额外设备动作，或
  保存后 reader 回读不一致时，转换失败且不能封存为可用 Memory。
- 空响应：有限重试仍携带同一条 RunLog teacher evidence，不会退回无约束 prompt。
- 离线验证：7 个 focused tests 通过，其中包含真实官方 Server/Agent/Memory
  代码路径的 end-to-end 测试；模型与 embedding 使用本地测试 provider，未调用
  外部 API。

## 实验组登记合同

每完成一组真实实验，才向独立工作簿的 `Codex Results` 页追加记录。记录至少
包含 task、method、device、source/evaluation seed、cell status、outcome、官方
validator、动作数、Function 命中与覆盖、VLM/model calls、平均与总 token、
memory lookup/hit rate、fallback、latency、energy、RunLog、result root 和失败分类。
没有完成真实执行的环境错误或仅离线转换验证，不得冒充 PASS/FAIL cell。
