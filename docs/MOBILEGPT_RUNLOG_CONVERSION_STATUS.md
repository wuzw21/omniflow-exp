# MobileGPT RunLog 转换记录

本文件只记录 `RunLog → MobileGPT Prepared Memory` 的离线转换工作。真实
AndroidWorld 实验结果仍以运行结果、官方 validator 和独立实验工作簿为准；
离线转换测试不占用、也不填充实验 cell。

## 2026-08-25：RunLog 到官方 Memory schema 的机械转换

- 状态：完成离线实现与 focused tests；本流程不启动模拟器，不执行实时任务。
- 输入：完整且 official validator 成功的 AndroidWorld RunLog。
- 转换：不调用 Explore/Select/Derive，也不重新规划动作；RunLog 的每个已验证
  transition 映射为 MobileGPT 官方 action/subtask，页面通过官方 XML encoder，
  memory 通过官方 `Memory`/`PageManager` API 保存。
- schema 约束：`click`、`double_tap`、`long_press`、`input_text`、`scroll/swipe`
  和 `navigate_back` 分别映射到官方 action 名称；滚动动作必须带官方 scroll
  container `index`。`input_text` 保留 RunLog 的 concrete 文本，因为官方
  `adapt_action_to_arguments` 不会展开 `input_text` 字段中的占位符。
- 失败条件：官方 reader 无法直接读回动作、动作名/目标 index/方向/输入文本
  与 RunLog 不一致、finish 顺序错误，或 memory 图不完整时，转换失败且不能
  封存为可用 Memory。读回为 `examples` fallback 的记录会被审计为非 direct hit。
- 离线验证：focused tests 覆盖官方 Memory 写入、reader 读回、输入文本和滚动
  schema；模型调用为 0，embedding 使用测试 provider，未调用外部 API。

## 实验组登记合同

每完成一组真实实验，才向独立工作簿的 `Codex Results` 页追加记录。记录至少
包含 task、method、device、source/evaluation seed、cell status、outcome、官方
validator、动作数、Function 命中与覆盖、VLM/model calls、平均与总 token、
memory lookup/hit rate、fallback、latency、energy、RunLog、result root 和失败分类。
没有完成真实执行的环境错误或仅离线转换验证，不得冒充 PASS/FAIL cell。
