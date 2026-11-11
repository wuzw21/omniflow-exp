# new_exp

只放 MobiCom'27 rebuttal 补充实验的产出。

目录：
- `omni-device-agent/` — Agent Migration 新方向源码（11/10 从项目根移入）
- `ANDROIDWORLD_EXPERIMENT_DESIGN.md` — 重新设计后的 AndroidWorld 补实验方案（v2）
- `REBUTTAL_EVIDENCE_FROM_FROZEN.md` — 从 9/3 冻结数据可直接写进 rebuttal 的数字与口径边界
- `REBUTTAL_ANSWERS_FROM_FROZEN.md` — 上述四个结论针对 C/E 原问题的正式回复文字（可定稿，补现有草稿未覆盖的 4 问）
- `run_rebuttal.sh` — 分层启动器（默认 dry-run，L0 门禁未过拒绝开跑）
- `runs/` — 各轮实验结果，按 `<date>-<name>/` 落盘；`runs/20261110-l0/L0_RESULT.md` 是开跑前的环境核查结论

核心依据（原位引用，不复制、不改动）：
- 论文定稿：`../mobicom27_gui_agent.pdf`
- rebuttal：`../rebuttal.md`
- 回复草稿：`../src/OmniFlow-exp/docs/REBUTTAL_DRAFT_20261107.md`

rebuttal 承诺的 6 项补实验：
1. state-matching / action-transfer 与 prior methods 对比（A）
2. 工作流中断鲁棒性：app 更新、重复弹窗（B/C/D）
3. 真机 full-task success / latency / energy（C/D）
4. Function 分解与中间态重放消融、阈值敏感性（B/C/D）
5. GUI 状态匹配与动作迁移的 false-positive/negative 人工量化（A/B/C/D/E）
6. 全部 attempts 的 token 节省（C）

执行入口：`../src/OmniFlow-exp/scripts/exp/rebuttal_matrix.sh`
（L1/L2/L4、no_reentry、no_checker、threshold、page）与 `--baseline all`。

口径：论文 53.2% / 37.1% 带 oracle 条件，跨方法比较前须统一 `--completion-oracle off`；
`rebuttal_matrix.sh` 默认约 55 个任务，不能写成 116 全量。
新结果写本目录，不回填 9/3 冻结数据。
