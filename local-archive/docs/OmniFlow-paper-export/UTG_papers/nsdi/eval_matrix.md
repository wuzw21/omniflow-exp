# NSDI Evaluation Matrix

本文件冻结当前论文的评测矩阵：**实验内容、指标、数据源、脚本、输出文件**。

原则：

- 先定实验内容和输出，再补跑数据。
- 没有登记在这里的评测，不进入正文主结果。
- 没有真实输出文件的项，必须标记为 `partial` 或 `missing`。

## Unified Runner

论文评测总入口固定为：

```bash
python scripts/nsdi_eval.py --all --device <adb-serial>
```

无设备时只允许做离线预检：

```bash
python scripts/nsdi_eval.py --offline
```

单项实验用 `--experiment <name>` 选择；统一 manifest 固定写到
`runtime/evals/nsdi/manifest.json`，主字段为 `experiments`。`--offline`
只能把 live 项标成 blocked 或 partial，不能把 offline oracle 写成 live
success。旧 selector 和旧输出目录仅作为兼容路径保留，不作为论文组织口径。

## Claim Support Ladder

本节把 `core_claims.md` 的 C1-C5 映射到从小到大的实验设计。目标是先用最小
可复现证据支撑机制，再逐步扩到论文主表；不要为了追求复杂度而引入新 workflow。

| Claim | `S` 最小机制证据 | `M` 小规模可冻结结果 | `L` 扩展 / live 结果 | 当前正文可写边界 |
|---|---|---|---|---|
| C1 cache-first Function execution | cache lookup regression、provider function-id contract；Contacts key-function sidecar | Omni representation recall：`runtime/evals/nsdi/recall/summary.json`；end-to-end success / efficiency：`runtime/evals/nsdi/e2e/summary.json` | Cross-device Function recall live：`runtime/evals/nsdi/recall/cross_device_live.json`，缺设备时 `blocked_live_device` | 可写 labeled Function recall 质量、0-token hit 和 route-only precondition；不可写未跑完的跨设备 headline |
| C2 privacy-safe page-vector representation / action transfer | page-vector schema/provider tests；`tests/vector/test_page_match_accuracy.py` 中 fixed/text/resource/row-anchor/null outcome cases | Source-to-current transfer offline oracle：`runtime/evals/nsdi/page_replay_baselines/headline/summary.json`；Rico baseline-failure recovery queue 需 human/live gold 后冻结 | Source-to-current live verify：`runtime/evals/nsdi/page_replay_baselines/headline/live_verify.json` with `--device` | 可写 offline representation/action-compatibility 与 privacy boundary；不可把 offline oracle 写成 live execution success |
| C3 run_log -> Function asset | `tests/integrations/test_utg_api.py` import/writeback tests；`tests/multi_function/` slot/noise/outlier recorded-run cases | Asset-yield compression：`runtime/evals/nsdi/rq2/summary.json` | 更多 AndroidWorld/OmniFlow tasks 的 asset-yield curve，尚未冻结 | 可写 7→1、4→1、11→2 当前压缩；不可写成熟线上 promotion/eviction |
| C4 control-aware execution and observability | control/offline replay/terminal verify tests；key-function wrong-start sidecar | Case-study cache-hit timeline：`runtime/evals/nsdi/rq6_case_study/summary.json`；end-to-end success / timing as efficiency support | 三 regime live event stream（hit + recovery + fallback），当前缺失 | 可写命中不是裸 replay、当前 cache-hit timeline；不可写完整 recovery 覆盖率或三 regime 图已完成 |
| C5 co-evolution through trajectory reuse | `tests/multi_user/` privacy/tagging/reuse tests；cross-user replay report | Multi-user report：`runtime/evals/nsdi/multi_user/report.json`；asset yield + multi-user reuse 组合推导集体探索降低 | 更多 user/task/app sweep 与真实 OmniFlow 上传，尚未冻结 | 可写当前 cross-user HIT、spurious hit、privacy leakage、token reduction；不可写生产级网络效应 |

### Minimal-To-Larger Experiment Order

每条 claim 的推荐执行顺序固定为：

1. 先跑 `S`：定向 pytest 或现有 sidecar，确认机制没有回退。
2. 再跑 `M`：使用对应实验名生成或刷新 frozen summary。
3. 最后跑 `L`：只有在设备、人工 gold 或真实 OmniFlow 上传证据齐备时，才把 live / scale
   数字写进正文。

最小离线命令：

```bash
python scripts/nsdi_eval.py --offline
pytest -q tests/multi_function tests/multi_user
pytest -q tests/vector/test_page_match_accuracy.py
pytest -q tests/integrations/test_utg_api.py -k 'import_run_log or merge_preview_route'
```

最小 live 命令：

```bash
python scripts/nsdi_eval.py --all --device <adb-serial>
python scripts/nsdi_e2e_live.py --devices emulator-5554,emulator-5556
```

Live 命令默认只 attach 已在线设备；不要隐式执行 emulator setup。没有设备时，输出必须保持
`blocked_live_device` / `partial`，不能用离线结果补 live 表格。

## Paper Narrative Order

正文组织按系统会议投稿的读者预期排序：先报告端到端任务是否成功，再解释时间、在线模型成本和系统开销，
最后给出机制证据与次级实验。正文按实验内容组织，不按问题编号组织。

1. End-to-End Success Rate and Efficiency：成功率是 headline，延迟和 token 是成功路径上的效率解释。
2. Time and Cost Breakdown：解释 cache hit 后剩余开销来自 device interaction，而不是 lookup 或 planner。
3. Omni Representation Recall：证明结构化检索的准确性、隔离性、速度和 payload-size 评测口径。
4. Source-to-Current Function Transfer：证明历史 Function 到当前 GUI 的映射质量和 false-positive 边界。
5. Trajectory Compression and Asset Yield：证明 run log 能沉淀为可复用 Function asset。
6. Multi-User Co-Evolution and Privacy Contract：证明跨用户复用、隔离和隐私边界。
7. Boundary Ablation：证明 Function 必须满足 start-node precondition、route-only bridge 和 terminal/effect verify。
8. Resource and Energy Cost：次级系统指标，只在 runner 输出冻结数字后写结果。
9. External Exposure and Tool Surface：证明 OmniFlow 作为外部 agent 可调用的 cache / Function runtime 暴露面。
10. Case Study: Hit, Recover, Fall Back：只使用真实 event stream；缺 regime 时必须标 partial。

## E2E Workload Plan

End-to-end evaluation uses two task sources and one shared device pool:

- AndroidWorld: 100 task instances from benchmark apps, with parameterized goals varying target content, account state, and random seed.
- Self-collected mobile-app tasks: 100 task instances from common mobile apps, covering list, form, tab, popup, WebView, and image-card interfaces.
- Device pool: three physical Android phones (`vivo X100`, `OnePlus`, `Pixel`) and two virtual devices.
- Purpose: test variation across device model, screen density, system theme, emulator/physical-device behavior, and account-specific list content.
- Source traces: one successful human-recorded source run per task; reuse runs receive no human assistance.
- Status: workload plan registered; paper claims should use only the latest
  runtime index generated by
  `python scripts/androidworld_baseline_eval.py --organize-latest-results`.
  The current latest index is
  `/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/`
  (`generated_at=2026-06-15T02:10:11`). It keeps one latest record per
  `(task, backend)`, excludes dryrun/smoke/probe/debug/pre-fix runs, and normalizes
  legacy native backend labels to `omniflow-native`. Current aggregate: 73 latest rows
  across 38 tasks; `omniflow-native` has 29/33 official AndroidWorld-validator
  successes, AutoDroid has 0/9 official successes, MobileGPT has one latest
  official-validator failure row, and `mobilegpt-self` has 16/30 finish-marker successes
  that are not official AndroidWorld-validator successes. Older accepted-30
  records are excluded from the current paper-facing result version.

## Matrix

| Experiment | 问题 | 主指标 | 当前数据源 | 当前脚本 / 命令 | 输出文件 | 状态 |
|---|---|---|---|---|---|---|
| Omni Representation Recall | Omni representation 能否准确、高效地召回当前 goal/state 相关的 node 与 Function？ | Function top-1 accuracy, candidate coverage, context isolation, miss/ambiguous rate, lookup latency, recalled payload size, payload reduction vs full Function catalog, screen-only VL embedding baseline Top1/Top3, encoding latency, tokenized encoding cost, cross-device live status | executable cache lookup regression; payload-size measurement is computed by `scripts/nsdi_eval.py` from the actual returned candidate payloads; cross-device page-localization manual subset frozen under `UTG_papers/nsdi/evidence/cross_device_page_localization_20260605/`; screen-only VL embedding baseline terminal collection frozen under `UTG_papers/nsdi/evidence/representation_screen_vl_baseline_20260610/` | `python scripts/nsdi_eval.py --experiment representation-recall`; cross-device subset generated from `scripts/cross_device_page_annotation_server.py` annotations; screen-only VL embedding values collected with `node UTG_papers/nsdi/evidence/representation_screen_vl_baseline_20260610/collect_terminal_values.mjs` | `runtime/evals/nsdi/recall/summary.json`, `runtime/evals/nsdi/recall/cross_device_live.json`, `UTG_papers/nsdi/evidence/cross_device_page_localization_20260605/paper_values.json`, `UTG_papers/nsdi/evidence/representation_screen_vl_baseline_20260610/paper_values.json`, `UTG_papers/nsdi/evidence/representation_screen_vl_baseline_20260610/terminal_capture.log` | `partial+manual_subset+screen_vl_terminal_collected`：labeled eval and recalled payload-size fields are available；cross-device page localization reviewed subset has 94 reviewed pairs, 73 valid pairs after 21 duplicate/invalid exclusions, Ours top-1 72/73=98.63%, top-3 73/73=100.0%, lookup P50/P95 0.0035/0.0066 ms, average recalled payload 5.27 KB；terminal-collected screen-only VL embedding baseline has Top1=69.44%, Top3=87.50%, avg encode=188.69 ms/item, total tokens=267,254, avg tokens=1,008.51/item；OmniFlow UDEG/PageVector has Top1=98.63%, Top3=100.0%, total encode=1.09 s, avg encode=4.13 ms/item, total tokens=0, avg tokens=0/item；reported avg encode speedup=45.73x and token reduction=267,254 total / 1,008.51 per encoded item；当前不是 30 goals x 3 device headline |
| Trajectory Compression and Asset Yield | 一组 run log 能压成多强的可复用 function 资产？ | `input_runs`, `aggregated_functions`, `compression_ratio`, `slot_count`, `cache_hit` | `tests/multi_function/fixtures/*`, `runtime/demo/multi_function_demo/phase1_runs/*.json` | `python scripts/nsdi_eval.py --experiment asset-yield` | `runtime/evals/nsdi/rq2/summary.json`（legacy output path） | `available`：compression 7x (7→1), slots=2, cache_hit=True；只报告可运行聚合指标 |
| Source-to-Current Function Transfer | Flow 能否把历史 Function 的 source context 安全迁移到当前 GUI context？ | page-family recall@k, target_found_rate, offline oracle action_success_rate, offline oracle false_positive_rate, abstain/null-target rate, wrong-click / false-positive rate, live_verify action_success_rate / false_positive_rate / failures | `runtime/page_collections/contacts_5554_targeted_20260506_151100_clean/`, `runtime/evals/nsdi/page_replay_baselines/contacts_clean_20260506/family_map.json` | `python scripts/nsdi_eval.py --experiment function-transfer` | `runtime/evals/nsdi/page_replay_baselines/headline/summary.json`, `runtime/evals/nsdi/page_replay_baselines/headline/live_verify.json` | `available+bounded_live`：B0/B1/B2/B3 offline oracle available；`--device emulator-5554` live verify 覆盖 16 safe Contacts cases，positive action success=93.3%，negative false positive=1/1；不能写成大规模 live success |
| End-to-End Success Rate and Efficiency | cache-first execution 能否在 live 设备上完成任务，并在成功路径上减少在线开销？ | official AndroidWorld validator success, run duration, step count, model calls, token usage, zero-token HIT evidence | latest AndroidWorld runtime index only: `/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/` | `python scripts/androidworld_baseline_eval.py --organize-latest-results` | `latest_results_index/manifest.json`, `latest_results_index/summary.md`, `latest_results_index/all_latest.jsonl` | `latest_only`：当前 latest index 有 73 条 `(task, backend)` 最新记录、38 个任务。`omniflow-native` 为 29/33 official AndroidWorld-validator success，0 model calls / 0 tokens for replay rows；AutoDroid 为 0/9 official success；MobileGPT 当前 strict row 是 1 条 official-validator failure；`mobilegpt-self` 为 16/30 finish-marker success，但不是 official AndroidWorld-validator success。旧 accepted-30 记录不进入当前 paper-facing result version。 |
| External Mobile Reuse Baselines | 其它 mobile memory / programmatic GUI-agent 系统在已有成功轨迹时能否复用？ | AndroidWorld official validator success where available, native finish-marker success where no official validator is present, model calls, tokens, elapsed time | same latest AndroidWorld runtime index: `/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/latest_results_index/` | `python scripts/androidworld_baseline_eval.py --organize-latest-results` | `latest_results_index/all_latest.jsonl`, `latest_results_index/by_task/*/results.jsonl` | `latest_only`：AutoDroid latest strict rows are 0/9 official success. MobileGPT strict has one official-validator failure row. MobileGPT self-memory has 16/30 target finish-marker successes and is labeled non-official; do not report it as AndroidWorld validator SR. |
| AITW Prompt Similarity Characterization | AITW workload 中有多少 instruction 具有可复用意图候选？ | prompts with a parameter-template peer; prompts with a template or semantic peer at cosine $\geq 0.98$; threshold sensitivity at $\geq 0.95$; per-subset strict rate | official AITW five-subset prompt vocabulary, extracted through the processed public index whose per-subset vocabulary matches the official counts | normalize and mask parameters; embed global distinct prompts with `BAAI/bge-small-en-v1.5`; nearest-neighbor cosine search; remove explicit opposite-intent pairs; count each official subset-level prompt entry if it has at least one accepted peer | `UTG_papers/nsdi/evidence/aitw_prompt_similarity_20260714/paper_values.json`; methodology and tables in `overleaf/appendix.tex` | `available+prompt_only`：official denominator 30,378; deterministic template peers 11,052 (36.4%); strict template-or-cosine-$\geq 0.98$ peers 22,834 (75.2%); cosine-$\geq 0.95$ sensitivity 26,386 (86.9%). This is a Function-reuse candidate upper bound, not trace equivalence, cache-hit rate, or execution success. |
| Time and Cost Breakdown | cache hit 后剩余开销来自哪里？ | canonical duration, launcher overhead, function execution, observe/act I/O, cache lookup latency, public argument fill latency, VLM step-by-step comparison latency, token/model calls | `runtime/evals/nsdi/e2e_live/summary.json`, `runtime/evals/nsdi/e2e/summary.json`, `UTG_papers/nsdi/evidence/function_lookup_timing_20260611/contacts_add_contact_recall_lookup_microbench.json` | `python scripts/nsdi_eval.py --experiment time-cost-breakdown`; `python scripts/build_function_lookup_timeline.py --iterations 1000 --warmup 20 --param-mode llm --param-provider dashscope --param-model qwen-turbo --param-iterations 3 --minimal --figure-stem F23_contacts_add_contact_true_scale_timeline --json-name contacts_add_contact_recall_lookup_microbench.json` | `runtime/evals/nsdi/time_cost/summary.json`, `runtime/evals/nsdi/e2e_live/summary.json`, `runtime/paper_figures/e2e_live/figure_payload.json`, `UTG_papers/nsdi/overleaf/figs/F23_contacts_add_contact_true_scale_timeline.{svg,png,pdf}` | `available+contacts_timeline`：runner 直接从 frozen/live E2E summary 派生 stage timing；ContactsAddContact true-scale sidecar shows local Function recall lookup P50 0.006167 ms, public argument fill for text-only arguments P50 1160.884458 ms, replay run duration 15712.970495 ms, and full VLMPlanner comparison 9.742 s per step x 5 = 48.710 s；9.742 s 只能作为 VLM step-by-step 对比，不是 Function recall lookup 或 public argument fill；解释为 device interaction / observe / verify 主导，不能推导 energy |
| Multi-User Co-Evolution and Privacy Contract | 多用户共享链是否做到匿名、可导入、可分发？ | raw identifier leakage, import/export success, anonymous scope counts, cross-user HIT rate, tokenized context cost | `tests/multi_user/*`, `paper_results/android_control_eval_20260509_165641.json` | `python scripts/nsdi_eval.py --experiment multi-user-coevolution` | `runtime/evals/nsdi/multi_user/report.json` | `available`：cross-user HIT=100%, isolation=100%, token reduction 88.1%, 58 multi_user tests pass；只报告 runner 实际输出字段 |
| Boundary Ablation | Function 是否必须受 start-node precondition、route-only bridge 和 terminal/effect verify 约束？ | full-function success, semantic-tail correct-start success, wrong-start terminal failure, controller bridge recovery, route-only bridge tests | `runtime/evals/key_function_ablation/key_function_goto_bridge_report.md`, `tests/integrations/test_utg_api.py` route-bridge tests | `python scripts/nsdi_eval.py --experiment boundary-ablation` | `runtime/evals/nsdi/boundary_ablation/summary.json`, `runtime/evals/key_function_ablation/key_function_goto_bridge_report.md` | `available`：wrong-start semantic tail returned action true but terminal verify failed；route-only bridge tests 由 runner 验证 |
| Resource and Energy Cost | 延迟和 token 收益是否把成本转移到设备能耗、CPU、内存或网络？ | device active time, battery drain, host CPU time, peak RSS, network bytes, wall-clock latency, token/model calls | `runtime/evals/nsdi/e2e_live/summary.json` plus one-run adb/resource sidecar counters | `python scripts/nsdi_eval.py --experiment resource-energy` | `runtime/evals/nsdi/resource_energy/summary.json` | `available+secondary`：one-run emulator counters available；可写为 secondary accounting，不能写成能耗节省结论；physical-device repeated protocol 仍未冻结 |
| External Exposure and Tool Surface | OmniFlow 能否作为外部 agent 可调用的 cache / Function runtime 暴露？ | MCP tool schema, JSON-RPC envelope, recall candidate schema, call-function delegation, run-log ingest, Function execute route contract | `tests/integrations/test_mcp_server.py`, `tests/multi_user/test_workbench_mcp.py`, `tests/utg_core/test_external_tool_mcp_compatibility.py`, selected public-route tests in `tests/integrations/test_utg_api.py` | `python scripts/nsdi_eval.py --experiment external-exposure` | `runtime/evals/nsdi/external_exposure/summary.json` | `available` after runner passes；该实验只证明外部暴露契约，不证明生产部署规模 |
| Case Study: Hit, Recover, Fall Back | 单条 run 的 case study 是否能清楚展示 cache hit/control/fallback 行为？ | step count, control event count, slowest step, action breakdown, observed/missing regimes | `runtime/paper_figures/runlog_timeline/*` real event-stream payloads | `python scripts/nsdi_eval.py --experiment case-study` | `runtime/evals/nsdi/rq6_case_study/summary.json`（legacy output path）, `runtime/paper_figures/runlog_timeline/nsdi_rq6_case_study/figure_payload.json`（legacy output path） | `partial`：当前 real nonzero event stream 只观察到 cache_hit；control/fallback 缺失，不能作为最终三 regime 图证据 |

## AITW Prompt Similarity Snapshot

The frozen AITW workload characterization uses the official sum of unique
prompt counts across the five subsets as its denominator.  This produces
30,378 subset-level prompt entries; global string deduplication produces 30,165
strings because 213 strings occur in multiple subsets.  The processed public
index covers 5,687,349 interaction steps and 714,327 episodes, and its
per-subset prompt vocabulary exactly matches the official AITW counts.

The main statistic marks prompt $i$ as similar when at least one other prompt
shares its parameter-masked deterministic template or has
`BAAI/bge-small-en-v1.5` cosine similarity at least 0.98 after explicit
opposite-intent pairs are removed.  The strict result is 22,834/30,378 = 75.2%.
Template matching alone gives 11,052/30,378 = 36.4%; lowering the semantic
threshold to 0.95 gives 26,386/30,378 = 86.9%.  The strict per-subset counts are
General 69/545, GoogleApps 81/306, Install 549/688, Single 9,151/15,366, and
WebShopping 12,984/13,473.

The paper may interpret this snapshot only as recurring-intent workload
evidence and a prompt-level Function-reuse candidate upper bound.  A separate
AITW trace-pair generalization protocol should use distinct source/target
episodes, same-app parameter-template positives, same-app near-miss and
opposite-intent negatives, and leave-one-template-out or leave-one-app-out
splits.  Until that protocol produces a frozen output, no trace-alignment,
Function-recall, cache-hit, or executable-replay result may be attributed to
AITW.

## Representation Baseline Snapshot

The screen-only VL embedding baseline is registered as a terminal-collected
paper-value snapshot.  The frozen values are generated by
`node UTG_papers/nsdi/evidence/representation_screen_vl_baseline_20260610/collect_terminal_values.mjs`
and stored in
`UTG_papers/nsdi/evidence/representation_screen_vl_baseline_20260610/paper_values.json`
with terminal output preserved in
`UTG_papers/nsdi/evidence/representation_screen_vl_baseline_20260610/terminal_capture.log`.

| Method | Top1 | Top3 | Total encode | Avg encode | Total tokens | Avg tokens |
|---|---:|---:|---:|---:|---:|---:|
| Screen-only VL embedding | 69.44% | 87.50% | 50.00 s derived | 188.69 ms/item | 267,254 | 1,008.51/item |
| OmniFlow UDEG / PageVector | 98.63% | 100.0% | 1.09 s | 4.13 ms/item | 0 | 0/item |

Registered comparison: average encoding speedup is 45.73x; token reduction is
267,254 total tokens and 1,008.51 tokens per encoded item.  The 50.00 s
screen-only total encode time is derived from 188.69 ms/item over approximately
265 encoded items; the supplied speedup is preserved because displayed per-item
encode values are rounded.

## Appendix-Only Single Inference Cost Snapshot

This diagnostic snapshot records the per-inference cost of converting one
mobile observation into a complete textual description.  It is used only in the
appendix to contextualize why OmniFlow avoids repeated online reasoning over
raw screenshots or full XML text.  It is not an end-to-end task result, a
success-rate result, or an energy result.

The frozen source is
`UTG_papers/nsdi/evidence/single_inference_cost_20260611/paper_values.json`.
Screenshot and XML inputs are both counted as complete textual descriptions of
the observed screen.

| Location | Model / backend / device | tokens | Time |
|---|---|---:|---:|
| On-device | Llama.cpp + 0.8B-W4 + CPU | screenshot ~2500 | 60s+ |
| On-device | MNN-LMM + 0.8B-W4 + CPU | screenshot ~2500 | 10s+ |
| On-device | Omniinfer + 0.8B-W4 + NPU | XML ~1k--10k+ | 3s |
| Cloud | Qwen-VL-Plus + cloud | screenshot ~3200 | 3s |

## Current Inputs Already In Repo

- Function recall eval 当前输入与计划：
  - cache lookup unit regression cases
  - Function recall scenario regression cases
  - verb-gate boundary regression cases
  - `tests/fixtures/utg_smoke_test.json`
  - app-goal dataset plan
- offline replay 输入：
  - `tests/fixtures/offline_replay/androidworld_contacts_add_contact/live_run_refs.json`
  - `tests/fixtures/offline_replay/androidworld_open_app_contacts/live_run_refs.json`
- 当前 function pool / event stream：
  - `runtime/utg/utg_store.json`
  - `runtime/logs/agent_runs.jsonl`
- multi-function 已有输出：
  - `runtime/demo/multi_function_demo/experiment_summary.json`
  - `runtime/demo/multi_function_demo/shared_pool_report.json`
- timeline / 论文评测已有输出：
  - `runtime/evals/nsdi/manifest.json`
  - `runtime/evals/nsdi/e2e_live/summary.json`
  - `runtime/paper_figures/runlog_timeline/`
- key-function / goto bridge 消融：
  - `runtime/evals/key_function_ablation/key_function_goto_bridge_report.md`
  - `runtime/evals/key_function_ablation/contacts_semantic_tail.json`

## Source-to-Current Transfer Required Baselines

Source-to-current transfer 的命中单位不是 exact XML page id，而是人工确认或 benchmark 标注的 `UTG node / page family`。同一 node/family 可以包含多个 XML page variant；只要候选命中同一 node/family，并且目标 action 能通过 anchor/control 成功执行，就应计为页面级复用成功。

必须登记并对比以下 baseline：

| Baseline | 输入 | 判定方式 | 论文解释边界 |
|---|---|---|---|
| `B0 coordinate replay` | recorded target bounds / center | 直接点原坐标，统计 action success 和 false positive | 最弱但必要，用于证明裸坐标 replay 脆弱 |
| `B1 UI tree exact replay` | raw UI tree + recorded target element attributes | 按稳定 `resource-id/text/class/index/bounds` 等 exact 或近 exact 规则找同一元素，再执行 | 同设备、resource-id 稳定且允许保留 raw UI tree / target element 时很强；它不证明 privacy-safe sharing |
| `B2 UI tree + anchor replay` | raw UI tree + recorded target element + anchor mapping | 先用 UI tree 找候选元素，再用 anchor mapping / geometry consistency rerank，最后执行 | 预计是最强本地 replay baseline；OmniFlow 不能用“单已知元素 replay 更准”来攻击它 |
| `Ours` | page-vector page/family top-k candidates + element/anchor mapping + control-aware verify | 先做候选召回，再迁移 target/action，最后用 verify 判断成功 | 优势在 `run_log -> Function asset`、privacy-safe candidate recall、跨状态/跨用户可复用 cache，而不是保留 raw UI tree 的单步已知目标重放 |

Source-to-current transfer 另登记一个子协议 `Baseline-Failure Recovery Set`，用于论文中展示“自动构造的笨拙对应方法失败，而 OmniFlow action mapping 恢复成功”的样本。该子协议和 guard / recovery 实验分开，默认只比较：

| Baseline | 角色 | 判定边界 |
|---|---|---|
| `fixed` | 原坐标 replay | 旧坐标落到错误目标或目标外时记为 baseline failure |
| `text_feature` | feature-like UI-tree proposer / baseline | 使用 `resource-id`、resource tail、`text`、`content-desc`、`class` 和 actionable flags 自动提出 `candidate_target` / `proposal_target`；它不是 gold oracle |
| `ours_default` | `get_new_coordinates(x, y, {"page": source_xml, "element": source_action}, {"page": target_xml})` | 生产 action mapping 主链；只在最终 gold target 上成功时计为 recovered |

构造链路固定为 Rico `unique_uis` JSON + screenshot -> Android XML -> same-app similar page pairs -> clickable/focusable/editable source actions -> feature-like candidate proposer -> hard sample selection -> human review or live/postcondition gold。历史字段 `auto_label_target` 只能解释为 `candidate_target` / `proposal_target`，不能写成 ground truth。当前状态为 `review_required`：headline 数字必须等人工裁决、trace-grounded label 或 live verify 冻结后才能进入正文。

MobileViews 只进入“真实 touch action 的 selector 信号缺口 / 人工审核样例”协议，不直接并入 Rico baseline-failure recovery 数字。构造入口固定为 `python scripts/signal_coverage_analysis.py --mobileviews-review-output ... --mobileviews-review-html ...`：从 `actions.csv` 映射到 `state_*.json`、`window_dump_*.xml` 与 `screen_*.jpg`，优先收集无 `resource-id/text/content-desc`、非 actionable 子节点、WebView/SurfaceView、泛化/混淆 resource-id 和少量 semantic control 对照。输出仍是 `gold_review_required` 的 `summary + pairs` review queue，只有人工裁决、trace-grounded label 或 live/postcondition verify 后才能进入最终 accuracy 分母。

自采 `page_match_utg_walk` 进入“真实 before/action/after transition / effect 人工审核样例”协议，也不并入 Rico baseline-failure recovery 数字。构造入口固定为 `python scripts/signal_coverage_analysis.py --page-match-walk-review-output ... --page-match-walk-review-html ...`：从 `runtime/captures/page_match_utg_walk/*/utg_edge_candidates.json` 读取真实 probe click、before/after XML 与截图，优先收集 `page_transition` edge candidate、需要 page-family curation 的样本和 `no_observable_change` 负例。输出同样是 `gold_review_required` 的 `summary + pairs` review queue；它审核的是可复用 edge / Function transition 证据，不是 source action 到另一页 target click 的 final gold。

子协议指标固定为：

- `ours_recovered_from_fixed_failure`
- `ours_recovered_from_text_feature_failure`
- `baseline_failure_recovery_rate`

Source-to-current transfer 报告时必须同时给出 retrieval 子指标和 execution 结果：

- `recall@k` 只作为候选召回子指标，不单独代表最终 page-match 成功。
- `top1/top3` 应按 `UTG node / page family` 命中计算，而不是只按 exact XML filename。
- `action_success_rate` 和 `false_positive_rate` 是判断一个簇是否可执行复用的主指标；当前 `tests.vector` runner 先用 `signature_compatibility_offline` oracle，最终论文结果仍需接设备 verify。
- `abstain_rate`、`null_target_precision` 和 `wrong_click_rate` 是 Flow 安全边界指标；当前只能消费已冻结 runner 输出，不能从人工直觉补数。
- 如果 B1/B2 在单设备 raw UI tree replay 上领先，这是预期结果；论文主张应转向隐私、安全共享、资产化和跨状态迁移收益。
- `Baseline-Failure Recovery Set` 只报告 baseline failure 上的恢复率，不混入 `occluded_recoverable`、`ambiguous_target`、`target_absent_or_semantic_change` 等 guard/recovery 数字；这些属于单独的 action-compatible replay / recovery 评测。

## Retrieval Payload Size Metrics

为对齐 System Design 中 “offline layer returns only compact structured
information to the online layer” 的主张，评测需要显式区分三类 size：

| 指标 | 定义 | 进入正文条件 |
|---|---|---|
| `stored_asset_size_bytes` | 离线层中一个 node / page-vector payload / Function asset 的序列化大小 | 由统一 runner 输出后可写；不得手工估算 |
| `recalled_payload_size_bytes` | 一次 recall 返回给 online executor / host agent 的 candidate Function schema 与必要 metadata 的总字节数 | 由 `python scripts/nsdi_eval.py --experiment representation-recall` 写入 `runtime/evals/nsdi/recall/summary.json.payload_size` 后可写 |
| `tokenized_context_size` | miss 后注入 host agent 的 recalled context token 数 | End-to-end efficiency 成本分析可用；当前需 runner 或 tokenizer sidecar 冻结 |

当前可跑 payload reduction 的定义固定为：

```text
payload_reduction = 1 - bytes(OmniFlow recalled payload) / bytes(full Function catalog payload)
```

full Function catalog payload 使用与 recall 返回值相同的 public candidate fields 序列化。
该指标衡量在线层接收的信息量，不是磁盘上任意文件的大小，也不是最终 token cost 的替代指标。

## Representation-Layer Scalability Sidecar

10K MobileViews-derived UDEG scalability is registered as representation-layer
overhead evidence, not as an end-to-end task-success result.  The frozen
diagnostic source is
`runtime/evals/mobileviews_vector_benchmark/mobileviews_10k_omni_scalability_summary_20260603.json`
in the OmniFlow runtime workspace.  It uses 10,000 MobileViews local trace XML
pages, 512-dimensional page vectors, and 1,000 sampled queries.
Current paper-eligible values are limited to representation-layer
overhead: page encoding P50/P95 = 3.484/12.563 ms, UDEG lookup P50/P95 =
0.328/0.375 ms, online float32 index = 19.531 MB, and SQLite vector store =
37.613 MB.  These numbers support only the claim that the representation layer
is not the online execution bottleneck; they must not be reported as
cross-device accuracy or E2E success.

## Resource and Energy Metrics

Resource and Energy Cost 是次级系统实验。它的作用是验证 cache-first 的时间和 token 收益没有把
成本转移到设备或宿主机器。该实验不参与 headline success claim；只有 frozen sidecar 输出后，正文
才能写数值。

| 指标 | 定义 | 进入正文条件 |
|---|---|---|
| `device_active_time_ms` | 从任务开始到任务结束期间设备处于 active 执行状态的时间 | live runner 或 sidecar 输出 per-run 统计 |
| `battery_delta_mah` / `battery_delta_pct` | Android BatteryStats 或等价采集得到的电量变化 | 同一设备、同一初始条件、重复运行后冻结 |
| `host_cpu_time_ms` | 评测进程及子进程的 CPU 时间 | 由 runner 记录并写入 summary |
| `host_peak_rss_bytes` | 评测进程峰值内存 | 由 runner 或系统监控 sidecar 写入 summary |
| `network_bytes` | online model / provider / device bridge 产生的网络字节数 | 有明确计数边界，且和 token/model-call 统计对齐 |

当前可写的系统成本包括：wall-clock latency、stage timing、lookup latency、token/model calls，
以及 `resource_energy/summary.json` 中 one-run emulator sidecar 实际输出的 device active time、
battery、host CPU/RSS 和 network counters。这些资源数字只能作为 secondary accounting；
能耗节省或真实设备能效结论必须等 repeated physical-device protocol 冻结后再写。

## Key Function / Goto Bridge Sidecar

该 sidecar 只验证“关键语义 Function 是否足够”的执行前置条件，不替代页面复用或端到端 headline。

| 项 | 当前数据源 | 命令 / 检查 | 输出文件 | 状态 |
|---|---|---|---|---|
| Contacts 完整函数正例 | `runtime/utg/contacts_official_success_5554_latest.json` | `python scripts/device_cli.py --target emulator-5554 replay-function --function-id aw_function_515be7bf7e ... --no-skip-terminal-verify` | `runtime/evals/key_function_ablation/key_function_goto_bridge_report.md` | `available`：Contacts provider 找到 `Key Path` |
| 语义尾段正确起点正例 | `runtime/evals/key_function_ablation/contacts_semantic_tail.json` | `python scripts/device_cli.py --target emulator-5554 replay-function --function-id key_semantic_tail_create_contact ... --no-skip-terminal-verify` | 同上 | `available`：Contacts provider 找到 `Tail Only` |
| 语义尾段错误起点负例 | 同上 | 同命令追加 `--baseline-no-controller`，从 launcher 起步 | 同上 | `available`：action execution returned true，但 terminal verify failed，provider 未找到 `No Bridge` |
| route-only shortest-path guard | `tests/integrations/test_utg_api.py` | `pytest -q tests/integrations/test_utg_api.py -k 'merge_preview_route_bridges_selected_functions_by_shortest_path or merge_preview_route_does_not_bridge_through_semantic_function or merge_preview_route_builds_composed_function'` | pytest stdout | `available`：3 passed，semantic/input edge 不可作为 bridge |

论文正文允许写：关键语义 Function 足够的前提是先用 route-only goto/navigation bridge 满足
`start_node_id`，再执行语义 Function 并验收 terminal/effect。正文不能写成：只要找到关键语义
Function，就可以从任意 UTG node 直接执行成功；也不能写成 unrestricted shortest path over all edges。

## Reviewer-Identified Gaps (2026-05-22)

下列问题由论文 reviewer 角色分析识别，需要在投稿前解决或在 Limitations 中正式承认。

| ID | 问题 | 类型 | 处理方式 | 状态 |
|---|---|---|---|---|
| R1 | 评测规模太小：只有 Contacts/open_app 1 个任务、1–2 台设备、38 个 recall 案例 | 实验规模 | 扩大到 ≥10 个 AndroidWorld 任务、≥3 台设备；或在 Limitations 承认当前边界 | 🔴 open |
| R4 | page-vector 相似度阈值未说明、无 sensitivity analysis | 设计细节 | 在 Implementation §5.1 或 Appendix B 中说明阈值选取和敏感性；或登记为独立实验 | 🟡 paper fix needed |
| R6 | page-vector 碰撞率/跨 App 假阳性未量化；38 cases 太少 | 实验规模 | 扩大 context-isolation 案例集；或在 recall eval 中报告 cross-app false positive rate | 🟡 register as metric |
| R7 | Argument extraction 机制未说明 | 写作/实现 | 在 Implementation write-back 段说明推断机制 | 🟡 paper fix needed |
| R8 | Navigation graph 不连通时的行为未说明 | 写作 | 在 design 或 Limitations 中说明 | 🟡 paper fix needed |

## Current Blockers

1. Omni representation recall labeled eval 已由 `scripts/nsdi_eval.py --experiment representation-recall` 串联；payload-size 字段已进入 `runtime/evals/nsdi/recall/summary.json`；cross-device headline 仍需要 30 goals x 3 device fingerprints，当前 opportunistic device rows 不能写成大规模跨设备结论。
2. Source-to-current transfer headline summary 已固定在 `runtime/evals/nsdi/page_replay_baselines/headline/summary.json`；live verify 已接入 `scripts/nsdi_eval.py --experiment function-transfer --device <adb-serial>`，但只覆盖 frozen Contacts clean collection 中有安全 route/effect-family evidence 的 B3 cases。最新 `emulator-5554` 结果有 1 个 negative false positive，最终 headline 仍缺更大的 negative set / human-live gold。
3. ~~End-to-end efficiency 当前只有 case-study 级 timeline，没有统一 e2e summary。~~ **CLEARED** (2026-05-10): unified e2e aggregator at `tests/control_plane_benchmark/rq4_e2e_aggregator.py`; summary frozen at `runtime/evals/nsdi/e2e/summary.json`. **LIVE SIDECAR** (2026-05-22): unified runner latest top-level output is single-device `emulator-5554` OpenAppTaskEval 1/1 success; the separate cross-device 5554/5556 sweep at `runtime/evals/nsdi/e2e_live/20260522T000228Z/summary.json` passed 2/2 with 2 cache hits and 2 zero-token hits, but remains bounded OpenAppTaskEval evidence and must not be implied by the single-device manifest.
4. ~~Multi-user report 目前仍以 test-only 原型为主，缺少统一导出的论文结果文件。~~ **CLEARED** (2026-05-10): multi-user aggregator at `tests/control_plane_benchmark/rq5_multiuser_aggregator.py`; report frozen at `runtime/evals/nsdi/multi_user/report.json`. 58 multi_user tests pass; cross-user HIT=100%, token reduction=88.1%.
5. Case-study timeline 已有真实 event-stream、非 0ms payload，但当前仅含 cache-hit regime；control/fallback 缺失时固定标 `partial`，最终论文三 regime 图需要新 live case。
6. Resource and Energy Cost 已有 `resource_energy/summary.json` sidecar，当前输出 one-run emulator battery/CPU/memory/network counters；这些只能作为 secondary accounting，不能写成 physical-device energy-saving 结论。
7. External Exposure and Tool Surface 已登记为独立实验，覆盖 MCP / JSON-RPC / public HTTP routes / run-log ingest；该实验证明外部调用契约，不证明生产部署规模。

## Freeze Order

1. ✅/🟡 Omni representation recall and asset yield — labeled eval + payload-size + compression aggregator done; cross-device headline still open
2. ✅ End-to-end success and efficiency — e2e summary done (HIT 3017ms vs StepWise 4850ms, 88% token reduction at scale); bounded live OpenApp evidence includes latest unified 1/1 `emulator-5554` plus separate run-dir 2/2 `emulator-5554,emulator-5556`
3. ✅ Multi-user co-evolution — multi-user report done (cross-user HIT=100%, isolation=100%, 58 tests pass)
4. 🟡 Source-to-current Function transfer — B0/B1/B2/B3 offline oracle ready; bounded live verifier ran on safe Contacts B3 cases, but final headline still depends on broader negative set and human/live gold
5. 🟡 Resource and energy cost — one-run emulator counters available; repeated physical-device energy protocol remains open
6. ✅ Boundary ablation — route-only bridge tests + Contacts key-function sidecar wired into unified runner
7. ✅ External exposure and tool surface — MCP / public route contract tests wired into unified runner
8. 🟡 Case-study timeline — supporting case study path is wired; current payload is nonzero real event stream but missing control/fallback regimes
