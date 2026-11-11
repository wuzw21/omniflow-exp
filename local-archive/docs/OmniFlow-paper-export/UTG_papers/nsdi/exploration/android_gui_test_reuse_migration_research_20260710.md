# Android GUI Test Reuse / Migration / Replay Research Audit

Status: exploration-only, not frozen paper result.
Date: 2026-07-10

This note audits the Android GUI test reuse / test migration / record-and-replay
line around FrUITeR, CraftDroid, TEMdroid, and recent LLM/VLM successors.  It is
intended to support OmniFlow related-work positioning.  It does not change
`core_claims.md`, `eval_matrix.md`, or the paper text.

## Executive Takeaway

This line of work is important for OmniFlow.  It shows that the core operation
"align a historical GUI action or test step to a current/target GUI state" is
already well established in mobile testing.  OmniFlow should not claim novelty
on GUI action transfer in isolation.

The defensible distinction is the runtime objective.  Prior work migrates test
cases, records, or test intentions for software testing.  OmniFlow turns
successful GUI-agent executions into reusable `Function` assets and executes
them as a cache-first runtime layer with live state recall, action transfer,
control-aware execution, terminal/effect verification, and VLM fallback.  The
system target is repeated user-facing mobile routines, not generating a test
suite for a target app.

Citation priority:

| Priority | Work | Why |
|---|---|---|
| Must cite | FrUITeR, CraftDroid, TEMdroid | Establishes the mature mobile test reuse / widget matching lineage most likely to be recognized by reviewers. |
| Strongly cite | AppTestMigrator / ATM, AppFlow, RIDA, ReuseDroid | Cover the adjacent axes: test migration, ML reusable tests, cross-app record-and-replay, and VLM-era migration. |
| Optional cite | TRASM/TReaDroid, SemFinder, MAPIT, MACdroid, ITeM, LLMigrate, SAIL | Useful if the related-work section has space for a more complete map. |
| Background only | RERAN, RANDR, Cross-device Record-and-Replay, V2S/V2S+ | Useful for record-and-replay history, but less directly about reusable agent execution. |

## Taxonomy

### 1. Low-level or same-app record-and-replay

These systems record events and replay them under constraints.  They are useful
for positioning "OmniFlow is not raw replay."

- RERAN records and replays low-level Android input events, emphasizing timing
  and touch sensitivity.
- RANDR records and replays Android UI and network inputs using targeted
  runtime instrumentation.
- Cross-device record-and-replay systems adapt replay across device layouts,
  screen sizes, and orientations.
- V2S/V2S+ extract replayable scenarios from videos or screen recordings.

OmniFlow relation: these works prove replay is a long-standing tool, but they
mainly seek faithful reproduction of an execution.  OmniFlow uses replay as a
cache hit path inside an agent runtime and authorizes each recalled `Function`
against live GUI state and task effect.

### 2. Cross-app Android test migration / test reuse

These systems migrate a source app's test to a target app with similar
functionality.  This is the most relevant prior-work cluster for "historical
context to current context alignment."

- AppTestMigrator / ATM and GTM map GUI widgets between same-category apps.
- CraftDroid transfers test inputs and oracles through semantic mapping of
  actionable GUI widgets.
- AppFlow learns reusable screen/widget abstractions for category-specific UI
  tests.
- TRASM/TReaDroid and SemFinder improve semantic matching choices.
- TEMdroid trains a widget matching model using contextual representations.

OmniFlow relation: this line already treats GUI reuse as source-to-target
mapping.  The difference is that the unit in OmniFlow is a `Function` generated
from successful agent/user run logs, and the consumer is an online execution
cache for a GUI agent rather than a testing tool generating target-app scripts.

### 3. LLM/VLM-era GUI test migration

Recent work uses LLMs/VLMs to go beyond direct widget matching by abstracting
intent, core logic, or operational steps.

- LLMigrate uses LLMs for usage-based UI test transfer across Android apps.
- MACdroid abstracts source tests into target functionality logic and then
  concretizes tests on the target app.
- ITeM generates test intentions and uses dynamic reasoning to fulfill them.
- ReuseDroid uses VLM-based agents and active feedback to migrate Android UI
  tests based on shared core logic.
- SAIL uses skill abstraction and adaptation for UI test reuse.

OmniFlow relation: these works are closer rhetorically because they convert
source traces into higher-level transferable knowledge.  The difference remains
the serving path: they spend LLM/VLM reasoning to create or migrate tests,
whereas OmniFlow's goal is to keep repeated routine execution off the online
VLM critical path when a cache hit is authorized.

### 4. Cross-platform GUI test migration and replay

This cluster transfers or replays GUI tests across platform boundaries, such as
iOS-to-Android, web-to-Android, Android-to-iOS sibling apps, or cross-device /
cross-platform visual replay.  It is useful for OmniFlow because it makes the
same underlying difficulty explicit: the source and target GUI can preserve
task intent while changing layout, widget type, interaction style, and the
number of concrete actions required.

- TestMig migrates iOS GUI event sequences to Android by recording source iOS
  tests and exploring the Android version under event-conversion guidance.  It
  reports five popular apps and 80.2% successfully converted iOS UI events.
- MAPIT transfers tests bidirectionally between sibling Android and iOS apps,
  without requiring source code, and can transfer tests containing oracles.
- TransDroid transfers GUI tests from web apps to Android apps.  It reports 20
  real-world web/Android apps, 110 test cases, 561 GUI/oracle events, 77%
  successful attempted transfers, 82% precision, and 99% recall for event and
  oracle mapping.
- Vision-based Widget Mapping for Test Migration across Mobile Platforms
  constructs a cross-platform widget-mapping dataset with 6,730 bidirectional
  mapped iOS/Android widget pairs from 50 app pairs and evaluates 89 visual
  matching configurations.
- NiCro studies non-intrusive cross-device and cross-platform GUI testing with
  visual widget detection and matching, evaluating replay on 8 devices and 107
  test cases from 28 popular apps plus the home page.

OmniFlow relation: cross-platform work is not a direct baseline unless
OmniFlow claims iOS/web portability.  Its main value is argumentative.  It
shows that layout drift, widget-type mismatch, missing metadata, action
incompatibility, and one-to-many / many-to-one step correspondence are already
known challenges in GUI reuse.  OmniFlow should therefore avoid claiming that
widget mapping or cross-state action transfer is new.  The stronger position is
that these migration mechanisms motivate the need for `Function`-level assets:
the reusable unit often has to be a short operation or logic step, with live
state admission and terminal/effect verification, rather than a single
coordinate or source widget.

## Core Works

### FrUITeR: A Framework for Evaluating UI Test Reuse

- Venue: ESEC/FSE 2020.
- Authors: Yixue Zhao, Justin Chen, Adriana Sejfia, Marcelo Schmitt Laser,
  Jie Zhang, Federica Sarro, Mark Harman, Nenad Medvidovic.
- Source: [ESEC/FSE 2020 page](https://2020.esec-fse.org/details/fse-2020-papers/42/FrUITeR-A-Framework-for-Evaluating-UI-Test-Reuse),
  [arXiv](https://arxiv.org/abs/2008.03427), [PDF](https://arxiv.org/pdf/2008.03427).

FrUITeR is not primarily a new migration algorithm.  Its contribution is an
automatic, reproducible framework for evaluating UI test reuse.  It identifies
problems in previous evaluations: event-transfer metrics alone miss whether the
generated test is useful, manual evaluation is expensive and hard to reproduce,
and existing techniques are difficult to compare because their components and
assumptions differ.

FrUITeR contributes new evaluation metrics, lower/upper-bound baselines, an
automated modular workflow, and a benchmark.  The paper reports 20 apps, 239
test cases, 1,082 GUI events, and 11,917 test-reuse instances.  It also
extracts core components from existing techniques such as AppFlow, ATM/GTM, and
CraftDroid for side-by-side comparison.

Value for OmniFlow:

- Must cite as the standard evaluation reference for UI test reuse.
- Useful vocabulary: source app, target app, source test, transferred test,
  ground-truth test, GUI event, canonical event, ancillary event, null event.
- It gives a reviewer-friendly precedent that event mapping alone is not enough;
  usefulness/executability matters.  This aligns with OmniFlow's terminal/effect
  verification.
- It also creates a risk: if OmniFlow says "we transfer historical actions,"
  reviewers may ask why this is not just test reuse.  We need to foreground the
  agent-runtime cache objective.

Recommended use:

> Prior UI test reuse evaluation work, such as FrUITeR, showed that GUI event
> mapping must be judged by executable usefulness rather than by local event
> similarity alone.  OmniFlow adopts the same caution in an agent runtime:
> recalled `Function`s are authorized against the live state and verified by
> terminal/effect checks before being counted as cache hits.

### CraftDroid: Test Transfer Across Mobile Apps Through Semantic Mapping

- Venue: ASE 2019.
- Authors: Jun-Wei Lin, Reyhaneh Jabbarvand, Sam Malek.
- Source: [ASE 2019 page](https://2019.ase-conferences.org/details/ase-2019-papers/22/Test-Transfer-Across-Mobile-Apps-Through-Semantic-Mapping),
  [PDF](https://seal.ics.uci.edu/publications/2019_ASE.pdf),
  [GitHub](https://github.com/seal-hub/CraftDroid).

CraftDroid transfers test cases across mobile apps through semantic mapping of
actionable GUI widgets.  It transfers both GUI events and test oracles, using a
combination of static analysis, dynamic analysis, and information retrieval.  A
key mechanism is semantic similarity over GUI widgets, including Word2Vec-style
representations and app-derived information.

The paper evaluates on 25 real-world apps from 5 categories.  Reported results
include 75% successful attempted transfers, with 73% precision and 90% recall
for transferred GUI and oracle events.

Value for OmniFlow:

- Must cite as the classic Android semantic mapping baseline.
- It directly undercuts any claim that semantic widget/action transfer is new.
- It is still not a direct OmniFlow competitor because the input is source app
  tests and the output is target app tests/oracles, not a cacheable agent
  `Function` executed inside a live user routine.
- The "transfer both event and oracle" point is important: OmniFlow should
  discuss terminal/effect verification as runtime authorization, not imply that
  prior testing work ignored correctness.

Recommended distinction:

> CraftDroid transfers GUI test events and oracles across apps by semantic
> widget mapping.  OmniFlow instead reuses a successful execution within a GUI
> agent runtime: a recalled `Function` is matched to a live PageVectorSet state,
> its action is transferred to the current UI, and the result is accepted only
> after control-aware execution and effect verification.

### TEMdroid: Learning-based Widget Matching for Migrating GUI Test Cases

- Venue: ICSE 2024.
- Authors: Yakun Zhang, Wenjie Zhang, Dezhi Ran, Qihao Zhu, Chengfeng Dou,
  Dan Hao, Tao Xie, Lu Zhang.
- Source: [ICSE 2024 page](https://conf.researchr.org/details/icse-2024/icse-2024-research-track/40/Learning-based-Widget-Matching-for-Migrating-GUI-Test-Cases),
  [PDF](https://yakzhang.github.io/2024-icse-temdroid.pdf),
  [ACM](https://dl.acm.org/doi/10.1145/3597503.3623322).

TEMdroid argues that previous widget-matching approaches depend on static word
embeddings and manually designed matching functions, which struggle with complex
relations in real apps.  TEMdroid uses BERT to capture widget context and trains
a matching model.  It designs a two-stage strategy: first mine hard negatives,
then train the matching model with positive and hard-negative samples.  It
reports evaluation on 34 apps; event matching Top-1 accuracy is 76%, and test
case migration F1 is 89%.

Value for OmniFlow:

- Must cite because it is the strongest "trainable widget matching" reference.
- It is important for the question "should OmniFlow's encoder/matcher be
  trainable?"  TEMdroid shows that trainable context-aware matching is a known
  route for test migration.
- OmniFlow can still choose a non-trained encoder if the goal is on-device,
  privacy-safe, fast recall and deterministic transfer, but we must explain the
  design point: PageVectorSet is an engineered state/action compatibility
  representation for cache admission, not a general learned widget matcher.

Recommended distinction:

> TEMdroid improves test migration by training a context-aware widget matcher.
> OmniFlow optimizes a different part of the stack: compact state recall and
> cache-hit execution for repeated tasks.  A learned matcher could replace or
> augment OmniFlow's transfer module, but the paper's main system contribution is
> the `run_log -> function asset -> cache-first execution` lifecycle.

## Additional Related Works

### AppTestMigrator / ATM and GTM

- Source: [ASE 2019 AppTestMigrator page](https://www.se.cs.uni-saarland.de/conferences/ASE/ase2019/details/ase-2019-papers/69/Test-Migration-Between-Mobile-Apps-with-Similar-Functionality.html),
  [ICSE 2020 demo](https://2020.icse-conferences.org/details/Demonstrations/17/AppTestMigrator-A-Tool-for-Automated-Test-Migration-for-Android-Apps).

AppTestMigrator migrates tests between apps in the same category by transforming
source-app event/oracle sequences into target-app event/oracle sequences.  It is
a foundational same-category Android test migration work.  FrUITeR treats ATM
and GTM as major extracted components for comparison.

OmniFlow value: cite if there is space; it reinforces that "same-category GUI
test transfer" is prior art, while OmniFlow is "same/recurrent user task cache."

### AppFlow

- Source: [ESEC/FSE 2018 page](https://2018.fseconference.org/details/fse-2018-research-papers/48/AppFlow-Using-Machine-Learning-to-Synthesize-Robust-Reusable-UI-Tests),
  [PDF](https://www.cs.columbia.edu/~junfeng/papers/appflow.pdf).

AppFlow learns common screens and widgets and synthesizes reusable tests for app
categories such as shopping and news.  It reports evaluation on 60 apps and
reduces effort for testing a new app by up to 90%.

OmniFlow value: cite as earlier ML-based reusable UI testing.  It is not replay
from successful agent trajectories, but it supports the general idea that UI
actions can be abstracted into reusable functional units.

### Semantic Matching / SemFinder / TRASM / TReaDroid

- Source: [Semantic matching in GUI test reuse: are we there yet?](https://dl.acm.org/doi/10.1145/3460319.3464827),
  [TRASM arXiv](https://arxiv.org/abs/2301.00530),
  [TReaDroid / SCP](https://www.sciencedirect.com/science/article/abs/pii/S016764232300134X).

This line focuses on the semantic matching component itself.  SemFinder studies
many semantic matching configurations over thousands of GUI events and proposes
an improved semantic matching algorithm.  TRASM/TReaDroid use adaptive semantic
matching and event deduplication to improve test reuse.

OmniFlow value: optional citation.  It is useful if reviewers challenge the
PageVectorSet/matcher design.  The answer should be that OmniFlow's current
matcher is a pragmatic on-device component, and learned/adaptive matchers are
orthogonal upgrades.

### RIDA: Cross-App Record and Replay for Android

- Venue: ICST 2023.
- Source: [PDF](https://yepangliu.github.io/files/ICST2023-Rida.pdf),
  [Zenodo artifact](https://zenodo.org/record/7234185).

RIDA records interaction sequences on a source app and replays them on another
app with similar functionality.  It is closer to replay than test migration
systems that output a test script.

OmniFlow value: strongly cite if the related-work section has a replay cluster.
It is a useful foil for "cross-app replay" versus OmniFlow's "agent cache hit
over recalled `Function`s with live authorization."

### ReuseDroid

- Source: [arXiv](https://arxiv.org/abs/2504.02357),
  [HTML](https://arxiv.org/html/2504.02357v1).

ReuseDroid is a VLM-powered multi-agent Android UI test migrator.  It analyzes
source tests and migrates based on shared core logic rather than requiring the
entire operational logic to match.  The paper reports LinPro, a dataset with
578 migration tasks for 39 apps across 4 categories, and 90.3% migration
success.

OmniFlow value: must consider if the paper is still current during submission.
It is the closest VLM-era test-migration work.  However, it still migrates tests
and invokes VLM agents for migration; OmniFlow's goal is to reduce online VLM
calls through cache-first execution.

### LLMigrate

- Source: [arXiv](https://arxiv.org/abs/2411.17933).

LLMigrate uses LLMs for automated usage-based UI test transfer across Android
apps.  It reports 97.5% success rate and 91.1% reduction in manual effort.

OmniFlow value: cite if discussing LLM/VLM test migration.  It increases the
reviewer pressure on any claim that LLM can solve GUI transfer offline.  The
OmniFlow response should be: yes, LLM/VLM can compile or migrate traces; our
runtime contribution is avoiding repeated online reasoning by serving compiled
experience as verified cache hits.

### MACdroid

- Source: [arXiv](https://arxiv.org/abs/2409.05028).

MACdroid introduces an abstraction-concretization paradigm: abstract source
tests into general test logic, then use an LLM to generate concrete target-app
tests.  It reports 64% target-functionality success on the FrUITeR dataset and
75% on the Lin dataset.

OmniFlow value: useful for the "offline compiler" framing.  It supports saying
that historical traces can be converted into reusable structured assets, but it
also means OmniFlow should avoid presenting this conversion alone as sufficient
novelty.

### ITeM

- Source: [ACM DOI](https://dl.acm.org/doi/10.1145/3728978),
  [PDF](https://seg.nju.edu.cn/uploadPublication/copyright/1251771328477.pdf),
  [GitHub](https://github.com/Vallxy/ITeM).

ITeM is an LLM-based intention-driven GUI test migration system.  It models
source actions as test intentions and uses dynamic reasoning to fulfill those
intentions on a target app.  It reports 35 Android apps, 280 migration tasks,
and higher F1/success than CraftDroid and TReaDroid.

OmniFlow value: relevant to Function granularity.  Their "test intention" is
close to the idea that one source action may expand into one or multiple target
actions.  OmniFlow should explicitly say whether a `Function` can contain a
short segment rather than only a single action.

### SAIL

- Source: [arXiv](https://arxiv.org/abs/2409.13311).

SAIL frames UI test migration as skill-adaptive imitation learning.  It argues
that even strong event mapping can be insufficient when source and target apps
have implementation differences; it extracts multi-level skills from source
tests and adapts them to the target context.

OmniFlow value: optional but conceptually relevant.  It is closest to the
"history becomes skill/function" framing, but still targets test generation.

## What This Means for OmniFlow Novelty

### Claims to avoid

- Avoid saying OmniFlow is the first system to align historical GUI context with
  current GUI context.
- Avoid saying action transfer / widget matching is itself the main novelty.
- Avoid saying prior replay systems cannot verify correctness; testing work has
  oracles, ground-truth tests, and effect checks.
- Avoid treating a single historical action transfer module as enough novelty.

### Claims that remain defensible

- OmniFlow is a GUI Agent cache system, not a GUI test migration tool.
- Its stored object is a reusable `Function` asset produced by `run_log ->
  function asset`, not merely a source test case or raw event trace.
- Its online objective is cache-first execution: cache hits skip online
  step-wise reasoning for matched segments.
- It performs live recall and authorization against current PageVectorSet state,
  then executes with control-aware verification and fallback.
- VLM/LLM reasoning can be used offline or on cache miss, while the repeated
  successful path is served by the cache.
- The system supports co-evolution through trajectory reuse: successful
  executions can improve a shared function store for future users/tasks.

### Technical gap to close in our paper

The strongest critique is: "Is OmniFlow just test migration/replay with a new
name?"  The paper needs one clean answer:

> Test migration systems use a source test to synthesize or replay a target
> test.  OmniFlow uses a successful agent execution to build a cacheable
> `Function` and serves that `Function` online when the live GUI state and task
> preconditions match.  The core system problem is not producing a target test
> artifact, but deciding when a cached operation can safely replace online model
> reasoning and how to recover when it cannot.

## Recommended Related-Work Paragraph

Short version:

> Android UI test reuse has long studied how to transfer GUI events from a
> source app to a target app.  CraftDroid maps actionable widgets using semantic
> analysis and transfers both events and oracles; FrUITeR later standardized
> evaluation for UI test reuse and showed that event mapping must be judged by
> executable usefulness; TEMdroid further improves migration with a learned
> context-aware widget matcher.  Recent LLM/VLM systems such as LLMigrate,
> MACdroid, ITeM, and ReuseDroid lift migration from widget matching toward
> intentions, core logic, and visual reasoning.  OmniFlow builds on the same
> observation that reuse requires aligning historical GUI context to a live GUI
> state, but targets a different runtime: it converts successful agent run logs
> into cacheable `Function` assets and uses cache-first execution, live
> verification, and VLM fallback to reduce repeated online reasoning.

More pointed version:

> Prior GUI test migration systems optimize developer testing effort: given a
> source test and a target app, they synthesize or replay a target test.  OmniFlow
> optimizes agent serving cost: given successful executions of recurring user
> tasks, it decides whether a `Function` cache hit can safely replace online
> VLM planning on the current device state.  This shifts the reuse problem from
> offline test generation to online cache admission and control-aware execution.

## Suggested Citation Set

Minimum set:

- FrUITeR: A Framework for Evaluating UI Test Reuse. ESEC/FSE 2020.
- Test Transfer Across Mobile Apps Through Semantic Mapping. ASE 2019
  (CraftDroid).
- Learning-based Widget Matching for Migrating GUI Test Cases. ICSE 2024
  (TEMdroid).
- ReuseDroid: A VLM-empowered Android UI Test Migrator Boosted by Active
  Feedback. arXiv 2025.

Expanded set:

- AppFlow: Using Machine Learning to Synthesize Robust, Reusable UI Tests.
  ESEC/FSE 2018.
- Test Migration Between Mobile Apps with Similar Functionality. ASE 2019
  (AppTestMigrator).
- RIDA: Cross-App Record and Replay for Android. ICST 2023.
- GUI Test Migration via Abstraction and Concretization. TOSEM 2025 / arXiv
  2024 (MACdroid).
- Intention-Based GUI Test Migration for Mobile Apps using Large Language
  Models. ISSTA 2025 (ITeM).
- Automated Test Transfer Across Android Apps Using Large Language Models.
  arXiv 2024 / ACM related DOI 2025 (LLMigrate).
- Test Reuse Based on Adaptive Semantic Matching across Android Mobile
  Applications. QRS 2022 (TRASM).
- Semantic matching of GUI events for test reuse: are we there yet? ISSTA 2021.

## Final Judgment

FrUITeR/CraftDroid/TEMdroid are valuable and should be cited.  The danger is
not that they kill OmniFlow; the danger is that they make a weak OmniFlow
framing look old.  If OmniFlow is framed as "we transfer historical GUI actions
to the current screen," it will look derivative.  If it is framed as "we build
an executable Function cache for GUI agents, with live recall, transfer,
verification, and fallback so repeated routines avoid online model reasoning,"
then these works become strong supporting background rather than direct
novelty blockers.
