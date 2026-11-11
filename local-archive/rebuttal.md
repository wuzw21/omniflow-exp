
We thank all reviewers for raising this concern. 
Unique contribution of OmniFlow and comparison with prior memory systems (Reviewer A/B/C/D) 
We deem OmniFlow's contribution lies in a workflow that alternates between memory-based and VLM-based execution cross devices, supported by two core components: (1) a lightweight GUI embedding layer and (2) an action transfer component.This allows OmniFlow to reuse pre-built memory whenever possible and adapt to different execution environments, as illustrated by our experiments on three types of phones with different layouts. Heterogeneous devices make sharing hard: privacy, layout differences, no shared state graph, and prior work's memory fails. OmniFlow is that a novel approach that enables cross user collection of unstructured data and friendly  deployment. We will revise the  manuscript and tone down our claims, especially those workflow-level overlap with prior memory systems.

We also acknowledge that the current manuscript does not adequately compare with prior memory baselines. We initially prioritized hierarchical subtask memory (MobileGPT), GUI transition graphs (AutoDroid), and element-level documentation (AppAgent), all using the same model backend. ActionEngine and LLM-Explorer require adaptation because they target web automation and app-exploration coverage, respectively.We acknowledge MobiMem as a directly relevant omission and will add it alongside AutoDroid-V2 under the same cross-device settings.

Distinguishment between this study and prior memory design (A, C1, E)
We do not claim function-like tool calls as novel; they fit agent workflows naturally. What is new is a portable, state-indexed memory with device-side encoding and a re-grounding module, enabling sharing without per-device LLM explorers. MobileGPT and former methods (e.g., AutoDroid,LLM-explorer) rely on xml-feature-based UI matching and state-graph features. Experiments show our advantages.

**Novelty of OmniTransfer (A1-A4, D4).**
Omnitransfer contributes to making re-grounding decisions as system 1 with context alignment.It significantly outperforms the feature-based algorithms reported by ASE, and also significantly outperforms on-device LLM generation. 
For traning, We used paired supervised datasets and focused on node representation and migration, which are specifically configured in the paper.We will construct richer data construction methods and test sets on the next versio.
For PAPT, In testing, we used 128 query-target pairs from 15 popular apps in the PAPT dataset, with the candidate pool shared across applications. Note that recall is coarse: top-k (usually 3) pages expose candidate Functions, and re-grounding decides executability. So the Papt dataset partially demonstrates the accuracy and speed of coarse recall. 
The XML baseline de-noises the hierarchy into structured text encoded with Qwen3-Embedding, like MobileGPT. 
For regrounding, we use 0.9 as the threshold because the numerical separation between hits and misses is clear. 

Function vs trajectory (B2-B3, D3). (Experiment)
Trajectories are compositions of state-indexed Functions, not indivisible replay units, so partial reuse and resumption from intermediate states work. Shared Checkers absorb known interruptions. We will add the whole-trajectory/Function/Function-with-different-granularity ablation in next version.

Cold start of OmniFlow (C1, C3, C4). (Explanation & Experiment)
We regard challenge is how to reuse memory. All frameworks start from the same initial state, produced by a pre-run of GPT-5.6 Sol (near-100% success here) to guarantee fairly measure.

Real device experiments (B2, D1, D6, C3, D5). (Experiment)
We report per-component and per-step level , meaning the speed of the cache-hit portion is close to the lower bound of physical operation/network requests. Limited by real-app permission issues, we have not conducted real-time Checker detection; we will design more realistic environments (with ad pop-ups) to detect Checkers and fine-grained Functions.

Scalability and practical value (E). The three properties must be achieved jointly. For N users, M devices, and K execution trajectories, device-dependent memory requires O(NMK) device-specific bindings, whereas OmniFlow constructs portable memory in O(K) and performs lightweight on-device adaptation across O(NM) targets. 


Additional experiments and explanations
We thank the reviewers for their suggestions. In the next revision, we will:
1.Compare state-matching and action-transfer accuracy with prior methods. (Reviewer A)
2.Test robustness to workflow interruptions (e.g., app updates, repeated pop-ups). (Reviewers B/C/D)
3.Report full-task success, latency, and energy on physical devices. (Reviewers C/D)
4.Ablate function decomposition and intermediate-state replay, and assess threshold sensitivity. (Reviewers B/C/D)
5.Manually quantify false-positive/negative rates and analyze semantic failures in GUI state match and action transfer. (Reviewers A/B/C/D/E)
6.Report token savings across all attempts. (Reviewer C)



审稿原文：

A：
核心意见
建议将重点放到【omnitransfer】的端侧模型并补充实验。质疑novelty。在前两段回答
原文
- 用于页面召回率评估的查询和候选页面数量是多少？候选页面是跨应用共享还是仅限于同一应用？

PAPT 中的页面匹配是否也意味着 OmniFlow 中的页面匹配正确？更具体地说，PAPT 中的手机-平板电脑页面配对是否保证这些页面具有相同的支持功能？

同样，页面匹配评估是否包含布局几乎相同但功能不同的页面？大多数现有的页面匹配算法都无法处理这类页面。通用 XML 和视觉嵌入的高准确率表明，评估中可能缺少或完全没有这类情况。如果没有这些特殊情况，结果可能无法反映 OmniFlow 的 Omni 表示法所要求的功能区别。
- XML 嵌入基线采用哪种模型？XML 是如何进行预处理和编码的？

B
核心意见
认可工作，对「function」这种细粒度的function设计比较感兴趣。希望补充更多的实验。希望看到checker的真实实验。
看中细节。
原文
●细粒度函数分解和组合的优势并未得到单独评估。
●工作流级别的自适应功能并未在评估中得到系统性的独立验证。
●已部署的功能选择和执行阈值并未直接表征

C
核心意见
接受，希望补充更多的实验，例如mobimem。疑似AI审稿。

原文
- 在失败的事件中，62.0% 的事件在没有动作级别失败的情况下完成了检索到的功能的结构性完成，但最终任务验证失败，这凸显了结构重放和语义正确性之间的差距，而这种差距依赖于宿主 VLM 来解决。

D
核心意见
拒绝，认为我们缺少真机实验和对mobimem，autodroid-v2等工作的比较。疑似AI审稿。
原文

1. 您能否报告在不使用 OmniFlow 的情况下，同一代理的配对完整任务延迟和能耗？

2. 您能否使用相同的源跟踪和 VLM，将结果与 MobiMem 或最近基于程序的代理进行比较？

3. 当函数分解或中间状态重放被禁用时，任务成功率和成本会发生什么变化？

4. 如何选择执行阈值，以及接受匹配错误率和重放覆盖率（包括缺失目标）是多少？

5. 在应用程序更新后或反复弹出窗口和登录中断的情况下，OmniFlow 的性能如何？

6.您能否报告在真实手机上的完整任务成功率、操作重用率和回退率？
E 
核心意见
认为没有创新点，命中率很低。不太懂对应领域。核心质疑只有一半的成功率。
原文


对比表格列出了应用探索、模型调用和跨环境可移植性，但并未解释实现每个属性的意义或挑战，也未说明为什么这三个属性共同作用对降低运行时成本至关重要。同时实现这三个属性的挑战是什么？之后，请更具体地说明每个所需属性对应的设计是什么？
介绍折叠屏为什么影响较大
提供具有代表性的成功标准及其在实践中的含义。



Paper #170 OmniFlow: Adaptive Memory Reuse for Mobile GUI Agents
Review #170A
Overall merit
2.Weak reject
Reviewer expertise
4.Expert
Paper summary
OmniFlow reuses successful mobile GUI executions to reduce repeated LLM/VLM calls. It organizes recorded executions into reusable Functions and uses OmniTransfer, a lightweight model, to match GUI states and relocate action targets when the interface changes. The paper evaluates task completion on modified AndroidWorld environments, matching accuracy on PAPT and ASE, and runtime overhead on a physical phone.
Strengths
●The paper is generally well written, and improving the efficiency of mobile GUI agents is an important problem.

●A lightweight on-device model for page matching and action re-grounding is a useful direction for making memory reuse more practical.

●The evaluation includes different device configurations, component ablations, and physical-device measurements.

Weaknesses
●The claimed contribution is broader than the actual novelty. Much of the memory and replay design overlaps with prior work.

●OmniTransfer appears to be the main technical contribution, but its architecture and training are not described in enough detail.

●The evaluation lacks direct comparisons with prior state-matching and action-adaptation methods, making OmniTransfer’s benefits difficult to assess.

Comments for authors
●I see value in this work, but my main concern is how the paper frames its contribution. Prior systems, particularly MobileGPT [1], LLM-Explorer [2], and AutoDroid v2 [3], already introduce many of these ideas, including reusable subtasks, state-based memory, parameter adaptation, and selective LLM fallback. Much of OmniFlow’s memory design reads as a restatement of these concepts.

●The more distinct contribution is the lightweight model for page matching and action re-grounding, yet it receives too little attention. Its architecture and training procedure are not described clearly enough to assess what is technically new.

●The evaluation should therefore directly compare page matching and action re-grounding with the corresponding mechanisms in prior systems, including MobileGPT and LLM-Explorer. Comparisons with generic embeddings and simplified matching baselines alone do not establish the model’s benefits over existing approaches.

Things to clarify
●How many queries and candidate pages are used for page recall eval? Are candidates pooled across apps or restricted to the same app?

●Does a correct page match in PAPT also imply correct page match in OmniFlow? More specifically, does phone-tablet page pair in PAPT guarantee that these pages have the same supporting functions?

●Similarly, does the page-matching evaluation include pages with nearly identical layouts but different functions? These are where most existing page-matching algorithms fail. The high accuracy of generic XML and visual embeddings suggests eval these cases may be underrepresented or absent from the evaluation. Without such difficult cases, the results may not reflect the functional distinctions required by OmniFlow’s Omni representation.

●Which model is used for XML embedding baseline, and how is the XML preprocessed and encoded?

I appreciate the authors’ effort to make memory-based GUI agents more practical and effective. However, the paper needs to focus more clearly on what it actually contributes. At present, it gives more attention to concepts already established in prior work than to its own technical contribution, leaving the most interesting part of the work underexplained.
[1] Lee, Sunjae, et al. “Mobilegpt: Augmenting llm with human-like app memory for mobile task automation.” Proceedings of the 30th Annual International Conference on Mobile Computing and Networking. 2024.
[2] Zhao, Shanhui, et al. “LLM-explorer: towards efficient and affordable LLM-based exploration for mobile apps.” Proceedings of the 31st Annual International Conference on Mobile Computing and Networking. 2025.
[3] Wen, Hao, et al. “Autodroid-v2: Boosting slm-based gui agents via code generation.” Proceedings of the 23rd Annual International Conference on Mobile Systems, Applications and Services. 2025.
What, if anything, could convince you to accept this paper?
A substantial rewrite centered on OmniTransfer, with a clear description of its design and training and direct comparisons against prior page-matching and action-adaptation methods.
Review #170B
Overall merit
2.Weak reject
Reviewer expertise
3.Knowledgeable
Paper summary
This paper presents OmniFlow, an adaptive replay system for mobile GUI agents. OmniFlow converts successful historical executions into fine-grained, composable Functions organized around GUI states. At runtime, it retrieves applicable Functions, re-grounds recorded actions to the current GUI using OmniTransfer, and replays them when sufficiently reliable. When reuse is not applicable, execution falls back to the underlying VLM agent and may later resume from a memorized state. The system is evaluated on AndroidWorld across three target device configurations. OmniFlow achieves a 53.2% task success rate, compared with 37.1% for the same online executor without memory reuse, and reports substantial reductions in model calls and token usage.
Strengths
●The perspective of reusing successful executions for mobile GUI agents is well motivated.
●The evaluation incoporates end-to-end gains, and effectiveness of state retrieval and action re-grounding.
Weaknesses
●The end-to-end evaluation assumes each AndroidWorld task has corresponding memory before target evaluation, so performance under partial historical coverage remains unclear.
●The benefit of fine-grained Function decomposition and composition is not isolated from the other mechanisms in evaluations of OmniFlow.
●The evaluation does not systematically isolate the workflow-level disruptions, which is important of being used to motivate the design.
●The deployed Function-selection and execution thresholds are not directly characterized; the reported top-k retrieval and grounding metrics do not characterize the deployed accept and reject decisions.
Comments for authors
I found the stop, fallback and resume design interesting, but I have several concerns about what the current evaluations.
●The end-to-end evaluation assumes each AndroidWorld task has corresponding memory before target evaluation. This measures reuse when related history is already available, but does not show how performance changes when only part of the future task set is covered by prior executions. How quickly do these gains appear as prior history accumulates? Reporting success rate, reuse rate, and model calls as source-history coverage increases would help answer this, and the offline construction cost could then be related to subsequent reuse.
●The benefit of fine-grained Function decomposition and composition is not isolated evaluated.The paper argues that complete trajectories or fixed subtasks are too coarse, and therefore decomposes executions into Functions that can be reused from intermediate GUI states. However, the current comparisons vary this representation together with state retrieval, action re-grounding, and fallback behavior. A matched baseline could keep the same source trajectories, parameter binding, state matching, OmniTransfer, backend VLM, and fallback policy, while replacing fine-grained Functions with a trajectory-level representation. Figure 7 shows that adding memory is beneficial relative to no memory, but does not isolate this design choice.
●The workflow-level adaptation claim is not systematically isolated in the evaluation. The paper motivates Function memory using workflow deviations such as login screens, pop-ups, loading delays, and alternative entry points. The main controlled AndroidWorld setting instead varies task parameters, screen size, form factor, resolution, and Android version. The onboarding-pop-up case is encouraging, but that case is handled by a shared Checker rather than directly exercising the proposed stop–VLM fallback–resume path. A controlled study with workflow interruptions or alternative entry states would show whether this mechanism actually helps under the workflow deviations used to motivate it.
●The deployed Function-selection and execution thresholds are not directly characterized. OmniFlow ranks candidate Functions using task relevance and GUI-state similarity, applies a confidence threshold and router, and executes recorded actions only when action correspondence satisfies an execution criterion. The micro-benchmarks instead report ranking metrics such as R@1 and Top-1. The reported 57.5% action-reuse rate shows how often memory is used, but not how reliable the deployed decisions are. What are the false-accept and false-reject rates at the deployed thresholds? Reporting this operating point together with reuse rate and task success would show how well the deployed thresholds are calibrated.
What, if anything, could convince you to accept this paper?
A controlled ablation that isolates the benefit of fine-grained Function memory, together with a clearer measurement of false-positive replay/action transfer.
A comparison with the closest recent executable-memory or programmatic GUI-agent systems would also strengthen the positioning.
Review #170C
Overall merit
3.Weak accept
Reviewer expertise
2.Some familiarity
Paper summary
OmniFlow reduces repeated vision-language-model reasoning in mobile GUI tasks by replaying reusable fragments of successful executions. It stores parameterized functions in a GUI transition graph, retrieves them from intermediate states, and uses a small multimodal matcher to relocate recorded action targets. Replay falls back to the host agent when matching fails. On Android World across three device configurations, success rises from 37% to 53%, while token use falls by 50% on jointly successful executions.
Strengths
(+) Resuming replay from intermediate GUI states addresses a useful limitation of whole-trajectory reuse. Combining this with action re-grounding gives the system a coherent response to workflow and layout changes.
(+) The compact action matcher uses local GUI context to distinguish otherwise similar elements. Removing context calibration lowers matching accuracy by 23.33 percentage points, providing substantial evidence for this design choice.
(+) The end-to-end evaluation changes task parameters and device layouts while retaining executable success validators. Comparing against the same executor without memory demonstrates a meaningful benefit beyond isolated retrieval accuracy.
(+) Physical-device measurements establish that GUI encoding and action matching have modest runtime costs. The resident observation service also targets a concrete communication bottleneck in mobile automation.
Weaknesses
(-) The evaluation starts with successful source executions for the benchmark tasks. It does not quantify cold-start failures, experience acquisition cost, or reuse rates in actual user histories. This favorable memory availability leaves the practical amortization of offline construction uncertain.
(-) The strongest closely related memory and programmatic agents discussed in the paper are not evaluated. MobiMem, AutoDroid-V2, and ActionEngine could clarify whether the gains require the proposed representation. The selected baselines alone do not establish the advance over contemporary execution reuse.
(-) Dynamic environments mainly change emulator configuration and task seeds. App-version evolution, intermittent network behavior, authentication changes, and systematic interruptions receive little controlled evaluation. These conditions are central to the claimed resilience of real-world replay.
(-) Action confidence thresholds are not characterized through acceptance coverage and wrong-action rates. Top-1 matching accuracy of 82.74% is insufficient to establish reliable execution by itself. The fallback policy needs evidence that it rejects dangerous mismatches without eliminating reuse.
(-) Most failed episodes complete their selected function without action-level failure but fail task validation. This exposes a gap between structural replay and semantic correctness that the system delegates to its host. Completion checking and parameter binding need stronger end-to-end validation.
(-) The main 50.8% token reduction applies to the subset solved by both variants. The paper reports broader successful-task savings but not total cost across all attempted tasks. Failed attempts and recovery can materially affect the benefit experienced by users.
(-) Physical-device results emphasize component latency and energy rather than paired task-completion measurements. Full-task latency distributions, total energy savings, memory growth, and repeated-use throughput are not established. Component efficiency alone does not validate the overall mobile deployment claims.
(-) The learned matcher is evaluated against mostly handcrafted or unimodal alternatives. Training details omit important encoder, objective, and threshold specifications, while app-disjoint generalization is unclear. This weakens both reproducibility and confidence in performance on unfamiliar applications.
Comments for authors
●The evaluation assumes pre-collected successful source executions, leaving cold-start acquisition overhead, experience collection failure rates, and real-world trace reuse frequencies unquantified.

●Closely related contemporary memory and programmatic agents like MobiMem, AutoDroid-V2, and ActionEngine are omitted from the baseline comparisons, leaving it uncertain whether gains require OmniFlow’s specific architecture over other modern execution reuse methods.

●Among failed episodes, 62.0% ran their retrieved functions to structural completion without action-level failure yet failed final task validation, highlighting a gap between structural replay and semantic correctness that relies on the host VLM for resolution.

Review #170D
Overall merit
2.Weak reject
Reviewer expertise
3.Knowledgeable
Paper summary
OmniFlow reuses successful mobile GUI executions as parameterized functions. A lightweight matcher adapts recorded actions to the current screen, while a VLM handles steps that cannot be replayed.
Strengths
1.Adding memory improves AndroidWorld success from 37.1% to 53.2% with the same online executor.

2.The compact action matcher has low measured CPU overhead on a physical phone.

Weaknesses
1.The phone results report component costs, but do not show full-task latency or energy savings.

2.Close alternatives such as MobiMem and program-based replay agents are discussed but not evaluated.

3.Beyond memory versus no memory, the end-to-end tests do not isolate fine-grained functions or replay from intermediate states.

4.It is unclear how often the confidence gate accepts a wrong target, especially when no valid target exists.

5.Most dynamic tests change device settings and task parameters. App updates and repeated interruptions are not systematically tested.

6.Full-task success and action-reuse rates are missing from the real-phone evaluation.

Comments for authors
1.Can you report paired full-task latency and energy against the same agent without OmniFlow?

2.Can you compare with MobiMem or a recent program-based agent using the same source traces and VLM?

3.What happens to task success and cost when function decomposition or intermediate-state replay is disabled?

4.How is the execution threshold chosen, and what are the accepted-match error rate and replay coverage, including missing targets?

5.How does OmniFlow perform after app updates or with repeated pop-ups and login interruptions?

6.Can you report full-task success, action reuse, and fallback rates on a real phone?

What, if anything, could convince you to accept this paper?
I would reconsider with stronger baseline comparisons and evidence of reliable replay and full-task savings on a phone.
Review #170E
Overall merit
3.Weak accept
Reviewer expertise
1.No familiarity
Paper summary
OmniFlow converts successful execution histories into parameterized, composable functions organized in a GUI transition graph. By reusing prior successful executions, it bypasses repeated reasoning. OmniFlow organizes successful executions into fine-grained, composable functions and rapidly retrieves reusable operations. Compared with the baseline method, OmniFlow reduces token consumption by half while improving task success rate.
Strengths
●Practical problems with the current repeated workflow for mobile GUI agents.
●Lightweight implementation
●Good performance: reduces token consumption and achieves 90+% GUI state retrieval accuracy
Weaknesses
●The novelty of the techniques used in the paper needs better articulation.
●Despite the improvement, the overall success rate is about half.
●More investigation into the sources of improvement in task success is needed.
Comments for authors
Clarify the contribution relative to prior work on reducing runtime costs. The comparison table lists app exploration, model invocation, and cross-environment transferability, but it doesn’t explain the significance or the challenge of achieving each property, or why these three properties matter together for reducing runtime cost. What is the challenge of achieving all three at the same time? Later, more explicitly, what is the design associated with each required property?
The evaluation spans Standard, Fold, and Tablet configurations, with Fold showing a relatively lower task success rate. Please analyze which configuration differences contribute to these results and whether they primarily affect GUI-state retrieval, action grounding, or the underlying agent’s reasoning. This would help identify which configuration factors OmniFlow is most sensitive to.
The definition and practical implications of task completion need more clarification. Although OmniFlow achieves a 53.2% success rate using AndroidWorld’s validators, please provide representative success criteria and what they mean in practice. Since nearly half of attempts remain unsuccessful, what does that mean in practice? How much user intervention is required? More info would help assess whether the reported reliability is sufficient for everyday use.
The failure analysis needs more justification. The paper provides a few examples and lists how the different methods handle the failure. More insight is needed into what these representative failures mean fundamentally for mobile GUI processing, and what it means for the design that these methods handle them differently. In addition, correct replay may not equal a correct semantic task completion. This part of the analysis is not explicit.




Reviewer A

1.[Exp & response & writing] Unique contribution of OmniFlow, especially compared with prior memory systems (MobileGPT, LLM-Explorer, AutoDroid v2)
●we should tone down the contribution of this study
2.[Writing & response] More details on OmniTransfer (architecture & training).
3.[Exp] comparison with state-matching & action adaptation methods
●Including those in MobileGPT & LLM-Explorer
4.[Response] Correct match in PAPT -> correct match in OmniFlow?
●first respond that the representativeness of PAPT.
●indicate that we should conduct human evaluation of our approach in realistic workflows.
5.[Response] Nearly identical page -> different functions?
4.[Response] Detail of the XML embedding baselines, and how is the XML preprocessed and encoded?

Reviewer B

1.[Response & exp] e2e evaluation assumes pre-built memory before evaluation, unclear of performance under partial histories
2.[Exp] fine-grained function (de)composition is not evaluated.
1.[Exp] workflow-level disruptions are not evaluated.
3.[Exp] function selection & exectuion thresholds are not evaluated.
5.[Exp] false-positive replay & action transfer through human validation.
5.[Exp] comparison with recent executable memory and programmatic GUI-agent systems.

Reviewer C
4.[Exp] partial/accumulated histories, and cold-start failures & experience acquisition cost during this process.
6.[Exp] comparision with related work, including MobiMem, AutoDroid-V2, ActionEngine
2.[Exp] More dynamics environments beyond photo target switching.
1.[Response] "Most failed episodes complete their selected function without action-level failure but fail task validation. This exposes a gap between structural replay and semantic correctness that the system delegates to its host. Completion checking and parameter binding need stronger end-to-end validation."
2.[Exp & response] results on all tasks rather than on successful tasks.
3.[Exp] On physical phone, report more metrics rather than pure component efficiency.
7.[Exp & Response] Training details of OmniTransfer

Reviewer D
3.[Exp] physicall phone result: more metrics
8.[Exp] compare with prior memory baselines
5.[Exp] isolate fine-grained functions and partial replay
4.[exp] confidence interval
5.[exp & response] app updates & repeated interruptions

Reviewer E
1.[writing & response] challenge of implementing all characteristics in the first table
2.[writing & response] no analysis on the low task success rate
3.[response] representiveness of 53% success rate on AndroidWorld
4.[writing & response] representiveness of the failure analysis
5.
We thank all reviewers for their thoughtful reviews. We respond to the key concerns below.

Unique contribution of OmniFlow and comparison with prior memory systems (Reviewer A/B/C/D) 
We thank all reviewers for raising this concern. We deem OmniFlow's contribution lies in a workflow that alternates between memory-based and VLM-based execution, supported by two core components: (1) a lightweight, device-friendly GUI embedding layer and (2) an action transfer component.This allows OmniFlow to reuse pre-built memory whenever possible and adapt to different execution environments, which outperforms the evaluated memory systems. We will revise the manuscript and tone down our claims, especially those workflow-level overlap with prior designs.

We also acknowledge that the current manuscript does not adequately compare with prior memory baselines. We initially prioritized hierarchical subtask memory (MobileGPT), GUI transition graphs (AutoDroid), and element-level documentation (AppAgent), all using the same model backend. ActionEngine and LLM-Explorer require adaptation because they target web automation and app-exploration coverage, respectively. We will incorporate MobiMem and AutoDroid-V2 under the same cross-device settings for a more comprehensive evaluation.

As stated in the paper, our core insight: does not require precise encoding of each pixel like embedding model, only two things need to be done: 1. Determine whether the original operation is transferable; 2. Calculate the new migration point. We implement it through a unified, deploied friendly model.
For training , we use ASE page pairs as training data: the same semantic element captured on iOS and Android pages. We treat OmniTrasnfer as a graph-alignment model. Two design decisions carry the novelty. First, a node is encoded with its local graph context, and each candidate pair carries a correspondence state that is refined over local-graph layers with bidirectional cross-page message passing, so the unit of decision is a context-level correspondence rather than a single node's features or coordinates. Second, the head is a partial assignment over all target nodes with an explicit NULL class, so the model can abstain instead of forcing a match.
The XML baseline：we use xml and de-noises the hierarchy into structured text encoded with text-Embedding-v4, like MobileGPT. 
We acknowledge that our experiment assumes existing successful trajectories. We aim to fairly compare the reuse rates of different methods for successful memory. We will add an experiment on omniflow with partial memory in the next version.
PAPT is used to search for similar function pages between tablets and mobile phones. He built a large number of pages with similar functions. We extracted 128 sets of similar pages from the top 15 apps and used all pages as a search set. We entered the original device page and searched for matching pairs. We emphasize that page similarity is only used for rough recall, and we still need to further investigate the capabilities of regularization and function detection. So, PAPT demonstrates our ability for efficient computation and coarse recall, and we will also add richer datasets and increase manual validation of omnitransfer.




Training details of OmniTransfer and corresponding baselines (Revie
wer A/C)
1.找到A/C原始的疑问，回答他们最关心的omnitransfer的training detail
2.描述XML baseline的detail

它的架构和训练程序描述得不够清楚，无法评估技术上的新东西。架构和训练描述
操作置信度阈值并非通过接受率和错误操作率来表征。82.74% 的 Top-1 匹配准确率本身不足以建立可靠的执行机制。回退策略需要证据证明，它能够在不排除重用的前提下拒绝危险的错误匹配

As stated in the paper, our core insight: does not require precise encoding of each pixel like embedding model, only two things need to be done: 1. Determine whether the original operation is transferable; 2. Calculate the new migration point. We implement it through a unified, deploied friendly model.
For training , we use ASE page pairs as training data: the same semantic element captured on iOS and Android pages. We treat OmniTrasnfer as a graph-alignment model. Two design decisions carry the novelty. First, a node is encoded with its local graph context, and each candidate pair carries a correspondence state that is refined over local-graph layers with bidirectional cross-page message passing, so the unit of decision is a context-level correspondence rather than a single node's features or coordinates. Second, the head is a partial assignment over all target nodes with an explicit NULL class, so the model can abstain instead of forcing a match.
The XML baseline：we use xml and de-noises the hierarchy into structured text encoded with text-Embedding-v4, like MobileGPT. 
PAPT is used to search for similar function pages between tablets and mobile phones. He built a large number of pages with similar functions. We extracted 128 sets of similar pages from the top 15 apps and used all pages as a search set. We entered the original device page and searched for matching pairs. We emphasize that page similarity is only used for rough recall, and we still need to further investigate the capabilities of regularization and function detection. So, PAPT demonstrates our ability for efficient computation and coarse recall, and we will also add richer datasets and increase manual validation of omnitransfer.

Additional evaluation on partial memories (Reviewer B/C/D)
1.承认我们当前的manuscript assume a pre-built memory，描述对于其他的memory方法，我们也是使用相同的pre-built memory来保证公平比较
2.我们会在next revision 添加 OmniFlow with partial memory 的实验（a complete memory by manually remove, e.g., the first 25%, the first 50%, the last 25%, the last 50%, or randomly remove one of the GUI node,)来展示不同memory方法在partial memory下的效果。

We acknowledge that our experiment assumes existing successful trajectories. We aim to fairly compare the reuse rates of different methods for successful memory. We will add an experiment on omniflow with partial memory in the next version.


Represetativeness of the PAPT benchmark (Reviewer A)
1.A关心PAPT上能成功match是否对应实际情况也能成功match，讲一下PAPT benchmark的结果是否有代表性
2.后边加一句说我们会在next revision添加OmniTransfer的人工验证，来找到false-positive/false-negative cases

PAPT用于检索平板——手机相似功能页面。他构建了大量相似功能的页面对。我们提取了前15个app里的128组相似页面，将所有页面作为检索集，输入原始设备页面，检索是否命中匹配对。我们强调，页面相似度仅用于粗召回，我们仍需进一步对regrounding和function能力检测。所以，PAPT展现了我们高效计算和粗召回的能力，我们也会添加更丰富的数据集，并增加omnitransfer的人工验证。

Challenges of implementaing all characteristics in Table1 (Reviewer E)
1.两三句话讲一下一起实现table 1中不同的几列有什么挑战吧，实际上还是描述我们这个工作的novelty/uniqueness

Analysis of low task success rate on Foldable phone (Reviewer E)
1.折叠屏上的低任务成功率解释一下是什么导致的


Representativeness of success rate on AndroidWorld (Reviewer E)
1.几句话精炼的解释下我们在AndroidWorld上仅达53%的任务成功率是什么因素导致的，是否会影响我们结论


Additional experiments and explanations
We thank the reviewers for their suggestions. In the next revision, we will:
1.Compare state-matching and action-transfer accuracy with prior methods. (Reviewer A)
2.Test robustness to workflow interruptions (e.g., app updates, repeated pop-ups). (Reviewers B/C/D)
3.Report full-task success, latency, and energy on physical devices. (Reviewers C/D)
4.Ablate function decomposition and intermediate-state replay, and assess threshold sensitivity. (Reviewers B/C/D)
5.Manually quantify false-positive/negative rates and analyze semantic failures in GUI state match and action transfer. (Reviewers A/B/C/D/E)
6.Report token savings across all attempts. (Reviewer C)


final version
We thank all reviewers and address the key concerns below.
Unique contribution of OmniFlow and comparison with prior memory systems (Reviewer A/B/C/D)
We appreciate this concern. OmniFlow's contribution is a workflow alternating between memory-based and VLM-based execution, with two components: (1) a lightweight, device-friendly GUI embedding layer and (2) an action transfer component. This workflow enables reuse of pre-built memory and adaptation to different execution environments, outperforming the evaluated systems. We will revise the manuscript and temper our claims, particularly regarding workflow-level overlap with prior designs.
Our manuscript also lacks adequate comparisons with prior memory baselines. We initially prioritized hierarchical subtask memory (MobileGPT), GUI transition graphs (AutoDroid), and element-level documentation (AppAgent), all using the same model backend. ActionEngine and LLM-Explorer require adaptation because they target web automation and app-exploration coverage, respectively. We will incorporate MobiMem alongside AutoDroid-V2 under the same cross-device settings for a more comprehensive evaluation.
OmniTransfer is trained on ASE page pairs — the same element captured on iOS and Android — as graph alignment over two view hierarchies. Nodes are encoded with local graph context (text, visual crop, widget attributes, geometry) under learned per-modality weights. Candidate pairs carry a correspondence state refined through local-graph layers with bidirectional cross-page message passing, so decisions are context-level rather than per-node. The head is a partial assignment with an explicit NULL class, letting the model abstain rather than force a match.
For the XML baseline, we use XML and denoise the hierarchy into structured text, which is then encoded with text-Embedding-v4, as in MobileGPT.
Additional evaluation on partial memories (Reviewer B/C/D)
We acknowledge that our experiment assumes existing successful trajectories be c. We aim to fairly compare the reuse rates of different methods for successful memories. We will add an experiment on OmniFlow with partial memory(e.g., the first/last 25%/50% or randomly remove one of the GUI node) in the next version.
Representativeness of the PAPT benchmark (Reviewer A)
We extract 128 sets of similar pages from the top 15 apps and use all pages as the retrieval corpus. Page similarity serves only as coarse recall; whether a matched page admits a valid transfer is checked downstream by OmniTransfer. PAPT therefore demonstrates efficiency and coarse-recall capability, and we will extend the dataset and add manual validation of OmniTransfer.
Additional experiments and explanations
We thank the reviewers for their suggestions. In the next revision, we will:
1.Compare state-matching and action-transfer accuracy with prior methods. (Reviewer A)
2.Test robustness to workflow interruptions (e.g., app updates, repeated pop-ups). (Reviewers B/C/D)
3.Report full-task success, latency, and energy on physical devices. (Reviewers C/D)
4.Ablate function decomposition and intermediate-state replay, and assess threshold sensitivity. (Reviewers B/C/D)
5.Manually quantify false-positive/negative rates and analyze semantic failures in GUI state match and action transfer. (Reviewers A/B/C/D/E)
6.Report token savings across all attempts. (Reviewer C)


