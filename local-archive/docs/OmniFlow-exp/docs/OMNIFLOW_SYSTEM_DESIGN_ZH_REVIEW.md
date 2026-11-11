# OmniFlow System Design 中文结构审阅稿

> **文档用途：** 将三段文字稿按照明显的层级关系排布为一个完整的 System Design 中文版本。
>
> **组织方式：** 总体架构 $ightarrow$ Offline Function Augmentation $ightarrow$ OmniTransfer $ightarrow$ Online Function Recall and Execution $ightarrow$ fallback 与闭环。
>
> **原文处理：** 正文保留三段文字稿中的原始内容，仅增加 Markdown 层级、中文标识和 LaTeX 篇幅建议。

## 0. 大纲与 LaTeX 三页排布

| LaTeX 章节 | 中文内容 | 建议篇幅 |
|---|---|---:|
| `Architecture Overview` | RunLog、Function、OmniTransfer 与在线执行之间的总体关系 | 0.25--0.35 页 |
| `Offline Function Augmentation` | 从 RunLog 提取稳定段落，构建 state-bounded Function | 0.75--0.90 页 |
| `OmniTransfer` | 统一 UI 表示、Function Recall 与 action Transfer | 1.10--1.25 页 |
| `Online Function Recall and Execution` | plugin 接入、Checker、逐步执行、fallback 与闭环 | 0.75--0.90 页 |

```text
真实执行产生的 RunLog
        ↓
语义稳定的段落
        ↓
Function 与 Function Store
        ↓
OmniTransfer：UI 表示、Function Recall、action Transfer
        ↓
在线 Function 执行、Checker 与 fallback
        ↓
成功执行重新保存为 RunLog
```

# 1. Architecture Overview

> **本节内容：** 从总体上说明 Function、OmniTransfer 和在线执行分别承担什么工作。
>
> **核心设计：** Function 保存可复用控制流；OmniTransfer 提供统一 UI 表示和重定位机制；在线执行负责 Recall、逐步执行、Checker 和 fallback。

OmniTransfer 的目标，是为 Function 复用提供一个统一的 UI 表示和重定位机制。通过 Function，我们将控制流和匹配流解耦，同时保留二者之间的必要联系，从而实现两个明确的功能：

1. 快速判断当前 UI 是否对应某个历史状态；
2. 在状态匹配之后，将历史 action 重定位到当前设备上的可执行 UI element。

## 1.1 六、结论:所需的核心组件支持

基于以上两点,这样的设计需要两个核心组件的支持:

1. 高效、快速、准确的实时状态定位:用于检测当前状态是否与之前匹配,并检测可能触发的异常情况;
2. 状态绑定机制:将已有的 UI representation 与当前界面进行绑定。

通过这种方式,我们实现了准确的记忆执行。

# 2. Function 设计思路整理：Offline Function Augmentation

> **本节内容：** 说明为什么不能直接把 RunLog 当作脚本，以及如何从 RunLog 中提取语义稳定的段落。
>
> **核心设计：** 将稳定段落保存为 Function；Function 具有起始状态，每一个 ordered step 都绑定一个状态；Function 的灵活性由组合提供。

## 2.1 一、动机与挑战

> **本段内容：** RunLog 容易获得，但直接转换为脚本难以复用。
>
> **核心设计：** 从真实 RunLog 出发，而不是要求用户编辑脚本。

我们能够获取到的是用户或 engine 在真实执行过程中产生的 RunLog。但这些 RunLog
存在大量噪声，且包含重复内容,如果直接将它们转换成脚本,难以覆盖任务的全集。

核心矛盾在于:RunLog 的收集是廉价的,但脚本的收集是困难的(用户难以编辑脚本)。

## 2.2 二、核心洞察(Key Insight)

> **本段内容：** 从已有 RunLog 中还原语义稳定的段落。
>
> **核心设计：** 将语义稳定的段落定义为 Function，并通过自顶向下的方式引出 Function。

正因为脚本收集困难,我们采取的方法完全依赖于:如何通过一个已有的
RunLog,去还原出可能存在的、语义稳定的段落。

一个自然的想法,是通过找出语义稳定的段落——我们称之为 Function。Function
是我们在现实世界执行中比较常见的元素。写作中我们用的是一个自顶向下的叙述方式来
引出它。

## 2.3 三、我们的核心观察与设计

> **本段内容：** 定义 Function、Function Store，以及 Function 与普通脚本的区别。
>
> **核心设计：** Function 是 API 或宏形式的可编程工具；Function 保存稳定段落，并与执行状态严格对应。

1. 我们将 Function 视作可以编程的工具(可以理解为一个
   API,或者是一个宏),并维护了一个 Function Store。
2. 与其他脚本的不同之处在于:我们认为执行需要严格的状态对应才能进行
   re-use。而这种 re-use 通常发生在 RunLog 的某些段落里。
3. 实现步骤:
   - 第一步,先保存 RunLog 的信息;
   - 在此基础上,将稳定段落视作真实执行的原子宏——也就是说,Function
     转换过程中需要把它当作原子宏来考虑。
4. Function 的必备元素/特征:
   - 它有起始状态;
   - 它的每一个节点都绑定一个状态。
   - 这是必要设计,因为我们要做实时的校验和恢复,别人没有做这一点。
5. 设计出发点:
   - 不追求 Function 自身的语义泛化性;
   - 我们的能力是由 Function 的组合提供的;
   - Function 本身就是"一段稳定的执行"。

## 2.4 四、这样设计的好处

> **本段内容：** 解释 Function 自身保持稳定、系统泛化性由 UI representation 和 Function 组合提供。
>
> **核心设计：** Function 不负责覆盖全部变化；UI representation 提供状态适配；plug-in 设计连接现有 GUI agent。

我们发现脚本的问题在于:泛化性如果指望脚本自身来覆盖,是有限的。

- 我们的泛化性是由 UI representation 表示出来的——只要知道当前状态就可以判断下
  一步该做什么,这样的做法能带来很大的泛化性;
- 但 Function 本身一定要是稳定的原子语义,可以理解为一个 API 或者一个宏。
- 这种插入式(plug-in)设计,能够让 Function 结合任意一个现有的 GUI agent 使用。

## 2.5 五、由此带来的挑战与应对

> **本段内容：** Function 的 ordered actions 如何与 UI state 绑定。
>
> **核心设计：** 每一步操作都必须满足当前状态要求，并沿着预期状态推进。

这种设计不可避免地带来一个挑战:我们怎么确保执行的正确性?

因此:

- 需要给它的每一步都绑定一个状态,让它始终沿着预期的状态推进;
- 为了应对可能出现的突发状况(例如广告弹窗),设计了一套 Checker 机制;
- Checker 挂载的位置是可插拔、可配置的:针对每一个
  Function,甚至每一个节点,都可以调用一组预定义的 Checker;
- 这些 Checker 本身也是通过 RunLog 生成出来的。

这样一来,我们同时解决了两件事:

1. 保证控制流始终稳定(即 ordered actions
   始终保持稳定),同时具备对异常状态的捕获与恢复能力;
2. 为每一个 ordered step
   实现状态绑定——换句话说,每一步操作都必须满足当前状态的要求。

# 3. OmniTransfer：统一 UI 表示与 Function 重定位

> **本节内容：** 说明为什么需要统一 UI 表示，以及 V10 如何同时支持页面级状态定位和 element-level action 重定位。
>
> **核心设计：** 多模态 UI element 编码、有界稀疏局部关系、Page Attention head 与 Cross-page matching head。

## 3.1 为什么传统的 UI 状态表示不够用

> **本段内容：** 不同设备、用户和 session 中的动态 UI 使单一 UTG、Resource ID、Index 和 XML 难以稳定复用。
>
> **核心设计：** 同时解决状态级定位和 action 重定位，而不是依赖固定状态图或单一 selector。

跨设备 Function 复用首先需要解决 UI 表征的泛化问题。现有常见思路通常是构建 UTG，也就是 UI Transition Graph，把应用表示为一组 UI 状态以及状态之间的转移关系。

这种方法在状态稳定的环境中比较有效。例如，在一些 Web 网络场景中，页面结构相对固定，状态之间的转移也比较容易复现。然而，在真实的 GUI 设备上，情况要复杂得多。不同设备和不同用户之间通常存在大量差异，包括：

- 广告弹窗；
- 个性化推荐；
- 动态加载的内容；
- 屏幕尺寸和布局变化；
- WebView 中重新渲染的卡片；
- 同一功能在不同设备上的组件变化。

因此，很难通过单一的 UTG 为所有设备构建稳定、完整并且可复用的状态转移图。在美团等真实应用页面中，大量卡片和广告内容是动态渲染的，这会导致传统 UTG 在不同设备上的状态匹配表现明显下降。

我们要解决的本质问题有两个：

1. 如何实现高效、快速的状态级定位；
2. 在定位到相关状态之后，如何判断当前任务是否仍然能够完成，以及如何将过去的 action 重定位到当前 UI 上。

之所以需要第二步，是因为现实中的点击操作和运行日志是相互独立、彼此解耦的。历史 run log 记录了过去执行过什么操作，但它并不保证当前设备上仍然存在完全相同的布局、节点编号或坐标。

因此，我们无法仅通过当前 action 做出确定判断，也不能依赖单一的 Resource ID 或 Index。不同用户的手机布局、个性化推荐和动态内容都会导致 XML 节点发生变化，单纯依赖 XML 会存在很大的局限。

## 3.2 Function 带来的系统设计思路

> **本段内容：** Function 保存控制流，UI matching 模块连接历史状态与当前 UI。
>
> **核心设计：** 将全局状态图拆成页面级历史 Function 查询和 element-level matching 两个局部任务。

OmniTransfer 的基本思路是：Function 负责保存可复用的控制流，而 UI matching 模块负责判断这个控制流能否在当前页面上执行。

换句话说，我们不再要求整个应用预先构建一个庞大而稳定的全局状态转移图，而是将问题拆成两个更明确的局部任务：

- 用页面级表示快速找到可能复用的历史 Function；
- 用 element-level matching 判断历史 action 在当前 UI 中对应哪个目标 element。

这样，Function 的控制逻辑和 UI 匹配逻辑可以分别维护。系统不需要为每一种设备、每一种弹窗和每一种动态布局建立一套完整的状态图，而只需要在运行时根据当前 UI 进行匹配和验证。

因此，OmniTransfer 被设计成一个可以在端侧部署的轻量化模块。它不负责重新规划整个任务，也不依赖源设备坐标回放，而是为 Function 提供 UI 状态定位和 action 重定位这两个核心能力。

## 3.3 OmniTransfer V10 的整体设计

> **本段内容：** OmniTransfer V10 的三个处理层次。
>
> **核心设计：** 先编码 UI element，再通过局部关系增强，最后分别服务于 Function Recall 和 action Transfer。

OmniTransfer V10 大致可以分成三个层次。

### 3.3.1 第一层：多模态 UI element 编码

> **本段内容：** 从 screenshot 和 UI hierarchy 中构建每个 UI element 的基础表示。
>
> **核心设计：** 统一结合文字、视觉、组件类型、interaction state 和 geometry。

对于每一个观察到的 UI state，系统接收两类输入：

- 当前设备的 screenshot；
- 与 screenshot 对应的 UI hierarchy。

UI hierarchy 中的每一个 element 都会被编码为一个多模态节点表示。输入信息包括：

- text 和 content-description；
- widget type；
- interaction state；
- normalized geometry；
- element 在截图中的视觉区域；
- element 附近的局部视觉区域。

这一步的目标不是简单地把 screenshot 或 XML 压缩成一个向量，而是为每一个 element 构建一个统一的表示，使文字、视觉外观、组件类型和布局信息能够在同一个表示空间中协同工作。

例如，一个有明确文字的按钮可以利用文字和 widget type；一个没有文字、没有 Resource ID 的图标，则需要依赖它在截图中的视觉外观以及它在 UI hierarchy 中的结构位置。

这一层输出的是每个 UI element 的基础节点表示。在 V10 中，每个节点表示被压缩到统一的 128 维空间，作为后续局部关系建模和两个输出头的共同输入。

### 3.3.2 第二层：局部关系学习与上下文校验

> **本段内容：** 单个 UI element 属性不足时，如何利用附近 element 的结构和空间关系。
>
> **核心设计：** 构建有界的稀疏局部图，避免整页 $N^2$ 关系，同时学习 element 在局部 UI 中的作用。

仅靠 element 自身的多模态属性，仍然无法覆盖所有真实情况。实际环境中经常存在大量属性缺失的节点。例如，某些 XML 节点只有 Node 类型、空间位置或层级关系，没有 Text，也没有可靠的 Resource ID。

这正是局部关系发挥作用的地方。

虽然某个节点自身可能缺少明确标注，但它周围的局部结构通常具有更强的稳定性。我们预定义了若干结构和空间关系，例如：

- parent--child 关系；
- sibling 关系；
- 邻近关系；
- 相对方向；
- 对齐关系；
- 包含关系。

在这些基础关系之上，V10 让每个节点自适应地学习：它与附近哪些节点真正相关，以及这些邻居的信息应该如何影响当前节点。

这里的关键不是构建整页的稠密关系图，而是构建一个有界的稀疏局部图。每个节点只在有限的局部候选范围内选择邻居，并通过多层关系更新得到上下文增强后的节点表示。

这样做有两个好处：

1. 计算量不会随着整页节点数产生 $N^2$ 爆炸；
2. 节点表示中不仅包含“它是什么”，还包含“它在周围 UI 中扮演什么角色”。

例如：

- 如果某个点是一个操作节点，它周围通常会存在与之配套的关联组件；
- 搜索框的输入区域、搜索按钮和相关标签往往构成稳定的局部结构；
- 如果我们需要查找一个联系人，那么联系人姓名以及该行对应的操作图标，通常会处于同一个局部组件结构中；
- 当我们要“给 Alice 打电话”时，电话图标可能没有明确的 Resource ID，界面上也可能有多个相似图标，但 Alice 姓名与附近操作图标之间的局部关系仍然可以帮助模型确定正确目标。

因此，局部关系学习相当于对多模态 element 表示进行一次结构校验。它弥补了单个节点属性缺失的问题，也降低了模型对绝对位置和固定节点标识的依赖。

### 3.3.3 第三层：两个任务输出头

> **本段内容：** 上下文增强后的 element representations 如何分别支持页面级和 element-level 任务。
>
> **核心设计：** Page Attention head 用于页面级 Function Recall；Cross-page matching head 用于 action Transfer。

经过多模态编码和局部关系更新之后，系统得到一组带有局部上下文的 element representations。V10 不把这些表示直接压缩成一个结果，而是通过两个不同的输出头服务于两个不同的任务。

#### 3.3.3.1 Page Attention head：页面级 Function Recall

第一个输出头用于页面匹配和 Function Recall。

系统将页面中所有经过局部关系增强的节点表示，通过一个 learned Page Attention readout 进行聚合。这个 readout 使用多个 attention slots，从不同角度关注页面中的重要区域和结构信息，然后将这些节点表示聚合为一个紧凑的页面向量。

最终得到一个归一化的 1024 维 page embedding，用于：

- 表示当前 UI state；
- 与 Function Store 中历史页面状态进行相似度比较；
- 快速召回可能复用的历史 Function。

这个页面向量表达的是页面整体配置和结构，而不是某一个具体 action 的目标。它的作用是回答：

> 当前页面最像哪些历史页面，哪些 Function 值得尝试？

#### 3.3.3.2 Cross-page matching head：action Transfer

第二个输出头用于 action Transfer。它不是只接收一个页面，而是同时接收：

- 记录 action 时的 source UI state；
- 当前设备上的 target UI state。

系统分别对两个页面进行编码，得到 source element representations 和 target element representations。然后，matching head 对 source element 与 target candidate element 之间的对应关系进行打分。

输出不是一个简单的页面相似度，而是一个候选匹配分数集合，用于判断：

- 历史 action 的目标 element 在当前页面中对应哪个节点；
- 哪个候选节点的局部结构和语义关系最符合；
- 当前 action 是否具有足够的执行置信度。

因此，页面级输出解决的是“应该尝试哪个 Function”，而 element-level matching 解决的是“这个 Function 中的 action 现在应该作用于哪个 element”。

如果没有候选节点达到置信度阈值，系统会明确报告 Transfer failure，并交给正常的 fallback 机制处理，而不会静默地回放源设备坐标。

## 3.4 OmniTransfer 的实际运行流程

> **本段内容：** 从 current UI observation 到 Function Recall、action Transfer 和 fallback 的完整过程。
>
> **核心设计：** 共享 UI 编码器连接页面级 Function Recall 与 element-level action Transfer。

OmniTransfer 的运行过程可以概括为：

1. 观察当前设备的 screenshot 和 UI hierarchy；
2. 对当前 UI element 进行多模态编码；
3. 通过稀疏局部关系获得带上下文的 element representations；
4. 使用 Page Attention head 生成页面级 embedding；
5. 从 Function Store 中召回相近的历史 Function；
6. 对历史 source state 和当前 target state 进行双页面编码；
7. 使用 cross-page matching head 为历史 action 寻找目标 element；
8. 通过置信度判断 action 是否可以执行；
9. 如果匹配失败，则进入正常 fallback，而不是使用源设备坐标。

因此，OmniTransfer 的核心不是“用一个向量解决所有问题”，而是：

> 用共享的 UI 编码器保留多模态和局部结构信息，再通过页面级读出完成 Function Recall，通过 element-level matching 完成 action Transfer。

Resource ID、XML path、source node ID 和绝对坐标都只是当前 UI state 的属性，不被当作跨设备保持不变的身份标识。真正的跨设备重定位依赖的是 element 的多模态证据、局部结构关系和 source-target 之间的 learned matching。

# 4. 在线的 Function 执行与闭环：Online Function Recall and Execution

> **本节内容：** 说明 Function 如何作为 cache layer 插入现有 Agent loop，以及如何执行 Checker、fallback 和经验回流。
>
> **核心设计：** 动态 Function tool search、Observe-Check-Act 顺序执行、携带 Function 上下文的 fallback，以及成功 RunLog 的重新保存。

## 4.1 Function Recall 与 plugin 式接入

> **本段内容：** 在 observe 与 plan 之间完成 Function Recall，并只暴露相关 Function。
>
> **核心设计：** 只需要 Agent 提供 Loop、Plan 和 tool calling，即可接入 OmniFlow。

在真实的执行中，我们会遇到两个挑战：

1. Function Recall（刚刚已讨论）  
   我们在实际执行中通过一个简单的 Observe-Plan-Act 工作流，实现方法的 plugin 式插入接入，可以理解为构建了一个真实执行中的缓存层。在每一步 plan 之前的 observe 里，我们首先将其导入 OmniTransfer 查询器，定位它所处的 Function 入口。定位过程中，会首先召回与其相近的配置，以及从这些配置出发可能触发的 Function；同时根据 goal 进行简单的描述过滤。这种基于工具调用的 tool search 方式，避免了将所有 Function 直接喂给智能体，大幅减少了 token 消耗并增强了复用性。

现有的 challenge 通常是无法在 memory use 和实际执行中进行快速稳定的切换。例如，现有智能体通过调用 API 可以实现自由组合，而现有 memory 工作通常只定义了一些固定模板（如跨 App 比价通常是 3 个 Function 的组合；在小红书完成长程任务也是若干局部 Function 的组合）。因此核心挑战在于如何在复杂任务中精准召回对应内容。

## 4.2 Step-level Execution 与 Checker

> **本段内容：** Function 内部 action 的顺序执行和异常 UI 处理。
>
> **核心设计：** Observe-Check-Act 顺序执行器，以及只依赖 observe/act 接口的快速接入。

2. Step-level 的 Execution  
   实际执行时，会通过 Observe-Check-Act 的顺序执行器，首先检查广告弹窗等异常状态。如果 Checker 触发，系统会进行回退并处理异常，否则直接执行当前的 act。这种设计的好处在于只需提供简单的 observe 和 act 接口，即可快速插入到其他 GUI agent 中。

## 4.3 Fallback 与闭环

> **本段内容：** Function 执行失败后如何交还 VLM，以及成功执行如何重新保存。
>
> **核心设计：** fallback 携带未完成 Function 和当前页面信息，避免完全从零开始；成功操作重新打包保存。

3. Fallback 与闭环  
   完成上述步骤后，我们需要将其 forward 给 VLM。如果直接 fallback 给 VLM，相当于让其从零开始探索。而我们的 fallback 具备以下优势：  
   (a) 携带未完成的 Function 信息  
   (b) 结合当前页面与 OmniTransfer 给出的候选操作，以缩小执行范围

当完成全部操作后，即可将其打包保存。

## 4.4 接入优势和需要说明的设计难点

> **本段内容：** 对比与特定 Prompt、Agent 执行流和解码条件强绑定的 Memory Use 方法。
>
> **核心设计：** OmniFlow 只要求真实 Agent 提供 Loop、Plan 和 tool calling，不与特定 Prompt 或解码条件绑定。

例如我们要说明其他工作，他们存在的问题是他们的 For Back

操作依然是正常，就其他的 Memory Use 工作需要自己和当前的 Prompt

进行，就是你要说明其他的方法它和现有的自己的 Prompt 以及自己的 Agent

的执行流是强耦合强绑定的。而我们的方法只需要一个接口式的声明和接入就行了，你

需要把这一点写出来，这是我们现实的一个挑战，就是很多 Memory

工作的方法是只有在自己的环境和自己的 Prompt

以及自己的这个解码条件下适用的，而我们只需要真实的 Agent 的提供，第一个是一个

Loop，第二个是这个 Loop 的 Plan，这只要支持工具调用就行了。你把这个作为我们

的接入的一个优势和核心设计的难点。

# 5. 总体设计闭环

> **本节内容：** 用一个顺序关系连接 Offline Function、OmniTransfer 和 Online Execution。
>
> **核心设计：** RunLog 产生 Function，Function 通过 UI representation 连接当前 UI，在线执行结果再次形成可以保存的 RunLog。

```text
RunLog 的收集是廉价的
        ↓
从已有 RunLog 还原语义稳定的段落
        ↓
Function 保存 ordered actions、起始状态和每一步状态
        ↓
UI representation 判断当前状态并支持 action 重定位
        ↓
Function Recall 与 tool search
        ↓
Observe-Check-Act 执行
        ↓
Fallback 给 VLM 或完成全部操作
        ↓
将成功操作打包保存
```
