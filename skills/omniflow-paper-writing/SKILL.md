---
name: omniflow-paper-writing
description: Write, revise, translate, and review the OmniFlow paper about cross-device GUI experience sharing, callable Functions, UI Representation, Function Recall, and action Transfer. Use this skill whenever the task involves the OmniFlow abstract, Introduction, Challenges and Motivations, System Design, Method, Implementation, Evaluation, captions, titles, or translations. Keep the paper centered on transferring reusable GUI experience to a changed UI.
compatibility: No external tools required. Check the project's terminology and implementation before making technical claims.
---

# OmniFlow Paper Writing

Use this skill to keep the OmniFlow paper consistent with its actual system and
with the terminology established by the authors. The paper should tell one
technical story rather than present Function storage, UI matching, and agent
execution as unrelated contributions.

## Central story

The paper addresses cross-device sharing of executable GUI experience. Its
central design principle is **Transfer rather than generalization**: a known
operation is reused, and its recorded UI actions are transferred to the
current UI.

The complete story is:

```text
successful RunLog
  -> stable semantic Function
  -> Function Recall from the current UI and task goal
  -> action Transfer to the current UI
  -> direct execution or VLM/planner fallback
```

VLM-based GUI agents normally combine two capabilities in an
`observe--reason--act` loop:

1. **Workflow/planning** decides which operation should be performed and how
   several operations should be combined.
2. **action Transfer** transfers an intended action to the current UI.

OmniFlow separates these capabilities for reuse. A successful RunLog provides
the operation and action order that can be reused, while the UI target of each
action must be transferred again when the device, layout, content, or session
changes.

## Fixed vocabulary

Use these terms consistently in the title, abstract, captions, equations,
method, and evaluation.

| Term | Meaning |
|---|---|
| `OmniFlow` / `\\sysname` | The complete system. Use `\\sysname` when the paper macro is defined. |
| `Function` | A callable, state-bounded GUI operation with a semantic description, parameters, and recorded actions. |
| `Function Store` | The collection of Functions and the UI states stored with their actions. |
| `RunLog` | A successful recorded GUI execution used to create or extend Functions. |
| `UI` | The graphical interface observed and acted on by the agent. |
| `UI state` | One UI observation at a particular point in execution. |
| `source UI` | The UI state where a recorded action was performed. |
| `current UI` | The UI state where the recorded action is to be used. |
| `UI element` | A visible or interactive element in a UI. Prefer this over `node`. |
| `UI Representation` | The encoded representation produced from the screenshot and UI hierarchy. |
| `local information` | Information from nearby UI elements and their structural or spatial relations. |
| `Function Recall` | Selecting Functions that may apply to the current UI and task goal. |
| `action Transfer` | Transferring a recorded action target from the source UI to the current UI. |
| `OmniTransfer` | The learned component that provides the shared UI Representation and action Transfer. |
| `VLM` | The original vision-language model used by the GUI agent. |
| `planner` | The agent component that chooses operations and combines Functions. |
| `checker` | A reusable rule for handling a recurring UI condition. |
| `observe` / `act` | The physical interfaces used by the existing GUI agent. |
| `MCP tool` / `tool list` | The interface through which Functions are exposed to the agent. |

Use `page` only when discussing page-level matching or Function entry states.
Use `source UI` and `current UI` when describing action Transfer.

## Preferred wording

Use the left-hand expression. Replace the right-hand expression when revising
the paper.

| Prefer | Avoid |
|---|---|
| `UI Representation` | operational representation, contextual representation, state embedding as a general name |
| `UI elements` | controls, entities, widgets, nodes as the default term |
| `local information` | local operational context, neighborhood topology, structural prior |
| `relations between UI elements` | graph topology, relational substrate |
| `Function Recall` | memory addressing, retrieval substrate |
| `action Transfer` | action mapping, action relocation, grounding decoder, cross-page association |
| `recorded action target` | source anchor, action endpoint |
| `current UI` | target environment, deployment surface |
| `Transfer score` | confidence score, affinity logit, evidence strength |
| `VLM or planner fallback` | recovery branch, abstention pathway |
| `call`, `use`, or `select` | invoke, invoked |
| `Transfer` | match, corresponding, correspondence |
| `GUI action` or `device action` | primitive action |
| `cross-device` | heterogeneous deployment environment |
| `RunLog` | trace |
| `UI information` or `recorded UI state` | evidence as a newly defined object |

Do not use `UDEG`, `Universal-Dynamic Execution Graph`, `state index`,
`online encoder`, or `functional subspaces`. OmniTransfer is the unified model
for UI Representation and action Transfer; do not introduce a second page
encoder or a second action mapper.

Use `replay` only when discussing a baseline, fixed scripts, or related work.
Do not call OmniFlow a replay system in the title or when stating its main
contribution.

Use `Transfer` consistently for the action operation. Do not use `mapping`,
`map`, `grounding`, `relocation`, or `locating` in the paper. Use `Function
Recall` consistently for selecting Functions; do not alternate between
`Recall`, `retrieval`, and `page matching` as names for the same operation.
Use `Transfer score` consistently for the action decision; do not replace it
with `confidence` or another score name.

## Writing style

Write concise, direct, scientific English. Explain why a design is needed
before describing how it is implemented.

Prefer concrete verbs such as `encodes`, `recalls`, `transfers`, and `scores`,
`recalls`, `executes`, `checks`, and `returns`.

Avoid inflated claims such as `revolutionizes`, `fully understands`,
`general-purpose`, or `solves all GUI changes`.

Avoid long chains of negative statements. First state the capability provided
by OmniFlow, then state the limitation that motivates it.

Keep one main claim per paragraph. Do not introduce a new name for a concept
that already has a fixed term. Do not use several synonyms for the same object
in adjacent paragraphs.

## Introduction logic

The Introduction should normally follow this order:

1. Explain the VLM `observe--reason--act` loop and its repeated latency and
   computation cost for routine mobile tasks.
2. Explain that prior experience reuse stores either semantic memory or
   executable scripts.
3. State the two limitations: semantic memory still requires reasoning, while
   fixed scripts and simple selectors are difficult to transfer across changed
   UIs.
4. Give the key observation: operation and action order can be reused, but the
   UI target must be transferred again.
5. Introduce OmniFlow as callable Functions plus OmniTransfer.
6. Explain that stable Function steps execute directly and uncertain steps
   return to the VLM or planner.
7. State the contributions and report only verified results.

Use this bridge when needed:

```latex
Our key observation is that a successful GUI execution contains two different
types of information. The operation and action order can be reused across
devices, while the UI target of each action must be transferred again on the
current UI. A practical reuse system should therefore store stable semantic
segments as callable Functions and provide a reliable action Transfer method
for using them in a changed UI.
```

## Challenges and Motivations

Use two challenges rather than a long list of loosely related problems.

### Reusable workflow units

Describe the tension between a complete fixed workflow and a semantic memory:

```latex
\\paragraph{Reusable workflow units.}

A successful GUI task may contain both stable operations and steps that depend
on the current UI. A complete fixed sequence ties reuse to the recorded
workflow from beginning to end, while a semantic memory requires the agent to
read the stored experience and plan again. The challenge is to extract stable
semantic segments as callable Functions, while allowing the agent to combine
multiple Functions and return to VLM reasoning when a new decision is needed.
```

### Cross-device action Transfer

Describe why action Transfer is needed:

```latex
\\paragraph{Cross-device action Transfer.}

The operation and action order in a successful Function can be reused, but the
UI target of each action may change across devices, layouts, app versions, and
sessions. Coordinates, selectors, resource identifiers, and single-element
attributes provide useful signals for the recorded UI, but they do not
remain stable across these changes. The challenge is to build a shared UI
Representation and use local information to transfer each recorded action to
the current UI with a reliable execution threshold.
```

Close the section with the system path:

```latex
A successful RunLog is converted into a state-bounded Function, recalled from
the current UI, transferred by \\textsc{OmniTransfer}, and either executed
directly or returned to the VLM when the match is uncertain.
```

## System Design logic

Use the following structure:

```text
System Design
├── Architecture and Workflow
├── Offline Function Augmentation
├── OmniTransfer
│   ├── Local UI Representation
│   └── Function Recall and action Transfer
└── Online Function Recall and Execution
    ├── observe--check--transfer--act
    ├── checker
    └── VLM fallback
```

### Architecture and Workflow

State the system boundary briefly:

```latex
Mobile applications expose GUI screens and touch actions but rarely provide
stable APIs for user operations. \\sysname builds such APIs from successful GUI
executions by organizing reusable operations as callable Functions. Offline,
successful RunLogs are divided into stable semantic segments and stored with
their ordered actions, reusable parameters, and source UI states.

Online, the current UI and task goal are used to recall relevant Functions.
The agent can use one Function or combine several Functions in a larger
workflow. \\textsc{OmniTransfer} transfers each recorded action to the current UI.
Reliable matches execute directly, while uncertain steps return to the VLM or
planner.
```

### Offline Function Augmentation

The offline section explains Function design and storage. It should not describe
online retrieval in detail.

A Function contains an operation description, input schema, parameter bindings,
ordered actions, and the source UI state for each action. A Function may retain
multiple source UI states from different devices, layouts, or sessions.

Use `successful RunLogs are divided into stable semantic segments`. Do not
describe the RunLog as a mechanically copied complete script. Explain that
tightly related actions remain together while stable parts can be used
independently.

### OmniTransfer

Introduce OmniTransfer as the shared model for both operations:

```latex
\\textsc{OmniTransfer} provides the shared UI Representation used by both
Function Recall and action Transfer. It encodes screenshots and UI hierarchies
into a compact representation, using multimodal UI element information and
relations between nearby UI elements.
```

Explain local information with concrete examples:

- several UI elements may share text or widget type;
- an icon may have no text;
- layout and absolute position may change across devices;
- nearby elements and their relations can preserve the interaction role.

The page-level Representation supports Function Recall. The element-level
Representation supports action Transfer. Do not present these as unrelated
models.

### Online execution

Describe the execution contract plainly:

```latex
\\textsc{Flow} executes recalled Functions through an
observe--check--transfer--act loop. Before each action, the runtime observes
the current UI, checks recurring UI conditions, transfers the recorded action,
and executes it when the Transfer score satisfies the execution threshold. If
Transfer fails, the current step returns to the VLM or planner.
```

Always state the alternating workflow:

```latex
A Function may execute several stable steps directly, return control to the
VLM when the current UI requires a new decision, and continue with another
Function after the planner selects the next operation.
```

Do not imply that source coordinates are executed after Transfer failure.

## Implementation and Evaluation

Keep the four implementation layers distinct:

```text
Implementation
├── Plugin integration
├── OmniTransfer model architecture
├── Training data and objective
└── Deployment
```

Put model dimensions, layer counts, loss, training data, optimizer, epochs,
checkpoint selection, and parameter count in Implementation or the Appendix.
In Evaluation Setup, only summarize the frozen checkpoint and training data
scale. Check every number against the final deployed OmniTransfer checkpoint.

Separate claims in Evaluation:

- end-to-end task success evaluates the complete OmniFlow system;
- conditional Function reuse evaluates whether a successful Function can run
  on a target device;
- page-level Function Recall evaluates Function Recall;
- element-level action Transfer evaluates UI Transfer;
- latency and model usage evaluate execution efficiency.

Do not use end-to-end success alone to claim Transfer accuracy. Do not use
action Transfer accuracy alone to claim workflow planning.

Use `TBD` for missing results and do not invent numerical results.

## Claims and citations

Match every claim to an experiment or a verified citation. Use `our results
show` only for completed measurements. Use `we evaluate whether` for an
experiment that has not yet been completed.

Never invent references or BibTeX entries. If a citation has not been checked,
write a visible placeholder and flag it for verification.

Do not claim that existing methods fail universally. Prefer precise statements
such as:

```text
These representations provide limited support for action Transfer when the UI
changes across devices or sessions.
```

## Translation rules

Keep established technical terms unchanged when translating:

```text
Function          -> Function
Function Store    -> Function Store
Function Recall   -> Function Recall
action Transfer   -> action Transfer
UI Representation -> UI Representation
RunLog            -> RunLog
source UI         -> source UI
current UI        -> current UI
VLM               -> VLM
planner           -> planner
```

Translate “定位” as `Transfer` when referring to a recorded action target.
Translate “复用” as
`reuse` or `Function reuse`. Translate “稳定段” as `stable semantic segment`
or `state-bounded segment`. Translate “交替执行” as `alternating Function
execution and VLM reasoning`.

When the user requests bilingual output, provide the English LaTeX first and
the Chinese explanation after it. Keep the English technical terms unchanged
in the Chinese version.

## Final review checklist

Before returning a revision, check:

- Does the paragraph explain why the design is needed before how it works?
- Is cross-device GUI experience sharing the central problem?
- Is Transfer, rather than generalization, the central design principle?
- Are workflow reuse and UI action Transfer separated clearly?
- Is Function described as a callable, stable semantic segment?
- Does each action retain its source UI state?
- Is OmniTransfer the only UI Representation and action Transfer model?
- Has all UDEG terminology been removed?
- Has the obsolete separate encoder been removed?
- Is Function Recall distinguished from action Transfer?
- Is alternating Function execution and VLM reasoning stated?
- Is the execution threshold and VLM fallback stated?
- Are speed claims limited to measured results?
- Are model statistics consistent with the final checkpoint?
- Are `invoked`, `corresponding`, and `primitive` absent?
- Are unsupported accuracy, robustness, and privacy claims avoided?
- Are citations verified rather than guessed?
