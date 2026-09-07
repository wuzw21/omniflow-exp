# Releases

## 1.1.0.dev1 — two Function services, 2026-09-07

The public MCP surface now consists of `omniflow_recall` and `omniflow_execute`.
Both use the existing recall and Check → Transfer → Act → Observe owners.
The host owns planning, primitive fallback, completion and cancellation; a
Function result never implicitly starts another Planner. Task identity keeps
repeated recall within one deadline; request identity deduplicates execution.
The development seven-tool MCP surface is superseded; invocation v1 and stored
Function/RunLog schemas are unchanged.

MCP, GUI-Owl, V-Droid and Mobilerun adapters share the two input definitions.
An explicit Host factory supports synchronous and asynchronous backends, while
formal AndroidWorld remains OOB-only. Optional Host application inventory is
resolved under the invocation deadline. The OOB backend queries real installed
packages before executing open_app; absent inventory and absent packages still
fail explicitly. This addresses a live integration failure, **待真机验证**.

The full suite passes 169 tests, with two optional upstream Droidrun registry
checks skipped because its dependency cannot import. The installed Droidrun
0.5.6 imports `mobilerun`, while installed mobilerun-sdk 5.1.0 exposes
`mobilerun_sdk`. Contract fixtures cover four harness adapters × three Host
forms × success/partial failure, plus application inventory, cancellation,
task retirement, request deduplication and real MCP stdio discovery. These
fixtures do not establish upstream LLM-driven or physical-device acceptance.

The dev1 wheel imports from an isolated installation outside the repository,
discovers exactly two tools and contains the distributable Skill. It excludes
screenshots, APKs and model weights. No physical device is available; 9207
and 4090 inventories contain emulators. Memory ON/OFF timing remains pending.

Live evidence under `data/runtime/validation/20260907-two-tools-live*` records
canonical 1024D OmniTransfer recall of the explicit Bluetooth Function on
9207/emulator-45562 with OOB 0.6.1 (versionCode 7). The first execution exposed
the missing inventory; subsequent runs encountered SSH timeouts. Preserve these
failed attempts as integration evidence, not benchmark success or latency.

## 1.1.0.dev0 — host protocol refactor, 2026-09-07

Observation evidence now uses immutable, content-addressed PNG files within
each existing bundle. Host and recorder share one capture; the post-action
Fast Pass reads the recorder's canonical XML rather than an undefined Host
attribute. The observation reporting index holds compact references. Raw
image resolution and the RunLog schema are unchanged.

The outer Planner loop and one-invocation execution now have separate entry
methods and reuse the same Function kernel. Completion checks, cooperative
cancellation, shared deadlines, Router iteration limits, and no-progress
stops use one control contract. External Function failure returns execution
facts and the current observation without starting another Planner.

The MCP stdio adapter and distributable `omniflow-gui` Skill expose canonical
tools, explicit Memory, session identity, request deduplication, cancellation,
and completion. Kernel import no longer loads AndroidWorld's experiment file.
An isolated wheel installation outside the checkout can import the kernel,
construct the MCP server, discover tools, and find the packaged Skill/schema.
No screenshots, APKs, or model weights are included in that wheel.

Focused lifecycle/protocol/packaging regression: 28 passed. The full repository
suite passes **131 tests** with `python -m pytest -q tests`; bare pytest additionally collects
an unrelated, untracked AutoDroid vendor RL suite whose optional `gym` dependency
is absent. Transport checks include a real MCP stdio subprocess, but its test
device is deliberately synthetic and does not validate Android execution.

All device behavior remains **待真机验证**. Local devices were disconnected at
the final inventory; 4090 and 9207 inventories contained emulators only. No
physical-device acceptance or comparative speed/RSS result is claimed. The
protocol is cooperative, has no durable public resume API, and does not yet
provide an HTTP service. Details: [protocol](HARNESS_PROTOCOL.md),
[MCP installation](HARNESS_MCP.md).

Local historical archive cleanup removed approximately 16.72 GB after protecting
731 dependency/unique-success files. The deduplicated deletion journal and audit
remain under `data/runtime/archive_cleanup/20260907/`, outside Git releases.

Follow-up cleanup removed the entire remaining local `.archive` directory
(790 files, 11.41 MB). The only live reference was an obsolete registry search
root, which was removed. The previously presumed unique source was proven
equivalent to its ordinary-directory copy after path normalization, with
byte-identical screenshots; ordinary evidence was unchanged. Final audit:
`data/runtime/archive_cleanup/20260907-final/`.

A subsequent live OOB protocol smoke passed on 9207 / emulator-45562 with OOB
0.6.1 (versionCode 7): capture, one wait action, request deduplication without
additional device I/O, post-cancel rejection, and session restart. This remains
supplemental emulator evidence; it does not validate physical-device behavior
or comparative performance.

## 1.0.0 — 2026-09-07

Preserved the pre-refactor working version, including Function semantic
authoring, shared Checker recovery, GUI-agent adapters, and runtime timing.
The imported working-tree schema snapshot is kept in Git history. Its initial
validation had 108 passing and 3 failing tests: the action and RunLog mirrors
did not match the existing swipe and compact-observation consumers. Release
preparation reconciled those mirrors with the established runtime contract and
made external scroll adapters emit canonical endpoints. The complete existing
suite plus the GUI-Owl swipe regression passes: **112 tests**. `git diff --check`
also passes. The raw imported changes remain available in earlier commits.

This release records a code baseline; device behavior remains **待真机验证**.
The local ADB inventory contained only emulator-5560. Historical physical-phone
measurements do not validate the current changes.

Experiment assets, source/target RunLogs, credentials, APKs, model weights,
paper build products, and temporary troubleshooting files are not Git release
contents. Reusable Memory remains an explicitly supplied external asset.
