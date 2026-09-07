# Releases

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
