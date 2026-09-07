# Releases

## Harness refactor — in progress

Observation evidence now uses immutable, content-addressed PNG files within
each existing bundle. Host and recorder share one capture; the post-action
Fast Pass reads the recorder's canonical XML rather than an undefined Host
attribute. The observation reporting index holds compact references. Raw
image resolution and the RunLog schema are unchanged. The evidence, Checker,
completion, and AppAgent focused regressions pass (33 tests); **待真机验证**.

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
