# AndroidWorld Connector Coverage Exploration

Status: exploration-only, not frozen paper result.
Date: 2026-07-10

This note answers one question: if we take an existing mobile GUI benchmark
instead of a hand-picked Top-50 app list, how many tasks can be expressed by
public app APIs from Composio or OpenConnector?

The short answer is conservative and useful for the paper: on the strict
AndroidWorld terminal-state criterion, current public connector catalogs do not
directly replace GUI execution for this workload.  Some tasks have cloud-service
analogues, but they do not operate on the same local Android app state that
AndroidWorld validates.

## Dataset Choice

Use AndroidWorld as the dataset for this audit.

Why this dataset is better than the previous Top-50 app list:

- It is a public, citable mobile GUI benchmark, not a manually defined app
  ranking.
- It is already in the OmniFlow evaluation path, so the audit aligns with the
  current paper workload.
- It contains executable goals, local app state, and official validators,
  making "API coverage" falsifiable: an API-only solution must create the same
  terminal state that AndroidWorld validates.
- It covers real mobile app categories that connector catalogs usually miss:
  system settings, local files, camera/audio hardware, local notes, SMS, local
  calendar, contacts, recipes, and expense logging.

Local data sources inspected:

- AndroidWorld task notes:
  `/Users/wuzewen/Projects/Omni/OmniFlow/androidworld.md`
- Curated successful source-runlog index:
  `/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/core_archive/success_source_runlogs/index_by_task.json`
- Latest result summary archive:
  `/Users/wuzewen/Projects/Omni/OmniFlow/runtime/evals/androidworld_validator/core_archive/latest_results_summary/`
- Upstream AndroidWorld code:
  `/Users/wuzewen/Projects/Omni/OmniFlow/runtime/external/droidrun-android-world/android_world/`

Public references:

- AndroidWorld: https://github.com/google-research/android_world
- Composio: https://github.com/ComposioHQ/composio
- OpenConnector: https://github.com/oomol-lab/open-connector

## Connector Catalogs

Composio catalog snapshot:

- Local repo: `/tmp/composio-research`
- Branch / commit: `next` / `30ba4d2`
- Catalog file: `docs/public/data/toolkits-list.json`
- Size: 1403 toolkits, 44602 tools
- Tool-count distribution: median 10, mean 31.79 tools/toolkit

OpenConnector catalog snapshot:

- Local repo: `/tmp/open-connector-research`
- Branch / commit: `main` / `f9f1647`
- Catalog path: `catalog/apps/*.json`
- Size: 1044 apps, 10090 actions
- Action-count distribution: median 6, mean 9.66 actions/app

Relevant connector examples found:

| Domain | Composio | OpenConnector |
|---|---|---|
| Calendar | `googlecalendar` 49 tools | `googlecalendar` 37 actions |
| Contacts | `googlecontacts` 24 tools | no write-capable Google Contacts provider found; contact enrichment providers are not task-equivalent |
| SMS | `sms_alert` 23, `team_sms` 3, `twilio` 1 | `sms_alert` 6, `twilio` 5 |
| Browser automation | `browserbase_tool` 19, `browser_tool` 5, `browserless` 7 | `browserbase` 11, `browserless` 3, `cloudflare_browser_rendering` 6 |
| Files/cloud storage | `googledrive` 90, `one_drive` 83, `files_com` 7 | `googledrive` 43, `one_drive` 13, `files_com` 5 |

These are related domains, not strict AndroidWorld coverage.  For example,
Google Calendar does not directly mutate Simple Calendar Pro's local app state;
Twilio does not send an SMS through Simple SMS Messenger on the emulator; and
Google Drive does not delete a file from the Android local filesystem.

## Coverage Definition

This audit uses two coverage notions.

`strict API coverage`: a connector can complete the task without GUI execution
and produce the same AndroidWorld validator-visible state in the same local app
or device subsystem.

`related-domain coverage`: a connector exists for an analogous cloud service or
subproblem, but it does not guarantee the AndroidWorld terminal state.  This is
an upper bound on what API connectors might replace after rewriting the task
environment.

The strict metric is the one that matters for the paper claim.  The related
metric is useful only for explaining why public APIs help some app classes but
do not remove the need for GUI Function recovery.

## Source-Runlog Workload Summary

The curated source-runlog index contains 51 AndroidWorld tasks with successful
source traces.  Total source steps are 447, with 8.76 steps per task on average.

| App / domain group | Tasks | Composio related-domain | OpenConnector related-domain | Strict API coverage | Reason |
|---|---:|---:|---:|---:|---|
| Android system settings | 12 | 0 | 0 | 0 | WiFi, Bluetooth, brightness, clipboard, and open-app state are local device state. |
| Audio Recorder | 1 | 0 | 0 | 0 | Requires local microphone and Audio Recorder app state. |
| Broccoli Recipe | 4 | 0 | 0 | 0 | Local recipe app; no public connector for the app database. |
| Browser / local HTML | 2 | 2 | 2 | 0 | Browser automation exists, but the task uses a local file opened through Android Files/Chrome. |
| Camera | 2 | 0 | 0 | 0 | Requires local camera hardware and media capture. |
| Clock | 3 | 0 | 0 | 0 | Local Clock stopwatch/timer UI state. |
| Contacts | 2 | 2 | 0 | 0 | Google Contacts is analogous, but AndroidWorld validates local contacts/draft state. |
| Files / local storage | 1 | 1 | 1 | 0 | Cloud file APIs are analogous, but the task edits Android local storage. |
| Markor | 7 | 0 | 0 | 0 | Local Markdown files through Markor; no connector for Markor app state. |
| Markor + Simple SMS | 1 | 0 | 0 | 0 | Composite local note creation plus SMS; no whole-task connector. |
| Pro Expense | 3 | 0 | 0 | 0 | Local expense app; no connector for the app database. |
| Settings / open app | 1 | 0 | 0 | 0 | Permission dialogs and foreground app state require device UI. |
| Simple Calendar Pro | 6 | 6 | 6 | 0 | Google Calendar is analogous, but not the local Simple Calendar Pro state. |
| Simple SMS Messenger | 6 | 6 | 6 | 0 | SMS APIs are analogous, but not the local Simple SMS Messenger/provider state. |

Aggregate coverage on the 51-task source-runlog workload:

| Catalog | Strict API coverage | Related-domain upper bound | No related connector or blocked by local/device state |
|---|---:|---:|---:|
| Composio | 0/51 = 0.0% | 17/51 = 33.3% | 34/51 = 66.7% |
| OpenConnector | 0/51 = 0.0% | 15/51 = 29.4% | 36/51 = 70.6% |

## Latest-Result Workload Summary

The latest result summary archive contains 72 latest rows across 38 tasks:

- `omniflow-native`: 32 rows
- `mobilegpt-self`: 30 rows
- `autodroid`: 9 rows
- `mobilegpt`: 1 row

The task-level connector coverage is similar:

| Catalog | Strict API coverage | Related-domain upper bound | No related connector or blocked by local/device state |
|---|---:|---:|---:|
| Composio | 0/38 = 0.0% | 12/38 = 31.6% | 26/38 = 68.4% |
| OpenConnector | 0/38 = 0.0% | 10/38 = 26.3% | 28/38 = 73.7% |

## Interpretation

The result supports a narrow and defensible argument:

Public API connectors are valuable for SaaS and cloud-account workflows, but
they are not a substitute for GUI-level Function recovery on AndroidWorld-style
mobile tasks.  In this workload, the task success condition is tied to local
device state, app-specific local databases, permission dialogs, and hardware
interactions.  Existing connector catalogs mostly expose cloud services or
backend APIs, so even their related-domain tools do not directly pass the
AndroidWorld validator.

This gives OmniFlow a clearer role: it does not compete with cloud APIs where a
developer-provided API already exists.  It synthesizes cacheable `Function`
assets from successful GUI executions for the large class of mobile workflows
where no suitable public API exists or where an API would not modify the same
user-facing device state.

## Suggested Paper Use

Do not write the above numbers into the paper until this audit is registered in
`UTG_papers/nsdi/eval_matrix.md`.

Safe story-outline wording:

> We audited the AndroidWorld task set against two large public connector
> catalogs.  Under a strict terminal-state criterion, existing cloud/API
> connectors do not directly cover the benchmark's local mobile tasks; at best,
> roughly one third have an analogous cloud-service connector that would require
> rewriting the task environment.  This motivates treating GUI-derived
> `Function` assets as a complementary API surface for mobile-only and
> device-local workflows.

If promoted to evaluation, the protocol should freeze:

1. Dataset split: curated source-runlog 51 tasks, latest-result 38 tasks, or the
   full upstream AndroidWorld registry.
2. Strict coverage rule: same app/device terminal state, not merely same
   semantic domain.
3. Manual adjudication file: task -> connector label -> reason.
4. Reproducible catalog snapshot: connector repo commit and generated catalog
   file hash.
