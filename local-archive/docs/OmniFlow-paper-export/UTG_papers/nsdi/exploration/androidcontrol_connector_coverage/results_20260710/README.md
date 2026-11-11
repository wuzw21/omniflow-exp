# AndroidControl-Curated Connector Coverage Exploration

Status: exploration-only, not frozen paper result.
Generated: 2026-07-10T12:25:22.778981+00:00

## Scope

This audit analyzes the local AndroidControl-Curated snapshot under
`/Users/wuzewen/Projects/Omni/android-control`.  The primary workload is
`android_control_high_bbox.json` because it preserves grouped task images, task
instructions, history, step instructions, and action labels.
This is not the original Google AndroidControl release; keep the
curated-dataset provenance explicit if the audit is later cited.

The script reports two levels of evidence:

- `catalog match`: an app name extracted from a trajectory has a matching
  Composio or OpenConnector surface.
- `strict API candidate`: a conservative automatic candidate where the app
  is API-rich and the task text contains an object-mutation verb.  These
  rows still require manual adjudication before entering the paper.

## Primary Workload Summary

- Trajectories / grouped tasks: 1543
- Steps: 8435
- Average steps per trajectory: 5.467
- Extracted unique apps: 528
- Unknown-app trajectories: 300 / 1543 = 19.4%

| Coverage label | Trajectories | Rate | Meaning |
|---|---:|---:|---|
| direct catalog match | 92 | 6.0% | A non-empty connector exists for the extracted app name or alias. |
| empty catalog match | 56 | 3.6% | A toolkit/provider name exists but exposes no actions/tools. |
| no catalog match | 1395 | 90.4% | No matching connector surface was found automatically. |
| strict API candidate | 46 | 3.0% | Conservative candidate for same-effect API completion; manual review required. |

Interpretation: catalog matches are much broader than true replacement.
Most AndroidControl-Curated samples still consist of GUI-local clicks,
swipes, searches, shopping flows, drawing/content apps, or mobile app
states that a connector cannot safely certify without replaying the GUI.

## Task Categories

| Category | Trajectories |
|---|---:|
| media_or_content | 442 |
| other | 435 |
| email_or_share | 169 |
| shopping_or_cart | 151 |
| calendar_or_reminder | 105 |
| file_or_document | 85 |
| local_device_or_utility | 79 |
| travel_or_maps | 77 |

## Top Extracted Apps

| App | Trajectories | Best catalog label | Composio | OpenConnector |
|---|---:|---|---|---|
| <unknown> | 300 | none | - (0) | - (0) |
| Artsy | 34 | none | - (0) | - (0) |
| Pinterest | 33 | empty | pinterest (0) | - (0) |
| Gmail | 23 | direct | gmail (63) | gmail (46) |
| Artier | 22 | none | - (0) | - (0) |
| Kitchen Stories | 21 | none | - (0) | - (0) |
| Art & Culture | 19 | none | - (0) | - (0) |
| Google News | 17 | none | - (0) | - (0) |
| Phases of the Moon | 17 | none | - (0) | - (0) |
| DeviantArt | 16 | none | - (0) | - (0) |
| Flipboard | 15 | none | - (0) | - (0) |
| Maps | 14 | direct | google_maps (23) | - (0) |
| eBay | 14 | none | - (0) | - (0) |
| Amazon | 13 | empty | amazon (0) | - (0) |
| Arts & Culture | 13 | none | - (0) | - (0) |
| CNN News | 13 | none | - (0) | - (0) |
| Guardian | 13 | none | - (0) | - (0) |
| Stellarium | 13 | none | - (0) | - (0) |
| Clock | 12 | none | - (0) | - (0) |
| Flipkart | 12 | none | - (0) | - (0) |

## Largest No-Match Apps

| App | Trajectories | Dominant categories |
|---|---:|---|
| Artsy | 34 | media_or_content:18;other:10;email_or_share:6 |
| Artier | 22 | media_or_content:12;other:5;email_or_share:4;shopping_or_cart:1 |
| Kitchen Stories | 21 | media_or_content:14;other:7 |
| Art & Culture | 19 | media_or_content:16;email_or_share:3 |
| Google News | 17 | media_or_content:13;email_or_share:3;travel_or_maps:1 |
| Phases of the Moon | 17 | other:9;calendar_or_reminder:6;email_or_share:1;local_device_or_utility:1 |
| DeviantArt | 16 | media_or_content:12;email_or_share:3;shopping_or_cart:1 |
| Flipboard | 15 | media_or_content:11;other:2;email_or_share:2 |
| eBay | 14 | shopping_or_cart:6;other:6;email_or_share:2 |
| Arts & Culture | 13 | media_or_content:9;other:2;email_or_share:1;file_or_document:1 |
| CNN News | 13 | media_or_content:11;email_or_share:2 |
| Guardian | 13 | media_or_content:10;email_or_share:2;other:1 |
| Stellarium | 13 | other:13 |
| Clock | 12 | local_device_or_utility:11;calendar_or_reminder:1 |
| Flipkart | 12 | other:8;shopping_or_cart:3;media_or_content:1 |
| Calculator Unit Converter | 11 | local_device_or_utility:11 |
| Hindu | 11 | media_or_content:7;email_or_share:3;other:1 |
| Moon Phase | 11 | other:5;calendar_or_reminder:4;local_device_or_utility:1;email_or_share:1 |
| SmartNews | 11 | media_or_content:9;email_or_share:2 |
| BBC News | 10 | media_or_content:8;email_or_share:2 |

## Example Strict API Candidates

These are not final coverage labels; they are candidates for manual
review because the task might be expressible by a cloud/API connector.

| Trajectory | App | Connector | Task |
|---|---|---|---|
| 10464 | Gmail | gmail | My wife is interested in the details of the moon phases and she asked me to share these moon phase details, so share all the details of moon phase to her at clarawagner98b@gmail.co |
| 10558 | Gmail | gmail | I want to share a recording of Grocery Shopping Updates with my sister at maredharris@gmail.com via the Gmail app. |
| 10590 | Gmail | gmail | I'd want to share my created art to my friend Rick at rick.weinstein@dayrep.com so that I can obtain feedback on it. |
| 11520 | Drive | googledrive | I wanted to create a new folder named My Collections in the Drive app. |
| 127 | Contacts | googlecontacts | Delete an existing contact from Contacts+ app. |
| 12709 | YouTube | youtube | I want to create a playlist named My Favorite Videos in the YouTube app and add The Lives of Others official trailer video to this playlist. |
| 13089 | Calendar | googlecalendar | In the calendar app, delete the casual meeting because it has been cancelled. |
| 14129 | Drive | googledrive | I want to download this uploaded file germanycountryst00sols.pdf to save a copy of this file in my phone's storage, so download the germanycountryst00sols.pdf file in the Drive app |
| 14212 | Gmail | gmail | My friend Roger requested the Simple & Tasty recipe, so I'd like to share the recipe for Uramaki sushi with him at roger.norred@teleworm.us via Gmail. |
| 14639 | Contacts | googlecontacts | I want to know how many email contacts are in my contact list, so view email contacts of my contact list in the Contacts app. |
| 14928 | Contacts | googlecontacts | I forgot to add many details about my wife like surname and email id  while saving her contact in my contact list, so edit this contact and mention full name Clara Wagner, add emai |
| 15020 | Gmail | gmail | In the Sound Recorder Plus app, I want to Share a recording Titled meditation thoughts  with Coledbwpersonatest@gmail.com via the Gmail app |
| 15241 | Drive | googledrive | My phone is not functioning properly, therefore I'm going to turn it on service. Upload the DIY project file to Google Drive to save it. |
| 15244 | Calendar | googlecalendar | We as a team have discussed all the RCA issues and doubts have been cleared. So there will be no need for the RCA meeting. Delete the RCA Meet event in the calendar app. |
| 15438 | Calendar | googlecalendar | I have received a message from the team regarding the cancellation of the tennis tournament, so delete the event. |
| 15968 | Calendar | googlecalendar | As the wedding dance rehearsal is over, I want to delete the event from the Calendar app. |
| 16324 | Calendar | googlecalendar | There is something wrong with creating an event, so I want to delete it. |
| 16837 | OneDrive | one_drive | To make seeing the agents.txt file easier for me in the future, upload it to the OneDrive app. |
| 17334 | Gmail | gmail | Natalie Larson has emailed me, thank you so, please reply, welcome to her on the Gmail App. |
| 1736 | Gmail | gmail | My wife is interested in the details of the moon phases, and she asked me to share these moon phase details, so share all the details of the moon phase with her at clarawagner98b@g |

## Suggested Paper Use

Use this only as an exploration result until registered in
`UTG_papers/nsdi/eval_matrix.md`.  The defensible claim is narrow:
public connectors cover some API-rich cloud apps, but AndroidControl-
Curated contains many real mobile GUI workflows where a GUI-derived
`Function` remains the relevant reusable asset.

## Output Files

- `summary.json`: machine-readable aggregate summary.
- `task_coverage_primary.csv`: one row per primary benchmark trajectory.
- `app_coverage_primary.csv`: app-level rollup for the primary benchmark.
- `task_coverage_all.csv`: all benchmark files combined.
